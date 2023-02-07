#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_string.h>
#include <npp.h>
#include <string.h>
#include <tgmath.h>

#include <filesystem>
#include <iostream>

#include "FreeImage.h"

void drawLine(FIBITMAP *inputImage, Npp32f rho, Npp32f theta, Npp32f nDeltaRho) {
    if ((rho == 0.0 && theta == 0.0) || std::isnan(rho) || std::isnan(theta)) {
        return;
    }
    RGBQUAD color = {255, 0, 0};

    int height = FreeImage_GetHeight(inputImage);
    int width = FreeImage_GetWidth(inputImage);

    Npp32f nHough = ((sqrt(2.0F) * static_cast<Npp32f>(fmax(height, width))) / 2.0F);
    int nAccumulatorsHeight = nDeltaRho > 1.0F ? static_cast<int>(ceil(nHough * 2.0F))
                                               : static_cast<int>(ceil((nHough * 2.0F) / nDeltaRho));
    int nCenterX = width >> 1;
    int nCenterY = height >> 1;
    Npp32f nThetaRad = static_cast<Npp32f>(theta) * 0.0174532925199433F;
    Npp32f nSinTheta = sin(nThetaRad);
    Npp32f nCosTheta = cos(nThetaRad);
    int x, y;
    if (theta >= 45 && theta <= 135) {
        for (x = 0; x < width; ++x) {
            int x1 = width - x - 1;
            y = static_cast<int>((static_cast<Npp32f>(rho - (nAccumulatorsHeight >> 1)) -
                                  ((x1 - nCenterX) * nCosTheta)) /
                                     nSinTheta +
                                 nCenterY);
            FreeImage_SetPixelColor(inputImage, x1, height - y - 1, &color);
        }
    } else {
        for (y = 0; y < height; ++y) {
            int y1 = height - y - 1;
            x = static_cast<int>((static_cast<Npp32f>(rho - (nAccumulatorsHeight >> 1)) -
                                  ((y1 - nCenterY) * nSinTheta)) /
                                     nCosTheta +
                                 nCenterX);
            FreeImage_SetPixelColor(inputImage, width - x - 1, y1, &color);
        }
    }
}

std::string getOutputFileName(std::string inputFileName) {
    int fileNameStartPos = inputFileName.rfind('/') + 1;
    int fileNameEndPos = inputFileName.rfind('.');
    return "output/" + inputFileName.substr(fileNameStartPos, fileNameEndPos - fileNameStartPos) + "_output.jpg";
}

void applyHoughTransform(std::string sFilename) {
    try {
        std::string sResultFilename = getOutputFileName(sFilename);
        npp::ImageCPU_8u_C1 oHostSrc;
        FIBITMAP *pSourceBitmap = npp::loadImage(sFilename);
        npp::convertImageToNpp(pSourceBitmap, oHostSrc);
        npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

        NppiSize oSrcSize = {(int)oDeviceSrc.size().nWidth - 4, (int)oDeviceSrc.size().nHeight - 4};

        // Gausian blur with 5x5 filter
        npp::ImageNPP_8u_C1 oDeviceDenoise(oSrcSize.width, oSrcSize.height);

        NPP_CHECK_NPP(nppiFilterGauss_8u_C1R(
            oDeviceSrc.data(2, 2), oDeviceSrc.pitch(), oDeviceDenoise.data(), oDeviceDenoise.pitch(),
            oSrcSize, NPP_MASK_SIZE_5_X_5));

        // Found mean pixel color to adjust canny edge detection threshold
        int nMeanBufferSize = 0;
        Npp8u *pMeanBufferNpp = 0;
        Npp64f *oDeviceMeanStdDev = 0;
        Npp64f oHostMean, oHostStdDev;
        NPP_CHECK_NPP(nppiMeanStdDevGetBufferHostSize_8u_C1R(oSrcSize, &nMeanBufferSize));

        NPP_CHECK_CUDA(cudaMalloc((void **)&pMeanBufferNpp, nMeanBufferSize));
        NPP_CHECK_CUDA(cudaMalloc((void **)&oDeviceMeanStdDev, sizeof(Npp64f) * 2));

        NPP_CHECK_NPP(nppiMean_StdDev_8u_C1R(
            oDeviceDenoise.data(), oDeviceDenoise.pitch(), oSrcSize, pMeanBufferNpp,
            &oDeviceMeanStdDev[0], &oDeviceMeanStdDev[1]));

        NPP_CHECK_CUDA(cudaMemcpy(&oHostMean, &oDeviceMeanStdDev[0], sizeof(Npp64f), cudaMemcpyDeviceToHost));
        NPP_CHECK_CUDA(cudaMemcpy(&oHostStdDev, &oDeviceMeanStdDev[1], sizeof(Npp64f), cudaMemcpyDeviceToHost));

        // Apply canny edge detection
        int nCannyBufferSize = 0;
        Npp8u *pCannyBuffer = 0;
        NPP_CHECK_NPP(nppiFilterCannyBorderGetBufferSize(oSrcSize, &nCannyBufferSize));

        NPP_CHECK_CUDA(cudaMalloc((void **)&pCannyBuffer, nCannyBufferSize));

        NppiPoint cannyOffset = {0, 0};
        npp::ImageNPP_8u_C1 oDeviceCannyEdge(oDeviceDenoise.size());

        Npp16s nLowThreshold = fmax(62, ceil(oHostMean - oHostStdDev * 0.7));
        Npp16s nHighThreshold = fmin(255, ceil(oHostMean + oHostStdDev * 0.7));
        NPP_CHECK_NPP(nppiFilterCannyBorder_8u_C1R(
            oDeviceDenoise.data(), oDeviceDenoise.pitch(), oSrcSize, cannyOffset,
            oDeviceCannyEdge.data(), oDeviceCannyEdge.pitch(), oSrcSize, NPP_FILTER_SOBEL,
            NPP_MASK_SIZE_3_X_3, nLowThreshold, nHighThreshold, nppiNormL2,
            NPP_BORDER_REPLICATE, pCannyBuffer));

        // Apply hough transform and fing parameters of lines
        int hHoughBufferSize = 0;
        Npp8u *pHoughBuffer = 0;

        int nMaxLineCount = 100;
        NppPointPolar *pDeviceLines = 0;
        int nThreshold = fmin(oSrcSize.height, oSrcSize.width) * 0.1;
        NPP_CHECK_CUDA(cudaMalloc((void **)&pDeviceLines, sizeof(NppPointPolar) * nMaxLineCount));
        NppPointPolar nDelta = {1, 1};

        NPP_CHECK_NPP(nppiFilterHoughLineGetBufferSize(oSrcSize, nDelta, nMaxLineCount, &hHoughBufferSize));

        NPP_CHECK_CUDA(cudaMalloc((void **)&pHoughBuffer, hHoughBufferSize));

        int *pDeviceLineCount;
        NPP_CHECK_CUDA(cudaMalloc((void **)&pDeviceLineCount, sizeof(int)));

        NPP_CHECK_NPP(nppiFilterHoughLine_8u32f_C1R(
            oDeviceCannyEdge.data(), oDeviceCannyEdge.pitch(), oSrcSize, nDelta,
            nThreshold, pDeviceLines, nMaxLineCount, pDeviceLineCount, pHoughBuffer));

        NppPointPolar *pHostLines = 0;
        NPP_CHECK_CUDA(cudaMallocHost((void **)&pHostLines, sizeof(NppPointPolar) * nMaxLineCount));
        NPP_CHECK_CUDA(cudaMemcpy(pHostLines, pDeviceLines, sizeof(NppPointPolar) * nMaxLineCount, cudaMemcpyDeviceToHost));

        if (FreeImage_GetColorType(pSourceBitmap) != FIC_RGB) {
            FIBITMAP *pGrayBitmap = pSourceBitmap;
            pSourceBitmap = FreeImage_Allocate(FreeImage_GetWidth(pGrayBitmap), FreeImage_GetHeight(pGrayBitmap), 24);
            for (int i = 0; i < FreeImage_GetWidth(pGrayBitmap); ++i) {
                for (int j = 0; j < FreeImage_GetHeight(pGrayBitmap); ++j) {
                    RGBQUAD color;
                    BYTE gray_color;
                    FreeImage_GetPixelIndex(pGrayBitmap, i, j, &gray_color);
                    color.rgbRed = gray_color;
                    color.rgbGreen = gray_color;
                    color.rgbBlue = gray_color;
                    FreeImage_SetPixelColor(pSourceBitmap, i, j, &color);
                }
            }
        }

        // Draw found lines on source image
        for (int i = 0; i < nMaxLineCount; ++i) {
            // printf("%f, %f\n", pHostLines[i].rho, pHostLines[i].theta);
            drawLine(pSourceBitmap, pHostLines[i].rho, pHostLines[i].theta, nDelta.rho);
        }

        nppiFree(oDeviceSrc.data());
        nppiFree(oDeviceDenoise.data());
        nppiFree(oDeviceCannyEdge.data());
        NPP_CHECK_CUDA(cudaFree(pMeanBufferNpp));
        NPP_CHECK_CUDA(cudaFree(oDeviceMeanStdDev));
        NPP_CHECK_CUDA(cudaFree(pCannyBuffer));
        NPP_CHECK_CUDA(cudaFreeHost(pHostLines));
        NPP_CHECK_CUDA(cudaFree(pDeviceLineCount));
        NPP_CHECK_CUDA(cudaFree(pDeviceLines));

        // Save the result
        npp::saveImage(sResultFilename.c_str(), pSourceBitmap);
        std::cout << "Saved image file " << sResultFilename << std::endl;
        FreeImage_Unload(pSourceBitmap);
    } catch (npp::Exception &rException) {
        std::cerr << "Program error wile processing " << sFilename << "!\n";
        std::cerr << "The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;
        exit(EXIT_FAILURE);
    } catch (...) {
        std::cerr << "Program error wile processing " << sFilename << "!\n";
        std::cerr << "An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {
    findCudaDevice(argc, (const char **)argv);
    std::filesystem::create_directory("output/");

    for (int imageIdx = 1; imageIdx < argc; ++imageIdx) {
        std::string sFilename(argv[imageIdx]);
        applyHoughTransform(sFilename);
    }
    exit(EXIT_SUCCESS);
}
