/*
 * main.c
 *
 *  Created on: 27 Jan 2010
 *      Author: sjf65
 */

#include <getopt.h>

#include "base.h"
#include "image.h"
#include "util.h"
#include "bayer.h"

#include "bilinear.h"

#include "ahd.h"
#include "ahd_mask.h"
#include "ahd_mask2.h"
#include "ahd_mask3.h"
#include "ahd_mask4.h"

#include "bilinear.cuh"
#include "cuda_utils.h"
#include "ahd.cuh"


settings_t _settings;
settings_t *settings = &_settings;

void usage(char *arg0) {
    printf("Usage: %s [-c] [-t] file\n"
            "    -c  Use CUDA device\n"
            "    -l  Use bilinear interpolation (default is AHD)\n"
            "    -m  Use AHD mask interpolation (default is AHD)\n"
            "    -j  Use AHD mask v2 interpolation (default is AHD)\n"
            "    -t <N> Run N tests\n"
            "    -n  No intermediate results\n"
            "    -b <1|2> Set the AHD homogenity neighbourhood size (AHD)\n"
            "    -a [1-3] The number of artefact removal steps (AHD and AHD mask)\n"
            "    -d n The number of dilation steps (AHD mask)\n"
            "    -e [1-512] Edge detection threshold (AHD mask)\n"
            "    -o print compile time options and exit\n"
            ,arg0);
}

void print_options(){
    printf("Compile time options: ");
#ifdef _TEST
    printf(" _TEST ");
    printf(" -O3 ");
#endif
#ifdef __CUDAEMU__
    printf(" emulation ");
#endif
#ifdef _DEBUG_
    printf(" _DEBUG ");
#endif
    printf("\n");
}

int main(int argc, char **argv) {
    if (argc == 1) {
        usage(argv[0]);
        exit(1);
    }
    init_settings(settings);

    int c;
    while ((c = getopt(argc, argv, "lct:nb:a:m:jod:e:")) != -1) {
        switch (c) {
        case 'l':
            settings->use_ahd = 0;
            break;
        case 'c':
            settings->use_cuda = 1;
            break;
        case 't':
            settings->run_tests = atoi(optarg);
            assert(settings->run_tests > 0);
            break;
        case 'n':
            settings->save_temps = 0;
            break;
        case 'b':
            settings->ball_distance = atoi(optarg);
            if (settings->ball_distance < 1 || settings->ball_distance > 2) {
                usage(argv[0]);
                exit(1);
            }
            break;
        case 'a':
            settings->median_filter_iterations = atoi(optarg);
            assert(settings->median_filter_iterations > 0);
            assert(settings->median_filter_iterations <= 3);
            break;
        case 'm':
            settings->ahd_mask = atoi(optarg);
            assert(settings->ahd_mask > 0);
            break;
        case 'o':
            print_options();
            exit(0);
            break;
        case 'd':
            settings->dilations = atoi(optarg);
            assert(settings->dilations >= 0);
            break;
        case 'e':
            settings->edge_threshold = atoi(optarg);
            assert(settings->edge_threshold > 1);
            break;
        case '?':
            // Opt arg error
            usage(argv[0]);
            exit(1);
        }
    }
    if (optind == argc) {
        usage(argv[0]);
        exit(1);
    }

    void (*interpolation_func)(img *) = NULL;

    void (*mask_funcs[])(img *) = {mask_ahd, mask_ahd2, mask_ahd3, mask_ahd3b};

    if (settings->use_ahd && !settings->use_cuda) {
        if (settings->ahd_mask) {
            assert(settings->ahd_mask <= sizeof(mask_funcs)/4);
            interpolation_func = mask_funcs[settings->ahd_mask-1];
        } else {
            interpolation_func = host_ahd;
        }
    } else if (settings->use_ahd && settings->use_cuda) {
        interpolation_func = cuda_ahd;
    } else if (!settings->use_ahd && !settings->use_cuda) {
        interpolation_func = host_bilinear;
    } else {
        interpolation_func = cuda_bilinear;
    }


    for (int i = optind; i < argc; i++) {
        char *file = argv[i];
        init_image_settings(file);

        img *image = read_image_from_file(file);
        Info("%s bayer type: %s", settings->image_name,
                get_bayer_name(settings->bayer_type));

        if (settings->use_cuda){
            if (settings->bayer_type != RGGB) {
                FatalError("Bayer type not support with CUDA");
            }
            cudaInit();

        }
        uint runs = MAX(settings->run_tests,1);
        while (runs--) {
            startTimer(main);
            interpolation_func(image);
            stopTimer(main,image->width,image->height);
        }

        write_result_to_file(file,image);
        free_image(image);
    }
}
