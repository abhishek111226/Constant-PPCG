#ifndef _CUDA_H
#define _CUDA_H

#include "ppcg_options.h"
#include "ppcg.h"

int generate_cuda(isl_ctx *ctx, struct ppcg_options *options,
	const char *input);

#endif
__isl_give isl_printer *print_cuda_macros(__isl_take isl_printer *p);
