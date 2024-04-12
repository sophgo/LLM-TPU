/*****************************************************************************
 *
 *    Copyright (c) 2016-2026 by Sophgo Technologies Inc. All rights reserved.
 *
 *    The material in this file is confidential and contains trade secrets
 *    of Sophgo Technologies Inc. This is proprietary information owned by
 *    Sophgo Technologies Inc. No part of this work may be disclosed,
 *    reproduced, copied, transmitted, or used in any way for any purpose,
 *    without the express written permission of Sophgo Technologies Inc.
 *
 *****************************************************************************/

inline uint16_t fp32_to_fp16_bits(float f)
{
	uint32_t x = *((uint32_t*)&f);
	uint16_t h = ((x>>16)&0x8000)|((((x&0x7f800000)-0x38000000)>>13)&0x7c00)|((x>>13)&0x03ff);

	return h;
}

inline uint16_t fp32_to_bf16_bits(float f)
{
	uint32_t x = *((uint32_t*)&f);
	uint16_t h = (x>>16);

	return h;
}