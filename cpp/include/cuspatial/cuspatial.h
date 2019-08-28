/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

namespace cuspatial
{
	template<typename T>
	struct location_3d
	{
		T latitude;
		T longitude;
		T altitude;
	};

	template<typename T>
	struct coord_2d
	{
		T x;
		T y;
	};

	struct its_timestamp
	{
		uint32_t y : 6;
		uint32_t m : 4;
		uint32_t d : 5;
		uint32_t hh : 5;
		uint32_t mm : 6;
		uint32_t ss : 6;
		uint32_t wd: 3;
		uint32_t yd: 9;
		uint32_t ms: 10;
		uint32_t pid:10;
	};
}// namespace cuspatial