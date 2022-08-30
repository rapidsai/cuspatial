/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

namespace cuspatial {
namespace detail {
namespace utility {

namespace {
// adapted from https://dl.acm.org/doi/10.1109/TC.2007.70814
static __device__ __constant__ uint16_t dilate_table[256] = {
  0x0000, 0x0001, 0x0004, 0x0005, 0x0010, 0x0011, 0x0014, 0x0015, 0x0040, 0x0041, 0x0044, 0x0045,
  0x0050, 0x0051, 0x0054, 0x0055, 0x0100, 0x0101, 0x0104, 0x0105, 0x0110, 0x0111, 0x0114, 0x0115,
  0x0140, 0x0141, 0x0144, 0x0145, 0x0150, 0x0151, 0x0154, 0x0155, 0x0400, 0x0401, 0x0404, 0x0405,
  0x0410, 0x0411, 0x0414, 0x0415, 0x0440, 0x0441, 0x0444, 0x0445, 0x0450, 0x0451, 0x0454, 0x0455,
  0x0500, 0x0501, 0x0504, 0x0505, 0x0510, 0x0511, 0x0514, 0x0515, 0x0540, 0x0541, 0x0544, 0x0545,
  0x0550, 0x0551, 0x0554, 0x0555, 0x1000, 0x1001, 0x1004, 0x1005, 0x1010, 0x1011, 0x1014, 0x1015,
  0x1040, 0x1041, 0x1044, 0x1045, 0x1050, 0x1051, 0x1054, 0x1055, 0x1100, 0x1101, 0x1104, 0x1105,
  0x1110, 0x1111, 0x1114, 0x1115, 0x1140, 0x1141, 0x1144, 0x1145, 0x1150, 0x1151, 0x1154, 0x1155,
  0x1400, 0x1401, 0x1404, 0x1405, 0x1410, 0x1411, 0x1414, 0x1415, 0x1440, 0x1441, 0x1444, 0x1445,
  0x1450, 0x1451, 0x1454, 0x1455, 0x1500, 0x1501, 0x1504, 0x1505, 0x1510, 0x1511, 0x1514, 0x1515,
  0x1540, 0x1541, 0x1544, 0x1545, 0x1550, 0x1551, 0x1554, 0x1555, 0x4000, 0x4001, 0x4004, 0x4005,
  0x4010, 0x4011, 0x4014, 0x4015, 0x4040, 0x4041, 0x4044, 0x4045, 0x4050, 0x4051, 0x4054, 0x4055,
  0x4100, 0x4101, 0x4104, 0x4105, 0x4110, 0x4111, 0x4114, 0x4115, 0x4140, 0x4141, 0x4144, 0x4145,
  0x4150, 0x4151, 0x4154, 0x4155, 0x4400, 0x4401, 0x4404, 0x4405, 0x4410, 0x4411, 0x4414, 0x4415,
  0x4440, 0x4441, 0x4444, 0x4445, 0x4450, 0x4451, 0x4454, 0x4455, 0x4500, 0x4501, 0x4504, 0x4505,
  0x4510, 0x4511, 0x4514, 0x4515, 0x4540, 0x4541, 0x4544, 0x4545, 0x4550, 0x4551, 0x4554, 0x4555,
  0x5000, 0x5001, 0x5004, 0x5005, 0x5010, 0x5011, 0x5014, 0x5015, 0x5040, 0x5041, 0x5044, 0x5045,
  0x5050, 0x5051, 0x5054, 0x5055, 0x5100, 0x5101, 0x5104, 0x5105, 0x5110, 0x5111, 0x5114, 0x5115,
  0x5140, 0x5141, 0x5144, 0x5145, 0x5150, 0x5151, 0x5154, 0x5155, 0x5400, 0x5401, 0x5404, 0x5405,
  0x5410, 0x5411, 0x5414, 0x5415, 0x5440, 0x5441, 0x5444, 0x5445, 0x5450, 0x5451, 0x5454, 0x5455,
  0x5500, 0x5501, 0x5504, 0x5505, 0x5510, 0x5511, 0x5514, 0x5515, 0x5540, 0x5541, 0x5544, 0x5545,
  0x5550, 0x5551, 0x5554, 0x5555};

static __device__ __constant__ unsigned char undilate_table[256] = {
  0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13, 0x20, 0x21, 0x30, 0x31, 0x22, 0x23, 0x32, 0x33,
  0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17, 0x24, 0x25, 0x34, 0x35, 0x26, 0x27, 0x36, 0x37,
  0x40, 0x41, 0x50, 0x51, 0x42, 0x43, 0x52, 0x53, 0x60, 0x61, 0x70, 0x71, 0x62, 0x63, 0x72, 0x73,
  0x44, 0x45, 0x54, 0x55, 0x46, 0x47, 0x56, 0x57, 0x64, 0x65, 0x74, 0x75, 0x66, 0x67, 0x76, 0x77,
  0x08, 0x09, 0x18, 0x19, 0x0A, 0x0B, 0x1A, 0x1B, 0x28, 0x29, 0x38, 0x39, 0x2A, 0x2B, 0x3A, 0x3B,
  0x0C, 0x0D, 0x1C, 0x1D, 0x0E, 0x0F, 0x1E, 0x1F, 0x2C, 0x2D, 0x3C, 0x3D, 0x2E, 0x2F, 0x3E, 0x3F,
  0x48, 0x49, 0x58, 0x59, 0x4A, 0x4B, 0x5A, 0x5B, 0x68, 0x69, 0x78, 0x79, 0x6A, 0x6B, 0x7A, 0x7B,
  0x4C, 0x4D, 0x5C, 0x5D, 0x4E, 0x4F, 0x5E, 0x5F, 0x6C, 0x6D, 0x7C, 0x7D, 0x6E, 0x6F, 0x7E, 0x7F,
  0x80, 0x81, 0x90, 0x91, 0x82, 0x83, 0x92, 0x93, 0xA0, 0xA1, 0xB0, 0xB1, 0xA2, 0xA3, 0xB2, 0xB3,
  0x84, 0x85, 0x94, 0x95, 0x86, 0x87, 0x96, 0x97, 0xA4, 0xA5, 0xB4, 0xB5, 0xA6, 0xA7, 0xB6, 0xB7,
  0xC0, 0xC1, 0xD0, 0xD1, 0xC2, 0xC3, 0xD2, 0xD3, 0xE0, 0xE1, 0xF0, 0xF1, 0xE2, 0xE3, 0xF2, 0xF3,
  0xC4, 0xC5, 0xD4, 0xD5, 0xC6, 0xC7, 0xD6, 0xD7, 0xE4, 0xE5, 0xF4, 0xF5, 0xE6, 0xE7, 0xF6, 0xF7,
  0x88, 0x89, 0x98, 0x99, 0x8A, 0x8B, 0x9A, 0x9B, 0xA8, 0xA9, 0xB8, 0xB9, 0xAA, 0xAB, 0xBA, 0xBB,
  0x8C, 0x8D, 0x9C, 0x9D, 0x8E, 0x8F, 0x9E, 0x9F, 0xAC, 0xAD, 0xBC, 0xBD, 0xAE, 0xAF, 0xBE, 0xBF,
  0xC8, 0xC9, 0xD8, 0xD9, 0xCA, 0xCB, 0xDA, 0xDB, 0xE8, 0xE9, 0xF8, 0xF9, 0xEA, 0xEB, 0xFA, 0xFB,
  0xCC, 0xCD, 0xDC, 0xDD, 0xCE, 0xCF, 0xDE, 0xDF, 0xEC, 0xED, 0xFC, 0xFD, 0xEE, 0xEF, 0xFE, 0xFF};
}  // namespace

/*  Algorithm 1  */
__device__ inline uint32_t dilate_2(uint16_t const x)
{
  return dilate_table[0xFF & x] | (dilate_table[(0xFFFF & x) >> 8] << 16);
}

__device__ inline uint32_t z_order(uint16_t const x, uint16_t const y)
{
  return (dilate_2(y) << 1) | dilate_2(x);
}

/*  Algorithm 2  */
__device__ inline uint16_t undilate_2(uint32_t const x)
{
  return undilate_table[0xFF & ((x >> 7) | x)] |
         (undilate_table[0xFF & (((x >> 7) | x) >> 16)] << 8);
}

__device__ inline uint16_t z_order_x(uint32_t const index)
{
  return undilate_2(index & 0x55555555);
}

__device__ inline uint16_t z_order_y(uint32_t const index)
{
  return undilate_2((index >> 1) & 0x55555555);
}

}  // namespace utility
}  // namespace detail
}  // namespace cuspatial
