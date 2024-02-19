// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#pragma once

#include "common.h"

namespace milvus {
namespace simd {

BitsetBlockType
GetBitsetBlockRef(const bool* src);

bool
AllTrueRef(const bool* src, int64_t size);

bool
AllFalseRef(const bool* src, int64_t size);

void
InvertBoolRef(bool* src, int64_t size);

void
AndBoolRef(bool* left, bool* right, int64_t size);

void
OrBoolRef(bool* left, bool* right, int64_t size);

template <typename T>
bool
FindTermRef(const T* src, size_t size, T val) {
    for (size_t i = 0; i < size; ++i) {
        if (src[i] == val) {
            return true;
        }
    }
    return false;
}

template <typename T>
void
EqualValRef(const T* src, size_t size, T val, bool* res) {
    for (size_t i = 0; i < size; ++i) {
        res[i] = src[i] == val;
    }
}

template <typename T>
void
LessValRef(const T* src, size_t size, T val, bool* res) {
    for (size_t i = 0; i < size; ++i) {
        res[i] = src[i] < val;
    }
}

template <typename T>
void
GreaterValRef(const T* src, size_t size, T val, bool* res) {
    for (size_t i = 0; i < size; ++i) {
        res[i] = src[i] > val;
    }
}

template <typename T>
void
LessEqualValRef(const T* src, size_t size, T val, bool* res) {
    for (size_t i = 0; i < size; ++i) {
        res[i] = src[i] <= val;
    }
}
template <typename T>
void
GreaterEqualValRef(const T* src, size_t size, T val, bool* res) {
    for (size_t i = 0; i < size; ++i) {
        res[i] = src[i] >= val;
    }
}
template <typename T>
void
NotEqualValRef(const T* src, size_t size, T val, bool* res) {
    for (size_t i = 0; i < size; ++i) {
        res[i] = src[i] != val;
    }
}

template <typename T>
void
EqualColumnRef(const T* left, const T* right, size_t size, bool* res) {
    for (size_t i = 0; i < size; ++i) {
        res[i] = left[i] == right[i];
    }
}

template <typename T>
void
LessColumnRef(const T* left, const T* right, size_t size, bool* res) {
    for (size_t i = 0; i < size; ++i) {
        res[i] = left[i] < right[i];
    }
}

template <typename T>
void
LessEqualColumnRef(const T* left, const T* right, size_t size, bool* res) {
    for (size_t i = 0; i < size; ++i) {
        res[i] = left[i] <= right[i];
    }
}

template <typename T>
void
GreaterColumnRef(const T* left, const T* right, size_t size, bool* res) {
    for (size_t i = 0; i < size; ++i) {
        res[i] = left[i] > right[i];
    }
}

template <typename T>
void
GreaterEqualColumnRef(const T* left, const T* right, size_t size, bool* res) {
    for (size_t i = 0; i < size; ++i) {
        res[i] = left[i] >= right[i];
    }
}

template <typename T>
void
NotEqualColumnRef(const T* left, const T* right, size_t size, bool* res) {
    for (size_t i = 0; i < size; ++i) {
        res[i] = left[i] != right[i];
    }
}

}  // namespace simd
}  // namespace milvus
