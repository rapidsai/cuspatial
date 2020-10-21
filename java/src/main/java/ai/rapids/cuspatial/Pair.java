/*
 *
 *  Copyright (c) 2020, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */
package ai.rapids.cuspatial;

import java.util.AbstractMap;

public class Pair<T1 extends AutoCloseable, T2 extends AutoCloseable> extends AbstractMap.SimpleEntry<T1, T2> implements AutoCloseable {
    public Pair(T1 l, T2 r) {
        super(l, r);
    }
    public T1 getLeft() {
        return getKey();
    }
    public T2 getRight() {
        return getValue();
    }

    @Override
    public synchronized void close() throws Exception {
        this.getLeft().close();
        this.getRight().close();
    }
}
