import os
import time
import random
import string
import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

fmt = "\n=== {:30} ===\n"
dim = 128

print(fmt.format("start connecting to Milvus"))
host = os.environ.get('MILVUS_HOST')
if host == None:
    host = "10.102.5.113"
print(fmt.format(f"Milvus host: {host}"))
connections.connect("default", host="10.102.9.235", port="19530")

default_fields = [
    FieldSchema(name="count", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="key", dtype=DataType.INT64, is_partition_key=True),
    FieldSchema(name="random", dtype=DataType.DOUBLE),
    FieldSchema(name="var", dtype=DataType.VARCHAR, max_length=10000),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
]
default_schema = CollectionSchema(fields=default_fields, description="test clustering-key collection")
collection_name = "major_compaction_collection_enable_scalar_partition_key_after_index"

if utility.has_collection(collection_name):
   collection = Collection(name=collection_name)
   collection.drop()
   print("drop the original collection")
hello_milvus = Collection(name=collection_name, schema=default_schema)

print("Starting major compaction")
start = time.time()
hello_milvus.compact(is_major=True)
res = hello_milvus.get_compaction_state(is_major=True)
print(res)
print("Waiting for major compaction complete")
hello_milvus.wait_for_compaction_completed(is_major=True)
end = time.time()
print("Major compaction complete in %f s" %(end - start))
res = hello_milvus.get_compaction_state(is_major=True)
print(res)


nb = 1000

rng = np.random.default_rng(seed=19530)
random_data = rng.random(nb).tolist()

vec_data = [[random.random() for _ in range(dim)] for _ in range(nb)]
_len = int(20)
_str = string.ascii_letters + string.digits
_s = _str
print("_str size ", len(_str))

for i in range(int(_len / len(_str))):
    _s += _str
    print("append str ", i)
values = [''.join(random.sample(_s, _len - 1)) for _ in range(nb)]
index = 0
while index < 100:
    # insert data
    data = [
        [index * nb + i for i in range(nb)],
        [random.randint(0,100) for i in range(nb)],
        random_data,
        values,
        vec_data,
    ]
    start = time.time()
    res = hello_milvus.insert(data)
    end = time.time() - start
    print("insert %d %d done in %f" % (index, nb, end))
    index += 1
#    hello_milvus.flush()
    
hello_milvus.flush()

print(f"Number of entities in Milvus: {hello_milvus.num_entities}")  # check the num_entites

# 4. create index
print(fmt.format("Start Creating index AUTOINDEX"))
index = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128},
}

print("creating index")
hello_milvus.create_index("embeddings", index)
print("waiting for index completed")
utility.wait_for_index_building_complete(collection_name)
res = utility.index_building_progress(collection_name)
print(res)

print(fmt.format("Load"))
hello_milvus.load()

res = utility.get_query_segment_info(collection_name)

print("before major compaction")
print(res)

# major compact

print("Starting major compaction")
start = time.time()
hello_milvus.compact(is_major=True)
res = hello_milvus.get_compaction_state(is_major=True)
print(res)
print("Waiting for major compaction complete")
hello_milvus.wait_for_compaction_completed(is_major=True)
end = time.time()
print("Major compaction complete in %f s" %(end - start))
res = hello_milvus.get_compaction_state(is_major=True)
print(res)

res = utility.get_query_segment_info(collection_name)
print("after major compaction")
print(res)

nb = 1
vectors = [[random.random() for _ in range(dim)] for _ in range(nb)]

nq = 1

default_search_params = {"metric_type": "L2", "params": {}}
res1 = hello_milvus.search(vectors[:nq], "embeddings", default_search_params, 10, "count >= 0")

print(res1[0].ids)
