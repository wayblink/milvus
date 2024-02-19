import multiprocessing
import numbers
import random
import numpy
import threading
import pytest
import pandas as pd
import decimal
from decimal import Decimal, getcontext
from time import sleep
import heapq

from base.client_base import TestcaseBase
from utils.util_log import test_log as log
from common import common_func as cf
from common import common_type as ct
from common.common_type import CaseLabel, CheckTasks
from utils.util_pymilvus import *
from common.constants import *
from pymilvus.orm.types import CONSISTENCY_STRONG, CONSISTENCY_BOUNDED, CONSISTENCY_SESSION, CONSISTENCY_EVENTUALLY
from base.high_level_api_wrapper import HighLevelApiWrapper
client_w = HighLevelApiWrapper()

prefix = "milvus_client_api_query"
epsilon = ct.epsilon
default_nb = ct.default_nb
default_nb_medium = ct.default_nb_medium
default_nq = ct.default_nq
default_dim = ct.default_dim
default_limit = ct.default_limit
default_search_exp = "id >= 0"
exp_res = "exp_res"
default_search_string_exp = "varchar >= \"0\""
default_search_mix_exp = "int64 >= 0 && varchar >= \"0\""
default_invaild_string_exp = "varchar >= 0"
default_json_search_exp = "json_field[\"number\"] >= 0"
perfix_expr = 'varchar like "0%"'
default_search_field = ct.default_float_vec_field_name
default_search_params = ct.default_search_params
default_primary_key_field_name = "id"
default_vector_field_name = "vector"
default_float_field_name = ct.default_float_field_name
default_bool_field_name = ct.default_bool_field_name
default_string_field_name = ct.default_string_field_name
default_int32_array_field_name = ct.default_int32_array_field_name
default_string_array_field_name = ct.default_string_array_field_name


class TestMilvusClientQueryInvalid(TestcaseBase):
    """ Test case of search interface """

    @pytest.fixture(scope="function", params=[False, True])
    def auto_id(self, request):
        yield request.param

    @pytest.fixture(scope="function", params=["COSINE", "L2"])
    def metric_type(self, request):
        yield request.param

    """
    ******************************************************************
    #  The following are invalid base cases
    ******************************************************************
    """

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_client_query_not_all_required_params(self):
        """
        target: test query (high level api) normal case
        method: create connection, collection, insert and search
        expected: search/query successfully
        """
        client = self._connect(enable_milvus_client_api=True)
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        client_w.create_collection(client, collection_name, default_dim, consistency_level="Strong")
        collections = client_w.list_collections(client)[0]
        assert collection_name in collections
        client_w.describe_collection(client, collection_name,
                                     check_task=CheckTasks.check_describe_collection_property,
                                     check_items={"collection_name": collection_name,
                                                  "dim": default_dim,
                                                  "consistency_level": 0})
        # 2. insert
        rng = np.random.default_rng(seed=19530)
        rows = [{default_primary_key_field_name: i, default_vector_field_name: list(rng.random((1, default_dim))[0]),
                 default_float_field_name: i * 1.0, default_string_field_name: str(i)} for i in range(default_nb)]
        client_w.insert(client, collection_name, rows)
        # 3. query using ids
        error = {ct.err_code: 65535, ct.err_msg: f"empty expression should be used with limit"}
        client_w.query(client, collection_name,
                       check_task=CheckTasks.err_res, check_items=error)


class TestMilvusClientQueryValid(TestcaseBase):
    """ Test case of search interface """

    @pytest.fixture(scope="function", params=[False, True])
    def auto_id(self, request):
        yield request.param

    @pytest.fixture(scope="function", params=["COSINE", "L2"])
    def metric_type(self, request):
        yield request.param

    """
    ******************************************************************
    #  The following are valid base cases
    ******************************************************************
    """

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_client_query_default(self):
        """
        target: test query (high level api) normal case
        method: create connection, collection, insert and search
        expected: search/query successfully
        """
        client = self._connect(enable_milvus_client_api=True)
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        client_w.create_collection(client, collection_name, default_dim, consistency_level="Strong")
        # 2. insert
        rng = np.random.default_rng(seed=19530)
        rows = [{default_primary_key_field_name: i, default_vector_field_name: list(rng.random((1, default_dim))[0]),
                 default_float_field_name: i * 1.0, default_string_field_name: str(i)} for i in range(default_nb)]
        client_w.insert(client, collection_name, rows)
        # 3. query using ids
        client_w.query(client, collection_name, ids=[i for i in range(default_nb)],
                       check_task=CheckTasks.check_query_results,
                       check_items={exp_res: rows,
                                    "with_vec": True,
                                    "primary_field": default_primary_key_field_name})
        # 4. query using filter
        client_w.query(client, collection_name, filter=default_search_exp,
                       check_task=CheckTasks.check_query_results,
                       check_items={exp_res: rows,
                                    "with_vec": True,
                                    "primary_field": default_primary_key_field_name})
        client_w.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_client_query_output_fields(self):
        """
        target: test query (high level api) normal case
        method: create connection, collection, insert and search
        expected: search/query successfully
        """
        client = self._connect(enable_milvus_client_api=True)
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        client_w.create_collection(client, collection_name, default_dim, consistency_level="Strong")
        # 2. insert
        rng = np.random.default_rng(seed=19530)
        rows = [{default_primary_key_field_name: i, default_vector_field_name: list(rng.random((1, default_dim))[0]),
                 default_float_field_name: i * 1.0, default_string_field_name: str(i)} for i in range(default_nb)]
        client_w.insert(client, collection_name, rows)
        # 3. query using ids
        client_w.query(client, collection_name, ids=[i for i in range(default_nb)],
                       check_task=CheckTasks.check_query_results,
                       check_items={exp_res: rows,
                                    "with_vec": True,
                                    "primary_field": default_primary_key_field_name})
        # 4. query using filter
        res = client_w.query(client, collection_name, filter=default_search_exp,
                             output_fields=[default_primary_key_field_name, default_float_field_name,
                                            default_string_field_name, default_vector_field_name],
                             check_task=CheckTasks.check_query_results,
                             check_items={exp_res: rows,
                                          "with_vec": True,
                                          "primary_field": default_primary_key_field_name})[0]
        assert set(res[0].keys()) == {default_primary_key_field_name, default_vector_field_name,
                                      default_float_field_name, default_string_field_name}
        client_w.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_client_query_output_fields_all(self):
        """
        target: test query (high level api) normal case
        method: create connection, collection, insert and search
        expected: search/query successfully
        """
        client = self._connect(enable_milvus_client_api=True)
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        client_w.create_collection(client, collection_name, default_dim, consistency_level="Strong")
        # 2. insert
        rng = np.random.default_rng(seed=19530)
        rows = [{default_primary_key_field_name: i, default_vector_field_name: list(rng.random((1, default_dim))[0]),
                 default_float_field_name: i * 1.0, default_string_field_name: str(i)} for i in range(default_nb)]
        client_w.insert(client, collection_name, rows)
        # 3. query using ids
        client_w.query(client, collection_name, ids=[i for i in range(default_nb)],
                       check_task=CheckTasks.check_query_results,
                       check_items={exp_res: rows,
                                    "with_vec": True,
                                    "primary_field": default_primary_key_field_name})
        # 4. query using filter
        res = client_w.query(client, collection_name, filter=default_search_exp,
                             output_fields=["*"],
                             check_task=CheckTasks.check_query_results,
                             check_items={exp_res: rows,
                                          "with_vec": True,
                                          "primary_field": default_primary_key_field_name})[0]
        assert set(res[0].keys()) == {default_primary_key_field_name, default_vector_field_name,
                                      default_float_field_name, default_string_field_name}
        client_w.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_client_query_limit(self):
        """
        target: test query (high level api) normal case
        method: create connection, collection, insert and search
        expected: search/query successfully
        """
        client = self._connect(enable_milvus_client_api=True)
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        client_w.create_collection(client, collection_name, default_dim, consistency_level="Strong")
        # 2. insert
        rng = np.random.default_rng(seed=19530)
        rows = [{default_primary_key_field_name: i, default_vector_field_name: list(rng.random((1, default_dim))[0]),
                 default_float_field_name: i * 1.0, default_string_field_name: str(i)} for i in range(default_nb)]
        client_w.insert(client, collection_name, rows)
        # 3. query using ids
        limit = 5
        client_w.query(client, collection_name, ids=[i for i in range(default_nb)],
                       limit=limit,
                       check_task=CheckTasks.check_query_results,
                       check_items={exp_res: rows[:limit],
                                    "with_vec": True,
                                    "primary_field": default_primary_key_field_name[:limit]})
        # 4. query using filter
        client_w.query(client, collection_name, filter=default_search_exp,
                       limit=limit,
                       check_task=CheckTasks.check_query_results,
                       check_items={exp_res: rows[:limit],
                                    "with_vec": True,
                                    "primary_field": default_primary_key_field_name[:limit]})[0]
        client_w.drop_collection(client, collection_name)
