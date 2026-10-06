from datetime import datetime, timezone
import os

import numpy
import ormsgpack
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver

class TestSqliteCheckpointer:

    def test_ormsgpack_with_date(self):
        event = {
            "type": "put",
            "time": datetime(1970, 1, 1),
            "uid": 1,
            "data": numpy.array([1, 2]),
        }
        result = ormsgpack.packb(event, option=ormsgpack.OPT_SERIALIZE_NUMPY)
        print(result)
        unpacked = ormsgpack.unpackb(result)
        print(unpacked)

    def test_ormsgpack_with_serde(self):
        from langgraph.checkpoint.serde.jsonplus import _msgpack_enc
        dt = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)
        b = _msgpack_enc({"time": dt})
        print(type(b), len(b))
        print(str(b))

    def test_checkpointing(self):
        # Source - https://stackoverflow.com/a/595315
        # Posted by Jason Coon
        # Retrieved 2026-10-06, License - CC BY-SA 2.5

        module_path: str = os.path.dirname(__file__)
        db_file_path: str = module_path + "/../test-data/memory/test_checkpointer.sqlite"

        write_config = {"configurable": {"thread_id": "3", "checkpoint_ns": ""}}
        read_config = {"configurable": {"thread_id": "3"}}

        with SqliteSaver.from_conn_string(db_file_path) as checkpointer:
            checkpoint = {
                "v": 4,
                "ts": datetime(1970, 1, 2),
                "id": f"{uuid4()}",
                "channel_values": {
                    "my_key": "meow",
                    "node": "node"
                },
                "channel_versions": {
                    "__start__": 2,
                    "my_key": 3,
                    "start:node": 3,
                    "node": 3
                },
                "versions_seen": {
                    "__input__": {},
                    "__start__": {
                        "__start__": 1
                    },
                    "node": {
                        "start:node": 2
                    }
                },
            }

            # store checkpoint
            checkpointer.put(write_config, checkpoint, {}, {})

            # load checkpoint
            checkpointer.get(read_config)

            # list checkpoints
            checkpoints =  list(checkpointer.list(read_config))
            for c in checkpoints:
                print(c)