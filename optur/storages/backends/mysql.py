import time
from typing import Any, Callable, List, Optional

from google.protobuf.timestamp_pb2 import Timestamp

from optur.errors import NotFoundError
from optur.proto.study_pb2 import StudyInfo
from optur.proto.study_pb2 import Trial as TrialProto
from optur.storages.backends.backend import StorageBackend


def _retry(func: Callable[..., Any]) -> Any:
    def wrapped_func(self: "MySQLBackend", *args: Any, **kwargs: Any) -> Any:
        import pymysql

        s = 0.1
        retry_limit = self.retry_limit
        while retry_limit > 0:
            try:
                if not self._connection.open:
                    self._connection.connect()
                return func(self, *args, **kwargs)
            except (
                pymysql.err.DatabaseError,
                pymysql.err.IntegrityError,
                pymysql.err.OperationalError,
            ):
                retry_limit -= 1
                if retry_limit <= 0:
                    raise
            time.sleep(s)
            s *= 2
            continue
        return func(self, *args, **kwargs)

    return wrapped_func


class MySQLBackend(StorageBackend):
    def __init__(
        self, *, host: str, user: str, port: int = 3306, password: str, database: str
    ) -> None:
        super().__init__()
        try:
            import pymysql
            import pymysql.cursors
        except ImportError:
            # TODO(tsuzuku)
            raise

        self._retry_limit = 1
        self._connection = pymysql.connect(
            user=user,
            host=host,
            port=port,
            password=password,
            database=database,
            cursorclass=pymysql.cursors.DictCursor,
        )

    @property
    def retry_limit(self) -> int:
        return self._retry_limit

    @_retry
    def drop_all(self) -> None:
        with self._connection.cursor() as cursor:
            cursor.execute(query="DELETE FROM trial_data;")
            cursor.execute(query="DELETE FROM trial;")
            cursor.execute(query="DELETE FROM study_info;")
            cursor.execute(query="DELETE FROM study;")

    @_retry
    def init(self) -> None:
        with self._connection.cursor() as cursor:
            query = """CREATE TABLE IF NOT EXISTS study (
                study_id varchar(32) PRIMARY KEY,
                timestamp TIMESTAMP(6) NOT NULL,
                INDEX study_timestamp (timestamp, study_id)
            );"""
            cursor.execute(query=query)
            query = """CREATE TABLE IF NOT EXISTS study_info
            (
                study_id varchar(32) NOT NULL PRIMARY KEY,
                info BLOB
            );
            """
            cursor.execute(query=query)
            query = """CREATE TABLE IF NOT EXISTS trial (
                trial_id varchar(32) PRIMARY KEY,
                study_id varchar(32),
                timestamp TIMESTAMP(6) NOT NULL,
                FOREIGN KEY(study_id) REFERENCES study(study_id),
                INDEX trial_study_timestamp (study_id, timestamp, trial_id),
                INDEX trial_timestamp (timestamp, trial_id)
            );
            """
            cursor.execute(query=query)
            query = """CREATE TABLE IF NOT EXISTS trial_data (
                trial_id varchar(32) PRIMARY KEY,
                data BLOB
            );
            """
            cursor.execute(query=query)
            self._connection.commit()

    @_retry
    def get_current_timestamp(self) -> Optional[Timestamp]:
        with self._connection.cursor() as cursor:
            query = """SELECT CURRENT_TIMESTAMP(6);"""
            cursor.execute(query=query)
            data = cursor.fetchall()
        timestamp = Timestamp()
        timestamp.FromDatetime(next(iter(data[0].values())))
        return timestamp

    def get_studies(self, timestamp: Optional[Timestamp] = None) -> List[StudyInfo]:
        with self._connection.cursor() as cursor:
            if timestamp is None:
                query = """SELECT info FROM study_info;"""
            else:
                ms = timestamp.ToMilliseconds()
                query = f"""SELECT info FROM study_info INNER JOIN (
                    SELECT study_id FROM study
                    WHERE timestamp >= FROM_UNIXTIME({ms}/1000)
                ) as ts ON study_info.study_id = ts.study_id;
                """
            cursor.execute(query=query)
            data = cursor.fetchall()
        return [StudyInfo.FromString(row["info"]) for row in data]

    @_retry
    def get_trials(
        self, study_id: Optional[str] = None, timestamp: Optional[Timestamp] = None
    ) -> List[TrialProto]:
        if study_id is None:
            if timestamp is None:
                query = """SELECT data from trial_data;"""
            else:
                ms = timestamp.ToMilliseconds()
                query = f"""SELECT data from trial_data INNER JOIN (
                    SELECT trial_id FROM trial WHERE timestamp >= FROM_UNIXTIME({ms}/1000)
                ) as tt ON tt.trial_id = trial_data.trial_id;"""
        else:
            if timestamp is None:
                query = f"""SELECT data from trial_data INNER JOIN (
                    SELECT trial_id FROM trial WHERE study_id = '{study_id}'
                ) as tt ON tt.trial_id = trial_data.trial_id;"""
            else:
                ms = timestamp.ToMilliseconds()
                query = f"""SELECT data from trial_data INNER JOIN (
                    SELECT trial_id FROM trial
                    WHERE study_id = '{study_id}' AND timestamp >= FROM_UNIXTIME({ms}/1000)
                ) as tt ON tt.trial_id = trial_data.trial_id;"""
        with self._connection.cursor() as cursor:
            cursor.execute(query=query)
            data = cursor.fetchall()
        return [TrialProto.FromString(row["data"]) for row in data]

    @_retry
    def get_trial(self, trial_id: str, study_id: Optional[str] = None) -> TrialProto:
        query = f"""SELECT data FROM trial_data WHERE trial_id = '{trial_id}';"""
        with self._connection.cursor() as cursor:
            cursor.execute(query=query)
            data = cursor.fetchall()
        if not data:
            raise NotFoundError("")
        return TrialProto.FromString(data[0]["data"])

    @_retry
    def write_study(self, study: StudyInfo) -> None:
        with self._connection.cursor() as cursor:
            self._connection.begin()
            query = f"""
            INSERT INTO study VALUES('{study.study_id}', CURRENT_TIMESTAMP(6))
            ON DUPLICATE KEY UPDATE timestamp = CURRENT_TIMESTAMP(6);
            """
            cursor.execute(query=query)
            hex_data = study.SerializeToString().hex()
            query = f"""
            INSERT INTO study_info VALUES('{study.study_id}', x'{hex_data}')
            ON DUPLICATE KEY UPDATE info = VALUES(info);
            """
            cursor.execute(query=query)
            self._connection.commit()

    @_retry
    def write_trial(self, trial: TrialProto) -> None:
        import pymysql

        with self._connection.cursor() as cursor:
            self._connection.begin()
            query = f"""
            INSERT INTO trial VALUES('{trial.trial_id}', '{trial.study_id}', CURRENT_TIMESTAMP(6))
            ON DUPLICATE KEY UPDATE study_id = VALUES(study_id), timestamp = CURRENT_TIMESTAMP(6);
            """
            try:
                cursor.execute(query=query)
            except pymysql.err.IntegrityError:
                raise NotFoundError("")  # TODO(tsuzuku)
            query = f"""
            INSERT INTO trial_data VALUES('{trial.trial_id}', x'{trial.SerializeToString().hex()}')
            ON DUPLICATE KEY UPDATE data = x'{trial.SerializeToString().hex()}';
            """
            cursor.execute(query=query)
