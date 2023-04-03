import sys
import traceback


class DuplicateWriter(object):
    def __init__(self, file_objects):
        self._file_objects = file_objects

    def write(self, data):
        for file_object in self._file_objects:
            file_object.write(data)
            file_object.flush()

    def writelines(self, data):
        for file_object in self._file_objects:
            file_object.write(data)
            file_object.flush()

    def flush(self):
        for file_object in self._file_objects:
            file_object.flush()

    def close(self):
        for file_object in self._file_objects:
            file_object.close()


class Tee(object):
    def __init__(self, stdout_path, stderr_path):
        self.stdout_file = open(stdout_path, 'w')
        self.stderr_file = open(stderr_path, 'w')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.stdout_writer = DuplicateWriter([sys.stdout, self.stdout_file])
        self.stderr_writer = DuplicateWriter([sys.stderr, self.stderr_file])

    def __enter__(self):
        sys.stdout = self.stdout_writer
        sys.stderr = self.stderr_writer

    def __exit__(self, exc_type, exc, exc_tb):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        if exc_type is not None:
            self.stderr_writer.write(traceback.format_exc())
        self.stderr_writer.flush()
        self.stdout_writer.flush()
        self.stderr_file.close()
        self.stdout_file.close()
