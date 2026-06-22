"""Bounded asynchronous image writing for render/evaluation loops."""

from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore

import torchvision


class AsyncImageWriter:
    def __init__(self, workers=4, max_pending=None):
        self.workers = max(0, int(workers))
        self.executor = (
            ThreadPoolExecutor(max_workers=self.workers)
            if self.workers > 0
            else None
        )
        pending = max_pending or max(1, self.workers * 2)
        self.slots = Semaphore(pending)
        self.futures = []

    @staticmethod
    def _save(tensor, path):
        torchvision.utils.save_image(tensor, path)

    def submit(self, tensor, path):
        # Move GPU data out of the render loop immediately. The bounded queue
        # prevents CPU tensors from growing without limit when storage is slow.
        tensor = tensor.detach().cpu()
        if self.executor is None:
            self._save(tensor, path)
            return

        self.slots.acquire()
        future = self.executor.submit(self._save, tensor, path)
        future.add_done_callback(lambda _: self.slots.release())
        self.futures.append(future)

    def close(self):
        first_error = None
        for future in self.futures:
            try:
                future.result()
            except Exception as error:
                if first_error is None:
                    first_error = error
        if self.executor is not None:
            self.executor.shutdown(wait=True)
        if first_error is not None:
            raise first_error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
