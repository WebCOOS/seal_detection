from prometheus_client import multiprocess

proc_name = 'webcoos-seal-detector-api'
bind = "0.0.0.0:8000"
workers = 4
threads = 1
worker_class = 'uvicorn.workers.UvicornWorker'
accesslog = '-'
errorlog = '-'
loglevel = 'info'
timeout = 120
capture_output = True
preload_app = True


# For Prometheus metrics
def child_exit(server, worker):
    try:
        multiprocess.mark_process_dead(worker.pid)
    except TypeError:
        pass
