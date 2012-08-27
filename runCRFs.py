import multiprocessing
import subprocess
import os

class WorkThread(multiprocessing.Process):
    def __init__(self, WorkQueue_, ResultsQueue_):
        super(WorkThread, self).__init__()
        self.WorkQueue_ = WorkQueue_
        self.ResultsQueue_ = ResultsQueue_

    def run(self):
        flag = 'ok'
        while (flag != 'stop'):
            args = self.WorkQueue_.get()
            if args == None:
                flag = 'stop'
            else:
                n = args[0]
                cwd = args[1]

                cmd = ['crf_test',
                        '-v2',
                        '-m',
                        os.path.join(cwd, 'model_%02d' %n),
                        os.path.join(cwd, 'crf_train_col%02d.txt' %n)]
                
                output = open(os.path.join(cwd, 'results_%02d.txt'%n), 'w')
                p = subprocess.Popen(cmd, stdout=output, stderr=output)
                p.wait()
                output.close()
                self.ResultsQueue_.put(n)

def runCRFs(num_threads=4):
    WorkQueue_ = multiprocessing.Queue()
    ResultsQueue_ = multiprocessing.Queue()

    for i in range(num_threads):
        thread = WorkThread(WorkQueue_, ResultsQueue_)
        thread.start()

    cwd = os.getcwd()
    for i in range(32):
        WorkQueue_.put([i, cwd])

    for i in range(num_threads):
        WorkQueue_.put(None)

    for i in range(32):
        result = ResultsQueue_.get()
        print "finished running CRF", result

if __name__ == "__main__":
    runCRFs()


