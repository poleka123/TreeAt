import os

class Logger:
    def __init__(self, exp_name):
        if not os.path.exists('./logs/PEMS07/'):
            os.makedirs('./logs/PEMS07/')
            self.file = open('./logs/PEMS07/{}.log'.format(exp_name), 'w')
        else:
            self.file = open('./logs/PEMS07/{}.log'.format(exp_name), 'w')

    def log(self, content):
        self.file.write(content + '\n')
        self.file.flush()
