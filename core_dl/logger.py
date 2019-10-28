import datetime
import os
import socket
import warnings
import torch
from torch.autograd import Variable

from core_dl.logger_file import FileLogger
from core_dl.logger_tensorflow import TensorboardLogger


class Logger:
    """
    Logger for training,
    usage:
    # >>> logger = Logger("./runs", "csv|txt|tensorboard")
    # >>> logger.add_keys(['Loss/class_layer', 'Accuracy/class_layer'])
    # >>> logger.log({'Loss/class_layer': 0.04, 'Accuracy/class_layer': 0.4, 'Iteration': 4})
    Use prefix to classify the term:
    - The Scalars:'Loss' and 'Accuracy' 'Scalar'
    - The Scalars: 'Scalars' used for visualize multiple records in single figure, use 'dict' for value (Only in tensorboard)
    - The Net instance: 'Net' used for parameter histogram (Only in tensorboard)
    - The Image visualization: 'Image' used for visualize bitmap (Only in tensorboard)
    """

    """ Logger instance """
    loggers = {}

    """ The place where log files are stored """
    log_base_dir = ''

    def __init__(self, base_dir=None, log_types='csv|tensorboard', comment='', continue_log=False):
        '''
        :param base_dir: The base directory stores the log file
        :param log_types:  the log file types including 'csv', 'txt', 'tensorboard'
        :param comment: Additional comments of the log
        :param continue_log: Enable True when the log will be write to the base_dir
        '''
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        self.log_base_dir = os.path.join(base_dir, current_time + '_' + socket.gethostname() + '_' + comment)

        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        if not os.path.exists(self.log_base_dir):
            os.mkdir(self.log_base_dir)

        log_types_token = log_types.split('|')
        for log_type in log_types_token:
            log_type = log_type.strip()
            logger = self.logger_factory(log_type)
            if logger is not None:
                self.loggers[log_type] = logger

    def add_keys(self, keys):
        for log_type, logger in self.loggers.items():
            logger.add_keys(keys)

    def log(self, log_dict):
        for log_type, logger in self.loggers.items():
            logger.log(log_dict)

    def get_logger_by_type(self, type):
        if type in self.loggers:
            return self.loggers[type]
        else:
            return None

    def logger_factory(self, logger_name):
        if logger_name == "csv":
            return FileLogger(os.path.join(self.log_base_dir, 'log.csv'))
        elif logger_name == "txt":
            return FileLogger(os.path.join(self.log_base_dir, 'log.txt'))
        elif logger_name == "tensorboard":
            return TensorboardLogger(self.log_base_dir)
        else:
            return None

    def draw_architecture(self, model, input_shape, verbose=False):
        if 'tensorboard' in self.loggers.keys():

            writer = self.loggers['tensorboard'].writer

            if torch.cuda.is_available():
                dtype = torch.cuda.FloatTensor
            else:
                dtype = torch.FloatTensor

            # check if there are multiple inputs to the network
            if isinstance(input_shape[0], (list, tuple)):
                x = [Variable(torch.rand(1, *in_size)).type(dtype) for in_size in input_shape]
            else:
                x = Variable(torch.rand(1, *input_shape)).type(dtype)

            # draw the graph
            writer.add_graph(model, (x, ), verbose=verbose)

        else:
            warnings.warn('No instance of tensorboard logger configured')