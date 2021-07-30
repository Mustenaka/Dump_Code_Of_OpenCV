import logging
import logging.handlers
import os
import time

import Decorator.Singleton

@Decorator.Singleton.Singleton
class Logs(object):
    """
    生成日志模块，采用单例模式生成
    Attributes:
    """
    def __init__(self):
        # getLogger不是获取模块名称，而是输入则可以返回一个特殊的日志名，否则则视为root日志
        # 就默认root日志就行
        self.logger = logging.getLogger('')
        # 设置输出的等级
        LEVELS = {'NOSET': logging.NOTSET,
                  'DEBUG': logging.DEBUG,
                  'INFO': logging.INFO,
                  'WARNING': logging.WARNING,
                  'ERROR': logging.ERROR,
                  'CRITICAL': logging.CRITICAL}
        # 创建文件目录
        logs_dir = "logs"
        if os.path.exists(logs_dir) and os.path.isdir(logs_dir):
            pass
        else:
            os.mkdir(logs_dir)
        # 修改log保存位置
        timestamp = time.strftime("%Y-%m-%d", time.localtime())
        logfilename = '%s.txt' % timestamp
        logfilepath = os.path.join(logs_dir, logfilename)
        rotatingFileHandler = logging.handlers.RotatingFileHandler(filename=logfilepath,
                                                                   maxBytes=1024 * 1024 * 50,
                                                                   backupCount=5)
        # 设置输出格式
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        rotatingFileHandler.setFormatter(formatter)
        # 控制台句柄
        console = logging.StreamHandler()
        console.setLevel(logging.NOTSET)
        console.setFormatter(formatter)
        # 添加内容到日志句柄中
        self.logger.addHandler(rotatingFileHandler)
        self.logger.addHandler(console)
        self.logger.setLevel(logging.NOTSET)


    def info(self, message):
        """
        info 等级，属于最低级，仅展示一些默认信息
        """
        self.logger.info(message)

    def debug(self, message):
        """
        debug 等级，属于测试信息，用来debug给程序员判断用
        """
        self.logger.debug(message)

    def warning(self, message):
        """
        warning 等级，属于警告信息，如果不避免很有可能发生错误或者崩溃现象
        """
        self.logger.warning(message)

    def error(self, message):
        """
        error 等级，属于错误信息，当程序发生错误或者崩溃的时候进行的记录
        """
        self.logger.error(message)