import logging


class ABC(object):
    def __init__(self):
        pass

    def main(self):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelno)s - %(name)s - '
                                   '%(levelname)s - %(filename)s - %(funcName)s - '
                                   '%(message)s')
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler("log111.txt")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        logger.addHandler(handler)
        # logger.addHandler(console)

        logger.info("Start print log")
        logger.debug("Do something")
        logger.warning("Something maybe fail.")
        logger.info("Finish")


if __name__ == '__main__':
    abc = ABC()
    abc.main()
