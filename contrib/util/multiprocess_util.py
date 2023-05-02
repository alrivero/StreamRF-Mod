from loguru import logger
from torch.multiprocessing  import Queue

frame_idx_queue = Queue()
dset_queue = Queue()
def pre_fetch_datasets():
    while True:
        try:
            logger.info(f'Waiting for new frame... (Queue Size: {frame_idx_queue.qsize()})')
            frame_idx = frame_idx_queue.get(block=True,timeout=60)
            logger.info(f'Frame received. (Queue Size: {frame_idx_queue.qsize()})')
        except Empty:
            logger.info('Ending data prefetch process.')
            return 
        
        logger.info(f"Finished loading frame: {frame_idx}")
        args.frame_id = frame_idx
        dataset = Data(args)
        memitem = dataset.initpc()
        dset_queue.put(memitem)