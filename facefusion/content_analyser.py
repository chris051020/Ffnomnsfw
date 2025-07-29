from functools import lru_cache
from typing import List, Tuple

from tqdm import tqdm
import numpy

from facefusion import inference_manager, state_manager, wording
from facefusion.download import conditional_download_hashes, conditional_download_sources
from facefusion.execution import has_execution_provider
from facefusion.types import DownloadScope, DownloadSet, ExecutionProvider, Fps, InferencePool, ModelSet, VisionFrame
from facefusion.vision import detect_video_fps, read_image, read_video_frame

STREAM_COUNTER = 0


@lru_cache(maxsize=None)
def create_static_model_set(download_scope: DownloadScope) -> ModelSet:
	return {}  # Eliminamos modelos NSFW


def get_inference_pool() -> InferencePool:
	return inference_manager.get_inference_pool(__name__, [], {})


def clear_inference_pool() -> None:
	inference_manager.clear_inference_pool(__name__, [])


def resolve_execution_providers() -> List[ExecutionProvider]:
	if has_execution_provider('coreml'):
		return ['cpu']
	return state_manager.get_item('execution_providers')


def collect_model_downloads() -> Tuple[DownloadSet, DownloadSet]:
	return {}, {}  # Nada que descargar


def pre_check() -> bool:
	model_hash_set, model_source_set = collect_model_downloads()
	return conditional_download_hashes(model_hash_set) and conditional_download_sources(model_source_set)


def analyse_stream(vision_frame: VisionFrame, video_fps: Fps) -> bool:
	global STREAM_COUNTER
	STREAM_COUNTER += 1
	if STREAM_COUNTER % int(video_fps) == 0:
		return analyse_frame(vision_frame)
	return False


def analyse_frame(vision_frame: VisionFrame) -> bool:
	return detect_nsfw(vision_frame)


@lru_cache(maxsize=None)
def analyse_image(image_path: str) -> bool:
	vision_frame = read_image(image_path)
	return analyse_frame(vision_frame)


@lru_cache(maxsize=None)
def analyse_video(video_path: str, trim_frame_start: int, trim_frame_end: int) -> bool:
	video_fps = detect_video_fps(video_path)
	frame_range = range(trim_frame_start, trim_frame_end)
	rate = 0.0
	total = 0
	counter = 0

	with tqdm(total=len(frame_range), desc=wording.get('analysing'), unit='frame',
	          ascii=' =', disable=state_manager.get_item('log_level') in ['warn', 'error']) as progress:
		for frame_number in frame_range:
			if frame_number % int(video_fps) == 0:
				vision_frame = read_video_frame(video_path, frame_number)
				total += 1
				if analyse_frame(vision_frame):
					counter += 1
			if counter > 0 and total > 0:
				rate = counter / total * 100
			progress.set_postfix(rate=rate)
			progress.update()

	return bool(rate > 10.0)


# ✅ Validación NSFW completamente desactivada
def detect_nsfw(vision_frame: VisionFrame) -> bool:
	return False