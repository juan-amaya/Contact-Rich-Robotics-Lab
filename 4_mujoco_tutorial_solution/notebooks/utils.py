import cv2

def save_video_from_frames(
	file_path : str,
	frames,
	fps : int
	):
	H, W = frames[0].shape[0], frames[0].shape[1]

	# Convert frames to BGR and write to video
    # Prepare video writer
	writer = cv2.VideoWriter(
		file_path,
		cv2.VideoWriter_fourcc(*"avc1"),
		fps,
		(W, H),
	)

	for frame in frames:
		# MuJoCo returns RGB, OpenCV expects BGR
		bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		writer.write(bgr)

	writer.release()
	print(f"Video saved at {file_path}")