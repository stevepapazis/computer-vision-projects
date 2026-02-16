function saveAnimation(frames, filename, fps=10)

  pkg load video;

  v = VideoWriter(filename);
  v.FrameRate = fps;

  open(v);

  nFrames = size(frames, 4);

  for k = 1:nFrames
    writeVideo(v, frames(:,:,:,k));
  end

  close(v);

end