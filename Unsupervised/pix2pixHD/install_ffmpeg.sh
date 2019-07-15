# Taken from https://gist.github.com/mindworker/b670903c3d8b8977afaa4440b50f48e1

sudo su -

cd /usr/local/bin
mkdir ffmpeg

cd ffmpeg
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar -xvf ./ffmpeg-release-amd64-static.tar.xz

ln -s /usr/local/bin/ffmpeg/ffmpeg-4.1.3-amd64-static/ffmpeg /usr/bin/ffmpeg
exit