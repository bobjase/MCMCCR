$bytes = [System.IO.File]::ReadAllBytes('testFiles/enwik8_5MB')
$head264 = $bytes[4224398..(4224398+10240-1)]
[System.IO.File]::WriteAllBytes('head_264.bin', $head264)