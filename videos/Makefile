download:
	@ echo Preparing sample videos ...
	@ test -s bus_station.mp4 || \
	  wget -N --continue "https://player.vimeo.com/external/188704553.sd.mp4?s=416fe058cc402a7b4e89cf7b4c6477e5abc7387e&profile_id=165&download=1" -O bus_station.mp4
	@ test -s motorcycle.mp4 || \
	  wget -N --continue "https://player.vimeo.com/external/188704572.sd.mp4?s=d64c6506b7aa37701e8c281a0889c81b1551aaa5&profile_id=165&download=1" -O motorcycle.mp4
	@ test -s scooters.mp4 || \
		wget -N --continue "https://player.vimeo.com/external/183850414.sd.mp4?s=8cbcff2250cad5dd87a7878ceeb062eb6eb168a7&profile_id=165&download=1" -O scooters.mp4

clean:
	rm -v *.mp4
