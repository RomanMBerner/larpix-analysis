############################################################################
#GEOMETRYFILE='multi_tile_layout-2.0.16.yaml'
GEOMETRYFILE='geometries/multi_tile_layout-2.1.16.yaml'

DATAPATH='/data/LArPix/SingleModule_March2021/TPC1+2/dataRuns/rawData'

#INPUTFILE='raw_2021_04_05_15_25_41_CEST.h5'
#INPUTFILE='raw_2021_04_06_05_41_21_CEST.h5'
#INPUTFILE='raw_2021_04_06_06_21_26_CEST.h5'
INPUTFILE='raw_2021_04_06_06_21_26_CEST.h5'
############################################################################

python scan_raw_larpix.py --infile=$DATAPATH/$INPUTFILE \
			  --geomfile=$GEOMETRYFILE
