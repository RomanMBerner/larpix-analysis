#!/usr/bin/python

############################################################################################
# Description:                                                                             #
# This (Python 2) script can be run over event-built data files.                           #
# It will generate ROOT trees which will be used for ArCLight studies/characterization.    #
# An event-built datafile is produced from the raw data file:                              #
# - Packets are time ordered, filtered on trigger type, etc.                               #
# - The events are constructed based on gaps in time.                                      #
# - Therefore, events can extend beyond a full drift period.                               #
# To build the event-built files, some cuts (e.g. nhit_cut = ...) have already been used.  #
############################################################################################

import argparse
from array import array
import collections
import glob
import h5py
import itertools
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import os
import ROOT
from ROOT import TCanvas, TFile, TProfile, TNtuple, TH1F, TH1D, TH2F, TH2D, TH1, TLine, TF1, TRandom3
from ROOT import gROOT, gBenchmark, gRandom, gSystem, gStyle
import scipy.stats
import time

import matplotlib as mpl
mpl.rc('image', cmap='viridis')

# Import plot functions
from plot_functions import *

# Import from other directories:
import sys
evd_libpath             = '/home/lhep/PACMAN/larpix-v2-testing-scripts/event-display/'
larpix_geometry_libpath = '/home/lhep/PACMAN/larpix-geometry/'
sys.path.append(evd_libpath)
sys.path.append(larpix_geometry_libpath)


def main(argv=None):
    start = time.time()

    # Turn on matplotlib interactive mode
    plt.ion()

    # ROOT batch mode
    ROOT.gROOT.SetBatch(1)


    '''
    # ============================================================
    # ArgumentParser
    # ============================================================
    parser = argparse.ArgumentParser(description='Cosmic Tracks')
    parser.add_argument("config_file")
    parser.add_argument("data_file",        nargs="+")
    parser.add_argument("-o", "--out_path")
    parser.add_argument("-i", "--uid",      type=int,            default=0,              help="Unique identifier used in output files.")
    parser.add_argument("-n", "--max_evts", type=int,            default=0, metavar="N", help="Stop after %(metavar)s events.")
    parser.add_argument("-n",               type=int,            default=0, metavar="N", help="Stop after %(metavar)s spills.")
    parser.add_argument("-s", "--seed",     type=int,                                    help="Set the RNG seed.")
    parser.add_argument("-m", "--measure",  action="store_true",                         help="Measure rho, phi and doca for each gamma and fill into histogram.")
    parser.add_argument("-a", "--analyse",  action="store_true",                         help="Run analysis.")
    parser.add_argument("-d", "--display",  action="store_true",                         help="Save event display CSVs.")
    parser.add_argument("-D", "--debug",    action="store_true",                         help="Only one event per spill, fixed vertex position.")
    args = parser.parse_args(argv)
    '''


    # ============================================================
    # Set paths
    # ============================================================
    datapath = '/home/lhep/PACMAN/DAQ/SingleModule_Nov2020/dataRuns/convertedData'
    print ' datapath: ', datapath

    outputpath = '/home/lhep/PACMAN/DAQ/SingleModule_Nov2020/dataRuns/rootTrees'
    print ' outputpath: ', outputpath

    files = sorted([os.path.basename(path) for path in glob.glob(datapath+'/*.h5')])
    print ' datafiles: '
    for f in files:
        print '\t ', f


    # ============================================================
    # Output Tree and File
    # ============================================================
    #inputFileName = (str(args.data_file)[34:])[:-7]   # excludes ending .root
    for file_number in range(len(files)):

        # Only process specific files
        if files[file_number] != 'datalog_2020_11_29_12_22_02_CET_evd.h5':
            continue

        #if not (file_number >= 0 and file_number < 10):
        #    continue

        inputFileName = files[file_number]

        # Define if plots are made or not
        make_plots = False

        print ' -------------------------------------- '
        print ' Processing file', inputFileName
        outFileName = inputFileName[:-7] + '.root'

        print ' Write data to: %s ' %(outputpath + '/' + outFileName)
        output_file = ROOT.TFile((outputpath + '/' + outFileName), "RECREATE")
        output_tree = ROOT.TTree("tracks", "tracks")

        if make_plots:
        # For plots, create folder
            output_folderName = outFileName[8:-9]
            try:
                os.system('rm -rf '+str(outputpath)+'/plots/'+str(output_folderName))
                os.system('mkdir '+str(outputpath)+'/plots/'+str(output_folderName))
                os.system('mkdir '+str(outputpath)+'/plots/'+str(output_folderName)+'/event_analysis')
                os.system('mkdir '+str(outputpath)+'/plots/'+str(output_folderName)+'/event_displays')
                os.system('mkdir '+str(outputpath)+'/plots/'+str(output_folderName)+'/track_analysis')
                os.system('mkdir '+str(outputpath)+'/plots/'+str(output_folderName)+'/track_displays')
            except:
                print ' --- WARNING: Exception in os commands --- '
                pass

        #os.system('ls /data/SingleModule_Nov2020/LArPix/dataRuns/rootTrees/')
        #os.system('ls /data/SingleModule_Nov2020/LArPix/dataRuns/rootTrees/plots/')
        #os.system('ls /data/SingleModule_Nov2020/LArPix/dataRuns/rootTrees/plots/'+str(output_folderName))

        MAXHITS = 5000

        # Event informations
        eventID              = array('i',[0])           # event ID [-]
        event_start_t        = array('i',[0])           # event timestamp start [UNITS?]
        event_end_t          = array('i',[0])           # event timestamp end [UNITS?]
        event_duration       = array('i',[0])           # event timestamp end - start [UNITS?]
        event_unix_ts        = array('i',[0])           # event unix timestamp [UNITS?]
        event_nhits          = array('i',[0])           # number of hits in the event [-]
        event_q              = array('f',[0.])          # total deposited charge [ke]
        event_q_raw          = array('f',[0.])          # total deposited raw charge [ke]
        event_ntracks        = array('i',[0])           # number of tracks [-]
        event_n_ext_trigs    = array('i',[0])           # number of external triggers [-]
	event_hits_x         = array('f',[0.]*MAXHITS)  # events hit coordinates (x)
        event_hits_y         = array('f',[0.]*MAXHITS)  # events hit coordinates (y)
        event_hits_z         = array('f',[0.]*MAXHITS)  # events hit coordinates (z)
        event_hits_ts        = array('f',[0.]*MAXHITS)  # events hit coordinates (timestamp)
	event_hits_q         = array('f',[0.]*MAXHITS)  # events hit charge (ke)

        # Track informations
        trackID              = array('i',[0])
        track_nhits          = array('i',[0])           # number of hits in the track [-]
        track_start_pos_x    = array('f',[0.])          # start position x of the track [mm]
        track_start_pos_y    = array('f',[0.])          # start position y of the track [mm]
        track_start_pos_z    = array('f',[0.])          # start position z of the track [mm]
        track_start_pos_t    = array('f',[0.])          # start position t of the track [0.1 us]
        track_end_pos_x      = array('f',[0.])          # end   position x of the track [mm]
        track_end_pos_y      = array('f',[0.])          # end   position y of the track [mm]
        track_end_pos_z      = array('f',[0.])          # end   position z of the track [mm]
        track_end_pos_t      = array('f',[0.])          # end   position t of the track [0.1 us]
        track_length         = array('f',[0.])          # length of the track [mm]
        track_nhits          = array('i',[0])           # number of hits in the track [-]
        track_q              = array('f',[0.])          # total deposited charge [ke]
        track_q_raw          = array('f',[0.])          # total deposited raw charge [ke]
        track_theta          = array('f',[0.])          # track theta
        track_phi            = array('f',[0.])          # track phi
        track_residual_x     = array('f',[0.])          # track residual x
        track_residual_y     = array('f',[0.])          # track residual y
        track_residual_z     = array('f',[0.])          # track residual z
        track_hits_x         = array('f',[0.]*MAXHITS)  # tracks hit coordinates (x)
        track_hits_y         = array('f',[0.]*MAXHITS)  # tracks hit coordinates (y)
        track_hits_z         = array('f',[0.]*MAXHITS)  # tracks hit coordinates (z)
        track_hits_ts        = array('f',[0.]*MAXHITS)  # tracks hit coordinates (timestamp)
        track_hits_q         = array('f',[0.]*MAXHITS)  # tracks hit charge (ke)

        # External Trigger informations
        trigID               = array('i',[0])     # trigger ID
        trig_type            = array('i',[0])     # trigger type (1-normal ; 2-external; 3-cross ; 4-periodic # TODO: IS THIS TRUE?)

        # Event
        output_tree.Branch("eventID"           ,eventID           ,"eventID/I")
        output_tree.Branch("event_start_t"     ,event_start_t     ,"event_start_t/I")     # 32 bit timestamp (2^32-1 = 2.147483647e9)
        output_tree.Branch("event_end_t"       ,event_end_t       ,"event_end_t/I")       # 32 bit timestamp (2^32-1 = 2.147483647e9)
        output_tree.Branch("event_duration"    ,event_duration    ,"event_duration/I")
        output_tree.Branch("event_unix_ts"     ,event_unix_ts     ,"event_unix_ts/I")
        output_tree.Branch("event_nhits"       ,event_nhits       ,"event_nhits/I")
        output_tree.Branch("event_q"           ,event_q           ,"event_q/F")
        output_tree.Branch("event_q_raw"       ,event_q_raw       ,"event_q_raw/F")
        output_tree.Branch("event_ntracks"     ,event_ntracks     ,"event_ntracks/I")
        output_tree.Branch("event_n_ext_trigs" ,event_n_ext_trigs ,"event_n_ext_trigs/I")
	output_tree.Branch("event_hits_x"      ,event_hits_x      ,"event_hits_x[event_nhits]/F")
	output_tree.Branch("event_hits_y"      ,event_hits_y      ,"event_hits_y[event_nhits]/F")
	output_tree.Branch("event_hits_z"      ,event_hits_z      ,"event_hits_z[event_nhits]/F")
	output_tree.Branch("event_hits_ts"     ,event_hits_ts     ,"event_hits_ts[event_nhits]/I")
	output_tree.Branch("event_hits_q"      ,event_hits_q      ,"event_hits_q[event_nhits]/F")

        # Tracks
        output_tree.Branch("trackID"          ,trackID          ,"trackID/I")
        output_tree.Branch("track_nhits"      ,track_nhits      ,"track_nhits/I")
        output_tree.Branch("track_start_pos_x",track_start_pos_x,"track_start_pos_x/F")
        output_tree.Branch("track_start_pos_y",track_start_pos_y,"track_start_pos_y/F")
        output_tree.Branch("track_start_pos_z",track_start_pos_z,"track_start_pos_z/F")
        output_tree.Branch("track_start_pos_t",track_start_pos_t,"track_start_pos_t/F")
        output_tree.Branch("track_end_pos_x"  ,track_end_pos_x  ,"track_end_pos_x/F")
        output_tree.Branch("track_end_pos_y"  ,track_end_pos_y  ,"track_end_pos_y/F")
        output_tree.Branch("track_end_pos_z"  ,track_end_pos_z  ,"track_end_pos_z/F")
        output_tree.Branch("track_end_pos_t"  ,track_end_pos_t  ,"track_end_pos_t/F")
        output_tree.Branch("track_length"     ,track_length     ,"track_length/F")
        output_tree.Branch("track_nhits"      ,track_nhits      ,"track_nhits/I")
        output_tree.Branch("track_q"          ,track_q          ,"track_q/F")
        output_tree.Branch("track_q_raw"      ,track_q_raw      ,"track_q_raw/F")
        output_tree.Branch("track_theta"      ,track_theta      ,"track_theta/F")
        output_tree.Branch("track_phi"        ,track_phi        ,"track_phi/F")
        output_tree.Branch("track_residual_x" ,track_residual_x ,"track_residual_x/F")
        output_tree.Branch("track_residual_y" ,track_residual_y ,"track_residual_y/F")
        output_tree.Branch("track_residual_z" ,track_residual_z ,"track_residual_z/F")
        output_tree.Branch("track_hits_x"     ,track_hits_x     ,"track_hits_x[track_nhits]/F")
        output_tree.Branch("track_hits_y"     ,track_hits_y     ,"track_hits_y[track_nhits]/F")
        output_tree.Branch("track_hits_z"     ,track_hits_z     ,"track_hits_z[track_nhits]/F")
        output_tree.Branch("track_hits_ts"    ,track_hits_ts    ,"track_hits_ts[track_nhits]/I")
        output_tree.Branch("track_hits_q"     ,track_hits_q     ,"track_hits_q[track_nhits]/F")

        # External Triggers
        output_tree.Branch("trigID"           ,trigID           ,"trigID/I")
        output_tree.Branch("trig_type"        ,trig_type        ,"trig_type/I")


        # ============================================================
        # Read input file
        # ============================================================
        f = h5py.File(os.path.join(datapath,inputFileName),'r')

        #print 'File has keys: ', [key for key in f.keys()]
        events    = f['events']
        hits      = f['hits']
        info      = f['info']
        tracks    = f['tracks']
        ext_trigs = f['ext_trigs']
        print ' n events:    ', len(events)
        print ' n hits:      ', len(hits)
        print ' n tracks:    ', len(tracks)
        print ' n ext_trigs: ', len(ext_trigs)


        # Loop over all events
        # ------------------------------------------
        for ev_index in range(len(f['events'])):
            event = f['events'][ev_index]
            if ev_index >= 0:
                break
            print ' ---- '
            print ' Event ID:              ', event['evid']
            print ' Event n hits:          ', event['nhit']
            print ' Event n tracks:        ', event['ntracks']
            print ' Event hit_ref:         ', event['hit_ref']
            print ' Event track_ref:       ', event['track_ref']
            print ' Event Q:               ', event['q']
            print ' Event Q raw:           ', event['q_raw']
            print ' Event ts_start:        ', event['ts_start']
            print ' Event ts_end:          ', event['ts_end']
            print ' Event duration:        ', event['ts_end'] - event['ts_start']
            print ' Event unix ts:         ', event['unix_ts']
            print ' Event n_ext_trigs:     ', event['n_ext_trigs']
            print ' Event ext_trig_ref:    ', event['ext_trig_ref']

        # Plot event summaries
        if make_plots:
            folder_name = str(outputpath)+'/plots/'+str(output_folderName)+'/event_analysis'
            plot_event_ts(f['events'],folder_name)
            plot_event_durations(f['events'],folder_name)
            plot_event_unix_ts(f['events'],folder_name)


        # Loop over tracks
        # ------------------------------------------
        for track_index in range(len(f['tracks'])):
            track = f['tracks'][track_index]
            if track_index >= 0:
                break
            print ' ---- '
            print ' Track ID:              ', track['track_id']
            print ' Track event_ref:       ', track['event_ref']
            print ' Track hit_ref:         ', track['hit_ref']
            print ' Track xp:              ', track['xp']
            print ' Track yp:              ', track['yp']
            print ' Track start:           ', track['start']
            print ' Track end:             ', track['end']
            print ' Track nhit:            ', track['nhit']
            print ' Track Q:               ', track['q']
            print ' Track Q raw:           ', track['q_raw']
            print ' Track ts_start:        ', track['ts_start']
            print ' Track ts_end:          ', track['ts_end']
            print ' Track duration:        ', track['ts_end']-track['ts_start']
            print ' Track lenght:          ', track['length']
            print ' Track theta:           ', track['theta']
            print ' Track phi:             ', track['phi']
            print ' Track residual:        ', track['residual']


        # Loop over external triggers and put event_IDs to list
        # ------------------------------------------
        # Types: 1-normal ; 2-external; 3-cross ; 4-periodic # TODO: IS THIS TRUE?
        event_IDs_externally_triggered = []
        for trig_index in range(len(f['ext_trigs'])):
            trig = f['ext_trigs'][trig_index]
            '''
            print ' ---- '
            print ' Trigger trig_id:       ', trig['trig_id']
            print ' Trigger event_ref:     ', trig['event_ref']
            print ' Trigger ts:            ', trig['ts']
            print ' Trigger type:          ', trig['type']
            print ' Trigger eventID:       ', f['events'][trig['event_ref']]['evid'][0]
            print ' Trigger ntracks in ev: ', f['events'][trig['event_ref']]['ntracks']
            '''
            #if trig['type'] == 2:
            event_IDs_externally_triggered.append(f['events'][trig['event_ref']]['evid'][0])
        #print ' eventIDs externally triggered: ', event_IDs_externally_triggered


        # Draw externally triggered events
        # ------------------------------------------
        if make_plots:
            for evID_index in range(len(event_IDs_externally_triggered)):
                event = f['events'][event_IDs_externally_triggered[evID_index]]
                #print ' evID:       ', event['evid']
                hit_ref = event['hit_ref']
                x = f['hits'][hit_ref]['px']
                y = f['hits'][hit_ref]['py']
                z = f['hits'][hit_ref]['ts'] - event['ts_start']
                q = f['hits'][hit_ref]['q']
                #q_raw = f['hits'][hit_ref]['q_raw']
                #iochain = f['hits'][hit_ref]['iochain']
                #chipid = f['hits'][hit_ref]['chipid']
                #channelid = f['hits'][hit_ref]['channelid']
                #geom = f['hits'][hit_ref]['geom']
                #event_ref = f['hits'][hit_ref]['event_ref']
                name='eventDisplay_evID_{}'.format(event['evid'])
                fig = vol3d_python2(x,y,z,q,name=name,fig=None)#,*geom_limits)
                plt.show()
                plt.savefig(str(outputpath)+'/plots/'+str(output_folderName)+'/event_displays/'+str(name)+'.png')
                plt.close()



        # Make track selection
        # ------------------------------------------
        good_track_mask = np.ones(len(f['tracks'])).astype(bool)
        #print 'good_track_mask (', len(good_track_mask), '): ', good_track_mask
        #print ' Track IDs of good tracks (', len(f['tracks']['track_id'][good_track_mask]), '): ', f['tracks']['track_id'][good_track_mask]


        if len(f['tracks']):
            # nhits
            # ----------------
            nhit_cut = 50
            good_track_mask = np.logical_and(f['tracks']['nhit'] > nhit_cut, good_track_mask)
            #print 'good_track_mask (', len(good_track_mask), '): ', good_track_mask
            #print ' Track IDs (', len(f['tracks']['track_id'][good_track_mask]), '): ', f['tracks']['track_id'][good_track_mask]
            #'''
            # length
            # ----------------
            length_cut = 300
            good_track_mask = np.logical_and(f['tracks']['length'] > length_cut, good_track_mask)
            '''
            # charge
            # ----------------
            #residual_cut =
            #good_track_mask = np.logical_and(np.linalg.norm(f['tracks']['q'][:,:3],axis=-1) < residual_cut, good_track_mask)
            # residual
            # ----------------
            #residual_cut =
            #good_track_mask = np.logical_and(np.linalg.norm(f['tracks']['residual'][:,:3],axis=-1) < residual_cut, good_track_mask)
            # theta
            # ----------------
            #theta_cut =
            #good_track_mask = np.logical_and(np.abs(f['tracks']['theta']) > theta_cut, good_track_mask)
            #good_track_mask = np.logical_and(np.abs(np.pi - f['tracks']['theta']) > theta_cut, good_track_mask)
            '''
            # hits per event fraction
            # ----------------
            event_frac_cut = 0.8
            event_hits = np.array([f['events'][ f['tracks'][i]['event_ref'] ]['nhit'][0] for i in range(len(f['tracks']))])
            event_fraction = f['tracks']['nhit']/event_hits
            good_track_mask = np.logical_and(event_fraction > event_frac_cut, good_track_mask)

            # Only accept those tracks which have an external trigger:
            print ' len(good_track_mask): ', len(good_track_mask)
            for track_index, track in enumerate(f['tracks']):
                if f['events'][track['event_ref']]['evid'][0] in event_IDs_externally_triggered:
                    good_track_mask[track_index] = True
                    #print ' ===== evID:    ', f['events'][track['event_ref']]['evid'][0]
                    #print ' ===== trackID: ', track['track_id']
                else:
                    good_track_mask[track_index] = False


            # Plot selected tracks
            if make_plots:
                folder_name = 'plots/'+str(output_folderName)+'/track_analysis/'
                plot_track_nhits      (f['tracks'],f['tracks'][good_track_mask],folder_name)
                plot_track_lengths    (f['tracks'],f['tracks'][good_track_mask],folder_name)
                plot_track_charge     (f['tracks'],f['tracks'][good_track_mask],folder_name)
                plot_track_charge_raw (f['tracks'],f['tracks'][good_track_mask],folder_name)
                plot_track_residuals  (f['tracks'],f['tracks'][good_track_mask],folder_name)
                plot_track_theta      (f['tracks'],f['tracks'][good_track_mask],folder_name)
                plot_track_phi        (f['tracks'],f['tracks'][good_track_mask],folder_name)
                plot_track_start_z    (f['tracks'],f['tracks'][good_track_mask],folder_name)
                plot_track_start_t    (f['tracks'],f['tracks'][good_track_mask],folder_name)
                plot_track_end_z      (f['tracks'],f['tracks'][good_track_mask],folder_name)
                plot_track_end_t      (f['tracks'],f['tracks'][good_track_mask],folder_name)
                plot_track_duration   (f['tracks'],f['tracks'][good_track_mask],folder_name)
                plot_track_start_xy   (f['tracks'],f['tracks'][good_track_mask],folder_name)
                plot_track_end_xy     (f['tracks'],f['tracks'][good_track_mask],folder_name)


        # Loop over all tracks in input file and fill output_tree
        # ------------------------------------------
        if len(f['tracks'][good_track_mask]) > 0:
            print ' Number of selected tracks in file', inputFileName, ':', len(f['tracks'][good_track_mask])

            for track_index, track in enumerate(f['tracks'][good_track_mask]):
                #if track_index > 0:
                #    break

                # Only get those tracks where the event got externally triggered
                if f['events'][track['event_ref']]['evid'][0] not in event_IDs_externally_triggered:
                    continue
                '''
                print ' track_index:    ', track_index
                print ' track:          ', track
                print ' event ID:       ', f['events'][track['event_ref']]['evid'][0]
                print ' track ID:       ', track['track_id']
                print ' event unix ts:  ', f['events'][track['event_ref']]['unix_ts'][0]
                print ' event start t:  ', f['events'][track['event_ref']]['ts_start'][0]
                print ' event end   t:  ', f['events'][track['event_ref']]['ts_end'][0]
                print ' event duration: ', f['events'][track['event_ref']]['ts_end'][0] -\
                                           f['events'][track['event_ref']]['ts_start'][0]
                print ' track start t:  ', track['start'][3]
                print ' track end t:    ', track['end'][3]
                print ' track start z:  ', track['start'][2]
                print ' track end z:    ', track['end'][2]
                print ' track ID:       ', track['track_id']
                print ' tr start:       ', track['start']
                print ' tr end:         ', track['end']
                print ' tr start-end:   ', track['start']-track['end']
                print ' tr. lenght:     ', track['length']
                '''

                eventID[0]           = f['events'][track['event_ref']]['evid'][0]
                event_start_t[0]     = f['events'][track['event_ref']]['ts_start'][0]
                event_end_t[0]       = f['events'][track['event_ref']]['ts_end'][0]
                event_duration[0]    = f['events'][track['event_ref']]['ts_end'][0] -\
                                       f['events'][track['event_ref']]['ts_start'][0]
                #print ' start:    ', event_start_t[0]
                #print ' end:      ', event_end_t[0]
                #print ' duration: ', event_duration[0]
                event_unix_ts[0]     = f['events'][track['event_ref']]['unix_ts']
                event_nhits[0]       = f['events'][track['event_ref']]['nhit'][0]
                event_q[0]           = f['events'][track['event_ref']]['q'][0]
                event_q_raw[0]       = f['events'][track['event_ref']]['q_raw'][0]
                event_ntracks[0]     = f['events'][track['event_ref']]['ntracks'][0]
                event_n_ext_trigs[0] = f['events'][track['event_ref']]['n_ext_trigs'][0]
                event_n_ext_trigs[0] = f['events'][track['event_ref']]['n_ext_trigs'][0]
                for hit in range(f['events'][track['event_ref']]['nhit'][0]):
                    hit_reference = f['events'][track['event_ref']]['hit_ref'][0]
                    #print ' == hit_reference: ', hit_reference
                    #print ' == x:  ', f['hits'][hit_reference]['px'][hit]
                    #print ' == y:  ', f['hits'][hit_reference]['py'][hit]
                    #print ' == z:  ', f['hits'][hit_reference]['pz'][hit]
                    #print ' == ts: ', f['hits'][hit_reference]['ts'][hit]
                    #print ' == q:  ', f['hits'][hit_reference]['q'][hit]
                    event_hits_x[hit]  = f['hits'][hit_reference]['px'][hit]
                    event_hits_y[hit]  = f['hits'][hit_reference]['py'][hit]
                    event_hits_z[hit]  = f['hits'][hit_reference]['pz'][hit]
                    event_hits_ts[hit] = f['hits'][hit_reference]['ts'][hit]
                    event_hits_q[hit]  = f['hits'][hit_reference]['q'][hit]

                trackID[0]           = track['track_id']
                track_nhits[0]       = track['nhit']
                track_start_pos_x[0] = track['start'][0]
                track_start_pos_y[0] = track['start'][1]
                track_start_pos_z[0] = track['start'][2]
                track_start_pos_t[0] = track['start'][3]
                track_end_pos_x[0]   = track['end'][0]
                track_end_pos_y[0]   = track['end'][1]
                track_end_pos_z[0]   = track['end'][2]
                track_end_pos_t[0]   = track['end'][3]
                track_length[0]      = track['length']
                track_theta[0]       = track['theta']
                track_phi[0]         = track['phi']
                track_residual_x[0]  = track['residual'][0]
                track_residual_y[0]  = track['residual'][1]
                track_residual_z[0]  = track['residual'][2]
                track_nhits[0]       = track['nhit']
                track_q[0]           = track['q']
                track_q_raw[0]       = track['q_raw']
                for hit in range(track['nhit']):
                    track_hits_x[hit]  = f['hits'][track['hit_ref']]['px'][hit]
                    track_hits_y[hit]  = f['hits'][track['hit_ref']]['py'][hit]
                    track_hits_z[hit]  = f['hits'][track['hit_ref']]['pz'][hit]
                    track_hits_ts[hit] = f['hits'][track['hit_ref']]['ts'][hit]
                    track_hits_q[hit]  = f['hits'][track['hit_ref']]['q'][hit]
                    #print ' hit: ', hit, ' \t x: ', f['hits'][track['hit_ref']]['px'][hit]
                    #print ' hit: ', hit, ' \t y: ', f['hits'][track['hit_ref']]['py'][hit]
                    #print ' hit: ', hit, ' \t z: ', f['hits'][track['hit_ref']]['pz'][hit]
                    #print ' hit: ', hit, ' \t t: ', f['hits'][track['hit_ref']]['ts'][hit]
                    #print ' hit: ', hit, ' \t q: ', f['hits'][track['hit_ref']]['q'][hit]

                #trigId               = f['ext_trigs'][f['events'][track['event_ref']]['ext_trig_ref']]['trig_id']
                #trig_type            = f['ext_trigs'][f['events'][track['event_ref']]['ext_trig_ref']]['trig_type']


                # Draw selected tracks (only if there are < max_tracks per selected event)
                if make_plots:
                    max_tracks = 10
                    folder_name = 'plots/'+str(output_folderName)+'/track_displays/'
                    if f['events'][track['event_ref']]['ntracks'] < 10:
                        draw_track(f,track,folder_name)
                    else:
                        command = '> '+str(folder_name)+'event_'\
                                      +str(f['events'][track['event_ref']]['evid'][0])+'_has_'\
                                      +str(f['events'][track['event_ref']]['ntracks'][0])\
                                      +'_tracks_thus_do_not_plot_them'
                        os.system(command)


                # Fill to output_tree
                output_tree.Fill()

                if track_index%2000 == 0:
                    print ' Processed track', track_index, 'of', len(f['tracks'][good_track_mask])


        # Write output_tree
        # ---------------------------------------------------------
        output_file.cd()
        output_tree.Write()
        print ' Data has been written to %s ' %(outputpath + '/' + outFileName)


if __name__ == "__main__":
    main()
