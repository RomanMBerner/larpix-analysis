#!/usr/local/bin/python3.8

# ------------------------------------------------------------------------------------------------------------- #
# Description:
# TODO
# ------------------------------------------------------------------------------------------------------------- #

#import argparse
#from array import array
import glob
import math
import numpy as np
import os
import ROOT
from ROOT import TCanvas, TFile, TProfile, TNtuple, TH1F, TH2F, TH3F, TH1, TLine
#from ROOT import gROOT, gBenchmark, gRandom, gSystem, Double, gStyle
import time
from plot_functions import *


def main(argv=None):
    start = time.time()

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
    datapath = '/data/SingleModule_Nov2020/LArPix/dataRuns/rootTrees/combined_with_light'
    print(' datapath:   ', datapath)

    outputpath = '/home/lhep/PACMAN/larpix-analysis/lightCharge_anticorrelation'
    print(' outputpath: ', outputpath)

    files = sorted([os.path.basename(path) for path in glob.glob(datapath+'/*.root')])
    print(' datafiles:  ')
    for f in files:
        print('              ', f)


    # ============================================================
    # Define voxelisation
    # ============================================================
    n_voxels_x = 70
    n_voxels_y = 70
    n_voxels_z = 70
    pitch_x = 4.434
    pitch_y = 4.434
    pitch_z = 4.434
    x_min = - pitch_x * n_voxels_x/2. #155.19
    x_max =   pitch_x * n_voxels_x/2. #155.19
    y_min = - pitch_y * n_voxels_y/2. #155.19
    y_max =   pitch_y * n_voxels_y/2. #155.19
    #z_min = - pitch_z * n_voxels_z/2. #155.19
    #z_max =   pitch_z * n_voxels_z/2. #155.19
    #y_min = -155.19
    #y_max = 155.19
    z_min = 0
    z_max = 400


    # ============================================================
    # Input tree
    # ============================================================
    #inputFileName = (str(args.data_file)[34:])[:-7]   # excludes ending .root
    for file_number in range(len(files)):

        # Only process specific file(s)
        #if files[file_number] != 'datalog_2020_11_29_12_22_02_CET_evd.h5':
        #    continue

        #if not (file_number >= 0 and file_number < 10):
        #    continue

        inputFileName = files[file_number]

        print(' -------------------------------------- ')
        print(' Processing file', inputFileName)
        outFileName = inputFileName[:-7] + '.root'


        input_tree = ROOT.TChain("t_out", "t_out")
        #for root_file in config["data_files"]:
        #    input_tree.Add(root_file)
        #input_tree.Add( "/path/to.root" )
        input_tree.Add(datapath + '/' + inputFileName)

        #not_used_files = [13,32,50,61,66,77,81,86,89,92,94,97,99]
        #print " Do not use files with numbers in {:} " .format(not_used_files)


        # Define if plots are made or not
        make_plots = True
        if make_plots:
            plot_folder = inputFileName[16:-5]
            os.system('rm -rf plots/' + str(plot_folder))
            os.system('mkdir plots/' + str(plot_folder))

        # Turn on all branches
        input_tree.SetBranchStatus( "*", 1 )


    # Define Histograms / NTuples
    # ---------------------------------------------------------
    makePlots = True
    h1_trLength          = TH1F('h1_trLength'         ,' ; Track length [mm] ; Entries [-]',                      150, 0, 500)
    h2_trLength_vs_nHits = TH2F('h2_trLength_vs_nHits',' ; Track Length [mm] ; Number of Hits [-] ; Entries [-]', 100, 0, 500, 100, 0, 500)
    h3_event_hits        = TH3F('h3_event_hits'       ,' ; x ; y; z'                                            , 70, -155, 155, 70, -155, 155, 100, -300, 3000)
    ntuple               = TNtuple('ntuple','data from ascii file','x:y:z:cont')
    plot4d               = TH3F('h3_ev_hits',' ; x ; y; z', 70, -155.19, 155.19, 70, -155.19, 155.19, 200, -500, 1500)


    # Make track selection
    # ---------------------------------------------------------
    # TODO: Make 3D histogram to test selection goodness
    # Event with only 1 track (right?)


    # Analyse input tree
    # ---------------------------------------------------------
    n_tracks = input_tree.GetEntries()
    print(' n_tracks: ', n_tracks)

    x_min = 100
    x_max = -100
    y_min = 100
    y_max = -100
    z_min = 100
    z_max = -100

    # Loop over all tracks in input_tree
    for track_id in range(n_tracks):
        input_tree.GetEntry(track_id)
        
        print(' Processing track', track_id, 'of', n_tracks, '...')

        if track_id > 5:
            break

        #print(' t_eventID:     ', input_tree.t_eventID)
        #print(' t_trackID:     ', input_tree.t_trackID)
        #print(' t_event_q:     ', input_tree.t_event_q)
        #print(' t_track_q:     ', input_tree.t_track_q)
        #print(' t_event_nhits: ', input_tree.t_event_nhits)
        #print(' t_track_nhits: ', input_tree.t_track_nhits)

        h1_trLength.Fill(input_tree.t_track_length)
        h2_trLength_vs_nHits.Fill(input_tree.t_track_length,input_tree.t_track_nhits)

        # Get all hits in the event
        voxels = np.zeros((n_voxels_x, n_voxels_y, n_voxels_z))

        for hit in range(10): #input_tree.t_event_nhits):
            if input_tree.t_event_hits_x[hit] < x_min:
                x_min = input_tree.t_event_hits_x[hit]
            if input_tree.t_event_hits_x[hit] > x_max:
                x_max = input_tree.t_event_hits_x[hit]
            if input_tree.t_event_hits_y[hit] < y_min:
                y_min = input_tree.t_event_hits_y[hit]
            if input_tree.t_event_hits_y[hit] > y_max:
                y_max = input_tree.t_event_hits_y[hit]
            if input_tree.t_event_hits_z[hit] < z_min:
                z_min = input_tree.t_event_hits_z[hit]
            if input_tree.t_event_hits_z[hit] > z_max:
                z_max = input_tree.t_event_hits_z[hit]

            #print(' hit: ', hit, ' \t x: ', input_tree.t_event_hits_x[hit], '\t y: ', input_tree.t_event_hits_y[hit], ' \t z: ', input_tree.t_event_hits_z[hit])

            voxel_x = math.floor((input_tree.t_event_hits_x[hit]+(pitch_x*(n_voxels_x)/2.))/pitch_x)
            voxel_y = math.floor((input_tree.t_event_hits_y[hit]+(pitch_y*(n_voxels_y)/2.))/pitch_y)
            voxel_z = math.floor((input_tree.t_event_hits_z[hit]+(pitch_z*(n_voxels_z)/2.))/pitch_z)

            #print(' voxel_x: ', voxel_x, ' \t voxel_y: ', voxel_y, ' \t voxel_z: ', voxel_z)
            if voxel_x<n_voxels_x and voxel_y<n_voxels_y and voxel_z<n_voxels_z:
                voxels[voxel_x][voxel_y][voxel_z] += input_tree.t_event_hits_q[hit]
            # TODO: make under- and overflow voxel for every coordinate


            h3_event_hits.Fill(input_tree.t_event_hits_x[hit],input_tree.t_event_hits_y[hit],input_tree.t_event_hits_z[hit],input_tree.t_event_hits_q[hit])

        for vox_x in range(n_voxels_x):
            vox_x_middle = x_min + (vox_x+0.5)*pitch_x
            for vox_y in range(n_voxels_y):
                vox_y_middle = y_min + (vox_y+0.5)*pitch_y
                for vox_z in range(n_voxels_z):
                    vox_z_middle = z_min + (vox_z+0.5)*pitch_z
                    if voxels[vox_x][vox_y][vox_z] > 0:
                        ntuple.Fill(vox_x_middle,vox_y_middle,vox_z_middle,voxels[vox_x][vox_y][vox_z])
                        #h3_event_hits.Fill(vox_x_middle,vox_y_middle,vox_z_middle,voxels[vox_x][vox_y][vox_z])

        if(track_id%2==0):
            now = time.time()
            print(' Processed', math.floor(track_id*100/n_tracks), 'of', n_tracks, 'tracks. \t Elapsed time:', (now-start), ' seconds ... \r')

    print(' x_min: ', x_min)
    print(' x_max: ', x_max)
    print(' y_min: ', y_min)
    print(' y_max: ', y_max)
    print(' z_min: ', z_min)
    print(' z_max: ', z_max)

    c0 = ROOT.TCanvas()
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)
    ntuple.Draw('x:y:z:cont>>plot4d','','COLZ')
    #plot4d.SetLabelSize(0.5)
    #plot4d.SetMarkerSize(300)
    #ntuple.SetMarkerSize(300)
    #ntuple.SetMarkerColor(2)
    #ntuple.SetFillColor(38)
    #h3_event_hits.Draw("COLZ")
    c0.Print('test.png')

    # Make plots
    # ---------------------------------------------------------
    if makePlots:
        plot_h1_trLength(h1_trLength,'h1_trLength',plot_folder)
        plot_h2_trLength_vs_nHits(h2_trLength_vs_nHits,'h2_trLength_vs_nHits',plot_folder)
        plot_h3_event_hits(h3_event_hits,'h3_event_hits',plot_folder)



if __name__ == "__main__":
    main()
