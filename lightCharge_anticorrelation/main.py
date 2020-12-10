#!/usr/local/bin/python3.7

# ------------------------------------------------------------------------------------------------------------- #
# Description:
# TODO
# ------------------------------------------------------------------------------------------------------------- #

import argparse
from array import array
import math
import numpy as np
import os
import ROOT
from ROOT import TCanvas, TFile, TProfile, TNtuple, TH1F, TH2F, TH1, TLine
from ROOT import gROOT, gBenchmark, gRandom, gSystem, Double, gStyle
import time


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
    print(' datapath: ', datapath)

    outputpath = '/home/lhep/PACMAN/larpix-analysis/lightCharge_anticorrelation'
    print(' outputpath: ', outputpath)

    files = sorted([os.path.basename(path) for path in glob.glob(datapath+'/*.root')])
    print(' datafiles: ')
    for f in files:
        print(' \t ', f)


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

        # Define if plots are made or not
        make_plots = False

        print(' -------------------------------------- ')
        print(' Processing file', inputFileName)
        outFileName = inputFileName[:-7] + '.root'


        input_tree = ROOT.TChain("pi0_showers", "pi0_showers")
        #for root_file in config["data_files"]:
        #    input_tree.Add(root_file)
        #input_tree.Add( "/home/roman/pi0_study/generated_trees/root_files/MINERvA_2x2_100evt.root" )
        input_tree.Add( "/home/roman/pi0_study/generated_trees/root_files/g4_ArgonCubeNuMI_0000.root" )

        #not_used_files = [13,32,50,61,66,77,81,86,89,92,94,97,99]
        #print " Do not use files with numbers in {:} " .format(not_used_files)

        #print " Used files for the plots are listed below: "
        #for filenumber in range(0,100):
        #    if filenumber in not_used_files:
        #        continue
        #    if filenumber<10:
        #        filename = "/home/roman/pi0_study/generated_trees/root_files/g4_ArgonCubeNuMI_000" + str(filenumber) + ".root"
        #    else:
        #        filename = "/home/roman/pi0_study/generated_trees/root_files/g4_ArgonCubeNuMI_00" + str(filenumber) + ".root"
        #    input_tree.Add(filename)
        #    print " \t {:} " .format(filename)


        # Turn on all branches
        input_tree.SetBranchStatus( "*", 1 )


    '''
    # Gauss smearing
    # ---------------------------------------------------------
    h1_gauss_smearing                              = TH1F( "h1_gauss_smearing", " ; #pi^{0} Kinetic Energy [GeV] ; # contained #pi^{0} induced showers [-]", 100, 0.4, 1.6)
    for num in range(10000):
        h1_gauss_smearing.Fill(smear_photon_energy(1))
    func = ROOT.TF1("Name", "gaus")
    c0 = ROOT.TCanvas()
    histo_file = ROOT.TFile(("/home/roman/pi0_study/test.root"), "RECREATE")
    h1_gauss_smearing.Write()
    h1_gauss_smearing.Draw()
    h1_gauss_smearing.Fit(func)
    c0.Print("/home/roman/pi0_study/test.png")
    histo_file.Close()
    '''

    '''
    # Define Histograms
    # ---------------------------------------------------------
    if plot_h2_true_vs_reco_pi0_etot:
        h2_true_vs_reco_pi0_etot                        = TH2F( 'h2_true_vs_reco_pi0_etot',
                                                                ' ; True #pi^{0} Total Energy [GeV] ; Reconstructed #pi^{0} Total Energy [GeV] ; Number of EM Shower Pairs',
                                                                100, 0, 5, 100, 0, 5 )
    if plot_h2_true_vs_reco_pi0_ekin:
        h2_true_vs_reco_pi0_ekin                        = TH2F( 'h2_true_vs_reco_pi0_ekin',
                                                                ' ; True #pi^{0} Kinetic Energy [GeV] ; Reconstructed #pi^{0} Kinetic Energy [GeV] ; Number of EM Shower Pairs',
                                                                100, 0, 5, 100, 0, 5 )
        h2_true_vs_reco_pi0_ekin_true_angle             = TH2F( 'h2_true_vs_reco_pi0_ekin_true_angle',
                                                                ' ; True #pi^{0} Kinetic Energy [GeV] ; Reconstructed #pi^{0} Kinetic Energy [GeV] ; Number of EM Shower Pairs',
                                                                100, 0, 5, 100, 0, 5 )
        h2_true_vs_reco_pi0_ekin_true_alpha             = TH2F( 'h2_true_vs_reco_pi0_ekin_true_alpha',
                                                                ' ; True #pi^{0} Kinetic Energy [GeV] ; Reconstructed #pi^{0} Kinetic Energy [GeV] ; Number of EM Shower Pairs',
                                                                100, 0, 5, 100, 0, 5 )
        h2_true_vs_reco_pi0_ekin_true_angle_true_alpha  = TH2F( 'h2_true_vs_reco_pi0_ekin_true_angle_true_alpha',
                                                                ' ; True #pi^{0} Kinetic Energy [GeV] ; Reconstructed #pi^{0} Kinetic Energy [GeV] ; Number of EM Shower Pairs',
                                                                100, 0, 5, 100, 0, 5 )
    if plot_h2_true_vs_reco_pi0_ekin:
        h1_pi0_mass_reco                                = TH1F( 'h1_pi0_mass_reco',
                                                                ' ; Reconstructed #pi^{0} Mass [GeV/c^{2}] ; Number of EM Shower Pairs',
                                                                150, -1.5, 1.5 )
    '''

    '''
    # Analyse input_tree
    # ---------------------------------------------------------
    n_evt = input_tree.GetEntries()
    print(" Processing {:d} events..." .format(n_evt))
    '''


    '''
    # Loop over all entries in input_tree; every entry is one neutrino interaction
    for evt_id in range(0,3): #(n_evt):
        input_tree.GetEntry(evt_id)
        if print_general_event_infos: print " =============================== \n        eventID. %d        \n =============================== " % evt_id

        # Get the track IDs of the pi0s and gammas
        pi0_IDs         = list()
        gamma_IDs       = list()

        for iq in range(input_tree.nq_in_all_EM_showers):
            if input_tree.from_gamma_ID[iq] not in gamma_IDs:
                gamma_IDs.append(input_tree.from_gamma_ID[iq])
                pi0_IDs.append(input_tree.from_pi0_ID[iq])

        if print_general_event_infos: print " gamma_IDs: \t {:} " .format(gamma_IDs)
        if print_general_event_infos: print " pi0_IDs: \t {:} " .format(pi0_IDs)

        n_EM_showers = len(gamma_IDs)
        #print " N EM showers: \t %d (where %d are useful) \n " %(input_tree.n_EM_showers_in_event,n_EM_showers)


        # Some definitions
        pi0_ekin_true       = [0.]*n_EM_showers
        pi0_ekin_reco       = [0.]*n_EM_showers
        gamma_start_in_FV   = [0]*n_EM_showers
        gamma_ekin_true     = [0.]*n_EM_showers
        gamma_ekin_smeared  = [0.]*n_EM_showers
        alpha_true          = [-1.]*n_EM_showers
        alpha_measured      = [-1.]*n_EM_showers
        gamma_start_pos_x   = [0.]*n_EM_showers
        gamma_start_pos_y   = [0.]*n_EM_showers
        gamma_start_pos_z   = [0.]*n_EM_showers
        gamma_dir_x         = [0.]*n_EM_showers
        gamma_dir_y         = [0.]*n_EM_showers
        gamma_dir_z         = [0.]*n_EM_showers
        gamma_dir_x_smeared = [0.]*n_EM_showers
        gamma_dir_y_smeared = [0.]*n_EM_showers
        gamma_dir_z_smeared = [0.]*n_EM_showers
        gamma_theta_true    = [0.]*n_EM_showers
        gamma_theta_smeared = [0.]*n_EM_showers
        gamma_phi_true      = [0.]*n_EM_showers
        gamma_phi_smeared   = [0.]*n_EM_showers
        gamma_angle_true    = [0.]*n_EM_showers
        gamma_angle_smeared = [0.]*n_EM_showers
        pi0_etot_reco       = [0.]*n_EM_showers
        pi0_mass_reco       = [0.]*n_EM_showers

        edeps_total         = [0.]*n_EM_showers
        edeps_contained     = [0.]*n_EM_showers
        edeps_not_contained = [0.]*n_EM_showers
        edeps_containment   = [0.]*n_EM_showers


        # Get the EM_shower's true energy, direction and true angles (theta and phi)
        for ID_in_list in range(n_EM_showers):
            for ID_in_tree in range(input_tree.n_EM_showers_in_event):
                if gamma_IDs[ID_in_list] == input_tree.gamma_ID[ID_in_tree]:
                    pi0_ekin_true[ID_in_list]       = input_tree.pi0_ekin[ID_in_tree]
                    gamma_ekin_true[ID_in_list]     = input_tree.gamma_ekin[ID_in_tree]
                    gamma_start_pos_x[ID_in_list]   = input_tree.gamma_start_pos_x[ID_in_tree]
                    gamma_start_pos_y[ID_in_list]   = input_tree.gamma_start_pos_y[ID_in_tree]
                    gamma_start_pos_z[ID_in_list]   = input_tree.gamma_start_pos_z[ID_in_tree]
                    gamma_dir_x[ID_in_list]         = input_tree.gamma_dir_x[ID_in_tree]
                    gamma_dir_y[ID_in_list]         = input_tree.gamma_dir_y[ID_in_tree]
                    gamma_dir_z[ID_in_list]         = input_tree.gamma_dir_z[ID_in_tree]
                    gamma_theta_true[ID_in_list]    = input_tree.gamma_theta[ID_in_tree]
                    gamma_phi_true[ID_in_list]      = input_tree.gamma_phi[ID_in_tree]
                    break


        # Loop over all EM showers
        for EM_shower in range(n_EM_showers):

            # Shower start in FV
            gamma_pos_start = np.array((gamma_start_pos_x[EM_shower], gamma_start_pos_x[EM_shower], gamma_start_pos_x[EM_shower]))
            if np.all(gamma_pos_start > fiducial_vol[0]) and np.all(gamma_pos_start < fiducial_vol[1]):
                gamma_start_in_FV[EM_shower] = 1
            else:
                gamma_start_in_FV[EM_shower] = 0
                continue

            # Energy deposits and containment (in the defined FV)
            for iq in range(input_tree.nq_in_all_EM_showers):
                if input_tree.from_gamma_ID[iq] == gamma_IDs[EM_shower]:
                    edep_pos = np.array((input_tree.xq[iq], input_tree.yq[iq], input_tree.zq[iq]))
                    edeps_total[EM_shower] += input_tree.dq[iq]
                    if np.all(edep_pos > fiducial_vol[0]) and np.all(edep_pos < fiducial_vol[1]):
                        edeps_contained[EM_shower] += input_tree.dq[iq]
                    else:
                        edeps_not_contained[EM_shower] += input_tree.dq[iq]

            # MeV -> GeV
            edeps_total[EM_shower] /= 1000
            edeps_contained[EM_shower] /= 1000
            edeps_not_contained[EM_shower] /= 1000

            edeps_containment[EM_shower]   = edeps_contained[EM_shower]/gamma_ekin_true[EM_shower]

            # Smear sum of contained edeps with a gaussian with sigma = 3% + sqrt(contained_energy)*5% of the energy
            if not edeps_contained[EM_shower]>0: continue
            gamma_ekin_smeared[EM_shower] = smear_photon_energy(edeps_contained[EM_shower])

            # Get the true alpha and angle between the gamma from this shower and the other gamma (from the same pi0)
            for EM_shower1 in range(n_EM_showers):
                for EM_shower2 in range(n_EM_showers):
                    if pi0_IDs[EM_shower1] == pi0_IDs[EM_shower2] and gamma_IDs[EM_shower1] != gamma_IDs[EM_shower2]:
                        gamma_angle_true[EM_shower1] = calc_gamma_opening_angle( gamma_dir_x[EM_shower1],gamma_dir_y[EM_shower1],gamma_dir_z[EM_shower1],
                                                                                gamma_dir_x[EM_shower2],gamma_dir_y[EM_shower2],gamma_dir_z[EM_shower2] )
                        alpha_true[EM_shower1] = calc_alpha(gamma_ekin_true[EM_shower1],gamma_ekin_true[EM_shower2])
                        break

        # Determine alpha_measured
        for EM_shower1 in range(n_EM_showers):
            if(gamma_start_in_FV[EM_shower1]==0): continue
            if not edeps_contained[EM_shower1]>0: continue
            for EM_shower2 in range(n_EM_showers):
                if(gamma_start_in_FV[EM_shower2]==0): continue
                if not edeps_contained[EM_shower2]>0: continue
                if gamma_IDs[EM_shower1] != gamma_IDs[EM_shower2] and pi0_IDs[EM_shower1] == pi0_IDs[EM_shower2]:
                    alpha_measured[EM_shower1] = calc_alpha(edeps_contained[EM_shower1],edeps_contained[EM_shower2])
                    alpha_measured[EM_shower2] = calc_alpha(edeps_contained[EM_shower1],edeps_contained[EM_shower2])
                    break

        # Smear the true gamma direction vector
        for EM_shower in range(n_EM_showers):
            if(gamma_start_in_FV[EM_shower]==0): continue
            if not edeps_contained[EM_shower]>0: continue
            smeared_direction = smear_photon_direction(gamma_dir_x[EM_shower],gamma_dir_y[EM_shower],gamma_dir_z[EM_shower])
            gamma_dir_x_smeared[EM_shower] = smeared_direction[0]
            gamma_dir_y_smeared[EM_shower] = smeared_direction[1]
            gamma_dir_z_smeared[EM_shower] = smeared_direction[2]

        # Calculate the smeared photon opening angle
        for EM_shower1 in range(n_EM_showers):
            if(gamma_start_in_FV[EM_shower1]==0): continue
            if not edeps_contained[EM_shower1]>0: continue
            for EM_shower2 in range(n_EM_showers):
                if(gamma_start_in_FV[EM_shower2]==0): continue
                if not edeps_contained[EM_shower2]>0: continue
                if pi0_IDs[EM_shower1] == pi0_IDs[EM_shower2] and gamma_IDs[EM_shower1] != gamma_IDs[EM_shower2]:
                    gamma_angle_smeared[EM_shower1] = calc_gamma_opening_angle(gamma_dir_x_smeared[EM_shower1],gamma_dir_y_smeared[EM_shower1],gamma_dir_z_smeared[EM_shower1],
                                                                               gamma_dir_x_smeared[EM_shower2],gamma_dir_y_smeared[EM_shower2],gamma_dir_z_smeared[EM_shower2])
                    break

        # Determine the pi0 total energy from the measured alpha and the smeared gamma opening angles
        used_pi0_IDs = list()
        del used_pi0_IDs[:]
        for EM_shower in range(n_EM_showers):
            if(gamma_start_in_FV[EM_shower]==0):    continue
            if not edeps_contained[EM_shower]>0:    continue
            if(gamma_angle_smeared[EM_shower]==0):  continue
            pi0_ekin_reco[EM_shower] = reconstruct_pi0_etot(alpha_measured[EM_shower],gamma_angle_smeared[EM_shower]) - pi0_mass
            pi0_etot_reco[EM_shower] = reconstruct_pi0_etot(alpha_measured[EM_shower],gamma_angle_smeared[EM_shower])
            pi0_mass_reco[EM_shower] = reconstruct_pi0_etot(alpha_measured[EM_shower],gamma_angle_smeared[EM_shower]) - pi0_ekin_true[EM_shower]

            if pi0_IDs[EM_shower] not in used_pi0_IDs:
                used_pi0_IDs.append(pi0_IDs[EM_shower])
                if plot_h2_true_vs_reco_pi0_etot:
                    h2_true_vs_reco_pi0_etot.Fill(pi0_ekin_true[EM_shower]+pi0_mass,pi0_etot_reco[EM_shower])
                if plot_h2_true_vs_reco_pi0_ekin:
                    h2_true_vs_reco_pi0_ekin.Fill(pi0_ekin_true[EM_shower],pi0_ekin_reco[EM_shower])
                    h2_true_vs_reco_pi0_ekin_true_angle.Fill(pi0_ekin_true[EM_shower],reconstruct_pi0_etot(alpha_measured[EM_shower],gamma_angle_true[EM_shower]) - pi0_mass)
                    h2_true_vs_reco_pi0_ekin_true_alpha.Fill(pi0_ekin_true[EM_shower],reconstruct_pi0_etot(alpha_true[EM_shower],gamma_angle_smeared[EM_shower]) - pi0_mass)
                    h2_true_vs_reco_pi0_ekin_true_angle_true_alpha.Fill(pi0_ekin_true[EM_shower],reconstruct_pi0_etot(alpha_true[EM_shower],gamma_angle_true[EM_shower]) - pi0_mass)
                if plot_h2_true_vs_reco_pi0_ekin:
                    h1_pi0_mass_reco.Fill(pi0_mass_reco[EM_shower])

            # Print informations
            if print_accepted_EM_shower_infos:
                print " ------------------------------------------------ \n Processing EM shower nr. %d (from gamma ID: %d) \n ------------------------------------------------ " %(EM_shower,gamma_IDs[EM_shower])
                print " \t pi0 ID: \t \t \t %d "                            %pi0_IDs[EM_shower]
                print " \t pi0 etot true [GeV]: \t \t %.4f "                %(pi0_ekin_true[EM_shower]+pi0_mass)
                print " \t pi0 etot reco [GeV]: \t \t %.4f "                %(pi0_etot_reco[EM_shower])
                print " \t pi0 ekin true [GeV]: \t \t %.4f "                %pi0_ekin_true[EM_shower]
                print " \t pi0 ekin reco: \t \t %.4f "                      %(pi0_ekin_reco[EM_shower])
                print " \t pi0 mass reco: \t \t %.4f "                      %pi0_mass_reco[EM_shower]
                print " \t gamma ID: \t \t \t %d "                          %gamma_IDs[EM_shower]
                print " \t gamma ekin true [GeV]: \t %.4f "                 %gamma_ekin_true[EM_shower]
                print " \t gamma smeared ekin [GeV]: \t %.4f "              %gamma_ekin_smeared[EM_shower]
                print " \t gamma direction: \t \t [ %.4f , %.4f , %.4f ] "  %(gamma_dir_x[EM_shower],gamma_dir_y[EM_shower],gamma_dir_z[EM_shower])
                print " \t gamma dir_x: \t \t \t %.4f "                     %gamma_dir_x[EM_shower]
                print " \t gamma dir_y: \t \t \t %.4f "                     %gamma_dir_y[EM_shower]
                print " \t gamma dir_z: \t \t \t %.4f "                     %gamma_dir_z[EM_shower]
                print " \t gamma angle true [rad]: \t %.4f "                %gamma_angle_true[EM_shower]
                print " \t gamma angle_smeared: \t \t %.4f "                %(gamma_angle_smeared[EM_shower])
                print " \t alpha true: \t \t \t %.4f "                      %alpha_true[EM_shower]
                print " \t alpha measured: \t \t %.4f "                     %alpha_measured[EM_shower]
                print " \t 1-alpha^2 true: \t \t %.4f "                     %(1-alpha_true[EM_shower]*alpha_true[EM_shower])
                print " \t 1 - cos(th) true: \t \t %.4f "                   %(1-np.cos(gamma_angle_true[EM_shower]))
                print " \t contained edeps [GeV]: \t %.3f "                 %edeps_contained[EM_shower]
                print " \t not contained edeps [GeV]:\t %.3f "              %edeps_not_contained[EM_shower]
                print " \t total edeps [GeV]: \t \t %.3f "                  %edeps_total[EM_shower]
                print " \t contained edeps: \t \t %.3f "                    %(edeps_contained[EM_shower]/edeps_total[EM_shower])
                print " \t energy containment: \t \t %.3f "                 %(edeps_contained[EM_shower]/gamma_ekin_true[EM_shower])
    '''

    '''
        if(evt_id%100==0):
            now = time.time()
            print(" Processed \t {:}% of all entries (event {:} of {:}, elapsed time: {:.2} hours) " .format( math.floor(evt_id*100/n_evt), evt_id, n_evt, (now-start)/3600 ))
    '''


    '''
    # Draw histograms
    # ---------------------------------------------------------
    # Define color palette for TH2F
    NRGBs = 3
    NCont = 200
    red   = [1.00, 0.00, 0.00]
    green = [0.00, 1.00, 0.00]
    blue  = [1.00, 0.00, 1.00]
    stops = [0.00, 0.50, 1.00]
    redArray    = array('d',red)
    greenArray  = array('d',green)
    blueArray   = array('d',blue)
    stopsArray  = array('d',stops)
    ROOT.TColor.CreateGradientColorTable(NRGBs, stopsArray, redArray, greenArray, blueArray, NCont)
    '''

    '''
    if plot_h1_pi0_mass_reco:
        c0 = ROOT.TCanvas()
        c0.SetGrid(1)
        c0.SetLeftMargin(0.14)
        c0.SetRightMargin(0.05)
        c0.SetBottomMargin(0.14)
        ROOT.gStyle.SetOptStat(0) # (1110)
        ROOT.gStyle.SetStatX(0.90);
        ROOT.gStyle.SetStatY(0.85);
        ROOT.gStyle.SetStatW(0.2);
        ROOT.gStyle.SetStatH(0.2);
        ROOT.gStyle.SetOptTitle(False)
        ROOT.gStyle.SetLineWidth(2)
        ROOT.gStyle.SetNumberContours(200)
        ROOT.gPad.SetTickx()
        ROOT.gPad.SetTicky()
        ROOT.gPad.SetLogx(0)
        ROOT.gPad.SetLogy(0)
        histo_file = ROOT.TFile(("/home/roman/pi0_study/testplots/h1_pi0_mass_reco.root"), "RECREATE")
        h1_pi0_mass_reco.Write()
        h1_pi0_mass_reco.GetXaxis().SetLabelSize(0.05)
        h1_pi0_mass_reco.GetYaxis().SetLabelSize(0.05)
        h1_pi0_mass_reco.GetXaxis().SetTitleOffset(1.4)
        h1_pi0_mass_reco.GetYaxis().SetTitleOffset(1.4)
        h1_pi0_mass_reco.GetXaxis().SetTitleSize(0.045)
        h1_pi0_mass_reco.GetYaxis().SetTitleSize(0.045)
        h1_pi0_mass_reco.GetXaxis().SetTitleOffset(1.4)
        h1_pi0_mass_reco.GetYaxis().SetTitleOffset(1.5)
        h1_pi0_mass_reco.Draw()
        max_bincontent = h1_pi0_mass_reco.GetBinContent(h1_pi0_mass_reco.GetMaximumBin())
        line = ROOT.TLine (pi0_mass ,0 ,pi0_mass ,1.05*max_bincontent)
        line.SetLineColor(ROOT.kRed)
        line.SetLineWidth(2)
        line.Draw("same")
        c0.Print("/home/roman/pi0_study/testplots/h1_pi0_mass_reco.png")
        histo_file.Close()
    '''

    '''
    if plot_h2_true_vs_reco_pi0_etot:
        #c1 = ROOT.TCanvas("","",1600,1200)
        c1 = ROOT.TCanvas()
        c1.SetGrid(1)
        c1.SetLeftMargin(0.14)
        c1.SetRightMargin(0.18)
        c1.SetBottomMargin(0.14)
        ROOT.gStyle.SetOptStat(0) # (1110)
        ROOT.gStyle.SetStatX(0.80);
        ROOT.gStyle.SetStatY(0.85);
        ROOT.gStyle.SetStatW(0.2);
        ROOT.gStyle.SetStatH(0.2);
        ROOT.gStyle.SetOptTitle(0)
        ROOT.gStyle.SetLineWidth(2)
        ROOT.gStyle.SetNumberContours(200)
        ROOT.gPad.SetTickx()
        ROOT.gPad.SetTicky()
        ROOT.gPad.SetLogx(0)
        ROOT.gPad.SetLogy(0)
        ROOT.gPad.SetLogz(1)
        histo_file = ROOT.TFile(("/home/roman/pi0_study/testplots/h2_true_vs_reco_pi0_etot.root"), "RECREATE")
        h2_true_vs_reco_pi0_etot.Write()
        h2_true_vs_reco_pi0_etot.GetXaxis().SetLabelSize(0.05)
        h2_true_vs_reco_pi0_etot.GetYaxis().SetLabelSize(0.05)
        h2_true_vs_reco_pi0_etot.GetZaxis().SetLabelSize(0.05)
        h2_true_vs_reco_pi0_etot.GetXaxis().SetTitleOffset(1.4)
        h2_true_vs_reco_pi0_etot.GetYaxis().SetTitleOffset(1.4)
        h2_true_vs_reco_pi0_etot.GetZaxis().SetTitleOffset(1.4)
        h2_true_vs_reco_pi0_etot.GetXaxis().SetTitleSize(0.045)
        h2_true_vs_reco_pi0_etot.GetYaxis().SetTitleSize(0.045)
        h2_true_vs_reco_pi0_etot.GetZaxis().SetTitleSize(0.045)
        h2_true_vs_reco_pi0_etot.GetZaxis().RotateTitle(1)
        h2_true_vs_reco_pi0_etot.GetXaxis().SetTitleOffset(1.4)
        h2_true_vs_reco_pi0_etot.GetYaxis().SetTitleOffset(1.5)
        h2_true_vs_reco_pi0_etot.GetZaxis().SetTitleOffset(1.5)
        h2_true_vs_reco_pi0_etot.Draw("colz")
        c1.Print("/home/roman/pi0_study/testplots/h2_true_vs_reco_pi0_etot.png")
        histo_file.Close()
    '''

    '''
    if plot_h2_true_vs_reco_pi0_ekin:
        #c2 = ROOT.TCanvas("","",1600,1200)
        c2 = ROOT.TCanvas()
        c2.SetGrid(1)
        c2.SetLeftMargin(0.14)
        c2.SetRightMargin(0.18)
        c2.SetBottomMargin(0.14)
        ROOT.gStyle.SetOptStat(0) # (1110)
        ROOT.gStyle.SetStatX(0.80);
        ROOT.gStyle.SetStatY(0.85);
        ROOT.gStyle.SetStatW(0.2);
        ROOT.gStyle.SetStatH(0.2);
        ROOT.gStyle.SetOptTitle(0)
        ROOT.gStyle.SetLineWidth(2)
        ROOT.gStyle.SetNumberContours(200)
        ROOT.gPad.SetTickx()
        ROOT.gPad.SetTicky()
        ROOT.gPad.SetLogx(0)
        ROOT.gPad.SetLogy(0)
        ROOT.gPad.SetLogz(1)
        histo_file = ROOT.TFile(("/home/roman/pi0_study/testplots/h2_true_vs_reco_pi0_ekin.root"), "RECREATE")
        h2_true_vs_reco_pi0_ekin.Write()
        h2_true_vs_reco_pi0_ekin.GetXaxis().SetLabelSize(0.05)
        h2_true_vs_reco_pi0_ekin.GetYaxis().SetLabelSize(0.05)
        h2_true_vs_reco_pi0_ekin.GetZaxis().SetLabelSize(0.05)
        h2_true_vs_reco_pi0_ekin.GetXaxis().SetTitleOffset(1.4)
        h2_true_vs_reco_pi0_ekin.GetYaxis().SetTitleOffset(1.4)
        h2_true_vs_reco_pi0_ekin.GetZaxis().SetTitleOffset(1.4)
        h2_true_vs_reco_pi0_ekin.GetXaxis().SetTitleSize(0.045)
        h2_true_vs_reco_pi0_ekin.GetYaxis().SetTitleSize(0.045)
        h2_true_vs_reco_pi0_ekin.GetZaxis().SetTitleSize(0.045)
        h2_true_vs_reco_pi0_ekin.GetZaxis().RotateTitle(1)
        h2_true_vs_reco_pi0_ekin.GetXaxis().SetTitleOffset(1.4)
        h2_true_vs_reco_pi0_ekin.GetYaxis().SetTitleOffset(1.5)
        h2_true_vs_reco_pi0_ekin.GetZaxis().SetTitleOffset(1.5)
        h2_true_vs_reco_pi0_ekin.Draw("colz")
        c2.Print("/home/roman/pi0_study/testplots/h2_true_vs_reco_pi0_ekin.png")
        histo_file.Close()

        #c3 = ROOT.TCanvas("","",1600,1200)
        c3 = ROOT.TCanvas()
        c3.SetGrid(1)
        c3.SetLeftMargin(0.14)
        c3.SetRightMargin(0.18)
        c3.SetBottomMargin(0.14)
        ROOT.gStyle.SetOptStat(0) # (1110)
        ROOT.gStyle.SetStatX(0.80);
        ROOT.gStyle.SetStatY(0.85);
        ROOT.gStyle.SetStatW(0.2);
        ROOT.gStyle.SetStatH(0.2);
        ROOT.gStyle.SetOptTitle(0)
        ROOT.gStyle.SetLineWidth(2)
        ROOT.gStyle.SetNumberContours(200)
        ROOT.gPad.SetTickx()
        ROOT.gPad.SetTicky()
        ROOT.gPad.SetLogx(0)
        ROOT.gPad.SetLogy(0)
        ROOT.gPad.SetLogz(1)
        histo_file = ROOT.TFile(("/home/roman/pi0_study/testplots/h2_true_vs_reco_pi0_ekin_true_angle.root"), "RECREATE")
        h2_true_vs_reco_pi0_ekin_true_angle.Write()
        h2_true_vs_reco_pi0_ekin_true_angle.GetXaxis().SetLabelSize(0.05)
        h2_true_vs_reco_pi0_ekin_true_angle.GetYaxis().SetLabelSize(0.05)
        h2_true_vs_reco_pi0_ekin_true_angle.GetZaxis().SetLabelSize(0.05)
        h2_true_vs_reco_pi0_ekin_true_angle.GetXaxis().SetTitleOffset(1.4)
        h2_true_vs_reco_pi0_ekin_true_angle.GetYaxis().SetTitleOffset(1.4)
        h2_true_vs_reco_pi0_ekin_true_angle.GetZaxis().SetTitleOffset(1.4)
        h2_true_vs_reco_pi0_ekin_true_angle.GetXaxis().SetTitleSize(0.045)
        h2_true_vs_reco_pi0_ekin_true_angle.GetYaxis().SetTitleSize(0.045)
        h2_true_vs_reco_pi0_ekin_true_angle.GetZaxis().SetTitleSize(0.045)
        h2_true_vs_reco_pi0_ekin_true_angle.GetZaxis().RotateTitle(1)
        h2_true_vs_reco_pi0_ekin_true_angle.GetXaxis().SetTitleOffset(1.4)
        h2_true_vs_reco_pi0_ekin_true_angle.GetYaxis().SetTitleOffset(1.5)
        h2_true_vs_reco_pi0_ekin_true_angle.GetZaxis().SetTitleOffset(1.5)
        h2_true_vs_reco_pi0_ekin_true_angle.Draw("colz")
        c3.Print("/home/roman/pi0_study/testplots/h2_true_vs_reco_pi0_ekin_true_angle.png")
        histo_file.Close()

        #c4 = ROOT.TCanvas("","",1600,1200)
        c4 = ROOT.TCanvas()
        c4.SetGrid(1)
        c4.SetLeftMargin(0.14)
        c4.SetRightMargin(0.18)
        c4.SetBottomMargin(0.14)
        ROOT.gStyle.SetOptStat(0) # (1110)
        ROOT.gStyle.SetStatX(0.80);
        ROOT.gStyle.SetStatY(0.85);
        ROOT.gStyle.SetStatW(0.2);
        ROOT.gStyle.SetStatH(0.2);
        ROOT.gStyle.SetOptTitle(0)
        ROOT.gStyle.SetLineWidth(2)
        ROOT.gStyle.SetNumberContours(200)
        ROOT.gPad.SetTickx()
        ROOT.gPad.SetTicky()
        ROOT.gPad.SetLogx(0)
        ROOT.gPad.SetLogy(0)
        ROOT.gPad.SetLogz(1)
        histo_file = ROOT.TFile(("/home/roman/pi0_study/testplots/h2_true_vs_reco_pi0_ekin_true_alpha.root"), "RECREATE")
        h2_true_vs_reco_pi0_ekin_true_alpha.Write()
        h2_true_vs_reco_pi0_ekin_true_alpha.GetXaxis().SetLabelSize(0.05)
        h2_true_vs_reco_pi0_ekin_true_alpha.GetYaxis().SetLabelSize(0.05)
        h2_true_vs_reco_pi0_ekin_true_alpha.GetZaxis().SetLabelSize(0.05)
        h2_true_vs_reco_pi0_ekin_true_alpha.GetXaxis().SetTitleOffset(1.4)
        h2_true_vs_reco_pi0_ekin_true_alpha.GetYaxis().SetTitleOffset(1.4)
        h2_true_vs_reco_pi0_ekin_true_alpha.GetZaxis().SetTitleOffset(1.4)
        h2_true_vs_reco_pi0_ekin_true_alpha.GetXaxis().SetTitleSize(0.045)
        h2_true_vs_reco_pi0_ekin_true_alpha.GetYaxis().SetTitleSize(0.045)
        h2_true_vs_reco_pi0_ekin_true_alpha.GetZaxis().SetTitleSize(0.045)
        h2_true_vs_reco_pi0_ekin_true_alpha.GetZaxis().RotateTitle(1)
        h2_true_vs_reco_pi0_ekin_true_alpha.GetXaxis().SetTitleOffset(1.4)
        h2_true_vs_reco_pi0_ekin_true_alpha.GetYaxis().SetTitleOffset(1.5)
        h2_true_vs_reco_pi0_ekin_true_alpha.GetZaxis().SetTitleOffset(1.5)
        h2_true_vs_reco_pi0_ekin_true_alpha.Draw("colz")
        c4.Print("/home/roman/pi0_study/testplots/h2_true_vs_reco_pi0_ekin_true_alpha.png")
        histo_file.Close()

        #c5 = ROOT.TCanvas("","",1600,1200)
        c5 = ROOT.TCanvas()
        c5.SetGrid(1)
        c5.SetLeftMargin(0.14)
        c5.SetRightMargin(0.18)
        c5.SetBottomMargin(0.14)
        ROOT.gStyle.SetOptStat(0) # (1110)
        ROOT.gStyle.SetStatX(0.80);
        ROOT.gStyle.SetStatY(0.85);
        ROOT.gStyle.SetStatW(0.2);
        ROOT.gStyle.SetStatH(0.2);
        ROOT.gStyle.SetOptTitle(0)
        ROOT.gStyle.SetLineWidth(2)
        ROOT.gStyle.SetNumberContours(200)
        ROOT.gPad.SetTickx()
        ROOT.gPad.SetTicky()
        ROOT.gPad.SetLogx(0)
        ROOT.gPad.SetLogy(0)
        ROOT.gPad.SetLogz(1)
        histo_file = ROOT.TFile(("/home/roman/pi0_study/testplots/h2_true_vs_reco_pi0_ekin_true_angle_true_alpha.root"), "RECREATE")
        h2_true_vs_reco_pi0_ekin_true_angle_true_alpha.Write()
        h2_true_vs_reco_pi0_ekin_true_angle_true_alpha.GetXaxis().SetLabelSize(0.05)
        h2_true_vs_reco_pi0_ekin_true_angle_true_alpha.GetYaxis().SetLabelSize(0.05)
        h2_true_vs_reco_pi0_ekin_true_angle_true_alpha.GetZaxis().SetLabelSize(0.05)
        h2_true_vs_reco_pi0_ekin_true_angle_true_alpha.GetXaxis().SetTitleOffset(1.4)
        h2_true_vs_reco_pi0_ekin_true_angle_true_alpha.GetYaxis().SetTitleOffset(1.4)
        h2_true_vs_reco_pi0_ekin_true_angle_true_alpha.GetZaxis().SetTitleOffset(1.4)
        h2_true_vs_reco_pi0_ekin_true_angle_true_alpha.GetXaxis().SetTitleSize(0.045)
        h2_true_vs_reco_pi0_ekin_true_angle_true_alpha.GetYaxis().SetTitleSize(0.045)
        h2_true_vs_reco_pi0_ekin_true_angle_true_alpha.GetZaxis().SetTitleSize(0.045)
        h2_true_vs_reco_pi0_ekin_true_angle_true_alpha.GetZaxis().RotateTitle(1)
        h2_true_vs_reco_pi0_ekin_true_angle_true_alpha.GetXaxis().SetTitleOffset(1.4)
        h2_true_vs_reco_pi0_ekin_true_angle_true_alpha.GetYaxis().SetTitleOffset(1.5)
        h2_true_vs_reco_pi0_ekin_true_angle_true_alpha.GetZaxis().SetTitleOffset(1.5)
        h2_true_vs_reco_pi0_ekin_true_angle_true_alpha.Draw("colz")
        c5.Print("/home/roman/pi0_study/testplots/h2_true_vs_reco_pi0_ekin_true_angle_true_alpha.png")
        histo_file.Close()
    '''


if __name__ == "__main__":
    main()
