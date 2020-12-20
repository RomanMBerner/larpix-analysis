from array import array
import matplotlib.pyplot as plt
import numpy as np
import ROOT
from ROOT import TCanvas, TProfile, TNtuple, TH1F, TH2F, TH1, TLine

def plot_h1_trLength(h1,hist_name,folder_name):
    c0 = ROOT.TCanvas()
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
    #h1.Write()
    h1.GetXaxis().SetLabelSize(0.05)
    h1.GetYaxis().SetLabelSize(0.05)
    h1.GetXaxis().SetTitleOffset(1.4)
    h1.GetYaxis().SetTitleOffset(1.4)
    h1.GetXaxis().SetTitleSize(0.045)
    h1.GetYaxis().SetTitleSize(0.045)
    h1.GetXaxis().SetTitleOffset(1.4)
    h1.GetYaxis().SetTitleOffset(1.5)
    h1.Draw()
    #max_bincontent = h1_trLength.GetBinContent(h1_trLength.GetMaximumBin())
    #line = ROOT.TLine (pi0_mass ,0 ,pi0_mass ,1.05*max_bincontent)
    #line.SetLineColor(ROOT.kRed)
    #line.SetLineWidth(2)
    #line.Draw("same")
    c0.Print('plots/'+str(folder_name)+'/'+str(hist_name)+'.png')
    #histo_file = ROOT.TFile(("/path/to/histo.root"), "RECREATE")
    #histo_file.Close()

def plot_h2_trLength_vs_nHits(h2,hist_name,folder_name):
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

    c0 = ROOT.TCanvas()
    c0.SetGrid(1)
    c0.SetLeftMargin(0.14)
    c0.SetRightMargin(0.18)
    c0.SetBottomMargin(0.14)
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
    #h2.Write()
    h2.GetXaxis().SetLabelSize(0.05)
    h2.GetYaxis().SetLabelSize(0.05)
    h2.GetZaxis().SetLabelSize(0.05)
    h2.GetXaxis().SetTitleOffset(1.4)
    h2.GetYaxis().SetTitleOffset(1.4)
    h2.GetZaxis().SetTitleOffset(1.4)
    h2.GetXaxis().SetTitleSize(0.045)
    h2.GetYaxis().SetTitleSize(0.045)
    h2.GetZaxis().SetTitleSize(0.045)
    h2.GetZaxis().RotateTitle(1)
    h2.GetXaxis().SetTitleOffset(1.4)
    h2.GetYaxis().SetTitleOffset(1.5)
    h2.GetZaxis().SetTitleOffset(1.5)
    h2.Draw("colz")
    c0.Print('plots/'+str(folder_name)+'/'+str(hist_name)+'.png')

def plot_h3_event_hits(h3,hist_name,folder_name):
    # Define color palette for TH2F
    #NRGBs = 3
    #NCont = 200
    #red   = [1.00, 0.00, 0.00]
    #green = [0.00, 1.00, 0.00]
    #blue  = [1.00, 0.00, 1.00]
    #stops = [0.00, 0.50, 1.00]
    #redArray    = array('d',red)
    #greenArray  = array('d',green)
    #blueArray   = array('d',blue)
    #stopsArray  = array('d',stops)
    #ROOT.TColor.CreateGradientColorTable(NRGBs, stopsArray, redArray, greenArray, blueArray, NCont)

    c0 = ROOT.TCanvas()
    c0.SetGrid(1)
    #c0.SetLeftMargin(0.14)
    #c0.SetRightMargin(0.18)
    #c0.SetBottomMargin(0.14)
    ROOT.gStyle.SetOptStat(0) # (1110)
    #ROOT.gStyle.SetStatX(0.80);
    #ROOT.gStyle.SetStatY(0.85);
    #ROOT.gStyle.SetStatW(0.2);
    #ROOT.gStyle.SetStatH(0.2);
    ROOT.gStyle.SetOptTitle(0)
    ROOT.gStyle.SetLineWidth(2)
    ROOT.gStyle.SetNumberContours(200)
    #ROOT.gPad.SetTickx()
    #ROOT.gPad.SetTicky()
    #ROOT.gPad.SetLogx(0)
    #ROOT.gPad.SetLogy(0)
    #ROOT.gPad.SetLogz(1)
    h3.GetXaxis().SetLabelSize(0.05)
    h3.GetYaxis().SetLabelSize(0.05)
    h3.GetZaxis().SetLabelSize(0.05)
    h3.GetXaxis().SetTitleOffset(1.4)
    h3.GetYaxis().SetTitleOffset(1.4)
    h3.GetZaxis().SetTitleOffset(1.4)
    h3.GetXaxis().SetTitleSize(0.045)
    h3.GetYaxis().SetTitleSize(0.045)
    h3.GetZaxis().SetTitleSize(0.045)
    #h3.GetZaxis().RotateTitle(1)
    h3.GetXaxis().SetTitleOffset(1.4)
    h3.GetYaxis().SetTitleOffset(1.5)
    h3.GetZaxis().SetTitleOffset(1.5)
    h3.Draw('COLZ')
    c0.Print('plots/'+str(folder_name)+'/'+str(hist_name)+'.png')
