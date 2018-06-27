\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj: wcvar
Subject To
 w_0_ADM:                        w_0_ADM - 0.99 b_0_ADM + 1.01 s_0_ADM  = 0.1
 w_0_AXP:                        w_0_AXP - 0.99 b_0_AXP + 1.01 s_0_AXP  = 0.1
 w_0_DIS:                        w_0_DIS - 0.99 b_0_DIS + 1.01 s_0_DIS  = 0.1
 w_0_FDX:                        w_0_FDX - 0.99 b_0_FDX + 1.01 s_0_FDX  = 0.1
 w_0_IBM:                        w_0_IBM - 0.99 b_0_IBM + 1.01 s_0_IBM  = 0.1
 w_0_INTC:                       w_0_INTC - 0.99 b_0_INTC + 1.01 s_0_INTC
                                  = 0.1
 w_0_KO:                         w_0_KO - 0.99 b_0_KO + 1.01 s_0_KO  = 0.1
 w_0_MAT:                        w_0_MAT - 0.99 b_0_MAT + 1.01 s_0_MAT  = 0.1
 w_0_MSFT:                       w_0_MSFT - 0.99 b_0_MSFT + 1.01 s_0_MSFT
                                  = 0.1
 w_0_PG:                         w_0_PG - 0.99 b_0_PG + 1.01 s_0_PG  = 0.1
 w_0_balance_constraint:         b_0_ADM + b_0_AXP + b_0_DIS + b_0_FDX
                                 + b_0_IBM + b_0_INTC + b_0_KO + b_0_MAT
                                 + b_0_MSFT + b_0_PG - s_0_ADM - s_0_AXP
                                 - s_0_DIS - s_0_FDX - s_0_IBM - s_0_INTC
                                 - s_0_KO - s_0_MAT - s_0_MSFT - s_0_PG  = 0
 w_k1_ADM_s0_return_constraint:  1.17911 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.984384 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.18883 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.99859 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.996456 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.831946 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.86577 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.934004 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.20776 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.854784 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.911433 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.722809 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.06092 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.11522 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.817911 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.0519 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.750931 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.00771 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.17345 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.793575 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.795737 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.00139 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.798564 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.24509 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.1802 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.42883 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.817518 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.55317 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.05487 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.02264 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.657653 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.3896 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.07822 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.830916 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.16897 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.13382 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.17446 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.906074 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.21996 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.937783 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.39452 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.87286 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.1302 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.737557 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.02072 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.838298 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.03681 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.682458 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.00025 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.755425 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.718647 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.42148 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.996293 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.3512 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.74496 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.2099 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.3979 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.1988 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.751869 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.19576 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.653013 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.48737 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.00323 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.38719 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.768591 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.45595 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.16421 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.890108 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.639096 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.58806 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.35063 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.78198 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.01324 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.99482 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.17443 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.13196 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.36024 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.927161 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.3073 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.805206 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.15366 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.661585 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.189 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.71403 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.28637 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.07636 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.850376 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.688485 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.944313 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.54385 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.961589 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.34085 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.790564 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.46445 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.66677 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.933437 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.630954 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.14277 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.609145 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.45803 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.03196 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.98064 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.753399 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.55259 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.14393 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.902178 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.572048 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.85369 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.18748 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.775435 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.31212 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.131 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.12002 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.22108 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.24007 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.95109 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.2266 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.885489 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.05677 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.827287 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.01087 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.938946 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.2101 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.924613 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.779455 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.773303 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.913578 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.24731 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.00849 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.14575 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.834534 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.12624 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.4447 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.881497 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.837291 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.15018 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.723332 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.13719 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.10355 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.47099 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.804396 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.40917 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.11643 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.852075 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.727223 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.41253 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.988226 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.79728 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.26103 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.02685 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.11799 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.03906 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.989824 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.04244 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.20599 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.941024 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.17083 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.62617 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.44477 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.557667 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.751345 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.00208 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.75559 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.864285 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.711773 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.29599 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.881429 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.31175 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.983553 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.79146 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.07069 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.76871 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.553294 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.793159 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.580671 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.68191 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.708723 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.38207 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.905397 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.795657 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.03795 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.40195 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.676907 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.2303 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.81295 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.09358 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.796339 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.833652 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.907364 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.31021 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.10038 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.929448 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.17728 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.908514 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.06659 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.8585 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.938738 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.849815 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.17429 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.05821 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.843996 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.774732 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.985814 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.36487 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.16507 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.19226 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.885045 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.3168 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.32313 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.08619 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.928433 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.08548 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.819648 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.21677 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.07094 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.4265 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.910586 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.20128 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.14499 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.814031 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.820783 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.28439 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.01617 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.854236 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.314 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.0118 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.24391 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.06986 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.1094 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.02906 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.15092 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.901022 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.02515 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.661642 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.38344 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.67203 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.952565 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.11429 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.27863 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.886436 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.959805 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.40694 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.941395 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.19262 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.867764 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.57112 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.15566 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.3007 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.629805 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.00065 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.657722 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.56265 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.926146 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.54562 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.828784 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.14117 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.18655 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.989882 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.595367 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.55526 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.32014 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.940206 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.07055 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.01773 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.07145 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.14119 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.11917 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.00538 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.15399 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.738514 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.36125 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.764629 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.52243 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.648955 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.740797 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.785526 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.988 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.981629 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.739929 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.45674 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.908697 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.35336 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.05594 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.6576 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.44809 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.98599 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.56223 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.08205 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.424686 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.69237 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.554457 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.14213 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.752699 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.675109 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.893677 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.24753 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.545784 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.28858 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.25389 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.690332 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.49444 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.624138 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.928114 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.15098 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.15323 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.981037 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.21186 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.861431 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.07076 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.829853 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.04496 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.922403 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.2439 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.05986 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.859758 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.782936 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.996473 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.30298 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.04893 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.12511 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.910386 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.36466 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.48544 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.965736 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.83388 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.21398 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.779165 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.15713 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.01268 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.38943 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.790485 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.22629 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.95471 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.848083 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.738846 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.34858 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.00525 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.744498 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.22654 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.02365 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.04068 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.03967 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.08971 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.973453 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.13977 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.945305 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.06011 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.852015 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.1228 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.796234 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.01395 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.03991 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.04669 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.788941 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.909623 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.21773 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.967158 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.10509 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.895022 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.20985 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.17386 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.05542 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.868785 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.992713 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.77461 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.21316 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.964306 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.3496 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.814042 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.17574 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.01521 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.976536 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.715014 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.29188 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.10759 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.872453 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.07955 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.998756 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.991554 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.05734 w_k1_PG_s11 - w_k1_PG_s117
                                 + 0.99 b_k1_PG_s117 - 1.01 s_k1_PG_s117  = 0
 k1_s0_balance_constraint:       b_k1_ADM_s0 + b_k1_AXP_s0 + b_k1_DIS_s0
                                 + b_k1_FDX_s0 + b_k1_IBM_s0 + b_k1_INTC_s0
                                 + b_k1_KO_s0 + b_k1_MAT_s0 + b_k1_MSFT_s0
                                 + b_k1_PG_s0 - s_k1_ADM_s0 - s_k1_AXP_s0
                                 - s_k1_DIS_s0 - s_k1_FDX_s0 - s_k1_IBM_s0
                                 - s_k1_INTC_s0 - s_k1_KO_s0 - s_k1_MAT_s0
                                 - s_k1_MSFT_s0 - s_k1_PG_s0  = 0
 k1_s1_balance_constraint:       b_k1_ADM_s1 + b_k1_AXP_s1 + b_k1_DIS_s1
                                 + b_k1_FDX_s1 + b_k1_IBM_s1 + b_k1_INTC_s1
                                 + b_k1_KO_s1 + b_k1_MAT_s1 + b_k1_MSFT_s1
                                 + b_k1_PG_s1 - s_k1_ADM_s1 - s_k1_AXP_s1
                                 - s_k1_DIS_s1 - s_k1_FDX_s1 - s_k1_IBM_s1
                                 - s_k1_INTC_s1 - s_k1_KO_s1 - s_k1_MAT_s1
                                 - s_k1_MSFT_s1 - s_k1_PG_s1  = 0
 k1_s00_balance_constraint:      b_k1_ADM_s00 + b_k1_AXP_s00 + b_k1_DIS_s00
                                 + b_k1_FDX_s00 + b_k1_IBM_s00 + b_k1_INTC_s00
                                 + b_k1_KO_s00 + b_k1_MAT_s00 + b_k1_MSFT_s00
                                 + b_k1_PG_s00 - s_k1_ADM_s00 - s_k1_AXP_s00
                                 - s_k1_DIS_s00 - s_k1_FDX_s00 - s_k1_IBM_s00
                                 - s_k1_INTC_s00 - s_k1_KO_s00 - s_k1_MAT_s00
                                 - s_k1_MSFT_s00 - s_k1_PG_s00  = 0
 k1_s01_balance_constraint:      b_k1_ADM_s01 + b_k1_AXP_s01 + b_k1_DIS_s01
                                 + b_k1_FDX_s01 + b_k1_IBM_s01 + b_k1_INTC_s01
                                 + b_k1_KO_s01 + b_k1_MAT_s01 + b_k1_MSFT_s01
                                 + b_k1_PG_s01 - s_k1_ADM_s01 - s_k1_AXP_s01
                                 - s_k1_DIS_s01 - s_k1_FDX_s01 - s_k1_IBM_s01
                                 - s_k1_INTC_s01 - s_k1_KO_s01 - s_k1_MAT_s01
                                 - s_k1_MSFT_s01 - s_k1_PG_s01  = 0
 k1_s10_balance_constraint:      b_k1_ADM_s10 + b_k1_AXP_s10 + b_k1_DIS_s10
                                 + b_k1_FDX_s10 + b_k1_IBM_s10 + b_k1_INTC_s10
                                 + b_k1_KO_s10 + b_k1_MAT_s10 + b_k1_MSFT_s10
                                 + b_k1_PG_s10 - s_k1_ADM_s10 - s_k1_AXP_s10
                                 - s_k1_DIS_s10 - s_k1_FDX_s10 - s_k1_IBM_s10
                                 - s_k1_INTC_s10 - s_k1_KO_s10 - s_k1_MAT_s10
                                 - s_k1_MSFT_s10 - s_k1_PG_s10  = 0
 k1_s11_balance_constraint:      b_k1_ADM_s11 + b_k1_AXP_s11 + b_k1_DIS_s11
                                 + b_k1_FDX_s11 + b_k1_IBM_s11 + b_k1_INTC_s11
                                 + b_k1_KO_s11 + b_k1_MAT_s11 + b_k1_MSFT_s11
                                 + b_k1_PG_s11 - s_k1_ADM_s11 - s_k1_AXP_s11
                                 - s_k1_DIS_s11 - s_k1_FDX_s11 - s_k1_IBM_s11
                                 - s_k1_INTC_s11 - s_k1_KO_s11 - s_k1_MAT_s11
                                 - s_k1_MSFT_s11 - s_k1_PG_s11  = 0
 k1_s000_balance_constraint:     b_k1_ADM_s000 + b_k1_AXP_s000 + b_k1_DIS_s000
                                 + b_k1_FDX_s000 + b_k1_IBM_s000
                                 + b_k1_INTC_s000 + b_k1_KO_s000
                                 + b_k1_MAT_s000 + b_k1_MSFT_s000
                                 + b_k1_PG_s000 - s_k1_ADM_s000 - s_k1_AXP_s000
                                 - s_k1_DIS_s000 - s_k1_FDX_s000
                                 - s_k1_IBM_s000 - s_k1_INTC_s000
                                 - s_k1_KO_s000 - s_k1_MAT_s000
                                 - s_k1_MSFT_s000 - s_k1_PG_s000  = 0
 k1_s001_balance_constraint:     b_k1_ADM_s001 + b_k1_AXP_s001 + b_k1_DIS_s001
                                 + b_k1_FDX_s001 + b_k1_IBM_s001
                                 + b_k1_INTC_s001 + b_k1_KO_s001
                                 + b_k1_MAT_s001 + b_k1_MSFT_s001
                                 + b_k1_PG_s001 - s_k1_ADM_s001 - s_k1_AXP_s001
                                 - s_k1_DIS_s001 - s_k1_FDX_s001
                                 - s_k1_IBM_s001 - s_k1_INTC_s001
                                 - s_k1_KO_s001 - s_k1_MAT_s001
                                 - s_k1_MSFT_s001 - s_k1_PG_s001  = 0
 k1_s002_balance_constraint:     b_k1_ADM_s002 + b_k1_AXP_s002 + b_k1_DIS_s002
                                 + b_k1_FDX_s002 + b_k1_IBM_s002
                                 + b_k1_INTC_s002 + b_k1_KO_s002
                                 + b_k1_MAT_s002 + b_k1_MSFT_s002
                                 + b_k1_PG_s002 - s_k1_ADM_s002 - s_k1_AXP_s002
                                 - s_k1_DIS_s002 - s_k1_FDX_s002
                                 - s_k1_IBM_s002 - s_k1_INTC_s002
                                 - s_k1_KO_s002 - s_k1_MAT_s002
                                 - s_k1_MSFT_s002 - s_k1_PG_s002  = 0
 k1_s003_balance_constraint:     b_k1_ADM_s003 + b_k1_AXP_s003 + b_k1_DIS_s003
                                 + b_k1_FDX_s003 + b_k1_IBM_s003
                                 + b_k1_INTC_s003 + b_k1_KO_s003
                                 + b_k1_MAT_s003 + b_k1_MSFT_s003
                                 + b_k1_PG_s003 - s_k1_ADM_s003 - s_k1_AXP_s003
                                 - s_k1_DIS_s003 - s_k1_FDX_s003
                                 - s_k1_IBM_s003 - s_k1_INTC_s003
                                 - s_k1_KO_s003 - s_k1_MAT_s003
                                 - s_k1_MSFT_s003 - s_k1_PG_s003  = 0
 k1_s004_balance_constraint:     b_k1_ADM_s004 + b_k1_AXP_s004 + b_k1_DIS_s004
                                 + b_k1_FDX_s004 + b_k1_IBM_s004
                                 + b_k1_INTC_s004 + b_k1_KO_s004
                                 + b_k1_MAT_s004 + b_k1_MSFT_s004
                                 + b_k1_PG_s004 - s_k1_ADM_s004 - s_k1_AXP_s004
                                 - s_k1_DIS_s004 - s_k1_FDX_s004
                                 - s_k1_IBM_s004 - s_k1_INTC_s004
                                 - s_k1_KO_s004 - s_k1_MAT_s004
                                 - s_k1_MSFT_s004 - s_k1_PG_s004  = 0
 k1_s005_balance_constraint:     b_k1_ADM_s005 + b_k1_AXP_s005 + b_k1_DIS_s005
                                 + b_k1_FDX_s005 + b_k1_IBM_s005
                                 + b_k1_INTC_s005 + b_k1_KO_s005
                                 + b_k1_MAT_s005 + b_k1_MSFT_s005
                                 + b_k1_PG_s005 - s_k1_ADM_s005 - s_k1_AXP_s005
                                 - s_k1_DIS_s005 - s_k1_FDX_s005
                                 - s_k1_IBM_s005 - s_k1_INTC_s005
                                 - s_k1_KO_s005 - s_k1_MAT_s005
                                 - s_k1_MSFT_s005 - s_k1_PG_s005  = 0
 k1_s006_balance_constraint:     b_k1_ADM_s006 + b_k1_AXP_s006 + b_k1_DIS_s006
                                 + b_k1_FDX_s006 + b_k1_IBM_s006
                                 + b_k1_INTC_s006 + b_k1_KO_s006
                                 + b_k1_MAT_s006 + b_k1_MSFT_s006
                                 + b_k1_PG_s006 - s_k1_ADM_s006 - s_k1_AXP_s006
                                 - s_k1_DIS_s006 - s_k1_FDX_s006
                                 - s_k1_IBM_s006 - s_k1_INTC_s006
                                 - s_k1_KO_s006 - s_k1_MAT_s006
                                 - s_k1_MSFT_s006 - s_k1_PG_s006  = 0
 k1_s007_balance_constraint:     b_k1_ADM_s007 + b_k1_AXP_s007 + b_k1_DIS_s007
                                 + b_k1_FDX_s007 + b_k1_IBM_s007
                                 + b_k1_INTC_s007 + b_k1_KO_s007
                                 + b_k1_MAT_s007 + b_k1_MSFT_s007
                                 + b_k1_PG_s007 - s_k1_ADM_s007 - s_k1_AXP_s007
                                 - s_k1_DIS_s007 - s_k1_FDX_s007
                                 - s_k1_IBM_s007 - s_k1_INTC_s007
                                 - s_k1_KO_s007 - s_k1_MAT_s007
                                 - s_k1_MSFT_s007 - s_k1_PG_s007  = 0
 k1_s010_balance_constraint:     b_k1_ADM_s010 + b_k1_AXP_s010 + b_k1_DIS_s010
                                 + b_k1_FDX_s010 + b_k1_IBM_s010
                                 + b_k1_INTC_s010 + b_k1_KO_s010
                                 + b_k1_MAT_s010 + b_k1_MSFT_s010
                                 + b_k1_PG_s010 - s_k1_ADM_s010 - s_k1_AXP_s010
                                 - s_k1_DIS_s010 - s_k1_FDX_s010
                                 - s_k1_IBM_s010 - s_k1_INTC_s010
                                 - s_k1_KO_s010 - s_k1_MAT_s010
                                 - s_k1_MSFT_s010 - s_k1_PG_s010  = 0
 k1_s011_balance_constraint:     b_k1_ADM_s011 + b_k1_AXP_s011 + b_k1_DIS_s011
                                 + b_k1_FDX_s011 + b_k1_IBM_s011
                                 + b_k1_INTC_s011 + b_k1_KO_s011
                                 + b_k1_MAT_s011 + b_k1_MSFT_s011
                                 + b_k1_PG_s011 - s_k1_ADM_s011 - s_k1_AXP_s011
                                 - s_k1_DIS_s011 - s_k1_FDX_s011
                                 - s_k1_IBM_s011 - s_k1_INTC_s011
                                 - s_k1_KO_s011 - s_k1_MAT_s011
                                 - s_k1_MSFT_s011 - s_k1_PG_s011  = 0
 k1_s012_balance_constraint:     b_k1_ADM_s012 + b_k1_AXP_s012 + b_k1_DIS_s012
                                 + b_k1_FDX_s012 + b_k1_IBM_s012
                                 + b_k1_INTC_s012 + b_k1_KO_s012
                                 + b_k1_MAT_s012 + b_k1_MSFT_s012
                                 + b_k1_PG_s012 - s_k1_ADM_s012 - s_k1_AXP_s012
                                 - s_k1_DIS_s012 - s_k1_FDX_s012
                                 - s_k1_IBM_s012 - s_k1_INTC_s012
                                 - s_k1_KO_s012 - s_k1_MAT_s012
                                 - s_k1_MSFT_s012 - s_k1_PG_s012  = 0
 k1_s013_balance_constraint:     b_k1_ADM_s013 + b_k1_AXP_s013 + b_k1_DIS_s013
                                 + b_k1_FDX_s013 + b_k1_IBM_s013
                                 + b_k1_INTC_s013 + b_k1_KO_s013
                                 + b_k1_MAT_s013 + b_k1_MSFT_s013
                                 + b_k1_PG_s013 - s_k1_ADM_s013 - s_k1_AXP_s013
                                 - s_k1_DIS_s013 - s_k1_FDX_s013
                                 - s_k1_IBM_s013 - s_k1_INTC_s013
                                 - s_k1_KO_s013 - s_k1_MAT_s013
                                 - s_k1_MSFT_s013 - s_k1_PG_s013  = 0
 k1_s014_balance_constraint:     b_k1_ADM_s014 + b_k1_AXP_s014 + b_k1_DIS_s014
                                 + b_k1_FDX_s014 + b_k1_IBM_s014
                                 + b_k1_INTC_s014 + b_k1_KO_s014
                                 + b_k1_MAT_s014 + b_k1_MSFT_s014
                                 + b_k1_PG_s014 - s_k1_ADM_s014 - s_k1_AXP_s014
                                 - s_k1_DIS_s014 - s_k1_FDX_s014
                                 - s_k1_IBM_s014 - s_k1_INTC_s014
                                 - s_k1_KO_s014 - s_k1_MAT_s014
                                 - s_k1_MSFT_s014 - s_k1_PG_s014  = 0
 k1_s015_balance_constraint:     b_k1_ADM_s015 + b_k1_AXP_s015 + b_k1_DIS_s015
                                 + b_k1_FDX_s015 + b_k1_IBM_s015
                                 + b_k1_INTC_s015 + b_k1_KO_s015
                                 + b_k1_MAT_s015 + b_k1_MSFT_s015
                                 + b_k1_PG_s015 - s_k1_ADM_s015 - s_k1_AXP_s015
                                 - s_k1_DIS_s015 - s_k1_FDX_s015
                                 - s_k1_IBM_s015 - s_k1_INTC_s015
                                 - s_k1_KO_s015 - s_k1_MAT_s015
                                 - s_k1_MSFT_s015 - s_k1_PG_s015  = 0
 k1_s016_balance_constraint:     b_k1_ADM_s016 + b_k1_AXP_s016 + b_k1_DIS_s016
                                 + b_k1_FDX_s016 + b_k1_IBM_s016
                                 + b_k1_INTC_s016 + b_k1_KO_s016
                                 + b_k1_MAT_s016 + b_k1_MSFT_s016
                                 + b_k1_PG_s016 - s_k1_ADM_s016 - s_k1_AXP_s016
                                 - s_k1_DIS_s016 - s_k1_FDX_s016
                                 - s_k1_IBM_s016 - s_k1_INTC_s016
                                 - s_k1_KO_s016 - s_k1_MAT_s016
                                 - s_k1_MSFT_s016 - s_k1_PG_s016  = 0
 k1_s017_balance_constraint:     b_k1_ADM_s017 + b_k1_AXP_s017 + b_k1_DIS_s017
                                 + b_k1_FDX_s017 + b_k1_IBM_s017
                                 + b_k1_INTC_s017 + b_k1_KO_s017
                                 + b_k1_MAT_s017 + b_k1_MSFT_s017
                                 + b_k1_PG_s017 - s_k1_ADM_s017 - s_k1_AXP_s017
                                 - s_k1_DIS_s017 - s_k1_FDX_s017
                                 - s_k1_IBM_s017 - s_k1_INTC_s017
                                 - s_k1_KO_s017 - s_k1_MAT_s017
                                 - s_k1_MSFT_s017 - s_k1_PG_s017  = 0
 k1_s100_balance_constraint:     b_k1_ADM_s100 + b_k1_AXP_s100 + b_k1_DIS_s100
                                 + b_k1_FDX_s100 + b_k1_IBM_s100
                                 + b_k1_INTC_s100 + b_k1_KO_s100
                                 + b_k1_MAT_s100 + b_k1_MSFT_s100
                                 + b_k1_PG_s100 - s_k1_ADM_s100 - s_k1_AXP_s100
                                 - s_k1_DIS_s100 - s_k1_FDX_s100
                                 - s_k1_IBM_s100 - s_k1_INTC_s100
                                 - s_k1_KO_s100 - s_k1_MAT_s100
                                 - s_k1_MSFT_s100 - s_k1_PG_s100  = 0
 k1_s101_balance_constraint:     b_k1_ADM_s101 + b_k1_AXP_s101 + b_k1_DIS_s101
                                 + b_k1_FDX_s101 + b_k1_IBM_s101
                                 + b_k1_INTC_s101 + b_k1_KO_s101
                                 + b_k1_MAT_s101 + b_k1_MSFT_s101
                                 + b_k1_PG_s101 - s_k1_ADM_s101 - s_k1_AXP_s101
                                 - s_k1_DIS_s101 - s_k1_FDX_s101
                                 - s_k1_IBM_s101 - s_k1_INTC_s101
                                 - s_k1_KO_s101 - s_k1_MAT_s101
                                 - s_k1_MSFT_s101 - s_k1_PG_s101  = 0
 k1_s102_balance_constraint:     b_k1_ADM_s102 + b_k1_AXP_s102 + b_k1_DIS_s102
                                 + b_k1_FDX_s102 + b_k1_IBM_s102
                                 + b_k1_INTC_s102 + b_k1_KO_s102
                                 + b_k1_MAT_s102 + b_k1_MSFT_s102
                                 + b_k1_PG_s102 - s_k1_ADM_s102 - s_k1_AXP_s102
                                 - s_k1_DIS_s102 - s_k1_FDX_s102
                                 - s_k1_IBM_s102 - s_k1_INTC_s102
                                 - s_k1_KO_s102 - s_k1_MAT_s102
                                 - s_k1_MSFT_s102 - s_k1_PG_s102  = 0
 k1_s103_balance_constraint:     b_k1_ADM_s103 + b_k1_AXP_s103 + b_k1_DIS_s103
                                 + b_k1_FDX_s103 + b_k1_IBM_s103
                                 + b_k1_INTC_s103 + b_k1_KO_s103
                                 + b_k1_MAT_s103 + b_k1_MSFT_s103
                                 + b_k1_PG_s103 - s_k1_ADM_s103 - s_k1_AXP_s103
                                 - s_k1_DIS_s103 - s_k1_FDX_s103
                                 - s_k1_IBM_s103 - s_k1_INTC_s103
                                 - s_k1_KO_s103 - s_k1_MAT_s103
                                 - s_k1_MSFT_s103 - s_k1_PG_s103  = 0
 k1_s104_balance_constraint:     b_k1_ADM_s104 + b_k1_AXP_s104 + b_k1_DIS_s104
                                 + b_k1_FDX_s104 + b_k1_IBM_s104
                                 + b_k1_INTC_s104 + b_k1_KO_s104
                                 + b_k1_MAT_s104 + b_k1_MSFT_s104
                                 + b_k1_PG_s104 - s_k1_ADM_s104 - s_k1_AXP_s104
                                 - s_k1_DIS_s104 - s_k1_FDX_s104
                                 - s_k1_IBM_s104 - s_k1_INTC_s104
                                 - s_k1_KO_s104 - s_k1_MAT_s104
                                 - s_k1_MSFT_s104 - s_k1_PG_s104  = 0
 k1_s105_balance_constraint:     b_k1_ADM_s105 + b_k1_AXP_s105 + b_k1_DIS_s105
                                 + b_k1_FDX_s105 + b_k1_IBM_s105
                                 + b_k1_INTC_s105 + b_k1_KO_s105
                                 + b_k1_MAT_s105 + b_k1_MSFT_s105
                                 + b_k1_PG_s105 - s_k1_ADM_s105 - s_k1_AXP_s105
                                 - s_k1_DIS_s105 - s_k1_FDX_s105
                                 - s_k1_IBM_s105 - s_k1_INTC_s105
                                 - s_k1_KO_s105 - s_k1_MAT_s105
                                 - s_k1_MSFT_s105 - s_k1_PG_s105  = 0
 k1_s106_balance_constraint:     b_k1_ADM_s106 + b_k1_AXP_s106 + b_k1_DIS_s106
                                 + b_k1_FDX_s106 + b_k1_IBM_s106
                                 + b_k1_INTC_s106 + b_k1_KO_s106
                                 + b_k1_MAT_s106 + b_k1_MSFT_s106
                                 + b_k1_PG_s106 - s_k1_ADM_s106 - s_k1_AXP_s106
                                 - s_k1_DIS_s106 - s_k1_FDX_s106
                                 - s_k1_IBM_s106 - s_k1_INTC_s106
                                 - s_k1_KO_s106 - s_k1_MAT_s106
                                 - s_k1_MSFT_s106 - s_k1_PG_s106  = 0
 k1_s107_balance_constraint:     b_k1_ADM_s107 + b_k1_AXP_s107 + b_k1_DIS_s107
                                 + b_k1_FDX_s107 + b_k1_IBM_s107
                                 + b_k1_INTC_s107 + b_k1_KO_s107
                                 + b_k1_MAT_s107 + b_k1_MSFT_s107
                                 + b_k1_PG_s107 - s_k1_ADM_s107 - s_k1_AXP_s107
                                 - s_k1_DIS_s107 - s_k1_FDX_s107
                                 - s_k1_IBM_s107 - s_k1_INTC_s107
                                 - s_k1_KO_s107 - s_k1_MAT_s107
                                 - s_k1_MSFT_s107 - s_k1_PG_s107  = 0
 k1_s110_balance_constraint:     b_k1_ADM_s110 + b_k1_AXP_s110 + b_k1_DIS_s110
                                 + b_k1_FDX_s110 + b_k1_IBM_s110
                                 + b_k1_INTC_s110 + b_k1_KO_s110
                                 + b_k1_MAT_s110 + b_k1_MSFT_s110
                                 + b_k1_PG_s110 - s_k1_ADM_s110 - s_k1_AXP_s110
                                 - s_k1_DIS_s110 - s_k1_FDX_s110
                                 - s_k1_IBM_s110 - s_k1_INTC_s110
                                 - s_k1_KO_s110 - s_k1_MAT_s110
                                 - s_k1_MSFT_s110 - s_k1_PG_s110  = 0
 k1_s111_balance_constraint:     b_k1_ADM_s111 + b_k1_AXP_s111 + b_k1_DIS_s111
                                 + b_k1_FDX_s111 + b_k1_IBM_s111
                                 + b_k1_INTC_s111 + b_k1_KO_s111
                                 + b_k1_MAT_s111 + b_k1_MSFT_s111
                                 + b_k1_PG_s111 - s_k1_ADM_s111 - s_k1_AXP_s111
                                 - s_k1_DIS_s111 - s_k1_FDX_s111
                                 - s_k1_IBM_s111 - s_k1_INTC_s111
                                 - s_k1_KO_s111 - s_k1_MAT_s111
                                 - s_k1_MSFT_s111 - s_k1_PG_s111  = 0
 k1_s112_balance_constraint:     b_k1_ADM_s112 + b_k1_AXP_s112 + b_k1_DIS_s112
                                 + b_k1_FDX_s112 + b_k1_IBM_s112
                                 + b_k1_INTC_s112 + b_k1_KO_s112
                                 + b_k1_MAT_s112 + b_k1_MSFT_s112
                                 + b_k1_PG_s112 - s_k1_ADM_s112 - s_k1_AXP_s112
                                 - s_k1_DIS_s112 - s_k1_FDX_s112
                                 - s_k1_IBM_s112 - s_k1_INTC_s112
                                 - s_k1_KO_s112 - s_k1_MAT_s112
                                 - s_k1_MSFT_s112 - s_k1_PG_s112  = 0
 k1_s113_balance_constraint:     b_k1_ADM_s113 + b_k1_AXP_s113 + b_k1_DIS_s113
                                 + b_k1_FDX_s113 + b_k1_IBM_s113
                                 + b_k1_INTC_s113 + b_k1_KO_s113
                                 + b_k1_MAT_s113 + b_k1_MSFT_s113
                                 + b_k1_PG_s113 - s_k1_ADM_s113 - s_k1_AXP_s113
                                 - s_k1_DIS_s113 - s_k1_FDX_s113
                                 - s_k1_IBM_s113 - s_k1_INTC_s113
                                 - s_k1_KO_s113 - s_k1_MAT_s113
                                 - s_k1_MSFT_s113 - s_k1_PG_s113  = 0
 k1_s114_balance_constraint:     b_k1_ADM_s114 + b_k1_AXP_s114 + b_k1_DIS_s114
                                 + b_k1_FDX_s114 + b_k1_IBM_s114
                                 + b_k1_INTC_s114 + b_k1_KO_s114
                                 + b_k1_MAT_s114 + b_k1_MSFT_s114
                                 + b_k1_PG_s114 - s_k1_ADM_s114 - s_k1_AXP_s114
                                 - s_k1_DIS_s114 - s_k1_FDX_s114
                                 - s_k1_IBM_s114 - s_k1_INTC_s114
                                 - s_k1_KO_s114 - s_k1_MAT_s114
                                 - s_k1_MSFT_s114 - s_k1_PG_s114  = 0
 k1_s115_balance_constraint:     b_k1_ADM_s115 + b_k1_AXP_s115 + b_k1_DIS_s115
                                 + b_k1_FDX_s115 + b_k1_IBM_s115
                                 + b_k1_INTC_s115 + b_k1_KO_s115
                                 + b_k1_MAT_s115 + b_k1_MSFT_s115
                                 + b_k1_PG_s115 - s_k1_ADM_s115 - s_k1_AXP_s115
                                 - s_k1_DIS_s115 - s_k1_FDX_s115
                                 - s_k1_IBM_s115 - s_k1_INTC_s115
                                 - s_k1_KO_s115 - s_k1_MAT_s115
                                 - s_k1_MSFT_s115 - s_k1_PG_s115  = 0
 k1_s116_balance_constraint:     b_k1_ADM_s116 + b_k1_AXP_s116 + b_k1_DIS_s116
                                 + b_k1_FDX_s116 + b_k1_IBM_s116
                                 + b_k1_INTC_s116 + b_k1_KO_s116
                                 + b_k1_MAT_s116 + b_k1_MSFT_s116
                                 + b_k1_PG_s116 - s_k1_ADM_s116 - s_k1_AXP_s116
                                 - s_k1_DIS_s116 - s_k1_FDX_s116
                                 - s_k1_IBM_s116 - s_k1_INTC_s116
                                 - s_k1_KO_s116 - s_k1_MAT_s116
                                 - s_k1_MSFT_s116 - s_k1_PG_s116  = 0
 k1_s117_balance_constraint:     b_k1_ADM_s117 + b_k1_AXP_s117 + b_k1_DIS_s117
                                 + b_k1_FDX_s117 + b_k1_IBM_s117
                                 + b_k1_INTC_s117 + b_k1_KO_s117
                                 + b_k1_MAT_s117 + b_k1_MSFT_s117
                                 + b_k1_PG_s117 - s_k1_ADM_s117 - s_k1_AXP_s117
                                 - s_k1_DIS_s117 - s_k1_FDX_s117
                                 - s_k1_IBM_s117 - s_k1_INTC_s117
                                 - s_k1_KO_s117 - s_k1_MAT_s117
                                 - s_k1_MSFT_s117 - s_k1_PG_s117  = 0
 k1_wcvar_constraint:            wcvar - var_1 - 0.109999730351632 z_k1_s0000
                                 - 0.149999799661006 z_k1_s0001
                                 - 0.42799856923433 z_k1_s0002
                                 - 0.344999078971167 z_k1_s0003
                                 - 0.205000493285288 z_k1_s0004
                                 - 0.207999476730384 z_k1_s0005
                                 - 0.28599882025513 z_k1_s0006
                                 - 0.108999885103608 z_k1_s0007
                                 - 0.043999993354994 z_k1_s0010
                                 - 0.0809998731671195 z_k1_s0011
                                 - 0.0640001049345653 z_k1_s0012
                                 - 0.094999985652828 z_k1_s0013
                                 - 0.0469999929019254 z_k1_s0014
                                 - 0.123000210624245 z_k1_s0015
                                 - 0.0359999945631769 z_k1_s0016
                                 - 0.0830002166651594 z_k1_s0017
                                 - 0.238998395752053 z_k1_s0020
                                 - 0.144999088983219 z_k1_s0021
                                 - 0.115999541785515 z_k1_s0022
                                 - 0.213999112683787 z_k1_s0023
                                 - 0.213999112683787 z_k1_s0024
                                 - 0.120999533698638 z_k1_s0025
                                 - 0.227998548842652 z_k1_s0026
                                 - 0.0759997417800005 z_k1_s0027
                                 - 0.0849998354509369 z_k1_s0030
                                 - 0.0259999547967506 z_k1_s0031
                                 - 0.0849998354509369 z_k1_s0032
                                 - 0.0509997268707849 z_k1_s0033
                                 - 0.0559997684854804 z_k1_s0034
                                 - 0.0550000217622072 z_k1_s0035
                                 - 0.0279999714426288 z_k1_s0036
                                 - 0.0499999801475117 z_k1_s0037
                                 - 0.0769999313181717 z_k1_s0040
                                 - 0.0449999675898348 z_k1_s0041
                                 - 0.0839999250743692 z_k1_s0042
                                 - 0.0559999500495794 z_k1_s0043
                                 - 0.0629999438057769 z_k1_s0044
                                 - 0.146000101629087 z_k1_s0045
                                 - 0.0339999851300902 z_k1_s0046
                                 - 0.0359999524318841 z_k1_s0047
                                 - 0.164999881622807 z_k1_s0050
                                 - 0.207999716253149 z_k1_s0051
                                 - 0.265000022278808 z_k1_s0052
                                 - 0.141999802543981 z_k1_s0053
                                 - 0.327999028361341 z_k1_s0054
                                 - 0.315999681249846 z_k1_s0055
                                 - 0.354999993109388 z_k1_s0056
                                 - 0.167999815750568 z_k1_s0057
                                 - 0.12900003486237 z_k1_s0060
                                 - 0.236000572850072 z_k1_s0061
                                 - 0.122000791948975 z_k1_s0062
                                 - 0.120000837687642 z_k1_s0063
                                 - 0.105000225525605 z_k1_s0064
                                 - 0.0720001444118217 z_k1_s0065
                                 - 0.203000730536798 z_k1_s0066
                                 - 0.207000639059464 z_k1_s0067
                                 - 0.182999144187448 z_k1_s0070
                                 - 0.248998974385981 z_k1_s0071
                                 - 0.407999258317721 z_k1_s0072
                                 - 0.162999811641469 z_k1_s0073
                                 - 0.309000360018589 z_k1_s0074
                                 - 0.096999812043202 z_k1_s0075
                                 - 0.188999282750708 z_k1_s0076
                                 - 0.095999845415903 z_k1_s0077
                                 - 0.48900134281329 z_k1_s0100
                                 - 0.270000701528788 z_k1_s0101
                                 - 0.13400003501041 z_k1_s0102
                                 - 0.421999265706944 z_k1_s0103
                                 - 0.271000150082292 z_k1_s0104
                                 - 0.144000157348567 z_k1_s0105
                                 - 0.222001154946213 z_k1_s0106
                                 - 0.21600012762279 z_k1_s0107
                                 - 0.147999894994259 z_k1_s0110
                                 - 0.280000861880719 z_k1_s0111
                                 - 0.245000754145629 z_k1_s0112
                                 - 0.106999993018802 z_k1_s0113
                                 - 0.309999833082553 z_k1_s0114
                                 - 0.264000756573266 z_k1_s0115
                                 - 0.153999885434568 z_k1_s0116
                                 - 0.454000780849632 z_k1_s0117
                                 - 0.169000171392358 z_k1_s0120
                                 - 0.319999517333358 z_k1_s0121
                                 - 0.316001110969218 z_k1_s0122
                                 - 0.254999541874903 z_k1_s0123
                                 - 0.535000883544399 z_k1_s0124
                                 - 0.120000171800571 z_k1_s0125
                                 - 0.247999978733914 z_k1_s0126
                                 - 0.389000251159697 z_k1_s0127
                                 - 0.111000120404203 z_k1_s0130
                                 - 0.105000044709335 z_k1_s0131
                                 - 0.112000151570026 z_k1_s0132
                                 - 0.0790000134984385 z_k1_s0133
                                 - 0.173999857849611 z_k1_s0134
                                 - 0.104000124843585 z_k1_s0135
                                 - 0.116000276233321 z_k1_s0136
                                 - 0.311999706730311 z_k1_s0137
                                 - 0.0590000612717387 z_k1_s0140
                                 - 0.052000083049698 z_k1_s0141
                                 - 0.0730001079159132 z_k1_s0142
                                 - 0.0570000546082852 z_k1_s0143
                                 - 0.092999903950169 z_k1_s0144
                                 - 0.233000370391914 z_k1_s0145
                                 - 0.213000303757379 z_k1_s0146
                                 - 0.122000135870384 z_k1_s0147
                                 - 0.0770000115130944 z_k1_s0150
                                 - 0.154000023026189 z_k1_s0151
                                 - 0.100000034640407 z_k1_s0152
                                 - 0.0590000659178979 z_k1_s0153
                                 - 0.0490000624538572 z_k1_s0154
                                 - 0.0870003181775191 z_k1_s0155
                                 - 0.092999895775406 z_k1_s0156
                                 - 0.138999942030032 z_k1_s0157
                                 - 0.52299876322358 z_k1_s0160
                                 - 0.241999576596888 z_k1_s0161
                                 - 0.281999771303356 z_k1_s0162
                                 - 0.202999690467559 z_k1_s0163
                                 - 0.69399824270606 z_k1_s0164
                                 - 0.20099962551233 z_k1_s0165
                                 - 0.18899978798001 z_k1_s0166
                                 - 0.426998958566916 z_k1_s0167
                                 - 0.383001578092019 z_k1_s0170
                                 - 0.176000398320445 z_k1_s0171
                                 - 0.491000474803984 z_k1_s0172
                                 - 0.116000187656498 z_k1_s0173
                                 - 0.521000168335085 z_k1_s0174
                                 - 0.117000246074347 z_k1_s0175
                                 - 0.114000276721237 z_k1_s0176
                                 - 0.141000206799664 z_k1_s0177
                                 - 0.251000092416456 z_k1_s1000
                                 - 0.45799997960776 z_k1_s1001
                                 - 0.575999635784512 z_k1_s1002
                                 - 1.2539994843183 z_k1_s1003
                                 - 1.02699739984333 z_k1_s1004
                                 - 0.412000133254107 z_k1_s1005
                                 - 1.02899935537129 z_k1_s1006
                                 - 0.778997926319357 z_k1_s1007
                                 - 0.465000820971035 z_k1_s1010
                                 - 1.22600105021419 z_k1_s1011
                                 - 0.581000851659209 z_k1_s1012
                                 - 1.10400006872155 z_k1_s1013
                                 - 0.499000595453373 z_k1_s1014
                                 - 0.805999264012753 z_k1_s1015
                                 - 0.711000941887432 z_k1_s1016
                                 - 1.08500299510094 z_k1_s1017
                                 - 1.98199457274965 z_k1_s1020
                                 - 2.23699896364341 z_k1_s1021
                                 - 1.36899637853773 z_k1_s1022
                                 - 0.507998796621542 z_k1_s1023
                                 - 0.593998423299142 z_k1_s1024
                                 - 1.91600023491679 z_k1_s1025
                                 - 0.834998061303638 z_k1_s1026
                                 - 1.06599610558371 z_k1_s1027
                                 - 0.425999987843256 z_k1_s1030
                                 - 0.35499998986938 z_k1_s1031
                                 - 0.561999054162163 z_k1_s1032
                                 - 1.08299857439437 z_k1_s1033
                                 - 0.737001373668362 z_k1_s1034
                                 - 0.265999992409169 z_k1_s1035
                                 - 0.536001844304323 z_k1_s1036
                                 - 0.683998120880588 z_k1_s1037
                                 - 1.39099929876005 z_k1_s1040
                                 - 0.851999290995177 z_k1_s1041
                                 - 1.87399652136989 z_k1_s1042
                                 - 1.086998880706 z_k1_s1043
                                 - 0.948998653318339 z_k1_s1044
                                 - 1.92899593877496 z_k1_s1045
                                 - 3.18999479547002 z_k1_s1046
                                 - 3.02099450664528 z_k1_s1047
                                 - 0.368000317724641 z_k1_s1050
                                 - 0.364000421566863 z_k1_s1051
                                 - 0.39900043837327 z_k1_s1052
                                 - 0.511999047206921 z_k1_s1053
                                 - 1.12799950751984 z_k1_s1054
                                 - 0.712002081103364 z_k1_s1055
                                 - 0.260000242361991 z_k1_s1056
                                 - 0.370000471453718 z_k1_s1057
                                 - 0.370001559729217 z_k1_s1060
                                 - 1.17200752982843 z_k1_s1061
                                 - 1.29600098786555 z_k1_s1062
                                 - 1.2300022483216 z_k1_s1063
                                 - 1.12700595449417 z_k1_s1064
                                 - 0.843000943691419 z_k1_s1065
                                 - 1.18100631448973 z_k1_s1066
                                 - 0.433001469588847 z_k1_s1067
                                 - 0.811998117688579 z_k1_s1070
                                 - 0.463999878779358 z_k1_s1071
                                 - 1.19200154040075 z_k1_s1072
                                 - 0.490000140012301 z_k1_s1073
                                 - 0.863999667954648 z_k1_s1074
                                 - 0.301000019934687 z_k1_s1075
                                 - 0.65500163971364 z_k1_s1076
                                 - 0.360999911225967 z_k1_s1077
                                 - 0.131999872497993 z_k1_s1100
                                 - 0.135999778998056 z_k1_s1101
                                 - 0.262000139745477 z_k1_s1102
                                 - 0.350999277370191 z_k1_s1103
                                 - 0.326999490370295 z_k1_s1104
                                 - 0.256999386621595 z_k1_s1105
                                 - 0.200999303622635 z_k1_s1106
                                 - 0.0739998362490018 z_k1_s1107
                                 - 0.165999888811441 z_k1_s1110
                                 - 0.123999980738619 z_k1_s1111
                                 - 0.113999851340442 z_k1_s1112
                                 - 0.394999639732117 z_k1_s1113
                                 - 0.372998931456452 z_k1_s1114
                                 - 0.0880000091048088 z_k1_s1115
                                 - 0.281000141391406 z_k1_s1116
                                 - 0.223999509721722 z_k1_s1117
                                 - 0.221000241947584 z_k1_s1120
                                 - 0.0919999403447594 z_k1_s1121
                                 - 0.0929999615463242 z_k1_s1122
                                 - 0.068999981708559 z_k1_s1123
                                 - 0.129999582204684 z_k1_s1124
                                 - 0.179999584283344 z_k1_s1125
                                 - 0.222999861150882 z_k1_s1126
                                 - 0.0500000020786599 z_k1_s1127
                                 - 1.07699622915949 z_k1_s1130
                                 - 0.289999707434176 z_k1_s1131
                                 - 0.671998816270347 z_k1_s1132
                                 - 0.339999458943833 z_k1_s1133
                                 - 1.02899915384992 z_k1_s1134
                                 - 0.234999671514047 z_k1_s1135
                                 - 0.340999250746351 z_k1_s1136
                                 - 0.433999327549452 z_k1_s1137
                                 - 0.207000015482155 z_k1_s1140
                                 - 0.220999653051475 z_k1_s1141
                                 - 0.117999907051797 z_k1_s1142
                                 - 0.131999879020677 z_k1_s1143
                                 - 0.192000523229155 z_k1_s1144
                                 - 0.402999957446031 z_k1_s1145
                                 - 0.230000017202395 z_k1_s1146
                                 - 0.169000521508916 z_k1_s1147
                                 - 0.200999564982106 z_k1_s1150
                                 - 0.100999829708863 z_k1_s1151
                                 - 0.0879998683443361 z_k1_s1152
                                 - 0.181000490266442 z_k1_s1153
                                 - 0.283999805347362 z_k1_s1154
                                 - 0.229999208218672 z_k1_s1155
                                 - 0.189999651788982 z_k1_s1156
                                 - 0.131999802516504 z_k1_s1157
                                 - 0.340999200935733 z_k1_s1160
                                 - 0.327999221867817 z_k1_s1161
                                 - 0.195999709408537 z_k1_s1162
                                 - 0.567000587036683 z_k1_s1163
                                 - 0.496998949750723 z_k1_s1164
                                 - 0.44399803509083 z_k1_s1165
                                 - 0.496998949750723 z_k1_s1166
                                 - 0.379999138139481 z_k1_s1167
                                 - 0.2240001337888 z_k1_s1170
                                 - 0.242999561743856 z_k1_s1171
                                 - 0.422999237109675 z_k1_s1172
                                 - 0.269999513048729 z_k1_s1173
                                 - 0.269000052630255 z_k1_s1174
                                 - 0.165000025085469 z_k1_s1175
                                 - 0.497999908512438 z_k1_s1176
                                 - 0.327999677333383 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.22518 w_k1_ADM_s000
                                 + 1.71206 w_k1_AXP_s000
                                 + 2.04334 w_k1_DIS_s000
                                 + 1.47063 w_k1_FDX_s000
                                 + 1.34527 w_k1_IBM_s000
                                 + 1.4819 w_k1_INTC_s000 + 1.31977 w_k1_KO_s000
                                 + 1.77872 w_k1_MAT_s000
                                 + 1.42373 w_k1_MSFT_s000
                                 + 1.28755 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.18336 w_k1_ADM_s000
                                 + 1.03006 w_k1_AXP_s000
                                 + 1.09114 w_k1_DIS_s000
                                 + 0.961628 w_k1_FDX_s000
                                 + 0.917384 w_k1_IBM_s000
                                 + 1.08581 w_k1_INTC_s000
                                 + 0.907249 w_k1_KO_s000
                                 + 1.11213 w_k1_MAT_s000
                                 + 1.09228 w_k1_MSFT_s000
                                 + 0.977278 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.2169 w_k1_ADM_s000
                                 + 0.914555 w_k1_AXP_s000
                                 + 0.945114 w_k1_DIS_s000
                                 + 0.907044 w_k1_FDX_s000
                                 + 0.761918 w_k1_IBM_s000
                                 + 1.06316 w_k1_INTC_s000
                                 + 0.832513 w_k1_KO_s000
                                 + 0.712224 w_k1_MAT_s000
                                 + 1.03624 w_k1_MSFT_s000
                                 + 0.829191 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.92887 w_k1_ADM_s000
                                 + 1.2156 w_k1_AXP_s000 + 1.09774 w_k1_DIS_s000
                                 + 1.01129 w_k1_FDX_s000
                                 + 1.44817 w_k1_IBM_s000
                                 + 1.37428 w_k1_INTC_s000
                                 + 1.12271 w_k1_KO_s000 + 1.63186 w_k1_MAT_s000
                                 + 1.0133 w_k1_MSFT_s000 + 1.0518 w_k1_PG_s000
                                 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.24812 w_k1_ADM_s000
                                 + 1.69196 w_k1_AXP_s000
                                 + 1.44484 w_k1_DIS_s000
                                 + 1.29771 w_k1_FDX_s000
                                 + 1.32331 w_k1_IBM_s000
                                 + 1.52544 w_k1_INTC_s000 + 1.2523 w_k1_KO_s000
                                 + 1.51366 w_k1_MAT_s000
                                 + 1.17158 w_k1_MSFT_s000
                                 + 1.25757 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.33554 w_k1_ADM_s000
                                 + 1.19644 w_k1_AXP_s000
                                 + 0.915993 w_k1_DIS_s000
                                 + 1.0294 w_k1_FDX_s000
                                 + 0.971128 w_k1_IBM_s000
                                 + 1.05824 w_k1_INTC_s000
                                 + 0.888797 w_k1_KO_s000
                                 + 1.09744 w_k1_MAT_s000
                                 + 0.929849 w_k1_MSFT_s000
                                 + 1.00037 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.0957 w_k1_ADM_s000
                                 + 0.70404 w_k1_AXP_s000
                                 + 0.567709 w_k1_DIS_s000
                                 + 0.778992 w_k1_FDX_s000
                                 + 0.760243 w_k1_IBM_s000
                                 + 0.90661 w_k1_INTC_s000
                                 + 0.666222 w_k1_KO_s000
                                 + 0.751245 w_k1_MAT_s000
                                 + 0.781233 w_k1_MSFT_s000
                                 + 0.879074 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.32292 w_k1_ADM_s000
                                 + 0.951002 w_k1_AXP_s000
                                 + 1.34623 w_k1_DIS_s000
                                 + 1.03257 w_k1_FDX_s000
                                 + 0.986826 w_k1_IBM_s000
                                 + 1.22813 w_k1_INTC_s000
                                 + 1.05339 w_k1_KO_s000
                                 + 0.908795 w_k1_MAT_s000
                                 + 1.21767 w_k1_MSFT_s000
                                 + 1.09053 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.00208 w_k1_ADM_s001
                                 + 1.24635 w_k1_AXP_s001
                                 + 1.61012 w_k1_DIS_s001
                                 + 1.53419 w_k1_FDX_s001
                                 + 0.810518 w_k1_IBM_s001
                                 + 1.47866 w_k1_INTC_s001
                                 + 1.25699 w_k1_KO_s001 + 0.70277 w_k1_MAT_s001
                                 + 1.13712 w_k1_MSFT_s001
                                 + 1.19004 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.10688 w_k1_ADM_s001
                                 + 1.25023 w_k1_AXP_s001
                                 + 1.21782 w_k1_DIS_s001
                                 + 1.12456 w_k1_FDX_s001
                                 + 0.864387 w_k1_IBM_s001
                                 + 1.03304 w_k1_INTC_s001
                                 + 0.972661 w_k1_KO_s001
                                 + 0.764068 w_k1_MAT_s001
                                 + 0.889496 w_k1_MSFT_s001
                                 + 1.16826 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.904918 w_k1_ADM_s001
                                 + 0.933413 w_k1_AXP_s001
                                 + 0.89538 w_k1_DIS_s001
                                 + 1.08573 w_k1_FDX_s001
                                 + 0.650476 w_k1_IBM_s001
                                 + 1.3228 w_k1_INTC_s001
                                 + 0.932432 w_k1_KO_s001
                                 + 1.07287 w_k1_MAT_s001
                                 + 1.08536 w_k1_MSFT_s001
                                 + 1.11232 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.12218 w_k1_ADM_s001
                                 + 1.08983 w_k1_AXP_s001
                                 + 0.917951 w_k1_DIS_s001
                                 + 0.930004 w_k1_FDX_s001
                                 + 1.11305 w_k1_IBM_s001
                                 + 1.16308 w_k1_INTC_s001
                                 + 1.18249 w_k1_KO_s001
                                 + 0.923326 w_k1_MAT_s001
                                 + 0.920986 w_k1_MSFT_s001
                                 + 1.19425 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.89916 w_k1_ADM_s001
                                 + 1.80413 w_k1_AXP_s001
                                 + 1.69644 w_k1_DIS_s001
                                 + 1.49431 w_k1_FDX_s001
                                 + 1.49907 w_k1_IBM_s001
                                 + 1.24551 w_k1_INTC_s001
                                 + 1.63527 w_k1_KO_s001 + 1.62343 w_k1_MAT_s001
                                 + 1.3382 w_k1_MSFT_s001 + 1.3736 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.781548 w_k1_ADM_s001
                                 + 0.799938 w_k1_AXP_s001
                                 + 0.686278 w_k1_DIS_s001
                                 + 0.815344 w_k1_FDX_s001
                                 + 0.831706 w_k1_IBM_s001
                                 + 0.790857 w_k1_INTC_s001
                                 + 0.851945 w_k1_KO_s001
                                 + 0.725197 w_k1_MAT_s001
                                 + 0.74119 w_k1_MSFT_s001
                                 + 0.997349 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.63726 w_k1_ADM_s001
                                 + 0.735446 w_k1_AXP_s001
                                 + 0.815197 w_k1_DIS_s001
                                 + 0.870521 w_k1_FDX_s001
                                 + 1.12807 w_k1_IBM_s001
                                 + 1.01504 w_k1_INTC_s001
                                 + 0.959106 w_k1_KO_s001
                                 + 1.21141 w_k1_MAT_s001
                                 + 0.905613 w_k1_MSFT_s001
                                 + 0.977843 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.01873 w_k1_ADM_s001
                                 + 1.51311 w_k1_AXP_s001
                                 + 1.05942 w_k1_DIS_s001
                                 + 1.11621 w_k1_FDX_s001
                                 + 1.45423 w_k1_IBM_s001
                                 + 1.12908 w_k1_INTC_s001
                                 + 1.16944 w_k1_KO_s001 + 1.88712 w_k1_MAT_s001
                                 + 1.0284 w_k1_MSFT_s001 + 1.05034 w_k1_PG_s001
                                 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.878765 w_k1_ADM_s002
                                 + 1.29943 w_k1_AXP_s002
                                 + 1.50972 w_k1_DIS_s002
                                 + 1.22242 w_k1_FDX_s002
                                 + 1.15377 w_k1_IBM_s002
                                 + 1.17992 w_k1_INTC_s002 + 1.409 w_k1_KO_s002
                                 + 1.48552 w_k1_MAT_s002
                                 + 1.2715 w_k1_MSFT_s002 + 1.25124 w_k1_PG_s002
                                 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.06739 w_k1_ADM_s002
                                 + 1.08251 w_k1_AXP_s002
                                 + 0.861216 w_k1_DIS_s002
                                 + 0.937023 w_k1_FDX_s002
                                 + 1.13249 w_k1_IBM_s002
                                 + 0.845184 w_k1_INTC_s002
                                 + 1.03251 w_k1_KO_s002 + 1.73789 w_k1_MAT_s002
                                 + 0.804521 w_k1_MSFT_s002
                                 + 1.07071 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.94174 w_k1_ADM_s002
                                 + 0.933832 w_k1_AXP_s002
                                 + 1.18888 w_k1_DIS_s002
                                 + 1.03581 w_k1_FDX_s002
                                 + 1.03906 w_k1_IBM_s002
                                 + 0.974958 w_k1_INTC_s002
                                 + 1.20071 w_k1_KO_s002 + 1.16865 w_k1_MAT_s002
                                 + 0.985516 w_k1_MSFT_s002
                                 + 0.984602 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.14675 w_k1_ADM_s002
                                 + 1.11971 w_k1_AXP_s002
                                 + 0.986852 w_k1_DIS_s002
                                 + 1.04129 w_k1_FDX_s002
                                 + 0.893521 w_k1_IBM_s002
                                 + 1.06928 w_k1_INTC_s002
                                 + 1.07013 w_k1_KO_s002 + 1.12797 w_k1_MAT_s002
                                 + 1.02119 w_k1_MSFT_s002
                                 + 1.05934 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.982753 w_k1_ADM_s002
                                 + 0.818379 w_k1_AXP_s002
                                 + 0.702362 w_k1_DIS_s002
                                 + 0.862599 w_k1_FDX_s002
                                 + 0.755145 w_k1_IBM_s002
                                 + 0.957026 w_k1_INTC_s002
                                 + 0.852537 w_k1_KO_s002
                                 + 0.957949 w_k1_MAT_s002
                                 + 0.817061 w_k1_MSFT_s002
                                 + 0.905645 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.93054 w_k1_ADM_s002
                                 + 1.07412 w_k1_AXP_s002
                                 + 1.01438 w_k1_DIS_s002
                                 + 0.979875 w_k1_FDX_s002
                                 + 1.39158 w_k1_IBM_s002
                                 + 0.862782 w_k1_INTC_s002
                                 + 1.21478 w_k1_KO_s002 + 1.47216 w_k1_MAT_s002
                                 + 0.809702 w_k1_MSFT_s002
                                 + 1.16258 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.68123 w_k1_ADM_s002
                                 + 0.731693 w_k1_AXP_s002
                                 + 0.654255 w_k1_DIS_s002
                                 + 0.750488 w_k1_FDX_s002
                                 + 0.958144 w_k1_IBM_s002
                                 + 0.812402 w_k1_INTC_s002
                                 + 0.879293 w_k1_KO_s002
                                 + 1.03433 w_k1_MAT_s002
                                 + 0.812166 w_k1_MSFT_s002
                                 + 1.00185 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.90448 w_k1_ADM_s002
                                 + 0.692915 w_k1_AXP_s002
                                 + 0.822477 w_k1_DIS_s002
                                 + 0.872373 w_k1_FDX_s002
                                 + 1.25971 w_k1_IBM_s002
                                 + 0.987471 w_k1_INTC_s002
                                 + 1.16107 w_k1_KO_s002 + 1.15933 w_k1_MAT_s002
                                 + 0.892202 w_k1_MSFT_s002
                                 + 0.941462 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.14892 w_k1_ADM_s003
                                 + 1.37756 w_k1_AXP_s003
                                 + 1.11212 w_k1_DIS_s003
                                 + 1.07708 w_k1_FDX_s003
                                 + 0.855213 w_k1_IBM_s003
                                 + 0.925237 w_k1_INTC_s003
                                 + 0.949188 w_k1_KO_s003
                                 + 1.31744 w_k1_MAT_s003
                                 + 0.933282 w_k1_MSFT_s003
                                 + 0.957072 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.20697 w_k1_ADM_s003
                                 + 0.608485 w_k1_AXP_s003
                                 + 0.630677 w_k1_DIS_s003
                                 + 0.827022 w_k1_FDX_s003
                                 + 0.620035 w_k1_IBM_s003
                                 + 0.645559 w_k1_INTC_s003
                                 + 0.819281 w_k1_KO_s003
                                 + 0.553116 w_k1_MAT_s003
                                 + 0.791529 w_k1_MSFT_s003
                                 + 0.985995 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.45393 w_k1_ADM_s003
                                 + 1.40436 w_k1_AXP_s003
                                 + 1.58971 w_k1_DIS_s003
                                 + 1.38118 w_k1_FDX_s003
                                 + 0.931421 w_k1_IBM_s003
                                 + 1.22092 w_k1_INTC_s003
                                 + 1.20324 w_k1_KO_s003 + 1.11899 w_k1_MAT_s003
                                 + 1.22835 w_k1_MSFT_s003
                                 + 0.994435 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.764119 w_k1_ADM_s003
                                 + 1.30269 w_k1_AXP_s003 + 1.0851 w_k1_DIS_s003
                                 + 1.02177 w_k1_FDX_s003
                                 + 1.15619 w_k1_IBM_s003
                                 + 1.02829 w_k1_INTC_s003
                                 + 1.14681 w_k1_KO_s003 + 1.8526 w_k1_MAT_s003
                                 + 1.00067 w_k1_MSFT_s003
                                 + 0.961945 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.09341 w_k1_ADM_s003
                                 + 0.866715 w_k1_AXP_s003
                                 + 0.951734 w_k1_DIS_s003
                                 + 0.987939 w_k1_FDX_s003
                                 + 0.734941 w_k1_IBM_s003
                                 + 0.846044 w_k1_INTC_s003
                                 + 0.882843 w_k1_KO_s003
                                 + 0.960042 w_k1_MAT_s003
                                 + 0.961958 w_k1_MSFT_s003
                                 + 0.899248 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.820843 w_k1_ADM_s003
                                 + 0.939879 w_k1_AXP_s003
                                 + 0.687595 w_k1_DIS_s003
                                 + 0.74387 w_k1_FDX_s003
                                 + 0.807582 w_k1_IBM_s003
                                 + 0.741454 w_k1_INTC_s003
                                 + 0.745805 w_k1_KO_s003
                                 + 1.22932 w_k1_MAT_s003
                                 + 0.690912 w_k1_MSFT_s003
                                 + 0.870135 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.02129 w_k1_ADM_s003
                                 + 0.925628 w_k1_AXP_s003
                                 + 0.889777 w_k1_DIS_s003
                                 + 0.840982 w_k1_FDX_s003
                                 + 1.35813 w_k1_IBM_s003
                                 + 0.773138 w_k1_INTC_s003
                                 + 1.11508 w_k1_KO_s003 + 1.41087 w_k1_MAT_s003
                                 + 0.760682 w_k1_MSFT_s003
                                 + 1.12054 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.949919 w_k1_ADM_s003
                                 + 0.82931 w_k1_AXP_s003
                                 + 0.917321 w_k1_DIS_s003
                                 + 0.872408 w_k1_FDX_s003
                                 + 0.949329 w_k1_IBM_s003
                                 + 0.925976 w_k1_INTC_s003
                                 + 1.02695 w_k1_KO_s003 + 1.06114 w_k1_MAT_s003
                                 + 0.911849 w_k1_MSFT_s003
                                 + 0.819135 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.03612 w_k1_ADM_s004
                                 + 0.997981 w_k1_AXP_s004
                                 + 0.881915 w_k1_DIS_s004
                                 + 1.01175 w_k1_FDX_s004
                                 + 0.882381 w_k1_IBM_s004
                                 + 1.00886 w_k1_INTC_s004
                                 + 1.06071 w_k1_KO_s004 + 1.06878 w_k1_MAT_s004
                                 + 1.01715 w_k1_MSFT_s004
                                 + 1.02702 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.2135 w_k1_ADM_s004
                                 + 1.70778 w_k1_AXP_s004
                                 + 2.07336 w_k1_DIS_s004
                                 + 1.51159 w_k1_FDX_s004
                                 + 1.74847 w_k1_IBM_s004
                                 + 1.35183 w_k1_INTC_s004
                                 + 1.77217 w_k1_KO_s004 + 2.23482 w_k1_MAT_s004
                                 + 1.35644 w_k1_MSFT_s004
                                 + 1.34249 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.962089 w_k1_ADM_s004
                                 + 0.950675 w_k1_AXP_s004
                                 + 0.889173 w_k1_DIS_s004
                                 + 0.936783 w_k1_FDX_s004
                                 + 1.17042 w_k1_IBM_s004
                                 + 0.961536 w_k1_INTC_s004
                                 + 0.892455 w_k1_KO_s004
                                 + 1.33458 w_k1_MAT_s004
                                 + 0.827067 w_k1_MSFT_s004
                                 + 0.93399 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.26227 w_k1_ADM_s004
                                 + 1.3685 w_k1_AXP_s004 + 1.44426 w_k1_DIS_s004
                                 + 1.38712 w_k1_FDX_s004
                                 + 0.997447 w_k1_IBM_s004
                                 + 1.41152 w_k1_INTC_s004
                                 + 0.993456 w_k1_KO_s004
                                 + 0.735051 w_k1_MAT_s004
                                 + 1.19871 w_k1_MSFT_s004
                                 + 1.05441 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.10162 w_k1_ADM_s004
                                 + 1.03434 w_k1_AXP_s004
                                 + 1.29927 w_k1_DIS_s004
                                 + 1.17258 w_k1_FDX_s004
                                 + 0.89429 w_k1_IBM_s004
                                 + 1.20103 w_k1_INTC_s004
                                 + 1.15159 w_k1_KO_s004 + 0.89451 w_k1_MAT_s004
                                 + 1.38548 w_k1_MSFT_s004
                                 + 1.16186 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.65337 w_k1_ADM_s004
                                 + 0.798055 w_k1_AXP_s004
                                 + 0.861947 w_k1_DIS_s004
                                 + 0.943101 w_k1_FDX_s004
                                 + 0.643551 w_k1_IBM_s004
                                 + 1.00372 w_k1_INTC_s004
                                 + 0.692408 w_k1_KO_s004
                                 + 0.545374 w_k1_MAT_s004
                                 + 0.916165 w_k1_MSFT_s004
                                 + 0.847673 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.895351 w_k1_ADM_s004
                                 + 1.02167 w_k1_AXP_s004
                                 + 1.41722 w_k1_DIS_s004
                                 + 0.966054 w_k1_FDX_s004
                                 + 1.54971 w_k1_IBM_s004
                                 + 1.07732 w_k1_INTC_s004
                                 + 1.31388 w_k1_KO_s004 + 1.23172 w_k1_MAT_s004
                                 + 1.10588 w_k1_MSFT_s004 + 1.1441 w_k1_PG_s004
                                 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.07264 w_k1_ADM_s004
                                 + 0.838037 w_k1_AXP_s004
                                 + 1.25812 w_k1_DIS_s004
                                 + 1.08096 w_k1_FDX_s004
                                 + 1.20217 w_k1_IBM_s004
                                 + 1.21268 w_k1_INTC_s004
                                 + 1.28163 w_k1_KO_s004
                                 + 0.999853 w_k1_MAT_s004
                                 + 1.12963 w_k1_MSFT_s004
                                 + 1.08853 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.30045 w_k1_ADM_s005
                                 + 1.26495 w_k1_AXP_s005
                                 + 1.26375 w_k1_DIS_s005
                                 + 1.27764 w_k1_FDX_s005
                                 + 1.04538 w_k1_IBM_s005
                                 + 1.14344 w_k1_INTC_s005
                                 + 1.03405 w_k1_KO_s005 + 1.03983 w_k1_MAT_s005
                                 + 1.02907 w_k1_MSFT_s005
                                 + 1.25624 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.14894 w_k1_ADM_s005
                                 + 0.864802 w_k1_AXP_s005
                                 + 1.09712 w_k1_DIS_s005
                                 + 1.15722 w_k1_FDX_s005
                                 + 0.79686 w_k1_IBM_s005
                                 + 1.42156 w_k1_INTC_s005
                                 + 0.967628 w_k1_KO_s005
                                 + 0.78418 w_k1_MAT_s005
                                 + 1.22759 w_k1_MSFT_s005
                                 + 1.12475 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.800329 w_k1_ADM_s005
                                 + 0.656446 w_k1_AXP_s005
                                 + 0.605519 w_k1_DIS_s005
                                 + 0.722958 w_k1_FDX_s005
                                 + 0.773401 w_k1_IBM_s005
                                 + 0.905991 w_k1_INTC_s005
                                 + 0.648383 w_k1_KO_s005
                                 + 0.573458 w_k1_MAT_s005
                                 + 0.778629 w_k1_MSFT_s005
                                 + 0.880584 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.52761 w_k1_ADM_s005
                                 + 1.31875 w_k1_AXP_s005
                                 + 1.76968 w_k1_DIS_s005 + 1.4046 w_k1_FDX_s005
                                 + 1.17876 w_k1_IBM_s005
                                 + 1.19484 w_k1_INTC_s005
                                 + 1.55751 w_k1_KO_s005 + 1.47956 w_k1_MAT_s005
                                 + 1.45202 w_k1_MSFT_s005
                                 + 1.47885 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.10094 w_k1_ADM_s005
                                 + 1.01862 w_k1_AXP_s005
                                 + 0.819417 w_k1_DIS_s005
                                 + 1.04014 w_k1_FDX_s005
                                 + 0.800916 w_k1_IBM_s005
                                 + 1.08169 w_k1_INTC_s005
                                 + 0.816944 w_k1_KO_s005
                                 + 0.732656 w_k1_MAT_s005
                                 + 0.859191 w_k1_MSFT_s005
                                 + 1.05771 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.44544 w_k1_ADM_s005
                                 + 1.23754 w_k1_AXP_s005
                                 + 1.13884 w_k1_DIS_s005
                                 + 1.04087 w_k1_FDX_s005
                                 + 1.42695 w_k1_IBM_s005
                                 + 1.10766 w_k1_INTC_s005 + 1.1554 w_k1_KO_s005
                                 + 1.01576 w_k1_MAT_s005
                                 + 1.00546 w_k1_MSFT_s005
                                 + 1.11749 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.02704 w_k1_ADM_s005
                                 + 0.862484 w_k1_AXP_s005
                                 + 1.08326 w_k1_DIS_s005
                                 + 0.91076 w_k1_FDX_s005
                                 + 1.07978 w_k1_IBM_s005
                                 + 0.971648 w_k1_INTC_s005
                                 + 1.09808 w_k1_KO_s005
                                 + 0.841003 w_k1_MAT_s005
                                 + 0.962526 w_k1_MSFT_s005
                                 + 1.2473 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.15704 w_k1_ADM_s005
                                 + 0.570762 w_k1_AXP_s005
                                 + 0.796118 w_k1_DIS_s005
                                 + 0.876652 w_k1_FDX_s005
                                 + 0.524256 w_k1_IBM_s005
                                 + 1.05252 w_k1_INTC_s005
                                 + 0.759931 w_k1_KO_s005
                                 + 0.410065 w_k1_MAT_s005
                                 + 1.00849 w_k1_MSFT_s005
                                 + 0.84453 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.06124 w_k1_ADM_s006
                                 + 1.43889 w_k1_AXP_s006 + 1.0889 w_k1_DIS_s006
                                 + 1.12396 w_k1_FDX_s006
                                 + 1.46623 w_k1_IBM_s006
                                 + 0.911162 w_k1_INTC_s006
                                 + 1.12604 w_k1_KO_s006 + 1.84403 w_k1_MAT_s006
                                 + 0.941888 w_k1_MSFT_s006
                                 + 1.17644 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.709777 w_k1_ADM_s006
                                 + 1.27686 w_k1_AXP_s006
                                 + 1.20876 w_k1_DIS_s006
                                 + 1.13878 w_k1_FDX_s006
                                 + 1.19203 w_k1_IBM_s006
                                 + 1.00294 w_k1_INTC_s006
                                 + 1.11065 w_k1_KO_s006 + 1.53563 w_k1_MAT_s006
                                 + 1.09486 w_k1_MSFT_s006
                                 + 1.17859 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.91294 w_k1_ADM_s006
                                 + 1.26102 w_k1_AXP_s006
                                 + 1.06447 w_k1_DIS_s006
                                 + 1.17373 w_k1_FDX_s006
                                 + 0.881752 w_k1_IBM_s006
                                 + 0.855973 w_k1_INTC_s006
                                 + 0.893487 w_k1_KO_s006
                                 + 1.34794 w_k1_MAT_s006
                                 + 1.00849 w_k1_MSFT_s006
                                 + 1.04652 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.690483 w_k1_ADM_s006
                                 + 0.992308 w_k1_AXP_s006
                                 + 0.699732 w_k1_DIS_s006
                                 + 0.80323 w_k1_FDX_s006
                                 + 1.18937 w_k1_IBM_s006
                                 + 0.943205 w_k1_INTC_s006 + 1.01 w_k1_KO_s006
                                 + 1.02741 w_k1_MAT_s006
                                 + 0.860859 w_k1_MSFT_s006
                                 + 1.01992 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.23279 w_k1_ADM_s006
                                 + 1.5908 w_k1_AXP_s006 + 1.65146 w_k1_DIS_s006
                                 + 1.46852 w_k1_FDX_s006
                                 + 1.14339 w_k1_IBM_s006
                                 + 1.51186 w_k1_INTC_s006
                                 + 1.13022 w_k1_KO_s006
                                 + 0.894429 w_k1_MAT_s006
                                 + 1.33464 w_k1_MSFT_s006
                                 + 1.30147 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.511954 w_k1_ADM_s006
                                 + 0.678379 w_k1_AXP_s006
                                 + 0.570918 w_k1_DIS_s006
                                 + 0.720404 w_k1_FDX_s006
                                 + 0.67563 w_k1_IBM_s006
                                 + 0.966135 w_k1_INTC_s006
                                 + 0.716928 w_k1_KO_s006
                                 + 0.611774 w_k1_MAT_s006
                                 + 0.810049 w_k1_MSFT_s006
                                 + 0.764611 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.784235 w_k1_ADM_s006
                                 + 0.867669 w_k1_AXP_s006
                                 + 0.801952 w_k1_DIS_s006
                                 + 0.900719 w_k1_FDX_s006
                                 + 0.797231 w_k1_IBM_s006
                                 + 0.816489 w_k1_INTC_s006
                                 + 0.739601 w_k1_KO_s006
                                 + 0.967701 w_k1_MAT_s006
                                 + 0.8368 w_k1_MSFT_s006
                                 + 0.927577 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.785013 w_k1_ADM_s006
                                 + 0.958744 w_k1_AXP_s006
                                 + 0.875927 w_k1_DIS_s006
                                 + 1.02968 w_k1_FDX_s006
                                 + 0.786975 w_k1_IBM_s006
                                 + 1.12964 w_k1_INTC_s006
                                 + 0.810811 w_k1_KO_s006
                                 + 0.990697 w_k1_MAT_s006
                                 + 1.03426 w_k1_MSFT_s006
                                 + 1.03457 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.81877 w_k1_ADM_s007
                                 + 0.639965 w_k1_AXP_s007
                                 + 0.732197 w_k1_DIS_s007
                                 + 0.86551 w_k1_FDX_s007
                                 + 0.641251 w_k1_IBM_s007
                                 + 0.838943 w_k1_INTC_s007
                                 + 0.731105 w_k1_KO_s007
                                 + 0.670417 w_k1_MAT_s007
                                 + 0.779226 w_k1_MSFT_s007
                                 + 0.871881 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.785904 w_k1_ADM_s007
                                 + 0.800562 w_k1_AXP_s007
                                 + 0.74963 w_k1_DIS_s007
                                 + 0.786551 w_k1_FDX_s007
                                 + 1.19828 w_k1_IBM_s007
                                 + 0.96014 w_k1_INTC_s007 + 1.0102 w_k1_KO_s007
                                 + 1.29359 w_k1_MAT_s007
                                 + 0.788966 w_k1_MSFT_s007
                                 + 0.900155 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.37133 w_k1_ADM_s007
                                 + 1.05034 w_k1_AXP_s007
                                 + 1.11905 w_k1_DIS_s007
                                 + 1.10037 w_k1_FDX_s007
                                 + 1.07783 w_k1_IBM_s007
                                 + 0.856644 w_k1_INTC_s007
                                 + 1.03279 w_k1_KO_s007 + 1.37638 w_k1_MAT_s007
                                 + 1.01517 w_k1_MSFT_s007
                                 + 1.08389 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.974666 w_k1_ADM_s007
                                 + 0.789157 w_k1_AXP_s007
                                 + 0.8903 w_k1_DIS_s007
                                 + 0.913887 w_k1_FDX_s007
                                 + 0.870946 w_k1_IBM_s007
                                 + 0.650054 w_k1_INTC_s007
                                 + 0.754874 w_k1_KO_s007
                                 + 0.884763 w_k1_MAT_s007
                                 + 0.886586 w_k1_MSFT_s007
                                 + 0.913248 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.56876 w_k1_ADM_s007
                                 + 1.51725 w_k1_AXP_s007
                                 + 1.64725 w_k1_DIS_s007
                                 + 1.40865 w_k1_FDX_s007
                                 + 1.49238 w_k1_IBM_s007
                                 + 1.40051 w_k1_INTC_s007
                                 + 1.50368 w_k1_KO_s007 + 1.65775 w_k1_MAT_s007
                                 + 1.29113 w_k1_MSFT_s007 + 1.3091 w_k1_PG_s007
                                 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.997749 w_k1_ADM_s007
                                 + 0.852674 w_k1_AXP_s007
                                 + 1.3591 w_k1_DIS_s007 + 1.25461 w_k1_FDX_s007
                                 + 0.958885 w_k1_IBM_s007
                                 + 1.08272 w_k1_INTC_s007
                                 + 1.22056 w_k1_KO_s007 + 1.12112 w_k1_MAT_s007
                                 + 1.29534 w_k1_MSFT_s007
                                 + 1.17867 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.25326 w_k1_ADM_s007
                                 + 1.08199 w_k1_AXP_s007
                                 + 0.819363 w_k1_DIS_s007
                                 + 1.04952 w_k1_FDX_s007
                                 + 0.894716 w_k1_IBM_s007
                                 + 0.913131 w_k1_INTC_s007
                                 + 0.891763 w_k1_KO_s007
                                 + 0.745734 w_k1_MAT_s007
                                 + 0.749729 w_k1_MSFT_s007
                                 + 0.935568 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.07557 w_k1_ADM_s007
                                 + 0.706727 w_k1_AXP_s007
                                 + 0.495495 w_k1_DIS_s007
                                 + 0.795175 w_k1_FDX_s007
                                 + 0.685893 w_k1_IBM_s007
                                 + 0.724959 w_k1_INTC_s007
                                 + 0.612541 w_k1_KO_s007
                                 + 0.605521 w_k1_MAT_s007
                                 + 0.686136 w_k1_MSFT_s007
                                 + 0.859021 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.11389 w_k1_ADM_s010
                                 + 1.10171 w_k1_AXP_s010
                                 + 1.11361 w_k1_DIS_s010
                                 + 1.01901 w_k1_FDX_s010
                                 + 1.11135 w_k1_IBM_s010
                                 + 0.925395 w_k1_INTC_s010
                                 + 1.06415 w_k1_KO_s010 + 1.56637 w_k1_MAT_s010
                                 + 1.06139 w_k1_MSFT_s010
                                 + 1.06652 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.15684 w_k1_ADM_s010
                                 + 0.802855 w_k1_AXP_s010
                                 + 0.866535 w_k1_DIS_s010
                                 + 0.981195 w_k1_FDX_s010
                                 + 0.783008 w_k1_IBM_s010
                                 + 0.769332 w_k1_INTC_s010
                                 + 0.741111 w_k1_KO_s010
                                 + 0.859163 w_k1_MAT_s010
                                 + 0.962227 w_k1_MSFT_s010
                                 + 0.9834 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.01332 w_k1_ADM_s010
                                 + 1.10528 w_k1_AXP_s010
                                 + 1.61771 w_k1_DIS_s010
                                 + 1.20531 w_k1_FDX_s010
                                 + 1.06238 w_k1_IBM_s010 + 1.172 w_k1_INTC_s010
                                 + 1.27303 w_k1_KO_s010 + 1.04612 w_k1_MAT_s010
                                 + 1.36358 w_k1_MSFT_s010 + 1.1522 w_k1_PG_s010
                                 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.965057 w_k1_ADM_s010
                                 + 0.765222 w_k1_AXP_s010
                                 + 0.687666 w_k1_DIS_s010
                                 + 0.824489 w_k1_FDX_s010
                                 + 0.856302 w_k1_IBM_s010
                                 + 0.655835 w_k1_INTC_s010
                                 + 0.772373 w_k1_KO_s010 + 1.0801 w_k1_MAT_s010
                                 + 0.805945 w_k1_MSFT_s010
                                 + 0.957364 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.28869 w_k1_ADM_s010
                                 + 1.12217 w_k1_AXP_s010 + 1.2481 w_k1_DIS_s010
                                 + 1.03304 w_k1_FDX_s010
                                 + 1.54977 w_k1_IBM_s010
                                 + 0.988149 w_k1_INTC_s010
                                 + 1.2083 w_k1_KO_s010 + 1.92544 w_k1_MAT_s010
                                 + 1.11891 w_k1_MSFT_s010
                                 + 1.16048 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.37484 w_k1_ADM_s010
                                 + 1.50662 w_k1_AXP_s010
                                 + 1.66094 w_k1_DIS_s010
                                 + 1.34467 w_k1_FDX_s010
                                 + 1.11466 w_k1_IBM_s010
                                 + 1.09174 w_k1_INTC_s010
                                 + 1.40917 w_k1_KO_s010 + 1.80995 w_k1_MAT_s010
                                 + 1.28969 w_k1_MSFT_s010
                                 + 1.22599 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.909355 w_k1_ADM_s010
                                 + 0.782768 w_k1_AXP_s010
                                 + 0.814301 w_k1_DIS_s010
                                 + 0.791665 w_k1_FDX_s010
                                 + 1.15816 w_k1_IBM_s010
                                 + 0.621203 w_k1_INTC_s010
                                 + 0.937066 w_k1_KO_s010
                                 + 1.50394 w_k1_MAT_s010
                                 + 0.783318 w_k1_MSFT_s010
                                 + 0.928433 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.25891 w_k1_ADM_s010
                                 + 1.13851 w_k1_AXP_s010 + 1.1055 w_k1_DIS_s010
                                 + 1.13334 w_k1_FDX_s010
                                 + 0.880534 w_k1_IBM_s010
                                 + 0.934218 w_k1_INTC_s010
                                 + 0.960874 w_k1_KO_s010 + 1.2749 w_k1_MAT_s010
                                 + 1.05328 w_k1_MSFT_s010
                                 + 1.08307 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.882134 w_k1_ADM_s011
                                 + 1.15586 w_k1_AXP_s011
                                 + 1.10773 w_k1_DIS_s011
                                 + 0.955861 w_k1_FDX_s011
                                 + 1.36916 w_k1_IBM_s011
                                 + 0.924811 w_k1_INTC_s011
                                 + 1.24114 w_k1_KO_s011 + 1.73107 w_k1_MAT_s011
                                 + 1.00924 w_k1_MSFT_s011
                                 + 1.09618 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.958153 w_k1_ADM_s011
                                 + 0.9636 w_k1_AXP_s011 + 1.24183 w_k1_DIS_s011
                                 + 1.22732 w_k1_FDX_s011
                                 + 0.884054 w_k1_IBM_s011
                                 + 1.27754 w_k1_INTC_s011
                                 + 1.07433 w_k1_KO_s011
                                 + 0.889975 w_k1_MAT_s011
                                 + 1.3869 w_k1_MSFT_s011 + 1.08907 w_k1_PG_s011
                                 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.83465 w_k1_ADM_s011
                                 + 0.57356 w_k1_AXP_s011
                                 + 0.59539 w_k1_DIS_s011
                                 + 0.772822 w_k1_FDX_s011
                                 + 0.687819 w_k1_IBM_s011
                                 + 0.864491 w_k1_INTC_s011
                                 + 0.74995 w_k1_KO_s011
                                 + 0.677023 w_k1_MAT_s011
                                 + 0.871933 w_k1_MSFT_s011
                                 + 0.836142 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.02276 w_k1_ADM_s011
                                 + 1.35054 w_k1_AXP_s011
                                 + 0.919954 w_k1_DIS_s011
                                 + 1.04794 w_k1_FDX_s011
                                 + 1.25135 w_k1_IBM_s011
                                 + 1.10361 w_k1_INTC_s011
                                 + 1.23587 w_k1_KO_s011 + 1.79206 w_k1_MAT_s011
                                 + 1.04081 w_k1_MSFT_s011
                                 + 1.02645 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.23402 w_k1_ADM_s011
                                 + 1.24475 w_k1_AXP_s011 + 1.6672 w_k1_DIS_s011
                                 + 1.32193 w_k1_FDX_s011
                                 + 1.24106 w_k1_IBM_s011
                                 + 1.44134 w_k1_INTC_s011
                                 + 1.55638 w_k1_KO_s011 + 1.6374 w_k1_MAT_s011
                                 + 1.44553 w_k1_MSFT_s011
                                 + 1.31996 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.644904 w_k1_ADM_s011
                                 + 0.870934 w_k1_AXP_s011
                                 + 0.783937 w_k1_DIS_s011
                                 + 0.846862 w_k1_FDX_s011
                                 + 0.942053 w_k1_IBM_s011
                                 + 0.984552 w_k1_INTC_s011
                                 + 0.846358 w_k1_KO_s011
                                 + 1.20414 w_k1_MAT_s011
                                 + 0.891309 w_k1_MSFT_s011
                                 + 0.828435 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.828841 w_k1_ADM_s011
                                 + 0.652854 w_k1_AXP_s011
                                 + 0.783609 w_k1_DIS_s011
                                 + 0.75522 w_k1_FDX_s011
                                 + 1.05665 w_k1_IBM_s011
                                 + 0.807234 w_k1_INTC_s011
                                 + 1.07258 w_k1_KO_s011
                                 + 0.768661 w_k1_MAT_s011
                                 + 0.96188 w_k1_MSFT_s011
                                 + 0.850187 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.979063 w_k1_ADM_s011
                                 + 0.908932 w_k1_AXP_s011
                                 + 0.966186 w_k1_DIS_s011
                                 + 0.932162 w_k1_FDX_s011
                                 + 0.926678 w_k1_IBM_s011
                                 + 0.914209 w_k1_INTC_s011
                                 + 1.09132 w_k1_KO_s011 + 1.06846 w_k1_MAT_s011
                                 + 1.12089 w_k1_MSFT_s011
                                 + 1.05864 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.35201 w_k1_ADM_s012
                                 + 1.36956 w_k1_AXP_s012
                                 + 1.64054 w_k1_DIS_s012
                                 + 1.39075 w_k1_FDX_s012
                                 + 1.05132 w_k1_IBM_s012
                                 + 1.4834 w_k1_INTC_s012 + 1.37708 w_k1_KO_s012
                                 + 0.96264 w_k1_MAT_s012
                                 + 1.2708 w_k1_MSFT_s012 + 1.22445 w_k1_PG_s012
                                 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.836971 w_k1_ADM_s012
                                 + 0.945744 w_k1_AXP_s012
                                 + 0.744357 w_k1_DIS_s012
                                 + 0.78802 w_k1_FDX_s012
                                 + 0.66748 w_k1_IBM_s012
                                 + 0.853257 w_k1_INTC_s012
                                 + 0.790938 w_k1_KO_s012
                                 + 0.722482 w_k1_MAT_s012
                                 + 0.872598 w_k1_MSFT_s012
                                 + 0.809147 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.16799 w_k1_ADM_s012
                                 + 1.25063 w_k1_AXP_s012
                                 + 1.20554 w_k1_DIS_s012
                                 + 1.14361 w_k1_FDX_s012
                                 + 0.923158 w_k1_IBM_s012
                                 + 1.24846 w_k1_INTC_s012
                                 + 1.09789 w_k1_KO_s012 + 0.78454 w_k1_MAT_s012
                                 + 1.22361 w_k1_MSFT_s012
                                 + 1.11914 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.06803 w_k1_ADM_s012
                                 + 1.54136 w_k1_AXP_s012
                                 + 1.55219 w_k1_DIS_s012
                                 + 1.21708 w_k1_FDX_s012
                                 + 1.51734 w_k1_IBM_s012
                                 + 1.56728 w_k1_INTC_s012
                                 + 1.66596 w_k1_KO_s012 + 1.7803 w_k1_MAT_s012
                                 + 1.30323 w_k1_MSFT_s012
                                 + 1.13657 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.995354 w_k1_ADM_s012
                                 + 0.931089 w_k1_AXP_s012
                                 + 0.89461 w_k1_DIS_s012
                                 + 0.947446 w_k1_FDX_s012
                                 + 0.876184 w_k1_IBM_s012
                                 + 1.05809 w_k1_INTC_s012
                                 + 0.973466 w_k1_KO_s012
                                 + 0.839574 w_k1_MAT_s012
                                 + 1.01652 w_k1_MSFT_s012
                                 + 0.967648 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.866282 w_k1_ADM_s012
                                 + 0.978999 w_k1_AXP_s012
                                 + 1.38605 w_k1_DIS_s012
                                 + 1.05376 w_k1_FDX_s012
                                 + 1.23893 w_k1_IBM_s012
                                 + 1.00029 w_k1_INTC_s012
                                 + 1.40927 w_k1_KO_s012 + 1.13501 w_k1_MAT_s012
                                 + 1.21645 w_k1_MSFT_s012
                                 + 1.15354 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.960069 w_k1_ADM_s012
                                 + 1.00603 w_k1_AXP_s012
                                 + 0.984317 w_k1_DIS_s012
                                 + 0.952323 w_k1_FDX_s012
                                 + 1.40227 w_k1_IBM_s012
                                 + 1.01235 w_k1_INTC_s012
                                 + 1.23473 w_k1_KO_s012 + 1.67534 w_k1_MAT_s012
                                 + 1.02909 w_k1_MSFT_s012
                                 + 1.00821 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.99535 w_k1_ADM_s012
                                 + 0.786135 w_k1_AXP_s012
                                 + 0.736166 w_k1_DIS_s012
                                 + 0.732808 w_k1_FDX_s012
                                 + 1.10135 w_k1_IBM_s012
                                 + 0.790624 w_k1_INTC_s012
                                 + 1.03481 w_k1_KO_s012 + 1.10543 w_k1_MAT_s012
                                 + 0.817771 w_k1_MSFT_s012
                                 + 0.940081 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.25795 w_k1_ADM_s013
                                 + 1.58315 w_k1_AXP_s013
                                 + 1.46529 w_k1_DIS_s013
                                 + 1.40262 w_k1_FDX_s013
                                 + 0.868666 w_k1_IBM_s013
                                 + 1.19906 w_k1_INTC_s013
                                 + 0.975738 w_k1_KO_s013
                                 + 0.862065 w_k1_MAT_s013
                                 + 1.04697 w_k1_MSFT_s013
                                 + 1.00478 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.00186 w_k1_ADM_s013
                                 + 1.44751 w_k1_AXP_s013
                                 + 1.05391 w_k1_DIS_s013
                                 + 1.03957 w_k1_FDX_s013
                                 + 1.29963 w_k1_IBM_s013
                                 + 1.12356 w_k1_INTC_s013
                                 + 0.997895 w_k1_KO_s013
                                 + 1.78413 w_k1_MAT_s013
                                 + 0.897017 w_k1_MSFT_s013
                                 + 0.840825 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.925423 w_k1_ADM_s013
                                 + 1.13308 w_k1_AXP_s013
                                 + 1.42991 w_k1_DIS_s013
                                 + 1.06543 w_k1_FDX_s013
                                 + 1.15599 w_k1_IBM_s013
                                 + 1.14333 w_k1_INTC_s013
                                 + 1.18152 w_k1_KO_s013 + 1.19545 w_k1_MAT_s013
                                 + 1.11725 w_k1_MSFT_s013
                                 + 1.15854 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.22144 w_k1_ADM_s013
                                 + 1.33159 w_k1_AXP_s013
                                 + 1.28089 w_k1_DIS_s013
                                 + 1.35377 w_k1_FDX_s013
                                 + 1.01293 w_k1_IBM_s013
                                 + 1.01779 w_k1_INTC_s013
                                 + 0.922913 w_k1_KO_s013
                                 + 1.17986 w_k1_MAT_s013
                                 + 0.982361 w_k1_MSFT_s013
                                 + 1.0599 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.02678 w_k1_ADM_s013
                                 + 0.998625 w_k1_AXP_s013
                                 + 0.794994 w_k1_DIS_s013
                                 + 1.0105 w_k1_FDX_s013
                                 + 0.734132 w_k1_IBM_s013
                                 + 0.874338 w_k1_INTC_s013
                                 + 0.747865 w_k1_KO_s013
                                 + 0.947505 w_k1_MAT_s013
                                 + 0.843711 w_k1_MSFT_s013
                                 + 0.934116 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.741501 w_k1_ADM_s013
                                 + 0.705801 w_k1_AXP_s013
                                 + 0.721657 w_k1_DIS_s013
                                 + 0.80979 w_k1_FDX_s013
                                 + 0.645396 w_k1_IBM_s013
                                 + 0.864657 w_k1_INTC_s013
                                 + 0.707254 w_k1_KO_s013
                                 + 0.684554 w_k1_MAT_s013
                                 + 0.769602 w_k1_MSFT_s013
                                 + 0.842264 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.62945 w_k1_ADM_s013
                                 + 1.24968 w_k1_AXP_s013
                                 + 1.78702 w_k1_DIS_s013 + 1.3443 w_k1_FDX_s013
                                 + 1.59501 w_k1_IBM_s013
                                 + 1.04624 w_k1_INTC_s013
                                 + 1.71384 w_k1_KO_s013 + 2.20961 w_k1_MAT_s013
                                 + 1.31801 w_k1_MSFT_s013
                                 + 1.40868 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.20905 w_k1_ADM_s013
                                 + 0.871765 w_k1_AXP_s013
                                 + 0.833039 w_k1_DIS_s013
                                 + 0.866101 w_k1_FDX_s013
                                 + 0.993854 w_k1_IBM_s013
                                 + 0.759452 w_k1_INTC_s013
                                 + 0.875014 w_k1_KO_s013
                                 + 0.993318 w_k1_MAT_s013
                                 + 0.838925 w_k1_MSFT_s013
                                 + 0.933288 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.25495 w_k1_ADM_s014
                                 + 1.66245 w_k1_AXP_s014
                                 + 1.40247 w_k1_DIS_s014
                                 + 1.44107 w_k1_FDX_s014
                                 + 0.789847 w_k1_IBM_s014
                                 + 1.28545 w_k1_INTC_s014
                                 + 1.05746 w_k1_KO_s014 + 1.10589 w_k1_MAT_s014
                                 + 1.08724 w_k1_MSFT_s014
                                 + 1.15449 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.18877 w_k1_ADM_s014
                                 + 1.19288 w_k1_AXP_s014
                                 + 1.12676 w_k1_DIS_s014
                                 + 0.972644 w_k1_FDX_s014
                                 + 1.37702 w_k1_IBM_s014
                                 + 0.857764 w_k1_INTC_s014
                                 + 1.52717 w_k1_KO_s014 + 1.32145 w_k1_MAT_s014
                                 + 0.933017 w_k1_MSFT_s014
                                 + 1.08813 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.28616 w_k1_ADM_s014
                                 + 1.358 w_k1_AXP_s014 + 1.01899 w_k1_DIS_s014
                                 + 1.12429 w_k1_FDX_s014
                                 + 0.880264 w_k1_IBM_s014
                                 + 1.00426 w_k1_INTC_s014
                                 + 1.11598 w_k1_KO_s014 + 1.0547 w_k1_MAT_s014
                                 + 1.11613 w_k1_MSFT_s014
                                 + 0.945317 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.15085 w_k1_ADM_s014
                                 + 1.55192 w_k1_AXP_s014
                                 + 1.07885 w_k1_DIS_s014
                                 + 1.12157 w_k1_FDX_s014
                                 + 1.56573 w_k1_IBM_s014
                                 + 1.03191 w_k1_INTC_s014
                                 + 1.13232 w_k1_KO_s014 + 1.66522 w_k1_MAT_s014
                                 + 1.06855 w_k1_MSFT_s014
                                 + 1.15426 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.10996 w_k1_ADM_s014
                                 + 1.03772 w_k1_AXP_s014
                                 + 1.32782 w_k1_DIS_s014
                                 + 1.13335 w_k1_FDX_s014
                                 + 1.16732 w_k1_IBM_s014
                                 + 1.00957 w_k1_INTC_s014 + 1.3265 w_k1_KO_s014
                                 + 0.958831 w_k1_MAT_s014
                                 + 1.16228 w_k1_MSFT_s014
                                 + 1.14104 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.02235 w_k1_ADM_s014
                                 + 0.930638 w_k1_AXP_s014
                                 + 0.751862 w_k1_DIS_s014
                                 + 0.854813 w_k1_FDX_s014
                                 + 0.891394 w_k1_IBM_s014
                                 + 0.982403 w_k1_INTC_s014
                                 + 1.0039 w_k1_KO_s014 + 0.998244 w_k1_MAT_s014
                                 + 0.990935 w_k1_MSFT_s014
                                 + 0.933297 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.900639 w_k1_ADM_s014
                                 + 0.868504 w_k1_AXP_s014
                                 + 0.779792 w_k1_DIS_s014
                                 + 0.957164 w_k1_FDX_s014
                                 + 0.528392 w_k1_IBM_s014
                                 + 1.05596 w_k1_INTC_s014
                                 + 0.744819 w_k1_KO_s014
                                 + 0.536089 w_k1_MAT_s014
                                 + 0.945099 w_k1_MSFT_s014
                                 + 0.915185 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.26744 w_k1_ADM_s014
                                 + 1.65583 w_k1_AXP_s014
                                 + 1.62894 w_k1_DIS_s014
                                 + 1.30717 w_k1_FDX_s014
                                 + 1.56741 w_k1_IBM_s014
                                 + 1.42571 w_k1_INTC_s014
                                 + 1.55689 w_k1_KO_s014 + 1.51017 w_k1_MAT_s014
                                 + 1.28473 w_k1_MSFT_s014
                                 + 1.16233 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.53499 w_k1_ADM_s015
                                 + 1.23306 w_k1_AXP_s015 + 1.3852 w_k1_DIS_s015
                                 + 1.22234 w_k1_FDX_s015
                                 + 1.76383 w_k1_IBM_s015
                                 + 0.86507 w_k1_INTC_s015
                                 + 1.46028 w_k1_KO_s015 + 2.04001 w_k1_MAT_s015
                                 + 1.05305 w_k1_MSFT_s015
                                 + 1.22658 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.29711 w_k1_ADM_s015
                                 + 1.11987 w_k1_AXP_s015
                                 + 1.14476 w_k1_DIS_s015
                                 + 1.09875 w_k1_FDX_s015
                                 + 0.998662 w_k1_IBM_s015
                                 + 0.909481 w_k1_INTC_s015
                                 + 0.962948 w_k1_KO_s015
                                 + 1.02972 w_k1_MAT_s015
                                 + 1.0482 w_k1_MSFT_s015
                                 + 0.942865 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.40708 w_k1_ADM_s015
                                 + 1.57883 w_k1_AXP_s015
                                 + 1.54971 w_k1_DIS_s015 + 1.3652 w_k1_FDX_s015
                                 + 1.28157 w_k1_IBM_s015
                                 + 1.44315 w_k1_INTC_s015
                                 + 1.32602 w_k1_KO_s015 + 1.38608 w_k1_MAT_s015
                                 + 1.36436 w_k1_MSFT_s015
                                 + 1.22652 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.14486 w_k1_ADM_s015
                                 + 1.14227 w_k1_AXP_s015
                                 + 1.26068 w_k1_DIS_s015 + 1.324 w_k1_FDX_s015
                                 + 0.875247 w_k1_IBM_s015
                                 + 1.08543 w_k1_INTC_s015
                                 + 0.938337 w_k1_KO_s015
                                 + 1.03591 w_k1_MAT_s015
                                 + 1.03597 w_k1_MSFT_s015
                                 + 1.12818 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.21764 w_k1_ADM_s015
                                 + 0.844469 w_k1_AXP_s015
                                 + 0.876775 w_k1_DIS_s015
                                 + 1.01024 w_k1_FDX_s015
                                 + 0.664186 w_k1_IBM_s015
                                 + 1.0398 w_k1_INTC_s015
                                 + 0.726966 w_k1_KO_s015
                                 + 0.54529 w_k1_MAT_s015
                                 + 1.02582 w_k1_MSFT_s015
                                 + 0.950436 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.0449 w_k1_ADM_s015
                                 + 1.01134 w_k1_AXP_s015
                                 + 0.931977 w_k1_DIS_s015
                                 + 0.995382 w_k1_FDX_s015
                                 + 0.897213 w_k1_IBM_s015
                                 + 0.94646 w_k1_INTC_s015
                                 + 0.918855 w_k1_KO_s015
                                 + 1.04942 w_k1_MAT_s015
                                 + 1.00242 w_k1_MSFT_s015
                                 + 0.850744 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.964855 w_k1_ADM_s015
                                 + 0.90176 w_k1_AXP_s015
                                 + 1.08629 w_k1_DIS_s015
                                 + 0.927977 w_k1_FDX_s015
                                 + 1.29517 w_k1_IBM_s015
                                 + 0.793638 w_k1_INTC_s015
                                 + 1.07549 w_k1_KO_s015 + 1.27585 w_k1_MAT_s015
                                 + 0.969726 w_k1_MSFT_s015
                                 + 1.0972 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.03887 w_k1_ADM_s015
                                 + 0.638884 w_k1_AXP_s015
                                 + 0.723843 w_k1_DIS_s015
                                 + 0.822258 w_k1_FDX_s015
                                 + 0.737893 w_k1_IBM_s015
                                 + 0.766538 w_k1_INTC_s015
                                 + 0.725866 w_k1_KO_s015
                                 + 0.894454 w_k1_MAT_s015
                                 + 0.8575 w_k1_MSFT_s015
                                 + 0.827226 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.812842 w_k1_ADM_s016
                                 + 0.625659 w_k1_AXP_s016
                                 + 0.673424 w_k1_DIS_s016
                                 + 0.670832 w_k1_FDX_s016
                                 + 0.924719 w_k1_IBM_s016
                                 + 0.754742 w_k1_INTC_s016
                                 + 0.870607 w_k1_KO_s016
                                 + 0.747551 w_k1_MAT_s016
                                 + 0.745693 w_k1_MSFT_s016
                                 + 0.691987 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.26958 w_k1_ADM_s016
                                 + 1.41423 w_k1_AXP_s016
                                 + 1.76874 w_k1_DIS_s016 + 1.3835 w_k1_FDX_s016
                                 + 1.52408 w_k1_IBM_s016
                                 + 1.41947 w_k1_INTC_s016
                                 + 1.56131 w_k1_KO_s016 + 1.56065 w_k1_MAT_s016
                                 + 1.69918 w_k1_MSFT_s016 + 1.2376 w_k1_PG_s016
                                 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.06607 w_k1_ADM_s016
                                 + 1.3104 w_k1_AXP_s016 + 1.24554 w_k1_DIS_s016
                                 + 1.25572 w_k1_FDX_s016 + 1.0566 w_k1_IBM_s016
                                 + 1.15404 w_k1_INTC_s016
                                 + 1.19998 w_k1_KO_s016 + 1.27165 w_k1_MAT_s016
                                 + 1.14681 w_k1_MSFT_s016
                                 + 1.01041 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.05342 w_k1_ADM_s016
                                 + 0.971649 w_k1_AXP_s016
                                 + 1.50167 w_k1_DIS_s016
                                 + 1.35798 w_k1_FDX_s016
                                 + 0.886774 w_k1_IBM_s016
                                 + 1.17067 w_k1_INTC_s016
                                 + 1.11376 w_k1_KO_s016 + 1.15451 w_k1_MAT_s016
                                 + 1.49304 w_k1_MSFT_s016
                                 + 1.11278 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.927803 w_k1_ADM_s016
                                 + 0.877067 w_k1_AXP_s016
                                 + 1.01265 w_k1_DIS_s016
                                 + 0.98736 w_k1_FDX_s016
                                 + 0.879985 w_k1_IBM_s016
                                 + 1.12562 w_k1_INTC_s016
                                 + 0.961289 w_k1_KO_s016
                                 + 0.727638 w_k1_MAT_s016
                                 + 1.07698 w_k1_MSFT_s016
                                 + 0.906475 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.40673 w_k1_ADM_s016
                                 + 1.27939 w_k1_AXP_s016
                                 + 1.15397 w_k1_DIS_s016
                                 + 1.14478 w_k1_FDX_s016
                                 + 1.23988 w_k1_IBM_s016
                                 + 1.10974 w_k1_INTC_s016
                                 + 1.18505 w_k1_KO_s016 + 1.06648 w_k1_MAT_s016
                                 + 1.10345 w_k1_MSFT_s016
                                 + 1.03358 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.995732 w_k1_ADM_s016
                                 + 1.06152 w_k1_AXP_s016
                                 + 1.01519 w_k1_DIS_s016
                                 + 0.968293 w_k1_FDX_s016
                                 + 1.81037 w_k1_IBM_s016
                                 + 0.909086 w_k1_INTC_s016
                                 + 1.3494 w_k1_KO_s016 + 1.86417 w_k1_MAT_s016
                                 + 1.10249 w_k1_MSFT_s016
                                 + 1.07453 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.808351 w_k1_ADM_s016
                                 + 0.869029 w_k1_AXP_s016
                                 + 0.950706 w_k1_DIS_s016
                                 + 0.867647 w_k1_FDX_s016
                                 + 1.10835 w_k1_IBM_s016
                                 + 0.855162 w_k1_INTC_s016
                                 + 0.994431 w_k1_KO_s016
                                 + 1.07201 w_k1_MAT_s016
                                 + 0.967531 w_k1_MSFT_s016
                                 + 0.925954 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.07006 w_k1_ADM_s017
                                 + 0.958004 w_k1_AXP_s017
                                 + 0.882874 w_k1_DIS_s017
                                 + 1.08842 w_k1_FDX_s017
                                 + 0.684329 w_k1_IBM_s017
                                 + 1.07359 w_k1_INTC_s017
                                 + 0.821413 w_k1_KO_s017
                                 + 0.70212 w_k1_MAT_s017
                                 + 0.907984 w_k1_MSFT_s017
                                 + 1.02961 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.27709 w_k1_ADM_s017
                                 + 1.16955 w_k1_AXP_s017
                                 + 1.18548 w_k1_DIS_s017
                                 + 1.10683 w_k1_FDX_s017
                                 + 1.32171 w_k1_IBM_s017
                                 + 1.21205 w_k1_INTC_s017
                                 + 1.12102 w_k1_KO_s017 + 1.28158 w_k1_MAT_s017
                                 + 1.05734 w_k1_MSFT_s017 + 1.1585 w_k1_PG_s017
                                 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.858069 w_k1_ADM_s017
                                 + 0.976749 w_k1_AXP_s017
                                 + 0.948929 w_k1_DIS_s017
                                 + 0.974447 w_k1_FDX_s017
                                 + 1.10584 w_k1_IBM_s017
                                 + 1.24104 w_k1_INTC_s017
                                 + 0.951449 w_k1_KO_s017
                                 + 1.26345 w_k1_MAT_s017
                                 + 1.04394 w_k1_MSFT_s017 + 1.0565 w_k1_PG_s017
                                 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.995769 w_k1_ADM_s017
                                 + 1.33119 w_k1_AXP_s017
                                 + 1.44059 w_k1_DIS_s017
                                 + 1.28537 w_k1_FDX_s017
                                 + 1.35748 w_k1_IBM_s017
                                 + 1.57866 w_k1_INTC_s017
                                 + 1.21817 w_k1_KO_s017 + 1.22716 w_k1_MAT_s017
                                 + 1.18849 w_k1_MSFT_s017
                                 + 1.09779 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.969397 w_k1_ADM_s017
                                 + 0.837063 w_k1_AXP_s017
                                 + 0.696756 w_k1_DIS_s017
                                 + 0.812285 w_k1_FDX_s017
                                 + 0.948389 w_k1_IBM_s017
                                 + 0.766935 w_k1_INTC_s017
                                 + 0.865082 w_k1_KO_s017
                                 + 0.664059 w_k1_MAT_s017
                                 + 0.762782 w_k1_MSFT_s017
                                 + 0.950456 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.1799 w_k1_ADM_s017
                                 + 1.35509 w_k1_AXP_s017
                                 + 1.31875 w_k1_DIS_s017
                                 + 1.14871 w_k1_FDX_s017
                                 + 1.81176 w_k1_IBM_s017
                                 + 1.39002 w_k1_INTC_s017
                                 + 1.53099 w_k1_KO_s017 + 1.80687 w_k1_MAT_s017
                                 + 1.21161 w_k1_MSFT_s017 + 1.3993 w_k1_PG_s017
                                 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.983893 w_k1_ADM_s017
                                 + 0.859706 w_k1_AXP_s017
                                 + 1.35712 w_k1_DIS_s017
                                 + 1.18319 w_k1_FDX_s017
                                 + 0.96504 w_k1_IBM_s017
                                 + 1.26254 w_k1_INTC_s017
                                 + 1.13951 w_k1_KO_s017
                                 + 0.538198 w_k1_MAT_s017
                                 + 1.09723 w_k1_MSFT_s017
                                 + 1.18155 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.28 w_k1_ADM_s017
                                 + 1.52529 w_k1_AXP_s017
                                 + 1.57245 w_k1_DIS_s017 + 1.4525 w_k1_FDX_s017
                                 + 1.25228 w_k1_IBM_s017
                                 + 1.41231 w_k1_INTC_s017
                                 + 1.21925 w_k1_KO_s017 + 1.02559 w_k1_MAT_s017
                                 + 1.4887 w_k1_MSFT_s017 + 1.21681 w_k1_PG_s017
                                 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.562838 w_k1_ADM_s100
                                 + 0.473181 w_k1_AXP_s100
                                 + 0.436645 w_k1_DIS_s100
                                 + 0.583932 w_k1_FDX_s100
                                 + 0.996639 w_k1_IBM_s100
                                 + 0.637227 w_k1_INTC_s100
                                 + 0.755704 w_k1_KO_s100
                                 + 1.04494 w_k1_MAT_s100
                                 + 0.654863 w_k1_MSFT_s100
                                 + 0.828875 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.94694 w_k1_ADM_s100
                                 + 1.22699 w_k1_AXP_s100 + 1.2505 w_k1_DIS_s100
                                 + 1.19103 w_k1_FDX_s100
                                 + 1.56259 w_k1_IBM_s100
                                 + 1.12156 w_k1_INTC_s100
                                 + 1.38305 w_k1_KO_s100 + 1.94758 w_k1_MAT_s100
                                 + 1.10111 w_k1_MSFT_s100 + 1.2536 w_k1_PG_s100
                                 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.874111 w_k1_ADM_s100
                                 + 0.733821 w_k1_AXP_s100
                                 + 0.693481 w_k1_DIS_s100
                                 + 0.754736 w_k1_FDX_s100
                                 + 1.12399 w_k1_IBM_s100
                                 + 0.751633 w_k1_INTC_s100
                                 + 0.903591 w_k1_KO_s100
                                 + 1.01524 w_k1_MAT_s100
                                 + 0.773517 w_k1_MSFT_s100
                                 + 0.925318 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.975726 w_k1_ADM_s100
                                 + 0.75231 w_k1_AXP_s100
                                 + 0.798094 w_k1_DIS_s100
                                 + 0.968158 w_k1_FDX_s100
                                 + 0.674854 w_k1_IBM_s100
                                 + 0.910547 w_k1_INTC_s100
                                 + 0.8325 w_k1_KO_s100 + 0.640201 w_k1_MAT_s100
                                 + 0.938962 w_k1_MSFT_s100
                                 + 0.920516 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.05792 w_k1_ADM_s100
                                 + 1.13192 w_k1_AXP_s100
                                 + 1.34572 w_k1_DIS_s100
                                 + 1.24694 w_k1_FDX_s100
                                 + 1.01819 w_k1_IBM_s100
                                 + 1.19477 w_k1_INTC_s100
                                 + 1.22002 w_k1_KO_s100 + 1.12148 w_k1_MAT_s100
                                 + 1.25749 w_k1_MSFT_s100
                                 + 1.11382 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.777719 w_k1_ADM_s100
                                 + 0.971041 w_k1_AXP_s100
                                 + 1.03663 w_k1_DIS_s100
                                 + 0.880326 w_k1_FDX_s100
                                 + 1.50772 w_k1_IBM_s100
                                 + 1.07377 w_k1_INTC_s100
                                 + 1.29202 w_k1_KO_s100 + 1.69928 w_k1_MAT_s100
                                 + 0.965799 w_k1_MSFT_s100
                                 + 1.04244 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.908041 w_k1_ADM_s100
                                 + 1.06787 w_k1_AXP_s100
                                 + 0.886436 w_k1_DIS_s100
                                 + 0.939127 w_k1_FDX_s100
                                 + 1.00217 w_k1_IBM_s100
                                 + 0.873306 w_k1_INTC_s100
                                 + 0.85852 w_k1_KO_s100
                                 + 0.999344 w_k1_MAT_s100
                                 + 0.78703 w_k1_MSFT_s100
                                 + 0.95805 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.16685 w_k1_ADM_s100
                                 + 1.39794 w_k1_AXP_s100
                                 + 1.24253 w_k1_DIS_s100
                                 + 1.36547 w_k1_FDX_s100
                                 + 0.966162 w_k1_IBM_s100
                                 + 1.08079 w_k1_INTC_s100
                                 + 1.11363 w_k1_KO_s100 + 1.6643 w_k1_MAT_s100
                                 + 1.10207 w_k1_MSFT_s100
                                 + 1.22548 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.913111 w_k1_ADM_s101
                                 + 0.807106 w_k1_AXP_s101
                                 + 0.701916 w_k1_DIS_s101
                                 + 0.926343 w_k1_FDX_s101
                                 + 0.579135 w_k1_IBM_s101
                                 + 0.839006 w_k1_INTC_s101
                                 + 0.544314 w_k1_KO_s101
                                 + 0.561838 w_k1_MAT_s101
                                 + 0.766519 w_k1_MSFT_s101
                                 + 0.795457 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.15758 w_k1_ADM_s101
                                 + 1.29208 w_k1_AXP_s101
                                 + 1.26152 w_k1_DIS_s101
                                 + 1.15561 w_k1_FDX_s101
                                 + 1.62284 w_k1_IBM_s101
                                 + 1.11299 w_k1_INTC_s101
                                 + 1.22379 w_k1_KO_s101 + 1.49768 w_k1_MAT_s101
                                 + 1.095 w_k1_MSFT_s101 + 1.05329 w_k1_PG_s101
                                 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.722431 w_k1_ADM_s101
                                 + 0.681373 w_k1_AXP_s101
                                 + 0.764077 w_k1_DIS_s101
                                 + 0.882802 w_k1_FDX_s101
                                 + 0.74405 w_k1_IBM_s101
                                 + 0.915009 w_k1_INTC_s101
                                 + 0.792042 w_k1_KO_s101
                                 + 0.707197 w_k1_MAT_s101
                                 + 0.933565 w_k1_MSFT_s101
                                 + 0.898098 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.15435 w_k1_ADM_s101
                                 + 1.05755 w_k1_AXP_s101
                                 + 1.00387 w_k1_DIS_s101
                                 + 1.07774 w_k1_FDX_s101
                                 + 0.961365 w_k1_IBM_s101
                                 + 0.848977 w_k1_INTC_s101
                                 + 0.906212 w_k1_KO_s101
                                 + 0.955845 w_k1_MAT_s101
                                 + 0.991405 w_k1_MSFT_s101
                                 + 1.04779 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.88003 w_k1_ADM_s101
                                 + 1.03933 w_k1_AXP_s101
                                 + 0.663292 w_k1_DIS_s101
                                 + 0.917075 w_k1_FDX_s101
                                 + 0.924068 w_k1_IBM_s101
                                 + 0.866882 w_k1_INTC_s101
                                 + 0.788633 w_k1_KO_s101
                                 + 1.14566 w_k1_MAT_s101
                                 + 0.783366 w_k1_MSFT_s101
                                 + 0.900294 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.15782 w_k1_ADM_s101
                                 + 1.3506 w_k1_AXP_s101 + 1.40365 w_k1_DIS_s101
                                 + 1.329 w_k1_FDX_s101 + 1.07325 w_k1_IBM_s101
                                 + 0.840478 w_k1_INTC_s101
                                 + 1.06214 w_k1_KO_s101 + 1.4963 w_k1_MAT_s101
                                 + 1.08369 w_k1_MSFT_s101
                                 + 1.12469 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.738286 w_k1_ADM_s101
                                 + 0.836211 w_k1_AXP_s101
                                 + 0.634567 w_k1_DIS_s101
                                 + 0.773125 w_k1_FDX_s101
                                 + 0.796662 w_k1_IBM_s101
                                 + 0.786134 w_k1_INTC_s101
                                 + 0.681133 w_k1_KO_s101
                                 + 0.773608 w_k1_MAT_s101
                                 + 0.664374 w_k1_MSFT_s101
                                 + 0.813076 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.879772 w_k1_ADM_s101
                                 + 0.85162 w_k1_AXP_s101
                                 + 1.07181 w_k1_DIS_s101
                                 + 0.916065 w_k1_FDX_s101
                                 + 0.949815 w_k1_IBM_s101
                                 + 0.913708 w_k1_INTC_s101
                                 + 1.05685 w_k1_KO_s101
                                 + 0.803794 w_k1_MAT_s101
                                 + 0.944413 w_k1_MSFT_s101
                                 + 1.01493 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.87011 w_k1_ADM_s102
                                 + 0.924508 w_k1_AXP_s102
                                 + 0.743722 w_k1_DIS_s102
                                 + 0.803826 w_k1_FDX_s102
                                 + 1.05909 w_k1_IBM_s102
                                 + 0.979932 w_k1_INTC_s102
                                 + 0.92778 w_k1_KO_s102 + 1.45615 w_k1_MAT_s102
                                 + 0.87508 w_k1_MSFT_s102
                                 + 0.916811 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.08143 w_k1_ADM_s102
                                 + 0.922402 w_k1_AXP_s102
                                 + 0.975044 w_k1_DIS_s102
                                 + 1.0199 w_k1_FDX_s102 + 1.07416 w_k1_IBM_s102
                                 + 1.03969 w_k1_INTC_s102
                                 + 0.961319 w_k1_KO_s102
                                 + 1.19676 w_k1_MAT_s102
                                 + 1.00886 w_k1_MSFT_s102
                                 + 1.06084 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.24732 w_k1_ADM_s102
                                 + 1.26759 w_k1_AXP_s102
                                 + 1.14685 w_k1_DIS_s102
                                 + 1.22538 w_k1_FDX_s102
                                 + 0.845232 w_k1_IBM_s102
                                 + 1.23556 w_k1_INTC_s102
                                 + 1.08623 w_k1_KO_s102 + 1.07953 w_k1_MAT_s102
                                 + 1.08175 w_k1_MSFT_s102 + 1.143 w_k1_PG_s102
                                 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.867867 w_k1_ADM_s102
                                 + 0.745774 w_k1_AXP_s102
                                 + 1.07305 w_k1_DIS_s102
                                 + 0.817101 w_k1_FDX_s102
                                 + 1.23825 w_k1_IBM_s102
                                 + 0.772922 w_k1_INTC_s102
                                 + 1.10997 w_k1_KO_s102 + 1.48581 w_k1_MAT_s102
                                 + 0.868447 w_k1_MSFT_s102
                                 + 1.01275 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.72589 w_k1_ADM_s102
                                 + 0.58036 w_k1_AXP_s102
                                 + 0.547972 w_k1_DIS_s102
                                 + 0.711266 w_k1_FDX_s102
                                 + 0.667661 w_k1_IBM_s102
                                 + 0.788772 w_k1_INTC_s102
                                 + 0.62328 w_k1_KO_s102
                                 + 0.781283 w_k1_MAT_s102
                                 + 0.738388 w_k1_MSFT_s102
                                 + 0.760407 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.929692 w_k1_ADM_s102
                                 + 1.30238 w_k1_AXP_s102
                                 + 1.46169 w_k1_DIS_s102
                                 + 1.22586 w_k1_FDX_s102
                                 + 1.29356 w_k1_IBM_s102
                                 + 1.43106 w_k1_INTC_s102
                                 + 1.33046 w_k1_KO_s102 + 1.55522 w_k1_MAT_s102
                                 + 1.3144 w_k1_MSFT_s102 + 1.19279 w_k1_PG_s102
                                 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.920965 w_k1_ADM_s102
                                 + 1.09749 w_k1_AXP_s102
                                 + 0.86663 w_k1_DIS_s102
                                 + 1.05303 w_k1_FDX_s102
                                 + 0.812807 w_k1_IBM_s102
                                 + 1.02381 w_k1_INTC_s102
                                 + 0.917243 w_k1_KO_s102
                                 + 1.19539 w_k1_MAT_s102
                                 + 0.98124 w_k1_MSFT_s102
                                 + 0.984691 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.940333 w_k1_ADM_s102
                                 + 0.787767 w_k1_AXP_s102
                                 + 0.726482 w_k1_DIS_s102
                                 + 0.895113 w_k1_FDX_s102
                                 + 0.670916 w_k1_IBM_s102
                                 + 0.878648 w_k1_INTC_s102
                                 + 0.718534 w_k1_KO_s102
                                 + 0.647458 w_k1_MAT_s102
                                 + 0.853478 w_k1_MSFT_s102
                                 + 0.902081 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.734348 w_k1_ADM_s103
                                 + 0.637705 w_k1_AXP_s103
                                 + 0.577383 w_k1_DIS_s103
                                 + 0.699365 w_k1_FDX_s103
                                 + 0.78134 w_k1_IBM_s103
                                 + 0.770464 w_k1_INTC_s103
                                 + 0.727724 w_k1_KO_s103
                                 + 0.68272 w_k1_MAT_s103
                                 + 0.735682 w_k1_MSFT_s103
                                 + 0.763972 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.70387 w_k1_ADM_s103
                                 + 0.877738 w_k1_AXP_s103
                                 + 0.806031 w_k1_DIS_s103
                                 + 0.820154 w_k1_FDX_s103
                                 + 1.1159 w_k1_IBM_s103
                                 + 0.972005 w_k1_INTC_s103
                                 + 1.05671 w_k1_KO_s103 + 1.38258 w_k1_MAT_s103
                                 + 0.88138 w_k1_MSFT_s103
                                 + 0.952654 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.2477 w_k1_ADM_s103
                                 + 1.27828 w_k1_AXP_s103
                                 + 1.15057 w_k1_DIS_s103
                                 + 1.02116 w_k1_FDX_s103
                                 + 1.56375 w_k1_IBM_s103
                                 + 1.06637 w_k1_INTC_s103
                                 + 1.37187 w_k1_KO_s103 + 1.66455 w_k1_MAT_s103
                                 + 1.14871 w_k1_MSFT_s103
                                 + 1.17759 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.08041 w_k1_ADM_s103
                                 + 1.42801 w_k1_AXP_s103
                                 + 1.09593 w_k1_DIS_s103
                                 + 1.18916 w_k1_FDX_s103
                                 + 0.900533 w_k1_IBM_s103
                                 + 1.17109 w_k1_INTC_s103 + 1.0401 w_k1_KO_s103
                                 + 1.09033 w_k1_MAT_s103
                                 + 1.07955 w_k1_MSFT_s103
                                 + 0.989237 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.894506 w_k1_ADM_s103
                                 + 0.970955 w_k1_AXP_s103
                                 + 0.954579 w_k1_DIS_s103
                                 + 0.94722 w_k1_FDX_s103
                                 + 0.950547 w_k1_IBM_s103
                                 + 0.901836 w_k1_INTC_s103
                                 + 0.962977 w_k1_KO_s103
                                 + 0.935978 w_k1_MAT_s103
                                 + 0.797013 w_k1_MSFT_s103
                                 + 0.903583 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.897872 w_k1_ADM_s103
                                 + 0.762118 w_k1_AXP_s103
                                 + 0.793245 w_k1_DIS_s103
                                 + 0.792414 w_k1_FDX_s103
                                 + 0.582704 w_k1_IBM_s103
                                 + 0.797005 w_k1_INTC_s103
                                 + 0.846062 w_k1_KO_s103
                                 + 0.697933 w_k1_MAT_s103
                                 + 0.922994 w_k1_MSFT_s103
                                 + 0.868797 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.898768 w_k1_ADM_s103
                                 + 0.84671 w_k1_AXP_s103
                                 + 0.723135 w_k1_DIS_s103
                                 + 0.912426 w_k1_FDX_s103
                                 + 0.658819 w_k1_IBM_s103
                                 + 0.787834 w_k1_INTC_s103
                                 + 0.724252 w_k1_KO_s103
                                 + 0.743729 w_k1_MAT_s103
                                 + 0.87916 w_k1_MSFT_s103
                                 + 0.815893 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.905901 w_k1_ADM_s103
                                 + 1.57232 w_k1_AXP_s103
                                 + 1.80726 w_k1_DIS_s103
                                 + 1.48223 w_k1_FDX_s103
                                 + 1.24873 w_k1_IBM_s103
                                 + 1.27763 w_k1_INTC_s103
                                 + 1.46952 w_k1_KO_s103 + 1.88987 w_k1_MAT_s103
                                 + 1.49972 w_k1_MSFT_s103 + 1.1388 w_k1_PG_s103
                                 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.38932 w_k1_ADM_s104
                                 + 1.21187 w_k1_AXP_s104 + 1.0681 w_k1_DIS_s104
                                 + 1.09486 w_k1_FDX_s104
                                 + 1.05216 w_k1_IBM_s104
                                 + 0.923908 w_k1_INTC_s104
                                 + 1.14022 w_k1_KO_s104 + 1.44895 w_k1_MAT_s104
                                 + 1.07822 w_k1_MSFT_s104
                                 + 1.15592 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.24737 w_k1_ADM_s104
                                 + 1.18367 w_k1_AXP_s104
                                 + 1.10305 w_k1_DIS_s104
                                 + 0.971992 w_k1_FDX_s104
                                 + 1.76826 w_k1_IBM_s104
                                 + 0.967793 w_k1_INTC_s104
                                 + 1.48154 w_k1_KO_s104 + 1.70598 w_k1_MAT_s104
                                 + 1.03339 w_k1_MSFT_s104
                                 + 1.20625 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.10901 w_k1_ADM_s104
                                 + 1.31015 w_k1_AXP_s104
                                 + 1.54783 w_k1_DIS_s104
                                 + 1.26413 w_k1_FDX_s104
                                 + 1.10081 w_k1_IBM_s104
                                 + 1.19463 w_k1_INTC_s104
                                 + 1.20332 w_k1_KO_s104 + 1.09655 w_k1_MAT_s104
                                 + 1.27119 w_k1_MSFT_s104
                                 + 1.25675 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.3927 w_k1_ADM_s104
                                 + 1.43516 w_k1_AXP_s104
                                 + 1.09393 w_k1_DIS_s104
                                 + 1.13658 w_k1_FDX_s104
                                 + 1.41656 w_k1_IBM_s104
                                 + 1.07165 w_k1_INTC_s104
                                 + 1.29754 w_k1_KO_s104 + 1.56492 w_k1_MAT_s104
                                 + 1.02886 w_k1_MSFT_s104
                                 + 1.29446 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.840595 w_k1_ADM_s104
                                 + 0.947255 w_k1_AXP_s104
                                 + 0.900512 w_k1_DIS_s104
                                 + 0.851941 w_k1_FDX_s104
                                 + 1.23969 w_k1_IBM_s104
                                 + 0.84624 w_k1_INTC_s104 + 1.0416 w_k1_KO_s104
                                 + 0.998116 w_k1_MAT_s104
                                 + 0.919951 w_k1_MSFT_s104
                                 + 1.08591 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.880209 w_k1_ADM_s104
                                 + 0.744411 w_k1_AXP_s104
                                 + 0.579576 w_k1_DIS_s104
                                 + 0.71501 w_k1_FDX_s104
                                 + 0.945883 w_k1_IBM_s104
                                 + 0.761276 w_k1_INTC_s104
                                 + 0.966003 w_k1_KO_s104
                                 + 0.868756 w_k1_MAT_s104
                                 + 0.759009 w_k1_MSFT_s104
                                 + 0.912577 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.958047 w_k1_ADM_s104
                                 + 0.981105 w_k1_AXP_s104
                                 + 0.87714 w_k1_DIS_s104
                                 + 0.942649 w_k1_FDX_s104
                                 + 0.903227 w_k1_IBM_s104
                                 + 1.03221 w_k1_INTC_s104
                                 + 1.01735 w_k1_KO_s104
                                 + 0.804006 w_k1_MAT_s104
                                 + 1.00015 w_k1_MSFT_s104 + 1.0416 w_k1_PG_s104
                                 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.05795 w_k1_ADM_s104
                                 + 0.924424 w_k1_AXP_s104
                                 + 0.914601 w_k1_DIS_s104
                                 + 1.03066 w_k1_FDX_s104
                                 + 0.701111 w_k1_IBM_s104
                                 + 0.935463 w_k1_INTC_s104
                                 + 0.963607 w_k1_KO_s104
                                 + 0.725631 w_k1_MAT_s104
                                 + 1.07322 w_k1_MSFT_s104
                                 + 0.950107 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.639598 w_k1_ADM_s105
                                 + 0.621963 w_k1_AXP_s105
                                 + 0.562396 w_k1_DIS_s105
                                 + 0.71757 w_k1_FDX_s105
                                 + 0.760974 w_k1_IBM_s105
                                 + 0.670708 w_k1_INTC_s105
                                 + 0.661938 w_k1_KO_s105
                                 + 0.811545 w_k1_MAT_s105
                                 + 0.696537 w_k1_MSFT_s105
                                 + 0.73994 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.13652 w_k1_ADM_s105
                                 + 1.48043 w_k1_AXP_s105
                                 + 1.60105 w_k1_DIS_s105
                                 + 1.48777 w_k1_FDX_s105
                                 + 1.39871 w_k1_IBM_s105
                                 + 1.46124 w_k1_INTC_s105
                                 + 1.27245 w_k1_KO_s105 + 2.50649 w_k1_MAT_s105
                                 + 1.38192 w_k1_MSFT_s105
                                 + 1.15236 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.704995 w_k1_ADM_s105
                                 + 1.01853 w_k1_AXP_s105
                                 + 0.746499 w_k1_DIS_s105
                                 + 0.831296 w_k1_FDX_s105
                                 + 1.32334 w_k1_IBM_s105
                                 + 0.946111 w_k1_INTC_s105
                                 + 0.979212 w_k1_KO_s105
                                 + 1.31884 w_k1_MAT_s105
                                 + 0.866398 w_k1_MSFT_s105
                                 + 0.888666 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.709532 w_k1_ADM_s105
                                 + 0.591403 w_k1_AXP_s105
                                 + 0.870583 w_k1_DIS_s105
                                 + 0.815937 w_k1_FDX_s105
                                 + 0.91149 w_k1_IBM_s105
                                 + 0.954662 w_k1_INTC_s105
                                 + 0.906908 w_k1_KO_s105
                                 + 0.848665 w_k1_MAT_s105
                                 + 0.880908 w_k1_MSFT_s105
                                 + 0.862419 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.843545 w_k1_ADM_s105
                                 + 1.03303 w_k1_AXP_s105
                                 + 0.980408 w_k1_DIS_s105
                                 + 1.02649 w_k1_FDX_s105
                                 + 0.901645 w_k1_IBM_s105
                                 + 1.19139 w_k1_INTC_s105
                                 + 0.99846 w_k1_KO_s105 + 1.0317 w_k1_MAT_s105
                                 + 1.04392 w_k1_MSFT_s105
                                 + 0.922767 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.900443 w_k1_ADM_s105
                                 + 1.04357 w_k1_AXP_s105
                                 + 1.08612 w_k1_DIS_s105 + 1.0351 w_k1_FDX_s105
                                 + 1.35756 w_k1_IBM_s105
                                 + 1.01284 w_k1_INTC_s105
                                 + 1.10344 w_k1_KO_s105 + 1.39601 w_k1_MAT_s105
                                 + 0.987732 w_k1_MSFT_s105
                                 + 0.997645 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.12014 w_k1_ADM_s105
                                 + 1.34896 w_k1_AXP_s105 + 1.6568 w_k1_DIS_s105
                                 + 1.4267 w_k1_FDX_s105 + 1.18071 w_k1_IBM_s105
                                 + 1.43875 w_k1_INTC_s105
                                 + 1.21767 w_k1_KO_s105 + 1.32063 w_k1_MAT_s105
                                 + 1.33611 w_k1_MSFT_s105
                                 + 1.23324 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.06517 w_k1_ADM_s105
                                 + 0.940343 w_k1_AXP_s105
                                 + 1.31304 w_k1_DIS_s105
                                 + 1.17902 w_k1_FDX_s105
                                 + 0.872011 w_k1_IBM_s105
                                 + 1.17209 w_k1_INTC_s105
                                 + 1.11656 w_k1_KO_s105 + 1.11568 w_k1_MAT_s105
                                 + 1.41848 w_k1_MSFT_s105
                                 + 0.997608 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.42702 w_k1_ADM_s106
                                 + 1.66553 w_k1_AXP_s106
                                 + 1.49155 w_k1_DIS_s106
                                 + 1.40157 w_k1_FDX_s106
                                 + 1.13802 w_k1_IBM_s106
                                 + 1.08776 w_k1_INTC_s106
                                 + 1.16408 w_k1_KO_s106 + 1.49265 w_k1_MAT_s106
                                 + 1.38863 w_k1_MSFT_s106
                                 + 1.37852 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.778249 w_k1_ADM_s106
                                 + 0.814409 w_k1_AXP_s106
                                 + 0.838787 w_k1_DIS_s106
                                 + 0.900385 w_k1_FDX_s106
                                 + 0.90513 w_k1_IBM_s106
                                 + 0.821919 w_k1_INTC_s106
                                 + 0.80303 w_k1_KO_s106
                                 + 0.816847 w_k1_MAT_s106
                                 + 1.02844 w_k1_MSFT_s106
                                 + 0.938095 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.08415 w_k1_ADM_s106
                                 + 0.677901 w_k1_AXP_s106
                                 + 0.698239 w_k1_DIS_s106
                                 + 0.768773 w_k1_FDX_s106
                                 + 0.737858 w_k1_IBM_s106
                                 + 0.74412 w_k1_INTC_s106
                                 + 0.789761 w_k1_KO_s106
                                 + 0.934961 w_k1_MAT_s106
                                 + 0.958822 w_k1_MSFT_s106
                                 + 0.929634 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.03219 w_k1_ADM_s106
                                 + 0.983037 w_k1_AXP_s106
                                 + 0.912457 w_k1_DIS_s106
                                 + 0.922288 w_k1_FDX_s106
                                 + 1.06693 w_k1_IBM_s106
                                 + 0.846447 w_k1_INTC_s106
                                 + 0.943539 w_k1_KO_s106 + 1.0266 w_k1_MAT_s106
                                 + 0.965377 w_k1_MSFT_s106
                                 + 0.96623 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.06648 w_k1_ADM_s106
                                 + 1.25224 w_k1_AXP_s106
                                 + 0.979235 w_k1_DIS_s106
                                 + 1.02379 w_k1_FDX_s106
                                 + 0.882012 w_k1_IBM_s106
                                 + 1.12259 w_k1_INTC_s106
                                 + 0.825527 w_k1_KO_s106
                                 + 1.03142 w_k1_MAT_s106
                                 + 1.11202 w_k1_MSFT_s106
                                 + 1.13357 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.20638 w_k1_ADM_s106
                                 + 1.03447 w_k1_AXP_s106
                                 + 1.11069 w_k1_DIS_s106
                                 + 0.998781 w_k1_FDX_s106
                                 + 1.31431 w_k1_IBM_s106
                                 + 0.968736 w_k1_INTC_s106
                                 + 1.02409 w_k1_KO_s106 + 1.59394 w_k1_MAT_s106
                                 + 1.11571 w_k1_MSFT_s106 + 1.2022 w_k1_PG_s106
                                 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.24073 w_k1_ADM_s106
                                 + 1.23566 w_k1_AXP_s106
                                 + 1.42485 w_k1_DIS_s106
                                 + 1.23224 w_k1_FDX_s106
                                 + 1.06024 w_k1_IBM_s106
                                 + 1.08599 w_k1_INTC_s106
                                 + 0.990505 w_k1_KO_s106
                                 + 0.970978 w_k1_MAT_s106
                                 + 1.26826 w_k1_MSFT_s106 + 1.1629 w_k1_PG_s106
                                 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.08107 w_k1_ADM_s106
                                 + 1.20097 w_k1_AXP_s106 + 1.4585 w_k1_DIS_s106
                                 + 1.09873 w_k1_FDX_s106 + 1.7537 w_k1_IBM_s106
                                 + 1.18217 w_k1_INTC_s106
                                 + 1.30522 w_k1_KO_s106 + 1.56572 w_k1_MAT_s106
                                 + 1.31053 w_k1_MSFT_s106
                                 + 1.17728 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.952063 w_k1_ADM_s107
                                 + 0.807657 w_k1_AXP_s107
                                 + 0.825335 w_k1_DIS_s107
                                 + 0.786686 w_k1_FDX_s107
                                 + 1.07565 w_k1_IBM_s107
                                 + 0.93694 w_k1_INTC_s107
                                 + 1.10333 w_k1_KO_s107
                                 + 0.919931 w_k1_MAT_s107
                                 + 0.953466 w_k1_MSFT_s107
                                 + 1.00289 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.911812 w_k1_ADM_s107
                                 + 1.57142 w_k1_AXP_s107
                                 + 1.13339 w_k1_DIS_s107 + 1.183 w_k1_FDX_s107
                                 + 0.861239 w_k1_IBM_s107
                                 + 1.27172 w_k1_INTC_s107 + 1.183 w_k1_KO_s107
                                 + 0.991746 w_k1_MAT_s107
                                 + 1.14927 w_k1_MSFT_s107
                                 + 1.07938 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.02209 w_k1_ADM_s107
                                 + 1.26568 w_k1_AXP_s107
                                 + 1.17601 w_k1_DIS_s107
                                 + 1.06218 w_k1_FDX_s107
                                 + 1.29958 w_k1_IBM_s107
                                 + 1.27011 w_k1_INTC_s107
                                 + 1.38412 w_k1_KO_s107 + 1.16051 w_k1_MAT_s107
                                 + 1.12911 w_k1_MSFT_s107 + 1.1121 w_k1_PG_s107
                                 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.789348 w_k1_ADM_s107
                                 + 0.795595 w_k1_AXP_s107
                                 + 0.724373 w_k1_DIS_s107
                                 + 0.820103 w_k1_FDX_s107
                                 + 0.6768 w_k1_IBM_s107
                                 + 1.04544 w_k1_INTC_s107
                                 + 0.808108 w_k1_KO_s107
                                 + 0.575722 w_k1_MAT_s107
                                 + 0.832184 w_k1_MSFT_s107
                                 + 0.880363 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.718788 w_k1_ADM_s107
                                 + 1.0756 w_k1_AXP_s107 + 1.17246 w_k1_DIS_s107
                                 + 1.0629 w_k1_FDX_s107
                                 + 0.904512 w_k1_IBM_s107
                                 + 1.40557 w_k1_INTC_s107
                                 + 1.19179 w_k1_KO_s107
                                 + 0.810507 w_k1_MAT_s107
                                 + 1.10528 w_k1_MSFT_s107
                                 + 0.974492 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.960675 w_k1_ADM_s107
                                 + 1.12002 w_k1_AXP_s107
                                 + 1.00724 w_k1_DIS_s107
                                 + 1.09641 w_k1_FDX_s107
                                 + 0.678338 w_k1_IBM_s107
                                 + 1.29893 w_k1_INTC_s107
                                 + 0.96742 w_k1_KO_s107
                                 + 0.640426 w_k1_MAT_s107
                                 + 1.10086 w_k1_MSFT_s107
                                 + 0.898341 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.730968 w_k1_ADM_s107
                                 + 0.799612 w_k1_AXP_s107
                                 + 0.783484 w_k1_DIS_s107
                                 + 0.780738 w_k1_FDX_s107
                                 + 0.863136 w_k1_IBM_s107
                                 + 1.19958 w_k1_INTC_s107
                                 + 0.901399 w_k1_KO_s107
                                 + 0.671096 w_k1_MAT_s107
                                 + 0.931367 w_k1_MSFT_s107
                                 + 0.893587 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.11728 w_k1_ADM_s107
                                 + 1.40224 w_k1_AXP_s107
                                 + 1.22023 w_k1_DIS_s107
                                 + 1.16152 w_k1_FDX_s107
                                 + 1.84597 w_k1_IBM_s107
                                 + 1.39294 w_k1_INTC_s107
                                 + 1.73418 w_k1_KO_s107 + 1.92485 w_k1_MAT_s107
                                 + 1.11994 w_k1_MSFT_s107
                                 + 1.13873 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.9442 w_k1_ADM_s110
                                 + 0.733414 w_k1_AXP_s110
                                 + 0.756002 w_k1_DIS_s110
                                 + 0.711741 w_k1_FDX_s110
                                 + 1.14175 w_k1_IBM_s110
                                 + 0.869817 w_k1_INTC_s110
                                 + 1.14522 w_k1_KO_s110
                                 + 0.822965 w_k1_MAT_s110
                                 + 0.80515 w_k1_MSFT_s110
                                 + 0.971922 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.42674 w_k1_ADM_s110
                                 + 1.78453 w_k1_AXP_s110
                                 + 1.63457 w_k1_DIS_s110
                                 + 1.52903 w_k1_FDX_s110
                                 + 1.00789 w_k1_IBM_s110
                                 + 1.23618 w_k1_INTC_s110
                                 + 1.56476 w_k1_KO_s110 + 1.3193 w_k1_MAT_s110
                                 + 1.34553 w_k1_MSFT_s110
                                 + 1.48727 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.33622 w_k1_ADM_s110
                                 + 1.36016 w_k1_AXP_s110
                                 + 1.46389 w_k1_DIS_s110
                                 + 1.12784 w_k1_FDX_s110
                                 + 1.29851 w_k1_IBM_s110
                                 + 1.07569 w_k1_INTC_s110
                                 + 1.40956 w_k1_KO_s110 + 1.42769 w_k1_MAT_s110
                                 + 1.20214 w_k1_MSFT_s110
                                 + 1.20541 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.957616 w_k1_ADM_s110
                                 + 0.631058 w_k1_AXP_s110
                                 + 0.793313 w_k1_DIS_s110
                                 + 0.79133 w_k1_FDX_s110
                                 + 0.751207 w_k1_IBM_s110
                                 + 0.820825 w_k1_INTC_s110
                                 + 0.971866 w_k1_KO_s110
                                 + 0.734017 w_k1_MAT_s110
                                 + 0.900983 w_k1_MSFT_s110
                                 + 0.984873 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.14997 w_k1_ADM_s110
                                 + 0.975154 w_k1_AXP_s110
                                 + 1.18899 w_k1_DIS_s110
                                 + 1.05042 w_k1_FDX_s110
                                 + 0.950485 w_k1_IBM_s110
                                 + 1.08877 w_k1_INTC_s110
                                 + 1.13899 w_k1_KO_s110
                                 + 0.753663 w_k1_MAT_s110
                                 + 1.01539 w_k1_MSFT_s110
                                 + 1.05504 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.19781 w_k1_ADM_s110
                                 + 0.847557 w_k1_AXP_s110
                                 + 1.36987 w_k1_DIS_s110
                                 + 1.23494 w_k1_FDX_s110
                                 + 0.695902 w_k1_IBM_s110
                                 + 1.0769 w_k1_INTC_s110 + 1.27144 w_k1_KO_s110
                                 + 0.512476 w_k1_MAT_s110
                                 + 1.21599 w_k1_MSFT_s110
                                 + 1.15722 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.28542 w_k1_ADM_s110
                                 + 0.959157 w_k1_AXP_s110
                                 + 0.833525 w_k1_DIS_s110
                                 + 0.929079 w_k1_FDX_s110
                                 + 1.11332 w_k1_IBM_s110
                                 + 0.955384 w_k1_INTC_s110
                                 + 1.24166 w_k1_KO_s110 + 1.01534 w_k1_MAT_s110
                                 + 0.884484 w_k1_MSFT_s110
                                 + 1.04732 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.05792 w_k1_ADM_s110
                                 + 0.832752 w_k1_AXP_s110
                                 + 0.704742 w_k1_DIS_s110
                                 + 0.649367 w_k1_FDX_s110
                                 + 1.5555 w_k1_IBM_s110
                                 + 0.765242 w_k1_INTC_s110
                                 + 1.15892 w_k1_KO_s110
                                 + 0.848334 w_k1_MAT_s110
                                 + 0.675437 w_k1_MSFT_s110
                                 + 0.905885 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.02087 w_k1_ADM_s111
                                 + 0.67296 w_k1_AXP_s111
                                 + 0.714631 w_k1_DIS_s111
                                 + 0.777232 w_k1_FDX_s111
                                 + 0.782458 w_k1_IBM_s111
                                 + 0.865366 w_k1_INTC_s111
                                 + 0.739042 w_k1_KO_s111
                                 + 0.593922 w_k1_MAT_s111
                                 + 0.839825 w_k1_MSFT_s111
                                 + 0.851534 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.933304 w_k1_ADM_s111
                                 + 0.699235 w_k1_AXP_s111
                                 + 1.00424 w_k1_DIS_s111
                                 + 0.901153 w_k1_FDX_s111
                                 + 0.867371 w_k1_IBM_s111
                                 + 0.867232 w_k1_INTC_s111
                                 + 0.909713 w_k1_KO_s111
                                 + 0.704729 w_k1_MAT_s111
                                 + 1.05575 w_k1_MSFT_s111
                                 + 0.868741 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.950039 w_k1_ADM_s111
                                 + 1.25425 w_k1_AXP_s111
                                 + 0.883133 w_k1_DIS_s111
                                 + 1.04693 w_k1_FDX_s111
                                 + 0.915806 w_k1_IBM_s111
                                 + 0.972602 w_k1_INTC_s111
                                 + 0.890864 w_k1_KO_s111
                                 + 1.33273 w_k1_MAT_s111
                                 + 1.05741 w_k1_MSFT_s111
                                 + 0.853636 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.08393 w_k1_ADM_s111
                                 + 0.910014 w_k1_AXP_s111
                                 + 1.15062 w_k1_DIS_s111
                                 + 1.17259 w_k1_FDX_s111
                                 + 0.880301 w_k1_IBM_s111
                                 + 1.0113 w_k1_INTC_s111
                                 + 0.887131 w_k1_KO_s111
                                 + 0.593003 w_k1_MAT_s111
                                 + 1.0772 w_k1_MSFT_s111 + 1.03095 w_k1_PG_s111
                                 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.734704 w_k1_ADM_s111
                                 + 0.76443 w_k1_AXP_s111
                                 + 0.941912 w_k1_DIS_s111
                                 + 0.917844 w_k1_FDX_s111
                                 + 1.26836 w_k1_IBM_s111
                                 + 0.973097 w_k1_INTC_s111
                                 + 0.98709 w_k1_KO_s111 + 1.07157 w_k1_MAT_s111
                                 + 0.983246 w_k1_MSFT_s111
                                 + 0.938487 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.07791 w_k1_ADM_s111
                                 + 1.14108 w_k1_AXP_s111
                                 + 1.45246 w_k1_DIS_s111
                                 + 1.20463 w_k1_FDX_s111
                                 + 1.73602 w_k1_IBM_s111
                                 + 1.1526 w_k1_INTC_s111 + 1.45209 w_k1_KO_s111
                                 + 1.32746 w_k1_MAT_s111
                                 + 1.23939 w_k1_MSFT_s111
                                 + 1.03594 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.34274 w_k1_ADM_s111
                                 + 1.57203 w_k1_AXP_s111
                                 + 1.67736 w_k1_DIS_s111
                                 + 1.37031 w_k1_FDX_s111
                                 + 1.03737 w_k1_IBM_s111
                                 + 1.30778 w_k1_INTC_s111
                                 + 1.28351 w_k1_KO_s111
                                 + 0.995796 w_k1_MAT_s111
                                 + 1.24435 w_k1_MSFT_s111
                                 + 1.21579 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.883871 w_k1_ADM_s111
                                 + 0.631982 w_k1_AXP_s111
                                 + 0.753943 w_k1_DIS_s111
                                 + 0.93162 w_k1_FDX_s111
                                 + 0.687461 w_k1_IBM_s111
                                 + 0.91931 w_k1_INTC_s111
                                 + 0.757286 w_k1_KO_s111
                                 + 0.41767 w_k1_MAT_s111
                                 + 0.983829 w_k1_MSFT_s111
                                 + 0.815643 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.734686 w_k1_ADM_s112
                                 + 0.875157 w_k1_AXP_s112
                                 + 0.852235 w_k1_DIS_s112
                                 + 0.856025 w_k1_FDX_s112
                                 + 1.0525 w_k1_IBM_s112 + 1.0462 w_k1_INTC_s112
                                 + 1.03687 w_k1_KO_s112 + 1.1114 w_k1_MAT_s112
                                 + 0.772986 w_k1_MSFT_s112
                                 + 0.858846 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.0588 w_k1_ADM_s112
                                 + 1.42883 w_k1_AXP_s112 + 1.4821 w_k1_DIS_s112
                                 + 1.40271 w_k1_FDX_s112
                                 + 1.04037 w_k1_IBM_s112
                                 + 1.21518 w_k1_INTC_s112
                                 + 1.30677 w_k1_KO_s112 + 1.50618 w_k1_MAT_s112
                                 + 1.28033 w_k1_MSFT_s112
                                 + 1.22756 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.08551 w_k1_ADM_s112
                                 + 1.26519 w_k1_AXP_s112
                                 + 1.66419 w_k1_DIS_s112
                                 + 1.43156 w_k1_FDX_s112
                                 + 1.47791 w_k1_IBM_s112
                                 + 1.35355 w_k1_INTC_s112
                                 + 1.46372 w_k1_KO_s112 + 1.23693 w_k1_MAT_s112
                                 + 1.34876 w_k1_MSFT_s112 + 1.1723 w_k1_PG_s112
                                 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.856707 w_k1_ADM_s112
                                 + 0.946146 w_k1_AXP_s112
                                 + 0.851581 w_k1_DIS_s112
                                 + 0.861755 w_k1_FDX_s112
                                 + 1.54311 w_k1_IBM_s112
                                 + 1.08626 w_k1_INTC_s112
                                 + 1.19834 w_k1_KO_s112 + 1.86693 w_k1_MAT_s112
                                 + 0.990009 w_k1_MSFT_s112
                                 + 1.02384 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.934199 w_k1_ADM_s112
                                 + 1.09616 w_k1_AXP_s112
                                 + 1.36702 w_k1_DIS_s112
                                 + 1.13459 w_k1_FDX_s112 + 1.0863 w_k1_IBM_s112
                                 + 1.08483 w_k1_INTC_s112
                                 + 1.15304 w_k1_KO_s112 + 1.28159 w_k1_MAT_s112
                                 + 1.09955 w_k1_MSFT_s112
                                 + 1.09222 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.10395 w_k1_ADM_s112
                                 + 0.872906 w_k1_AXP_s112
                                 + 1.19989 w_k1_DIS_s112
                                 + 1.12435 w_k1_FDX_s112
                                 + 0.886456 w_k1_IBM_s112
                                 + 1.35751 w_k1_INTC_s112
                                 + 1.10305 w_k1_KO_s112 + 0.78027 w_k1_MAT_s112
                                 + 1.2588 w_k1_MSFT_s112 + 1.04158 w_k1_PG_s112
                                 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.753889 w_k1_ADM_s112
                                 + 0.666728 w_k1_AXP_s112
                                 + 0.708878 w_k1_DIS_s112
                                 + 0.85499 w_k1_FDX_s112
                                 + 0.70193 w_k1_IBM_s112
                                 + 0.961416 w_k1_INTC_s112
                                 + 0.731011 w_k1_KO_s112
                                 + 0.550382 w_k1_MAT_s112
                                 + 0.798196 w_k1_MSFT_s112
                                 + 0.842573 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.975176 w_k1_ADM_s112
                                 + 1.19723 w_k1_AXP_s112
                                 + 1.08997 w_k1_DIS_s112
                                 + 1.26227 w_k1_FDX_s112
                                 + 0.711046 w_k1_IBM_s112
                                 + 1.28541 w_k1_INTC_s112
                                 + 0.858585 w_k1_KO_s112
                                 + 0.64411 w_k1_MAT_s112
                                 + 1.01113 w_k1_MSFT_s112
                                 + 0.961955 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.04115 w_k1_ADM_s113
                                 + 1.07767 w_k1_AXP_s113
                                 + 1.08858 w_k1_DIS_s113
                                 + 1.03301 w_k1_FDX_s113
                                 + 0.837525 w_k1_IBM_s113
                                 + 1.11059 w_k1_INTC_s113
                                 + 0.929398 w_k1_KO_s113
                                 + 1.03838 w_k1_MAT_s113
                                 + 1.07667 w_k1_MSFT_s113 + 0.9539 w_k1_PG_s113
                                 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.27661 w_k1_ADM_s113
                                 + 1.15018 w_k1_AXP_s113
                                 + 1.48971 w_k1_DIS_s113
                                 + 1.32378 w_k1_FDX_s113
                                 + 0.856257 w_k1_IBM_s113
                                 + 1.16875 w_k1_INTC_s113
                                 + 1.23836 w_k1_KO_s113
                                 + 0.986357 w_k1_MAT_s113
                                 + 1.31246 w_k1_MSFT_s113
                                 + 1.17345 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.788332 w_k1_ADM_s113
                                 + 0.894051 w_k1_AXP_s113
                                 + 0.656504 w_k1_DIS_s113
                                 + 0.707974 w_k1_FDX_s113
                                 + 1.0895 w_k1_IBM_s113
                                 + 0.759265 w_k1_INTC_s113
                                 + 0.782073 w_k1_KO_s113
                                 + 1.27351 w_k1_MAT_s113
                                 + 0.803594 w_k1_MSFT_s113
                                 + 0.941149 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.867455 w_k1_ADM_s113
                                 + 1.09881 w_k1_AXP_s113
                                 + 0.828281 w_k1_DIS_s113
                                 + 1.01813 w_k1_FDX_s113
                                 + 0.68723 w_k1_IBM_s113 + 1.094 w_k1_INTC_s113
                                 + 0.794129 w_k1_KO_s113
                                 + 1.08706 w_k1_MAT_s113
                                 + 1.09287 w_k1_MSFT_s113
                                 + 0.960848 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.04002 w_k1_ADM_s113
                                 + 1.19736 w_k1_AXP_s113
                                 + 1.07803 w_k1_DIS_s113
                                 + 1.01999 w_k1_FDX_s113
                                 + 1.29572 w_k1_IBM_s113
                                 + 1.07739 w_k1_INTC_s113
                                 + 1.14431 w_k1_KO_s113 + 1.22381 w_k1_MAT_s113
                                 + 1.09241 w_k1_MSFT_s113
                                 + 1.07286 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.726353 w_k1_ADM_s113
                                 + 0.558113 w_k1_AXP_s113
                                 + 0.594611 w_k1_DIS_s113
                                 + 0.748695 w_k1_FDX_s113
                                 + 0.627099 w_k1_IBM_s113
                                 + 0.847407 w_k1_INTC_s113
                                 + 0.661912 w_k1_KO_s113
                                 + 0.811221 w_k1_MAT_s113
                                 + 0.89582 w_k1_MSFT_s113
                                 + 0.785309 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.946202 w_k1_ADM_s113
                                 + 0.795306 w_k1_AXP_s113
                                 + 0.859444 w_k1_DIS_s113
                                 + 0.800224 w_k1_FDX_s113
                                 + 0.819231 w_k1_IBM_s113
                                 + 0.891102 w_k1_INTC_s113
                                 + 0.949606 w_k1_KO_s113
                                 + 0.793256 w_k1_MAT_s113
                                 + 1.02666 w_k1_MSFT_s113
                                 + 0.933796 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.36201 w_k1_ADM_s113
                                 + 1.36057 w_k1_AXP_s113
                                 + 1.62916 w_k1_DIS_s113
                                 + 1.31144 w_k1_FDX_s113
                                 + 1.56669 w_k1_IBM_s113
                                 + 1.51765 w_k1_INTC_s113
                                 + 1.56278 w_k1_KO_s113 + 1.49536 w_k1_MAT_s113
                                 + 1.61754 w_k1_MSFT_s113
                                 + 1.28004 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.875478 w_k1_ADM_s114
                                 + 0.775245 w_k1_AXP_s114
                                 + 0.849785 w_k1_DIS_s114
                                 + 0.90677 w_k1_FDX_s114
                                 + 0.729899 w_k1_IBM_s114
                                 + 0.714502 w_k1_INTC_s114
                                 + 0.759279 w_k1_KO_s114
                                 + 0.783919 w_k1_MAT_s114
                                 + 0.802805 w_k1_MSFT_s114
                                 + 0.788724 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.05141 w_k1_ADM_s114
                                 + 1.61288 w_k1_AXP_s114
                                 + 1.17011 w_k1_DIS_s114
                                 + 1.28688 w_k1_FDX_s114
                                 + 0.843275 w_k1_IBM_s114
                                 + 0.960131 w_k1_INTC_s114
                                 + 0.943987 w_k1_KO_s114
                                 + 1.35531 w_k1_MAT_s114
                                 + 1.06569 w_k1_MSFT_s114
                                 + 0.940989 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.679328 w_k1_ADM_s114
                                 + 0.689385 w_k1_AXP_s114
                                 + 0.651843 w_k1_DIS_s114
                                 + 0.761151 w_k1_FDX_s114
                                 + 1.01371 w_k1_IBM_s114
                                 + 0.728978 w_k1_INTC_s114
                                 + 0.827616 w_k1_KO_s114
                                 + 1.00179 w_k1_MAT_s114
                                 + 0.798232 w_k1_MSFT_s114
                                 + 0.784442 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.751892 w_k1_ADM_s114
                                 + 0.741479 w_k1_AXP_s114
                                 + 0.56943 w_k1_DIS_s114
                                 + 0.804551 w_k1_FDX_s114
                                 + 0.504475 w_k1_IBM_s114
                                 + 0.712748 w_k1_INTC_s114
                                 + 0.611988 w_k1_KO_s114
                                 + 0.724765 w_k1_MAT_s114
                                 + 0.720851 w_k1_MSFT_s114
                                 + 0.795273 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.10159 w_k1_ADM_s114
                                 + 1.52904 w_k1_AXP_s114
                                 + 1.43098 w_k1_DIS_s114 + 1.3602 w_k1_FDX_s114
                                 + 1.41599 w_k1_IBM_s114
                                 + 1.0586 w_k1_INTC_s114 + 1.36012 w_k1_KO_s114
                                 + 2.2011 w_k1_MAT_s114
                                 + 1.27462 w_k1_MSFT_s114
                                 + 1.29427 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.04987 w_k1_ADM_s114
                                 + 1.01254 w_k1_AXP_s114
                                 + 0.990451 w_k1_DIS_s114
                                 + 0.93297 w_k1_FDX_s114
                                 + 1.00393 w_k1_IBM_s114
                                 + 0.888377 w_k1_INTC_s114
                                 + 0.898271 w_k1_KO_s114
                                 + 1.01309 w_k1_MAT_s114
                                 + 0.953665 w_k1_MSFT_s114
                                 + 0.955798 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.06348 w_k1_ADM_s114
                                 + 1.17571 w_k1_AXP_s114
                                 + 1.38909 w_k1_DIS_s114
                                 + 1.18414 w_k1_FDX_s114
                                 + 0.853096 w_k1_IBM_s114
                                 + 0.982059 w_k1_INTC_s114
                                 + 0.986627 w_k1_KO_s114
                                 + 1.06545 w_k1_MAT_s114
                                 + 1.17079 w_k1_MSFT_s114
                                 + 0.970681 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.797858 w_k1_ADM_s114
                                 + 1.31695 w_k1_AXP_s114
                                 + 1.26439 w_k1_DIS_s114
                                 + 1.03758 w_k1_FDX_s114
                                 + 1.29621 w_k1_IBM_s114
                                 + 0.922468 w_k1_INTC_s114
                                 + 1.28269 w_k1_KO_s114 + 2.1598 w_k1_MAT_s114
                                 + 0.977032 w_k1_MSFT_s114
                                 + 1.01187 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.18678 w_k1_ADM_s115
                                 + 1.06241 w_k1_AXP_s115
                                 + 1.31411 w_k1_DIS_s115
                                 + 1.22206 w_k1_FDX_s115
                                 + 0.897246 w_k1_IBM_s115
                                 + 1.07256 w_k1_INTC_s115
                                 + 1.04718 w_k1_KO_s115 + 1.09583 w_k1_MAT_s115
                                 + 1.19074 w_k1_MSFT_s115
                                 + 1.00825 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.896659 w_k1_ADM_s115
                                 + 0.901079 w_k1_AXP_s115
                                 + 1.04407 w_k1_DIS_s115
                                 + 0.987065 w_k1_FDX_s115
                                 + 0.661351 w_k1_IBM_s115
                                 + 0.985926 w_k1_INTC_s115
                                 + 0.823523 w_k1_KO_s115
                                 + 1.01931 w_k1_MAT_s115
                                 + 1.04769 w_k1_MSFT_s115
                                 + 0.897034 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.06774 w_k1_ADM_s115
                                 + 1.01188 w_k1_AXP_s115
                                 + 0.79689 w_k1_DIS_s115
                                 + 0.860445 w_k1_FDX_s115
                                 + 1.27532 w_k1_IBM_s115
                                 + 0.949138 w_k1_INTC_s115
                                 + 1.04225 w_k1_KO_s115 + 1.20126 w_k1_MAT_s115
                                 + 0.880409 w_k1_MSFT_s115
                                 + 0.978073 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.01099 w_k1_ADM_s115
                                 + 0.911408 w_k1_AXP_s115
                                 + 1.06498 w_k1_DIS_s115 + 1.0531 w_k1_FDX_s115
                                 + 1.18734 w_k1_IBM_s115
                                 + 1.09881 w_k1_INTC_s115
                                 + 1.09827 w_k1_KO_s115 + 1.14982 w_k1_MAT_s115
                                 + 1.05999 w_k1_MSFT_s115
                                 + 0.981722 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.657767 w_k1_ADM_s115
                                 + 0.706068 w_k1_AXP_s115
                                 + 0.800168 w_k1_DIS_s115
                                 + 0.778115 w_k1_FDX_s115
                                 + 0.89924 w_k1_IBM_s115
                                 + 0.859564 w_k1_INTC_s115
                                 + 0.93587 w_k1_KO_s115 + 1.07019 w_k1_MAT_s115
                                 + 0.867283 w_k1_MSFT_s115
                                 + 0.975679 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.35231 w_k1_ADM_s115
                                 + 1.49014 w_k1_AXP_s115
                                 + 1.55177 w_k1_DIS_s115
                                 + 1.25529 w_k1_FDX_s115
                                 + 1.45213 w_k1_IBM_s115
                                 + 1.16943 w_k1_INTC_s115 + 1.3253 w_k1_KO_s115
                                 + 1.44596 w_k1_MAT_s115
                                 + 1.14005 w_k1_MSFT_s115
                                 + 1.11084 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.911783 w_k1_ADM_s115
                                 + 0.67321 w_k1_AXP_s115
                                 + 0.741787 w_k1_DIS_s115
                                 + 0.890772 w_k1_FDX_s115
                                 + 0.582905 w_k1_IBM_s115
                                 + 0.899746 w_k1_INTC_s115
                                 + 0.732269 w_k1_KO_s115
                                 + 0.477701 w_k1_MAT_s115
                                 + 0.896501 w_k1_MSFT_s115
                                 + 0.800516 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.07788 w_k1_ADM_s115
                                 + 0.898234 w_k1_AXP_s115
                                 + 0.970239 w_k1_DIS_s115
                                 + 0.946458 w_k1_FDX_s115
                                 + 1.09249 w_k1_IBM_s115
                                 + 0.865795 w_k1_INTC_s115
                                 + 1.05579 w_k1_KO_s115 + 1.03091 w_k1_MAT_s115
                                 + 0.945998 w_k1_MSFT_s115
                                 + 0.959536 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.848521 w_k1_ADM_s116
                                 + 0.800929 w_k1_AXP_s116
                                 + 1.05627 w_k1_DIS_s116
                                 + 0.912433 w_k1_FDX_s116
                                 + 1.00548 w_k1_IBM_s116
                                 + 0.757855 w_k1_INTC_s116
                                 + 1.04716 w_k1_KO_s116
                                 + 0.868867 w_k1_MAT_s116
                                 + 1.0487 w_k1_MSFT_s116
                                 + 0.982488 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.984615 w_k1_ADM_s116
                                 + 1.33307 w_k1_AXP_s116
                                 + 0.998849 w_k1_DIS_s116
                                 + 1.0149 w_k1_FDX_s116
                                 + 0.978736 w_k1_IBM_s116
                                 + 0.984336 w_k1_INTC_s116
                                 + 0.867385 w_k1_KO_s116
                                 + 0.997265 w_k1_MAT_s116
                                 + 0.940145 w_k1_MSFT_s116
                                 + 1.00377 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.89965 w_k1_ADM_s116
                                 + 0.856562 w_k1_AXP_s116
                                 + 0.651705 w_k1_DIS_s116
                                 + 0.94406 w_k1_FDX_s116
                                 + 0.682694 w_k1_IBM_s116
                                 + 0.728352 w_k1_INTC_s116
                                 + 0.663985 w_k1_KO_s116
                                 + 0.498376 w_k1_MAT_s116
                                 + 0.877002 w_k1_MSFT_s116
                                 + 0.900637 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.10586 w_k1_ADM_s116
                                 + 1.50266 w_k1_AXP_s116 + 1.472 w_k1_DIS_s116
                                 + 1.34856 w_k1_FDX_s116
                                 + 1.26585 w_k1_IBM_s116
                                 + 1.08503 w_k1_INTC_s116
                                 + 1.13423 w_k1_KO_s116 + 1.07119 w_k1_MAT_s116
                                 + 1.30194 w_k1_MSFT_s116 + 1.0746 w_k1_PG_s116
                                 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.625108 w_k1_ADM_s116
                                 + 0.724499 w_k1_AXP_s116
                                 + 0.73564 w_k1_DIS_s116 + 0.8149 w_k1_FDX_s116
                                 + 0.770656 w_k1_IBM_s116
                                 + 0.776788 w_k1_INTC_s116
                                 + 0.744814 w_k1_KO_s116
                                 + 0.604214 w_k1_MAT_s116
                                 + 0.82959 w_k1_MSFT_s116
                                 + 0.915445 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.04327 w_k1_ADM_s116
                                 + 1.10562 w_k1_AXP_s116
                                 + 1.09829 w_k1_DIS_s116
                                 + 1.07002 w_k1_FDX_s116
                                 + 0.991052 w_k1_IBM_s116
                                 + 0.947249 w_k1_INTC_s116
                                 + 0.971996 w_k1_KO_s116
                                 + 0.71336 w_k1_MAT_s116
                                 + 1.05129 w_k1_MSFT_s116
                                 + 1.04802 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.00022 w_k1_ADM_s116
                                 + 1.18462 w_k1_AXP_s116
                                 + 0.998529 w_k1_DIS_s116
                                 + 1.01547 w_k1_FDX_s116
                                 + 1.47004 w_k1_IBM_s116
                                 + 0.981054 w_k1_INTC_s116
                                 + 1.10501 w_k1_KO_s116 + 1.20101 w_k1_MAT_s116
                                 + 0.988878 w_k1_MSFT_s116
                                 + 1.12106 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.821629 w_k1_ADM_s116
                                 + 0.71459 w_k1_AXP_s116
                                 + 0.543388 w_k1_DIS_s116
                                 + 0.777775 w_k1_FDX_s116
                                 + 0.91777 w_k1_IBM_s116
                                 + 0.787421 w_k1_INTC_s116
                                 + 0.840162 w_k1_KO_s116
                                 + 0.646404 w_k1_MAT_s116
                                 + 0.732814 w_k1_MSFT_s116
                                 + 0.848053 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.04487 w_k1_ADM_s117
                                 + 0.54491 w_k1_AXP_s117
                                 + 0.746583 w_k1_DIS_s117
                                 + 0.784807 w_k1_FDX_s117
                                 + 0.738349 w_k1_IBM_s117
                                 + 0.849803 w_k1_INTC_s117
                                 + 0.777037 w_k1_KO_s117
                                 + 0.46227 w_k1_MAT_s117
                                 + 0.85342 w_k1_MSFT_s117
                                 + 0.800906 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.56502 w_k1_ADM_s117
                                 + 1.21578 w_k1_AXP_s117
                                 + 1.53393 w_k1_DIS_s117
                                 + 1.31406 w_k1_FDX_s117
                                 + 0.97248 w_k1_IBM_s117
                                 + 1.25197 w_k1_INTC_s117
                                 + 1.14665 w_k1_KO_s117
                                 + 0.965376 w_k1_MAT_s117
                                 + 1.23748 w_k1_MSFT_s117
                                 + 1.02797 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.53028 w_k1_ADM_s117
                                 + 1.10367 w_k1_AXP_s117
                                 + 1.04056 w_k1_DIS_s117
                                 + 1.07661 w_k1_FDX_s117
                                 + 0.850681 w_k1_IBM_s117
                                 + 0.944865 w_k1_INTC_s117
                                 + 1.06689 w_k1_KO_s117
                                 + 0.887605 w_k1_MAT_s117
                                 + 1.04943 w_k1_MSFT_s117
                                 + 1.06577 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.12697 w_k1_ADM_s117
                                 + 0.888407 w_k1_AXP_s117
                                 + 0.746483 w_k1_DIS_s117
                                 + 0.823042 w_k1_FDX_s117
                                 + 0.879177 w_k1_IBM_s117
                                 + 0.741682 w_k1_INTC_s117
                                 + 0.871421 w_k1_KO_s117
                                 + 0.714975 w_k1_MAT_s117
                                 + 0.780391 w_k1_MSFT_s117
                                 + 0.875812 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.12842 w_k1_ADM_s117
                                 + 0.989547 w_k1_AXP_s117
                                 + 0.920311 w_k1_DIS_s117
                                 + 0.935235 w_k1_FDX_s117
                                 + 0.756045 w_k1_IBM_s117
                                 + 0.856864 w_k1_INTC_s117
                                 + 0.911323 w_k1_KO_s117
                                 + 0.66688 w_k1_MAT_s117
                                 + 1.04509 w_k1_MSFT_s117
                                 + 0.897256 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.02849 w_k1_ADM_s117
                                 + 0.675178 w_k1_AXP_s117
                                 + 0.86834 w_k1_DIS_s117
                                 + 0.823866 w_k1_FDX_s117
                                 + 1.05264 w_k1_IBM_s117
                                 + 0.755328 w_k1_INTC_s117
                                 + 1.02677 w_k1_KO_s117
                                 + 0.777431 w_k1_MAT_s117
                                 + 0.799712 w_k1_MSFT_s117
                                 + 0.89091 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.15421 w_k1_ADM_s117
                                 + 1.1661 w_k1_AXP_s117 + 1.06863 w_k1_DIS_s117
                                 + 1.01803 w_k1_FDX_s117
                                 + 1.19683 w_k1_IBM_s117
                                 + 0.955113 w_k1_INTC_s117
                                 + 1.14674 w_k1_KO_s117 + 1.12153 w_k1_MAT_s117
                                 + 1.12049 w_k1_MSFT_s117
                                 + 0.986799 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.56084 w_k1_ADM_s117
                                 + 1.18294 w_k1_AXP_s117
                                 + 1.41443 w_k1_DIS_s117
                                 + 1.20237 w_k1_FDX_s117
                                 + 1.39199 w_k1_IBM_s117
                                 + 1.25812 w_k1_INTC_s117 + 1.5621 w_k1_KO_s117
                                 + 1.13166 w_k1_MAT_s117
                                 + 1.23643 w_k1_MSFT_s117
                                 + 1.10413 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0214478785813081 w_k1_ADM_s000
                                 + 0.00591157826200668 w_k1_ADM_s001
                                 + 0.0126640176773259 w_k1_ADM_s002
                                 + 0.00474061959140734 w_k1_ADM_s003
                                 + 0.00519742084531071 w_k1_ADM_s004
                                 + 0.0225940834053187 w_k1_ADM_s005
                                 + 0.00986648177544555 w_k1_ADM_s006
                                 + 0.0198554416165757 w_k1_ADM_s007
                                 + 0.0242109451880533 w_k1_ADM_s010
                                 + 0.0183770252097414 w_k1_ADM_s011
                                 + 0.0239951128589788 w_k1_ADM_s012
                                 + 0.0126698490515402 w_k1_ADM_s013
                                 + 0.0098324493210504 w_k1_ADM_s014
                                 + 0.00910909434473697 w_k1_ADM_s015
                                 + 0.0270683264009357 w_k1_ADM_s016
                                 + 0.0210717362411561 w_k1_ADM_s017
                                 + 0.0555226716402941 w_k1_ADM_s100
                                 + 0.0638974202420269 w_k1_ADM_s101
                                 + 0.102760259528225 w_k1_ADM_s102
                                 + 0.0443345677482619 w_k1_ADM_s103
                                 + 0.15335314916066 w_k1_ADM_s104
                                 + 0.035716247564925 w_k1_ADM_s105
                                 + 0.0826709245764042 w_k1_ADM_s106
                                 + 0.0459358651321043 w_k1_ADM_s107
                                 + 0.0202541342588662 w_k1_ADM_s110
                                 + 0.0176584694603676 w_k1_ADM_s111
                                 + 0.00956872441296147 w_k1_ADM_s112
                                 + 0.0447086156554367 w_k1_ADM_s113
                                 + 0.0160703775675577 w_k1_ADM_s114
                                 + 0.0141940966399055 w_k1_ADM_s115
                                 + 0.0299886679550641 w_k1_ADM_s116
                                 + 0.0312593622365936 w_k1_ADM_s117
                                 + 0.0205436929057208 w_k1_AXP_s000
                                 + 0.00654631376979252 w_k1_AXP_s001
                                 + 0.0134005238202437 w_k1_AXP_s002
                                 + 0.0048633240482884 w_k1_AXP_s003
                                 + 0.00556772603446021 w_k1_AXP_s004
                                 + 0.0187705395580138 w_k1_AXP_s005
                                 + 0.0135035558781074 w_k1_AXP_s006
                                 + 0.0169750357824631 w_k1_AXP_s007
                                 + 0.0216729522425294 w_k1_AXP_s010
                                 + 0.0185490119835531 w_k1_AXP_s011
                                 + 0.0249325567843094 w_k1_AXP_s012
                                 + 0.0122393686475128 w_k1_AXP_s013
                                 + 0.0104805814204774 w_k1_AXP_s014
                                 + 0.0079471706047437 w_k1_AXP_s015
                                 + 0.0267378029149004 w_k1_AXP_s016
                                 + 0.0211448957693324 w_k1_AXP_s017
                                 + 0.0579718467145225 w_k1_AXP_s100
                                 + 0.0664857590969814 w_k1_AXP_s101
                                 + 0.106062124578528 w_k1_AXP_s102
                                 + 0.0529580484652981 w_k1_AXP_s103
                                 + 0.149667393618165 w_k1_AXP_s104
                                 + 0.0408389003801994 w_k1_AXP_s105
                                 + 0.0792112542416497 w_k1_AXP_s106
                                 + 0.0557988859612845 w_k1_AXP_s107
                                 + 0.0170847914301873 w_k1_AXP_s110
                                 + 0.0166970686138831 w_k1_AXP_s111
                                 + 0.0101597367460931 w_k1_AXP_s112
                                 + 0.0469351842266184 w_k1_AXP_s113
                                 + 0.0189075276561231 w_k1_AXP_s114
                                 + 0.0134829394589134 w_k1_AXP_s115
                                 + 0.0344152390653789 w_k1_AXP_s116
                                 + 0.0247052915912809 w_k1_AXP_s117
                                 + 0.0196748635611022 w_k1_DIS_s000
                                 + 0.00595422464240084 w_k1_DIS_s001
                                 + 0.0131951276300982 w_k1_DIS_s002
                                 + 0.00463286868537793 w_k1_DIS_s003
                                 + 0.0061795348613437 w_k1_DIS_s004
                                 + 0.0199542203482148 w_k1_DIS_s005
                                 + 0.0119819494108137 w_k1_DIS_s006
                                 + 0.0176559857817578 w_k1_DIS_s007
                                 + 0.0228246439918297 w_k1_DIS_s010
                                 + 0.020390800420722 w_k1_DIS_s011
                                 + 0.0246762787263486 w_k1_DIS_s012
                                 + 0.0121523242598312 w_k1_DIS_s013
                                 + 0.00940716409527236 w_k1_DIS_s014
                                 + 0.00837988056204181 w_k1_DIS_s015
                                 + 0.0296886081556016 w_k1_DIS_s016
                                 + 0.0207355228587376 w_k1_DIS_s017
                                 + 0.0577179418273357 w_k1_DIS_s100
                                 + 0.0650163706260041 w_k1_DIS_s101
                                 + 0.103945212306138 w_k1_DIS_s102
                                 + 0.0490390936307594 w_k1_DIS_s103
                                 + 0.140489139473598 w_k1_DIS_s104
                                 + 0.0432914913847588 w_k1_DIS_s105
                                 + 0.0791638898619311 w_k1_DIS_s106
                                 + 0.0522268242915384 w_k1_DIS_s107
                                 + 0.0194462861162202 w_k1_DIS_s110
                                 + 0.0191769584565336 w_k1_DIS_s111
                                 + 0.0114449586010008 w_k1_DIS_s112
                                 + 0.045763464740628 w_k1_DIS_s113
                                 + 0.0179365542958084 w_k1_DIS_s114
                                 + 0.0148563860342972 w_k1_DIS_s115
                                 + 0.0320617352079943 w_k1_DIS_s116
                                 + 0.0256863519735942 w_k1_DIS_s117
                                 + 0.018586072194379 w_k1_FDX_s000
                                 + 0.00610935233997232 w_k1_FDX_s001
                                 + 0.0131158265805945 w_k1_FDX_s002
                                 + 0.00445969297166626 w_k1_FDX_s003
                                 + 0.005856201802139 w_k1_FDX_s004
                                 + 0.0198322535494292 w_k1_FDX_s005
                                 + 0.0125538303018307 w_k1_FDX_s006
                                 + 0.0178381702996343 w_k1_FDX_s007
                                 + 0.0216680474705198 w_k1_FDX_s010
                                 + 0.0195946588835323 w_k1_FDX_s011
                                 + 0.023135142889004 w_k1_FDX_s012
                                 + 0.0117732960191426 w_k1_FDX_s013
                                 + 0.00949527956759518 w_k1_FDX_s014
                                 + 0.00814659806356586 w_k1_FDX_s015
                                 + 0.0278425098635328 w_k1_FDX_s016
                                 + 0.0213651296157372 w_k1_FDX_s017
                                 + 0.0601421393866002 w_k1_FDX_s100
                                 + 0.0662268294190143 w_k1_FDX_s101
                                 + 0.105720250305791 w_k1_FDX_s102
                                 + 0.048626227947195 w_k1_FDX_s103
                                 + 0.142639282475498 w_k1_FDX_s104
                                 + 0.0425711520043837 w_k1_FDX_s105
                                 + 0.0763141364419949 w_k1_FDX_s106
                                 + 0.0503472815481484 w_k1_FDX_s107
                                 + 0.0177081086997004 w_k1_FDX_s110
                                 + 0.0186538788134923 w_k1_FDX_s111
                                 + 0.0111448198509451 w_k1_FDX_s112
                                 + 0.0438591681502622 w_k1_FDX_s113
                                 + 0.0175296758294797 w_k1_FDX_s114
                                 + 0.0141553644184696 w_k1_FDX_s115
                                 + 0.0327402770632901 w_k1_FDX_s116
                                 + 0.024616110187738 w_k1_FDX_s117
                                 + 0.0190956972386196 w_k1_IBM_s000
                                 + 0.00587116501754502 w_k1_IBM_s001
                                 + 0.0139587634975337 w_k1_IBM_s002
                                 + 0.00398017437988275 w_k1_IBM_s003
                                 + 0.00547063455131464 w_k1_IBM_s004
                                 + 0.0189558141854583 w_k1_IBM_s005
                                 + 0.0121420779135695 w_k1_IBM_s006
                                 + 0.0178654067607208 w_k1_IBM_s007
                                 + 0.0228638955602785 w_k1_IBM_s010
                                 + 0.0196944964527737 w_k1_IBM_s011
                                 + 0.0246352588027676 w_k1_IBM_s012
                                 + 0.011323396629271 w_k1_IBM_s013
                                 + 0.00891739907764528 w_k1_IBM_s014
                                 + 0.00803026307580854 w_k1_IBM_s015
                                 + 0.0300577671508387 w_k1_IBM_s016
                                 + 0.0218783023368838 w_k1_IBM_s017
                                 + 0.0591023686977409 w_k1_IBM_s100
                                 + 0.0667567124902501 w_k1_IBM_s101
                                 + 0.105570914533466 w_k1_IBM_s102
                                 + 0.0464590817220095 w_k1_IBM_s103
                                 + 0.145732338483857 w_k1_IBM_s104
                                 + 0.0439713498353796 w_k1_IBM_s105
                                 + 0.0786398150451387 w_k1_IBM_s106
                                 + 0.0537120234413123 w_k1_IBM_s107
                                 + 0.0172020437504309 w_k1_IBM_s110
                                 + 0.0176092099814161 w_k1_IBM_s111
                                 + 0.0106509971917962 w_k1_IBM_s112
                                 + 0.0455608849111163 w_k1_IBM_s113
                                 + 0.0161538774030024 w_k1_IBM_s114
                                 + 0.0141861245473803 w_k1_IBM_s115
                                 + 0.0341783855007998 w_k1_IBM_s116
                                 + 0.0242857253704788 w_k1_IBM_s117
                                 + 0.0218102091404974 w_k1_INTC_s000
                                 + 0.00629958919118032 w_k1_INTC_s001
                                 + 0.013159425022006 w_k1_INTC_s002
                                 + 0.00407754883616961 w_k1_INTC_s003
                                 + 0.00600821789158554 w_k1_INTC_s004
                                 + 0.0212067789583558 w_k1_INTC_s005
                                 + 0.0119974068416448 w_k1_INTC_s006
                                 + 0.0162802730918347 w_k1_INTC_s007
                                 + 0.0185874775656932 w_k1_INTC_s010
                                 + 0.020705746169038 w_k1_INTC_s011
                                 + 0.0256263688115911 w_k1_INTC_s012
                                 + 0.0105990026558011 w_k1_INTC_s013
                                 + 0.00974222233097777 w_k1_INTC_s014
                                 + 0.00728675561371116 w_k1_INTC_s015
                                 + 0.028425309062919 w_k1_INTC_s016
                                 + 0.0232225589009185 w_k1_INTC_s017
                                 + 0.05658367842034 w_k1_INTC_s100
                                 + 0.0588387427700052 w_k1_INTC_s101
                                 + 0.113540813013672 w_k1_INTC_s102
                                 + 0.0471370184262104 w_k1_INTC_s103
                                 + 0.139036811894546 w_k1_INTC_s104
                                 + 0.0451777819363855 w_k1_INTC_s105
                                 + 0.0724753349311572 w_k1_INTC_s106
                                 + 0.0627107831757285 w_k1_INTC_s107
                                 + 0.0173432502803493 w_k1_INTC_s110
                                 + 0.0179933133604044 w_k1_INTC_s111
                                 + 0.0120788377831923 w_k1_INTC_s112
                                 + 0.0468752094832507 w_k1_INTC_s113
                                 + 0.014832313360165 w_k1_INTC_s114
                                 + 0.0139589179774209 w_k1_INTC_s115
                                 + 0.0293270044270356 w_k1_INTC_s116
                                 + 0.0233794984072967 w_k1_INTC_s117
                                 + 0.0177185932484753 w_k1_KO_s000
                                 + 0.00619343925156276 w_k1_KO_s001
                                 + 0.0147289895252683 w_k1_KO_s002
                                 + 0.00435772274166635 w_k1_KO_s003
                                 + 0.00556474092374963 w_k1_KO_s004
                                 + 0.0191542162929454 w_k1_KO_s005
                                 + 0.0112585041604161 w_k1_KO_s006
                                 + 0.0174013038768161 w_k1_KO_s007
                                 + 0.021629469962522 w_k1_KO_s010
                                 + 0.0216703233438224 w_k1_KO_s011
                                 + 0.0265625220662403 w_k1_KO_s012
                                 + 0.0109381954643054 w_k1_KO_s013
                                 + 0.00993673043095327 w_k1_KO_s014
                                 + 0.00765177428362338 w_k1_KO_s015
                                 + 0.0298263456222696 w_k1_KO_s016
                                 + 0.020520253329679 w_k1_KO_s017
                                 + 0.0592374307041337 w_k1_KO_s100
                                 + 0.0609469348372954 w_k1_KO_s101
                                 + 0.104914724742765 w_k1_KO_s102
                                 + 0.0491067609783542 w_k1_KO_s103
                                 + 0.155220437916564 w_k1_KO_s104
                                 + 0.0420344396607572 w_k1_KO_s105
                                 + 0.0708459226857559 w_k1_KO_s106
                                 + 0.0602801619767284 w_k1_KO_s107
                                 + 0.0210894622407224 w_k1_KO_s110
                                 + 0.0171372570593316 w_k1_KO_s111
                                 + 0.0112257048770124 w_k1_KO_s112
                                 + 0.0449074126758035 w_k1_KO_s113
                                 + 0.0161107894961313 w_k1_KO_s114
                                 + 0.0143326345344351 w_k1_KO_s115
                                 + 0.030850180064353 w_k1_KO_s116
                                 + 0.0263727646294302 w_k1_KO_s117
                                 + 0.0208278077426679 w_k1_MAT_s000
                                 + 0.00614933740262952 w_k1_MAT_s001
                                 + 0.0169104223644055 w_k1_MAT_s002
                                 + 0.00529895289217949 w_k1_MAT_s003
                                 + 0.00549982114735874 w_k1_MAT_s004
                                 + 0.0162548185102735 w_k1_MAT_s005
                                 + 0.014275112872409 w_k1_MAT_s006
                                 + 0.0197063343165295 w_k1_MAT_s007
                                 + 0.0298559576302878 w_k1_MAT_s010
                                 + 0.0229195647629734 w_k1_MAT_s011
                                 + 0.0252664247107867 w_k1_MAT_s012
                                 + 0.0131241231751599 w_k1_MAT_s013
                                 + 0.00926064409933868 w_k1_MAT_s014
                                 + 0.00876386592627639 w_k1_MAT_s015
                                 + 0.0289103153175666 w_k1_MAT_s016
                                 + 0.0202051920709639 w_k1_MAT_s017
                                 + 0.0671853244192788 w_k1_MAT_s100
                                 + 0.0676340402414072 w_k1_MAT_s101
                                 + 0.129281134238989 w_k1_MAT_s102
                                 + 0.0546473084771569 w_k1_MAT_s103
                                 + 0.14604919114313 w_k1_MAT_s104
                                 + 0.0508563100725437 w_k1_MAT_s105
                                 + 0.0831486396655745 w_k1_MAT_s106
                                 + 0.049000715096121 w_k1_MAT_s107
                                 + 0.01564762844187 w_k1_MAT_s110
                                 + 0.0146203201071516 w_k1_MAT_s111
                                 + 0.0109003576844097 w_k1_MAT_s112
                                 + 0.0499919266930099 w_k1_MAT_s113
                                 + 0.0211662272736973 w_k1_MAT_s114
                                 + 0.0150038654215246 w_k1_MAT_s115
                                 + 0.0278798983998656 w_k1_MAT_s116
                                 + 0.0214400646830667 w_k1_MAT_s117
                                 + 0.0190328869117205 w_k1_MSFT_s000
                                 + 0.0055106054881346 w_k1_MSFT_s001
                                 + 0.0127919811504878 w_k1_MSFT_s002
                                 + 0.00414113085716562 w_k1_MSFT_s003
                                 + 0.00575273377407583 w_k1_MSFT_s004
                                 + 0.019483197657338 w_k1_MSFT_s005
                                 + 0.0118865619007363 w_k1_MSFT_s006
                                 + 0.0162992593857892 w_k1_MSFT_s007
                                 + 0.0219199752219036 w_k1_MSFT_s010
                                 + 0.0220312616876947 w_k1_MSFT_s011
                                 + 0.024761222586159 w_k1_MSFT_s012
                                 + 0.010546174989596 w_k1_MSFT_s013
                                 + 0.00952072965938805 w_k1_MSFT_s014
                                 + 0.00786919002046245 w_k1_MSFT_s015
                                 + 0.0301840454272291 w_k1_MSFT_s016
                                 + 0.020584514147758 w_k1_MSFT_s017
                                 + 0.0564939428947594 w_k1_MSFT_s100
                                 + 0.0609723127873825 w_k1_MSFT_s101
                                 + 0.105994341870284 w_k1_MSFT_s102
                                 + 0.0477096857473252 w_k1_MSFT_s103
                                 + 0.146506486753576 w_k1_MSFT_s104
                                 + 0.0430909821639378 w_k1_MSFT_s105
                                 + 0.0840826705828631 w_k1_MSFT_s106
                                 + 0.053618104614585 w_k1_MSFT_s107
                                 + 0.0179278030202043 w_k1_MSFT_s110
                                 + 0.0186221780924485 w_k1_MSFT_s111
                                 + 0.0108044422425225 w_k1_MSFT_s112
                                 + 0.0483847990597461 w_k1_MSFT_s113
                                 + 0.0165449661329469 w_k1_MSFT_s114
                                 + 0.0141821454688142 w_k1_MSFT_s115
                                 + 0.0322508164078395 w_k1_MSFT_s116
                                 + 0.0252312089483136 w_k1_MSFT_s117
                                 + 0.0184214492840776 w_k1_PG_s000
                                 + 0.00641247111076791 w_k1_PG_s001
                                 + 0.0142965760748513 w_k1_PG_s002
                                 + 0.00411119477061436 w_k1_PG_s003
                                 + 0.00552038343370482 w_k1_PG_s004
                                 + 0.0216930503164522 w_k1_PG_s005
                                 + 0.0127413844504823 w_k1_PG_s006
                                 + 0.0175290742063648 w_k1_PG_s007
                                 + 0.0227653961096376 w_k1_PG_s010
                                 + 0.0202130756063426 w_k1_PG_s011
                                 + 0.0238117855876054 w_k1_PG_s012
                                 + 0.0111803059726497 w_k1_PG_s013
                                 + 0.00919813215162769 w_k1_PG_s014
                                 + 0.00766472916423246 w_k1_PG_s015
                                 + 0.0260754499149726 w_k1_PG_s016
                                 + 0.022094982328342 w_k1_PG_s017
                                 + 0.059833607702918 w_k1_PG_s100
                                 + 0.0637482518507591 w_k1_PG_s101
                                 + 0.10790347436952 w_k1_PG_s102
                                 + 0.0451009007043289 w_k1_PG_s103
                                 + 0.153816808146671 w_k1_PG_s104
                                 + 0.0392885767772896 w_k1_PG_s105
                                 + 0.0817693269614097 w_k1_PG_s106
                                 + 0.0518092369021664 w_k1_PG_s107
                                 + 0.0191201757597744 w_k1_PG_s110
                                 + 0.0171917584389471 w_k1_PG_s111
                                 + 0.0104787294144108 w_k1_PG_s112
                                 + 0.0448926529856795 w_k1_PG_s113
                                 + 0.015967144205364 w_k1_PG_s114
                                 + 0.0136836184147089 w_k1_PG_s115
                                 + 0.0324980589896999 w_k1_PG_s116
                                 + 0.0235843020012068 w_k1_PG_s117 >= 1.15
Bounds
 0 <= w_0_ADM <= 1
 0 <= w_0_AXP <= 1
 0 <= w_0_DIS <= 1
 0 <= w_0_FDX <= 1
 0 <= w_0_IBM <= 1
 0 <= w_0_INTC <= 1
 0 <= w_0_KO <= 1
 0 <= w_0_MAT <= 1
 0 <= w_0_MSFT <= 1
 0 <= w_0_PG <= 1
 0 <= w_k1_ADM_s0 <= 1
 0 <= w_k1_ADM_s1 <= 1
 0 <= w_k1_ADM_s00 <= 1
 0 <= w_k1_ADM_s01 <= 1
 0 <= w_k1_ADM_s10 <= 1
 0 <= w_k1_ADM_s11 <= 1
 0 <= w_k1_ADM_s000 <= 1
 0 <= w_k1_ADM_s001 <= 1
 0 <= w_k1_ADM_s002 <= 1
 0 <= w_k1_ADM_s003 <= 1
 0 <= w_k1_ADM_s004 <= 1
 0 <= w_k1_ADM_s005 <= 1
 0 <= w_k1_ADM_s006 <= 1
 0 <= w_k1_ADM_s007 <= 1
 0 <= w_k1_ADM_s010 <= 1
 0 <= w_k1_ADM_s011 <= 1
 0 <= w_k1_ADM_s012 <= 1
 0 <= w_k1_ADM_s013 <= 1
 0 <= w_k1_ADM_s014 <= 1
 0 <= w_k1_ADM_s015 <= 1
 0 <= w_k1_ADM_s016 <= 1
 0 <= w_k1_ADM_s017 <= 1
 0 <= w_k1_ADM_s100 <= 1
 0 <= w_k1_ADM_s101 <= 1
 0 <= w_k1_ADM_s102 <= 1
 0 <= w_k1_ADM_s103 <= 1
 0 <= w_k1_ADM_s104 <= 1
 0 <= w_k1_ADM_s105 <= 1
 0 <= w_k1_ADM_s106 <= 1
 0 <= w_k1_ADM_s107 <= 1
 0 <= w_k1_ADM_s110 <= 1
 0 <= w_k1_ADM_s111 <= 1
 0 <= w_k1_ADM_s112 <= 1
 0 <= w_k1_ADM_s113 <= 1
 0 <= w_k1_ADM_s114 <= 1
 0 <= w_k1_ADM_s115 <= 1
 0 <= w_k1_ADM_s116 <= 1
 0 <= w_k1_ADM_s117 <= 1
 0 <= w_k1_AXP_s0 <= 1
 0 <= w_k1_AXP_s1 <= 1
 0 <= w_k1_AXP_s00 <= 1
 0 <= w_k1_AXP_s01 <= 1
 0 <= w_k1_AXP_s10 <= 1
 0 <= w_k1_AXP_s11 <= 1
 0 <= w_k1_AXP_s000 <= 1
 0 <= w_k1_AXP_s001 <= 1
 0 <= w_k1_AXP_s002 <= 1
 0 <= w_k1_AXP_s003 <= 1
 0 <= w_k1_AXP_s004 <= 1
 0 <= w_k1_AXP_s005 <= 1
 0 <= w_k1_AXP_s006 <= 1
 0 <= w_k1_AXP_s007 <= 1
 0 <= w_k1_AXP_s010 <= 1
 0 <= w_k1_AXP_s011 <= 1
 0 <= w_k1_AXP_s012 <= 1
 0 <= w_k1_AXP_s013 <= 1
 0 <= w_k1_AXP_s014 <= 1
 0 <= w_k1_AXP_s015 <= 1
 0 <= w_k1_AXP_s016 <= 1
 0 <= w_k1_AXP_s017 <= 1
 0 <= w_k1_AXP_s100 <= 1
 0 <= w_k1_AXP_s101 <= 1
 0 <= w_k1_AXP_s102 <= 1
 0 <= w_k1_AXP_s103 <= 1
 0 <= w_k1_AXP_s104 <= 1
 0 <= w_k1_AXP_s105 <= 1
 0 <= w_k1_AXP_s106 <= 1
 0 <= w_k1_AXP_s107 <= 1
 0 <= w_k1_AXP_s110 <= 1
 0 <= w_k1_AXP_s111 <= 1
 0 <= w_k1_AXP_s112 <= 1
 0 <= w_k1_AXP_s113 <= 1
 0 <= w_k1_AXP_s114 <= 1
 0 <= w_k1_AXP_s115 <= 1
 0 <= w_k1_AXP_s116 <= 1
 0 <= w_k1_AXP_s117 <= 1
 0 <= w_k1_DIS_s0 <= 1
 0 <= w_k1_DIS_s1 <= 1
 0 <= w_k1_DIS_s00 <= 1
 0 <= w_k1_DIS_s01 <= 1
 0 <= w_k1_DIS_s10 <= 1
 0 <= w_k1_DIS_s11 <= 1
 0 <= w_k1_DIS_s000 <= 1
 0 <= w_k1_DIS_s001 <= 1
 0 <= w_k1_DIS_s002 <= 1
 0 <= w_k1_DIS_s003 <= 1
 0 <= w_k1_DIS_s004 <= 1
 0 <= w_k1_DIS_s005 <= 1
 0 <= w_k1_DIS_s006 <= 1
 0 <= w_k1_DIS_s007 <= 1
 0 <= w_k1_DIS_s010 <= 1
 0 <= w_k1_DIS_s011 <= 1
 0 <= w_k1_DIS_s012 <= 1
 0 <= w_k1_DIS_s013 <= 1
 0 <= w_k1_DIS_s014 <= 1
 0 <= w_k1_DIS_s015 <= 1
 0 <= w_k1_DIS_s016 <= 1
 0 <= w_k1_DIS_s017 <= 1
 0 <= w_k1_DIS_s100 <= 1
 0 <= w_k1_DIS_s101 <= 1
 0 <= w_k1_DIS_s102 <= 1
 0 <= w_k1_DIS_s103 <= 1
 0 <= w_k1_DIS_s104 <= 1
 0 <= w_k1_DIS_s105 <= 1
 0 <= w_k1_DIS_s106 <= 1
 0 <= w_k1_DIS_s107 <= 1
 0 <= w_k1_DIS_s110 <= 1
 0 <= w_k1_DIS_s111 <= 1
 0 <= w_k1_DIS_s112 <= 1
 0 <= w_k1_DIS_s113 <= 1
 0 <= w_k1_DIS_s114 <= 1
 0 <= w_k1_DIS_s115 <= 1
 0 <= w_k1_DIS_s116 <= 1
 0 <= w_k1_DIS_s117 <= 1
 0 <= w_k1_FDX_s0 <= 1
 0 <= w_k1_FDX_s1 <= 1
 0 <= w_k1_FDX_s00 <= 1
 0 <= w_k1_FDX_s01 <= 1
 0 <= w_k1_FDX_s10 <= 1
 0 <= w_k1_FDX_s11 <= 1
 0 <= w_k1_FDX_s000 <= 1
 0 <= w_k1_FDX_s001 <= 1
 0 <= w_k1_FDX_s002 <= 1
 0 <= w_k1_FDX_s003 <= 1
 0 <= w_k1_FDX_s004 <= 1
 0 <= w_k1_FDX_s005 <= 1
 0 <= w_k1_FDX_s006 <= 1
 0 <= w_k1_FDX_s007 <= 1
 0 <= w_k1_FDX_s010 <= 1
 0 <= w_k1_FDX_s011 <= 1
 0 <= w_k1_FDX_s012 <= 1
 0 <= w_k1_FDX_s013 <= 1
 0 <= w_k1_FDX_s014 <= 1
 0 <= w_k1_FDX_s015 <= 1
 0 <= w_k1_FDX_s016 <= 1
 0 <= w_k1_FDX_s017 <= 1
 0 <= w_k1_FDX_s100 <= 1
 0 <= w_k1_FDX_s101 <= 1
 0 <= w_k1_FDX_s102 <= 1
 0 <= w_k1_FDX_s103 <= 1
 0 <= w_k1_FDX_s104 <= 1
 0 <= w_k1_FDX_s105 <= 1
 0 <= w_k1_FDX_s106 <= 1
 0 <= w_k1_FDX_s107 <= 1
 0 <= w_k1_FDX_s110 <= 1
 0 <= w_k1_FDX_s111 <= 1
 0 <= w_k1_FDX_s112 <= 1
 0 <= w_k1_FDX_s113 <= 1
 0 <= w_k1_FDX_s114 <= 1
 0 <= w_k1_FDX_s115 <= 1
 0 <= w_k1_FDX_s116 <= 1
 0 <= w_k1_FDX_s117 <= 1
 0 <= w_k1_IBM_s0 <= 1
 0 <= w_k1_IBM_s1 <= 1
 0 <= w_k1_IBM_s00 <= 1
 0 <= w_k1_IBM_s01 <= 1
 0 <= w_k1_IBM_s10 <= 1
 0 <= w_k1_IBM_s11 <= 1
 0 <= w_k1_IBM_s000 <= 1
 0 <= w_k1_IBM_s001 <= 1
 0 <= w_k1_IBM_s002 <= 1
 0 <= w_k1_IBM_s003 <= 1
 0 <= w_k1_IBM_s004 <= 1
 0 <= w_k1_IBM_s005 <= 1
 0 <= w_k1_IBM_s006 <= 1
 0 <= w_k1_IBM_s007 <= 1
 0 <= w_k1_IBM_s010 <= 1
 0 <= w_k1_IBM_s011 <= 1
 0 <= w_k1_IBM_s012 <= 1
 0 <= w_k1_IBM_s013 <= 1
 0 <= w_k1_IBM_s014 <= 1
 0 <= w_k1_IBM_s015 <= 1
 0 <= w_k1_IBM_s016 <= 1
 0 <= w_k1_IBM_s017 <= 1
 0 <= w_k1_IBM_s100 <= 1
 0 <= w_k1_IBM_s101 <= 1
 0 <= w_k1_IBM_s102 <= 1
 0 <= w_k1_IBM_s103 <= 1
 0 <= w_k1_IBM_s104 <= 1
 0 <= w_k1_IBM_s105 <= 1
 0 <= w_k1_IBM_s106 <= 1
 0 <= w_k1_IBM_s107 <= 1
 0 <= w_k1_IBM_s110 <= 1
 0 <= w_k1_IBM_s111 <= 1
 0 <= w_k1_IBM_s112 <= 1
 0 <= w_k1_IBM_s113 <= 1
 0 <= w_k1_IBM_s114 <= 1
 0 <= w_k1_IBM_s115 <= 1
 0 <= w_k1_IBM_s116 <= 1
 0 <= w_k1_IBM_s117 <= 1
 0 <= w_k1_INTC_s0 <= 1
 0 <= w_k1_INTC_s1 <= 1
 0 <= w_k1_INTC_s00 <= 1
 0 <= w_k1_INTC_s01 <= 1
 0 <= w_k1_INTC_s10 <= 1
 0 <= w_k1_INTC_s11 <= 1
 0 <= w_k1_INTC_s000 <= 1
 0 <= w_k1_INTC_s001 <= 1
 0 <= w_k1_INTC_s002 <= 1
 0 <= w_k1_INTC_s003 <= 1
 0 <= w_k1_INTC_s004 <= 1
 0 <= w_k1_INTC_s005 <= 1
 0 <= w_k1_INTC_s006 <= 1
 0 <= w_k1_INTC_s007 <= 1
 0 <= w_k1_INTC_s010 <= 1
 0 <= w_k1_INTC_s011 <= 1
 0 <= w_k1_INTC_s012 <= 1
 0 <= w_k1_INTC_s013 <= 1
 0 <= w_k1_INTC_s014 <= 1
 0 <= w_k1_INTC_s015 <= 1
 0 <= w_k1_INTC_s016 <= 1
 0 <= w_k1_INTC_s017 <= 1
 0 <= w_k1_INTC_s100 <= 1
 0 <= w_k1_INTC_s101 <= 1
 0 <= w_k1_INTC_s102 <= 1
 0 <= w_k1_INTC_s103 <= 1
 0 <= w_k1_INTC_s104 <= 1
 0 <= w_k1_INTC_s105 <= 1
 0 <= w_k1_INTC_s106 <= 1
 0 <= w_k1_INTC_s107 <= 1
 0 <= w_k1_INTC_s110 <= 1
 0 <= w_k1_INTC_s111 <= 1
 0 <= w_k1_INTC_s112 <= 1
 0 <= w_k1_INTC_s113 <= 1
 0 <= w_k1_INTC_s114 <= 1
 0 <= w_k1_INTC_s115 <= 1
 0 <= w_k1_INTC_s116 <= 1
 0 <= w_k1_INTC_s117 <= 1
 0 <= w_k1_KO_s0 <= 1
 0 <= w_k1_KO_s1 <= 1
 0 <= w_k1_KO_s00 <= 1
 0 <= w_k1_KO_s01 <= 1
 0 <= w_k1_KO_s10 <= 1
 0 <= w_k1_KO_s11 <= 1
 0 <= w_k1_KO_s000 <= 1
 0 <= w_k1_KO_s001 <= 1
 0 <= w_k1_KO_s002 <= 1
 0 <= w_k1_KO_s003 <= 1
 0 <= w_k1_KO_s004 <= 1
 0 <= w_k1_KO_s005 <= 1
 0 <= w_k1_KO_s006 <= 1
 0 <= w_k1_KO_s007 <= 1
 0 <= w_k1_KO_s010 <= 1
 0 <= w_k1_KO_s011 <= 1
 0 <= w_k1_KO_s012 <= 1
 0 <= w_k1_KO_s013 <= 1
 0 <= w_k1_KO_s014 <= 1
 0 <= w_k1_KO_s015 <= 1
 0 <= w_k1_KO_s016 <= 1
 0 <= w_k1_KO_s017 <= 1
 0 <= w_k1_KO_s100 <= 1
 0 <= w_k1_KO_s101 <= 1
 0 <= w_k1_KO_s102 <= 1
 0 <= w_k1_KO_s103 <= 1
 0 <= w_k1_KO_s104 <= 1
 0 <= w_k1_KO_s105 <= 1
 0 <= w_k1_KO_s106 <= 1
 0 <= w_k1_KO_s107 <= 1
 0 <= w_k1_KO_s110 <= 1
 0 <= w_k1_KO_s111 <= 1
 0 <= w_k1_KO_s112 <= 1
 0 <= w_k1_KO_s113 <= 1
 0 <= w_k1_KO_s114 <= 1
 0 <= w_k1_KO_s115 <= 1
 0 <= w_k1_KO_s116 <= 1
 0 <= w_k1_KO_s117 <= 1
 0 <= w_k1_MAT_s0 <= 1
 0 <= w_k1_MAT_s1 <= 1
 0 <= w_k1_MAT_s00 <= 1
 0 <= w_k1_MAT_s01 <= 1
 0 <= w_k1_MAT_s10 <= 1
 0 <= w_k1_MAT_s11 <= 1
 0 <= w_k1_MAT_s000 <= 1
 0 <= w_k1_MAT_s001 <= 1
 0 <= w_k1_MAT_s002 <= 1
 0 <= w_k1_MAT_s003 <= 1
 0 <= w_k1_MAT_s004 <= 1
 0 <= w_k1_MAT_s005 <= 1
 0 <= w_k1_MAT_s006 <= 1
 0 <= w_k1_MAT_s007 <= 1
 0 <= w_k1_MAT_s010 <= 1
 0 <= w_k1_MAT_s011 <= 1
 0 <= w_k1_MAT_s012 <= 1
 0 <= w_k1_MAT_s013 <= 1
 0 <= w_k1_MAT_s014 <= 1
 0 <= w_k1_MAT_s015 <= 1
 0 <= w_k1_MAT_s016 <= 1
 0 <= w_k1_MAT_s017 <= 1
 0 <= w_k1_MAT_s100 <= 1
 0 <= w_k1_MAT_s101 <= 1
 0 <= w_k1_MAT_s102 <= 1
 0 <= w_k1_MAT_s103 <= 1
 0 <= w_k1_MAT_s104 <= 1
 0 <= w_k1_MAT_s105 <= 1
 0 <= w_k1_MAT_s106 <= 1
 0 <= w_k1_MAT_s107 <= 1
 0 <= w_k1_MAT_s110 <= 1
 0 <= w_k1_MAT_s111 <= 1
 0 <= w_k1_MAT_s112 <= 1
 0 <= w_k1_MAT_s113 <= 1
 0 <= w_k1_MAT_s114 <= 1
 0 <= w_k1_MAT_s115 <= 1
 0 <= w_k1_MAT_s116 <= 1
 0 <= w_k1_MAT_s117 <= 1
 0 <= w_k1_MSFT_s0 <= 1
 0 <= w_k1_MSFT_s1 <= 1
 0 <= w_k1_MSFT_s00 <= 1
 0 <= w_k1_MSFT_s01 <= 1
 0 <= w_k1_MSFT_s10 <= 1
 0 <= w_k1_MSFT_s11 <= 1
 0 <= w_k1_MSFT_s000 <= 1
 0 <= w_k1_MSFT_s001 <= 1
 0 <= w_k1_MSFT_s002 <= 1
 0 <= w_k1_MSFT_s003 <= 1
 0 <= w_k1_MSFT_s004 <= 1
 0 <= w_k1_MSFT_s005 <= 1
 0 <= w_k1_MSFT_s006 <= 1
 0 <= w_k1_MSFT_s007 <= 1
 0 <= w_k1_MSFT_s010 <= 1
 0 <= w_k1_MSFT_s011 <= 1
 0 <= w_k1_MSFT_s012 <= 1
 0 <= w_k1_MSFT_s013 <= 1
 0 <= w_k1_MSFT_s014 <= 1
 0 <= w_k1_MSFT_s015 <= 1
 0 <= w_k1_MSFT_s016 <= 1
 0 <= w_k1_MSFT_s017 <= 1
 0 <= w_k1_MSFT_s100 <= 1
 0 <= w_k1_MSFT_s101 <= 1
 0 <= w_k1_MSFT_s102 <= 1
 0 <= w_k1_MSFT_s103 <= 1
 0 <= w_k1_MSFT_s104 <= 1
 0 <= w_k1_MSFT_s105 <= 1
 0 <= w_k1_MSFT_s106 <= 1
 0 <= w_k1_MSFT_s107 <= 1
 0 <= w_k1_MSFT_s110 <= 1
 0 <= w_k1_MSFT_s111 <= 1
 0 <= w_k1_MSFT_s112 <= 1
 0 <= w_k1_MSFT_s113 <= 1
 0 <= w_k1_MSFT_s114 <= 1
 0 <= w_k1_MSFT_s115 <= 1
 0 <= w_k1_MSFT_s116 <= 1
 0 <= w_k1_MSFT_s117 <= 1
 0 <= w_k1_PG_s0 <= 1
 0 <= w_k1_PG_s1 <= 1
 0 <= w_k1_PG_s00 <= 1
 0 <= w_k1_PG_s01 <= 1
 0 <= w_k1_PG_s10 <= 1
 0 <= w_k1_PG_s11 <= 1
 0 <= w_k1_PG_s000 <= 1
 0 <= w_k1_PG_s001 <= 1
 0 <= w_k1_PG_s002 <= 1
 0 <= w_k1_PG_s003 <= 1
 0 <= w_k1_PG_s004 <= 1
 0 <= w_k1_PG_s005 <= 1
 0 <= w_k1_PG_s006 <= 1
 0 <= w_k1_PG_s007 <= 1
 0 <= w_k1_PG_s010 <= 1
 0 <= w_k1_PG_s011 <= 1
 0 <= w_k1_PG_s012 <= 1
 0 <= w_k1_PG_s013 <= 1
 0 <= w_k1_PG_s014 <= 1
 0 <= w_k1_PG_s015 <= 1
 0 <= w_k1_PG_s016 <= 1
 0 <= w_k1_PG_s017 <= 1
 0 <= w_k1_PG_s100 <= 1
 0 <= w_k1_PG_s101 <= 1
 0 <= w_k1_PG_s102 <= 1
 0 <= w_k1_PG_s103 <= 1
 0 <= w_k1_PG_s104 <= 1
 0 <= w_k1_PG_s105 <= 1
 0 <= w_k1_PG_s106 <= 1
 0 <= w_k1_PG_s107 <= 1
 0 <= w_k1_PG_s110 <= 1
 0 <= w_k1_PG_s111 <= 1
 0 <= w_k1_PG_s112 <= 1
 0 <= w_k1_PG_s113 <= 1
 0 <= w_k1_PG_s114 <= 1
 0 <= w_k1_PG_s115 <= 1
 0 <= w_k1_PG_s116 <= 1
 0 <= w_k1_PG_s117 <= 1
 0 <= b_0_ADM <= 0.2
 0 <= b_0_AXP <= 0.2
 0 <= b_0_DIS <= 0.2
 0 <= b_0_FDX <= 0.2
 0 <= b_0_IBM <= 0.2
 0 <= b_0_INTC <= 0.2
 0 <= b_0_KO <= 0.2
 0 <= b_0_MAT <= 0.2
 0 <= b_0_MSFT <= 0.2
 0 <= b_0_PG <= 0.2
 0 <= b_k1_ADM_s0 <= 0.2
 0 <= b_k1_ADM_s1 <= 0.2
 0 <= b_k1_ADM_s00 <= 0.2
 0 <= b_k1_ADM_s01 <= 0.2
 0 <= b_k1_ADM_s10 <= 0.2
 0 <= b_k1_ADM_s11 <= 0.2
 0 <= b_k1_ADM_s000 <= 0.2
 0 <= b_k1_ADM_s001 <= 0.2
 0 <= b_k1_ADM_s002 <= 0.2
 0 <= b_k1_ADM_s003 <= 0.2
 0 <= b_k1_ADM_s004 <= 0.2
 0 <= b_k1_ADM_s005 <= 0.2
 0 <= b_k1_ADM_s006 <= 0.2
 0 <= b_k1_ADM_s007 <= 0.2
 0 <= b_k1_ADM_s010 <= 0.2
 0 <= b_k1_ADM_s011 <= 0.2
 0 <= b_k1_ADM_s012 <= 0.2
 0 <= b_k1_ADM_s013 <= 0.2
 0 <= b_k1_ADM_s014 <= 0.2
 0 <= b_k1_ADM_s015 <= 0.2
 0 <= b_k1_ADM_s016 <= 0.2
 0 <= b_k1_ADM_s017 <= 0.2
 0 <= b_k1_ADM_s100 <= 0.2
 0 <= b_k1_ADM_s101 <= 0.2
 0 <= b_k1_ADM_s102 <= 0.2
 0 <= b_k1_ADM_s103 <= 0.2
 0 <= b_k1_ADM_s104 <= 0.2
 0 <= b_k1_ADM_s105 <= 0.2
 0 <= b_k1_ADM_s106 <= 0.2
 0 <= b_k1_ADM_s107 <= 0.2
 0 <= b_k1_ADM_s110 <= 0.2
 0 <= b_k1_ADM_s111 <= 0.2
 0 <= b_k1_ADM_s112 <= 0.2
 0 <= b_k1_ADM_s113 <= 0.2
 0 <= b_k1_ADM_s114 <= 0.2
 0 <= b_k1_ADM_s115 <= 0.2
 0 <= b_k1_ADM_s116 <= 0.2
 0 <= b_k1_ADM_s117 <= 0.2
 0 <= b_k1_AXP_s0 <= 0.2
 0 <= b_k1_AXP_s1 <= 0.2
 0 <= b_k1_AXP_s00 <= 0.2
 0 <= b_k1_AXP_s01 <= 0.2
 0 <= b_k1_AXP_s10 <= 0.2
 0 <= b_k1_AXP_s11 <= 0.2
 0 <= b_k1_AXP_s000 <= 0.2
 0 <= b_k1_AXP_s001 <= 0.2
 0 <= b_k1_AXP_s002 <= 0.2
 0 <= b_k1_AXP_s003 <= 0.2
 0 <= b_k1_AXP_s004 <= 0.2
 0 <= b_k1_AXP_s005 <= 0.2
 0 <= b_k1_AXP_s006 <= 0.2
 0 <= b_k1_AXP_s007 <= 0.2
 0 <= b_k1_AXP_s010 <= 0.2
 0 <= b_k1_AXP_s011 <= 0.2
 0 <= b_k1_AXP_s012 <= 0.2
 0 <= b_k1_AXP_s013 <= 0.2
 0 <= b_k1_AXP_s014 <= 0.2
 0 <= b_k1_AXP_s015 <= 0.2
 0 <= b_k1_AXP_s016 <= 0.2
 0 <= b_k1_AXP_s017 <= 0.2
 0 <= b_k1_AXP_s100 <= 0.2
 0 <= b_k1_AXP_s101 <= 0.2
 0 <= b_k1_AXP_s102 <= 0.2
 0 <= b_k1_AXP_s103 <= 0.2
 0 <= b_k1_AXP_s104 <= 0.2
 0 <= b_k1_AXP_s105 <= 0.2
 0 <= b_k1_AXP_s106 <= 0.2
 0 <= b_k1_AXP_s107 <= 0.2
 0 <= b_k1_AXP_s110 <= 0.2
 0 <= b_k1_AXP_s111 <= 0.2
 0 <= b_k1_AXP_s112 <= 0.2
 0 <= b_k1_AXP_s113 <= 0.2
 0 <= b_k1_AXP_s114 <= 0.2
 0 <= b_k1_AXP_s115 <= 0.2
 0 <= b_k1_AXP_s116 <= 0.2
 0 <= b_k1_AXP_s117 <= 0.2
 0 <= b_k1_DIS_s0 <= 0.2
 0 <= b_k1_DIS_s1 <= 0.2
 0 <= b_k1_DIS_s00 <= 0.2
 0 <= b_k1_DIS_s01 <= 0.2
 0 <= b_k1_DIS_s10 <= 0.2
 0 <= b_k1_DIS_s11 <= 0.2
 0 <= b_k1_DIS_s000 <= 0.2
 0 <= b_k1_DIS_s001 <= 0.2
 0 <= b_k1_DIS_s002 <= 0.2
 0 <= b_k1_DIS_s003 <= 0.2
 0 <= b_k1_DIS_s004 <= 0.2
 0 <= b_k1_DIS_s005 <= 0.2
 0 <= b_k1_DIS_s006 <= 0.2
 0 <= b_k1_DIS_s007 <= 0.2
 0 <= b_k1_DIS_s010 <= 0.2
 0 <= b_k1_DIS_s011 <= 0.2
 0 <= b_k1_DIS_s012 <= 0.2
 0 <= b_k1_DIS_s013 <= 0.2
 0 <= b_k1_DIS_s014 <= 0.2
 0 <= b_k1_DIS_s015 <= 0.2
 0 <= b_k1_DIS_s016 <= 0.2
 0 <= b_k1_DIS_s017 <= 0.2
 0 <= b_k1_DIS_s100 <= 0.2
 0 <= b_k1_DIS_s101 <= 0.2
 0 <= b_k1_DIS_s102 <= 0.2
 0 <= b_k1_DIS_s103 <= 0.2
 0 <= b_k1_DIS_s104 <= 0.2
 0 <= b_k1_DIS_s105 <= 0.2
 0 <= b_k1_DIS_s106 <= 0.2
 0 <= b_k1_DIS_s107 <= 0.2
 0 <= b_k1_DIS_s110 <= 0.2
 0 <= b_k1_DIS_s111 <= 0.2
 0 <= b_k1_DIS_s112 <= 0.2
 0 <= b_k1_DIS_s113 <= 0.2
 0 <= b_k1_DIS_s114 <= 0.2
 0 <= b_k1_DIS_s115 <= 0.2
 0 <= b_k1_DIS_s116 <= 0.2
 0 <= b_k1_DIS_s117 <= 0.2
 0 <= b_k1_FDX_s0 <= 0.2
 0 <= b_k1_FDX_s1 <= 0.2
 0 <= b_k1_FDX_s00 <= 0.2
 0 <= b_k1_FDX_s01 <= 0.2
 0 <= b_k1_FDX_s10 <= 0.2
 0 <= b_k1_FDX_s11 <= 0.2
 0 <= b_k1_FDX_s000 <= 0.2
 0 <= b_k1_FDX_s001 <= 0.2
 0 <= b_k1_FDX_s002 <= 0.2
 0 <= b_k1_FDX_s003 <= 0.2
 0 <= b_k1_FDX_s004 <= 0.2
 0 <= b_k1_FDX_s005 <= 0.2
 0 <= b_k1_FDX_s006 <= 0.2
 0 <= b_k1_FDX_s007 <= 0.2
 0 <= b_k1_FDX_s010 <= 0.2
 0 <= b_k1_FDX_s011 <= 0.2
 0 <= b_k1_FDX_s012 <= 0.2
 0 <= b_k1_FDX_s013 <= 0.2
 0 <= b_k1_FDX_s014 <= 0.2
 0 <= b_k1_FDX_s015 <= 0.2
 0 <= b_k1_FDX_s016 <= 0.2
 0 <= b_k1_FDX_s017 <= 0.2
 0 <= b_k1_FDX_s100 <= 0.2
 0 <= b_k1_FDX_s101 <= 0.2
 0 <= b_k1_FDX_s102 <= 0.2
 0 <= b_k1_FDX_s103 <= 0.2
 0 <= b_k1_FDX_s104 <= 0.2
 0 <= b_k1_FDX_s105 <= 0.2
 0 <= b_k1_FDX_s106 <= 0.2
 0 <= b_k1_FDX_s107 <= 0.2
 0 <= b_k1_FDX_s110 <= 0.2
 0 <= b_k1_FDX_s111 <= 0.2
 0 <= b_k1_FDX_s112 <= 0.2
 0 <= b_k1_FDX_s113 <= 0.2
 0 <= b_k1_FDX_s114 <= 0.2
 0 <= b_k1_FDX_s115 <= 0.2
 0 <= b_k1_FDX_s116 <= 0.2
 0 <= b_k1_FDX_s117 <= 0.2
 0 <= b_k1_IBM_s0 <= 0.2
 0 <= b_k1_IBM_s1 <= 0.2
 0 <= b_k1_IBM_s00 <= 0.2
 0 <= b_k1_IBM_s01 <= 0.2
 0 <= b_k1_IBM_s10 <= 0.2
 0 <= b_k1_IBM_s11 <= 0.2
 0 <= b_k1_IBM_s000 <= 0.2
 0 <= b_k1_IBM_s001 <= 0.2
 0 <= b_k1_IBM_s002 <= 0.2
 0 <= b_k1_IBM_s003 <= 0.2
 0 <= b_k1_IBM_s004 <= 0.2
 0 <= b_k1_IBM_s005 <= 0.2
 0 <= b_k1_IBM_s006 <= 0.2
 0 <= b_k1_IBM_s007 <= 0.2
 0 <= b_k1_IBM_s010 <= 0.2
 0 <= b_k1_IBM_s011 <= 0.2
 0 <= b_k1_IBM_s012 <= 0.2
 0 <= b_k1_IBM_s013 <= 0.2
 0 <= b_k1_IBM_s014 <= 0.2
 0 <= b_k1_IBM_s015 <= 0.2
 0 <= b_k1_IBM_s016 <= 0.2
 0 <= b_k1_IBM_s017 <= 0.2
 0 <= b_k1_IBM_s100 <= 0.2
 0 <= b_k1_IBM_s101 <= 0.2
 0 <= b_k1_IBM_s102 <= 0.2
 0 <= b_k1_IBM_s103 <= 0.2
 0 <= b_k1_IBM_s104 <= 0.2
 0 <= b_k1_IBM_s105 <= 0.2
 0 <= b_k1_IBM_s106 <= 0.2
 0 <= b_k1_IBM_s107 <= 0.2
 0 <= b_k1_IBM_s110 <= 0.2
 0 <= b_k1_IBM_s111 <= 0.2
 0 <= b_k1_IBM_s112 <= 0.2
 0 <= b_k1_IBM_s113 <= 0.2
 0 <= b_k1_IBM_s114 <= 0.2
 0 <= b_k1_IBM_s115 <= 0.2
 0 <= b_k1_IBM_s116 <= 0.2
 0 <= b_k1_IBM_s117 <= 0.2
 0 <= b_k1_INTC_s0 <= 0.2
 0 <= b_k1_INTC_s1 <= 0.2
 0 <= b_k1_INTC_s00 <= 0.2
 0 <= b_k1_INTC_s01 <= 0.2
 0 <= b_k1_INTC_s10 <= 0.2
 0 <= b_k1_INTC_s11 <= 0.2
 0 <= b_k1_INTC_s000 <= 0.2
 0 <= b_k1_INTC_s001 <= 0.2
 0 <= b_k1_INTC_s002 <= 0.2
 0 <= b_k1_INTC_s003 <= 0.2
 0 <= b_k1_INTC_s004 <= 0.2
 0 <= b_k1_INTC_s005 <= 0.2
 0 <= b_k1_INTC_s006 <= 0.2
 0 <= b_k1_INTC_s007 <= 0.2
 0 <= b_k1_INTC_s010 <= 0.2
 0 <= b_k1_INTC_s011 <= 0.2
 0 <= b_k1_INTC_s012 <= 0.2
 0 <= b_k1_INTC_s013 <= 0.2
 0 <= b_k1_INTC_s014 <= 0.2
 0 <= b_k1_INTC_s015 <= 0.2
 0 <= b_k1_INTC_s016 <= 0.2
 0 <= b_k1_INTC_s017 <= 0.2
 0 <= b_k1_INTC_s100 <= 0.2
 0 <= b_k1_INTC_s101 <= 0.2
 0 <= b_k1_INTC_s102 <= 0.2
 0 <= b_k1_INTC_s103 <= 0.2
 0 <= b_k1_INTC_s104 <= 0.2
 0 <= b_k1_INTC_s105 <= 0.2
 0 <= b_k1_INTC_s106 <= 0.2
 0 <= b_k1_INTC_s107 <= 0.2
 0 <= b_k1_INTC_s110 <= 0.2
 0 <= b_k1_INTC_s111 <= 0.2
 0 <= b_k1_INTC_s112 <= 0.2
 0 <= b_k1_INTC_s113 <= 0.2
 0 <= b_k1_INTC_s114 <= 0.2
 0 <= b_k1_INTC_s115 <= 0.2
 0 <= b_k1_INTC_s116 <= 0.2
 0 <= b_k1_INTC_s117 <= 0.2
 0 <= b_k1_KO_s0 <= 0.2
 0 <= b_k1_KO_s1 <= 0.2
 0 <= b_k1_KO_s00 <= 0.2
 0 <= b_k1_KO_s01 <= 0.2
 0 <= b_k1_KO_s10 <= 0.2
 0 <= b_k1_KO_s11 <= 0.2
 0 <= b_k1_KO_s000 <= 0.2
 0 <= b_k1_KO_s001 <= 0.2
 0 <= b_k1_KO_s002 <= 0.2
 0 <= b_k1_KO_s003 <= 0.2
 0 <= b_k1_KO_s004 <= 0.2
 0 <= b_k1_KO_s005 <= 0.2
 0 <= b_k1_KO_s006 <= 0.2
 0 <= b_k1_KO_s007 <= 0.2
 0 <= b_k1_KO_s010 <= 0.2
 0 <= b_k1_KO_s011 <= 0.2
 0 <= b_k1_KO_s012 <= 0.2
 0 <= b_k1_KO_s013 <= 0.2
 0 <= b_k1_KO_s014 <= 0.2
 0 <= b_k1_KO_s015 <= 0.2
 0 <= b_k1_KO_s016 <= 0.2
 0 <= b_k1_KO_s017 <= 0.2
 0 <= b_k1_KO_s100 <= 0.2
 0 <= b_k1_KO_s101 <= 0.2
 0 <= b_k1_KO_s102 <= 0.2
 0 <= b_k1_KO_s103 <= 0.2
 0 <= b_k1_KO_s104 <= 0.2
 0 <= b_k1_KO_s105 <= 0.2
 0 <= b_k1_KO_s106 <= 0.2
 0 <= b_k1_KO_s107 <= 0.2
 0 <= b_k1_KO_s110 <= 0.2
 0 <= b_k1_KO_s111 <= 0.2
 0 <= b_k1_KO_s112 <= 0.2
 0 <= b_k1_KO_s113 <= 0.2
 0 <= b_k1_KO_s114 <= 0.2
 0 <= b_k1_KO_s115 <= 0.2
 0 <= b_k1_KO_s116 <= 0.2
 0 <= b_k1_KO_s117 <= 0.2
 0 <= b_k1_MAT_s0 <= 0.2
 0 <= b_k1_MAT_s1 <= 0.2
 0 <= b_k1_MAT_s00 <= 0.2
 0 <= b_k1_MAT_s01 <= 0.2
 0 <= b_k1_MAT_s10 <= 0.2
 0 <= b_k1_MAT_s11 <= 0.2
 0 <= b_k1_MAT_s000 <= 0.2
 0 <= b_k1_MAT_s001 <= 0.2
 0 <= b_k1_MAT_s002 <= 0.2
 0 <= b_k1_MAT_s003 <= 0.2
 0 <= b_k1_MAT_s004 <= 0.2
 0 <= b_k1_MAT_s005 <= 0.2
 0 <= b_k1_MAT_s006 <= 0.2
 0 <= b_k1_MAT_s007 <= 0.2
 0 <= b_k1_MAT_s010 <= 0.2
 0 <= b_k1_MAT_s011 <= 0.2
 0 <= b_k1_MAT_s012 <= 0.2
 0 <= b_k1_MAT_s013 <= 0.2
 0 <= b_k1_MAT_s014 <= 0.2
 0 <= b_k1_MAT_s015 <= 0.2
 0 <= b_k1_MAT_s016 <= 0.2
 0 <= b_k1_MAT_s017 <= 0.2
 0 <= b_k1_MAT_s100 <= 0.2
 0 <= b_k1_MAT_s101 <= 0.2
 0 <= b_k1_MAT_s102 <= 0.2
 0 <= b_k1_MAT_s103 <= 0.2
 0 <= b_k1_MAT_s104 <= 0.2
 0 <= b_k1_MAT_s105 <= 0.2
 0 <= b_k1_MAT_s106 <= 0.2
 0 <= b_k1_MAT_s107 <= 0.2
 0 <= b_k1_MAT_s110 <= 0.2
 0 <= b_k1_MAT_s111 <= 0.2
 0 <= b_k1_MAT_s112 <= 0.2
 0 <= b_k1_MAT_s113 <= 0.2
 0 <= b_k1_MAT_s114 <= 0.2
 0 <= b_k1_MAT_s115 <= 0.2
 0 <= b_k1_MAT_s116 <= 0.2
 0 <= b_k1_MAT_s117 <= 0.2
 0 <= b_k1_MSFT_s0 <= 0.2
 0 <= b_k1_MSFT_s1 <= 0.2
 0 <= b_k1_MSFT_s00 <= 0.2
 0 <= b_k1_MSFT_s01 <= 0.2
 0 <= b_k1_MSFT_s10 <= 0.2
 0 <= b_k1_MSFT_s11 <= 0.2
 0 <= b_k1_MSFT_s000 <= 0.2
 0 <= b_k1_MSFT_s001 <= 0.2
 0 <= b_k1_MSFT_s002 <= 0.2
 0 <= b_k1_MSFT_s003 <= 0.2
 0 <= b_k1_MSFT_s004 <= 0.2
 0 <= b_k1_MSFT_s005 <= 0.2
 0 <= b_k1_MSFT_s006 <= 0.2
 0 <= b_k1_MSFT_s007 <= 0.2
 0 <= b_k1_MSFT_s010 <= 0.2
 0 <= b_k1_MSFT_s011 <= 0.2
 0 <= b_k1_MSFT_s012 <= 0.2
 0 <= b_k1_MSFT_s013 <= 0.2
 0 <= b_k1_MSFT_s014 <= 0.2
 0 <= b_k1_MSFT_s015 <= 0.2
 0 <= b_k1_MSFT_s016 <= 0.2
 0 <= b_k1_MSFT_s017 <= 0.2
 0 <= b_k1_MSFT_s100 <= 0.2
 0 <= b_k1_MSFT_s101 <= 0.2
 0 <= b_k1_MSFT_s102 <= 0.2
 0 <= b_k1_MSFT_s103 <= 0.2
 0 <= b_k1_MSFT_s104 <= 0.2
 0 <= b_k1_MSFT_s105 <= 0.2
 0 <= b_k1_MSFT_s106 <= 0.2
 0 <= b_k1_MSFT_s107 <= 0.2
 0 <= b_k1_MSFT_s110 <= 0.2
 0 <= b_k1_MSFT_s111 <= 0.2
 0 <= b_k1_MSFT_s112 <= 0.2
 0 <= b_k1_MSFT_s113 <= 0.2
 0 <= b_k1_MSFT_s114 <= 0.2
 0 <= b_k1_MSFT_s115 <= 0.2
 0 <= b_k1_MSFT_s116 <= 0.2
 0 <= b_k1_MSFT_s117 <= 0.2
 0 <= b_k1_PG_s0 <= 0.2
 0 <= b_k1_PG_s1 <= 0.2
 0 <= b_k1_PG_s00 <= 0.2
 0 <= b_k1_PG_s01 <= 0.2
 0 <= b_k1_PG_s10 <= 0.2
 0 <= b_k1_PG_s11 <= 0.2
 0 <= b_k1_PG_s000 <= 0.2
 0 <= b_k1_PG_s001 <= 0.2
 0 <= b_k1_PG_s002 <= 0.2
 0 <= b_k1_PG_s003 <= 0.2
 0 <= b_k1_PG_s004 <= 0.2
 0 <= b_k1_PG_s005 <= 0.2
 0 <= b_k1_PG_s006 <= 0.2
 0 <= b_k1_PG_s007 <= 0.2
 0 <= b_k1_PG_s010 <= 0.2
 0 <= b_k1_PG_s011 <= 0.2
 0 <= b_k1_PG_s012 <= 0.2
 0 <= b_k1_PG_s013 <= 0.2
 0 <= b_k1_PG_s014 <= 0.2
 0 <= b_k1_PG_s015 <= 0.2
 0 <= b_k1_PG_s016 <= 0.2
 0 <= b_k1_PG_s017 <= 0.2
 0 <= b_k1_PG_s100 <= 0.2
 0 <= b_k1_PG_s101 <= 0.2
 0 <= b_k1_PG_s102 <= 0.2
 0 <= b_k1_PG_s103 <= 0.2
 0 <= b_k1_PG_s104 <= 0.2
 0 <= b_k1_PG_s105 <= 0.2
 0 <= b_k1_PG_s106 <= 0.2
 0 <= b_k1_PG_s107 <= 0.2
 0 <= b_k1_PG_s110 <= 0.2
 0 <= b_k1_PG_s111 <= 0.2
 0 <= b_k1_PG_s112 <= 0.2
 0 <= b_k1_PG_s113 <= 0.2
 0 <= b_k1_PG_s114 <= 0.2
 0 <= b_k1_PG_s115 <= 0.2
 0 <= b_k1_PG_s116 <= 0.2
 0 <= b_k1_PG_s117 <= 0.2
 0 <= s_0_ADM <= 0.2
 0 <= s_0_AXP <= 0.2
 0 <= s_0_DIS <= 0.2
 0 <= s_0_FDX <= 0.2
 0 <= s_0_IBM <= 0.2
 0 <= s_0_INTC <= 0.2
 0 <= s_0_KO <= 0.2
 0 <= s_0_MAT <= 0.2
 0 <= s_0_MSFT <= 0.2
 0 <= s_0_PG <= 0.2
 0 <= s_k1_ADM_s0 <= 0.2
 0 <= s_k1_ADM_s1 <= 0.2
 0 <= s_k1_ADM_s00 <= 0.2
 0 <= s_k1_ADM_s01 <= 0.2
 0 <= s_k1_ADM_s10 <= 0.2
 0 <= s_k1_ADM_s11 <= 0.2
 0 <= s_k1_ADM_s000 <= 0.2
 0 <= s_k1_ADM_s001 <= 0.2
 0 <= s_k1_ADM_s002 <= 0.2
 0 <= s_k1_ADM_s003 <= 0.2
 0 <= s_k1_ADM_s004 <= 0.2
 0 <= s_k1_ADM_s005 <= 0.2
 0 <= s_k1_ADM_s006 <= 0.2
 0 <= s_k1_ADM_s007 <= 0.2
 0 <= s_k1_ADM_s010 <= 0.2
 0 <= s_k1_ADM_s011 <= 0.2
 0 <= s_k1_ADM_s012 <= 0.2
 0 <= s_k1_ADM_s013 <= 0.2
 0 <= s_k1_ADM_s014 <= 0.2
 0 <= s_k1_ADM_s015 <= 0.2
 0 <= s_k1_ADM_s016 <= 0.2
 0 <= s_k1_ADM_s017 <= 0.2
 0 <= s_k1_ADM_s100 <= 0.2
 0 <= s_k1_ADM_s101 <= 0.2
 0 <= s_k1_ADM_s102 <= 0.2
 0 <= s_k1_ADM_s103 <= 0.2
 0 <= s_k1_ADM_s104 <= 0.2
 0 <= s_k1_ADM_s105 <= 0.2
 0 <= s_k1_ADM_s106 <= 0.2
 0 <= s_k1_ADM_s107 <= 0.2
 0 <= s_k1_ADM_s110 <= 0.2
 0 <= s_k1_ADM_s111 <= 0.2
 0 <= s_k1_ADM_s112 <= 0.2
 0 <= s_k1_ADM_s113 <= 0.2
 0 <= s_k1_ADM_s114 <= 0.2
 0 <= s_k1_ADM_s115 <= 0.2
 0 <= s_k1_ADM_s116 <= 0.2
 0 <= s_k1_ADM_s117 <= 0.2
 0 <= s_k1_AXP_s0 <= 0.2
 0 <= s_k1_AXP_s1 <= 0.2
 0 <= s_k1_AXP_s00 <= 0.2
 0 <= s_k1_AXP_s01 <= 0.2
 0 <= s_k1_AXP_s10 <= 0.2
 0 <= s_k1_AXP_s11 <= 0.2
 0 <= s_k1_AXP_s000 <= 0.2
 0 <= s_k1_AXP_s001 <= 0.2
 0 <= s_k1_AXP_s002 <= 0.2
 0 <= s_k1_AXP_s003 <= 0.2
 0 <= s_k1_AXP_s004 <= 0.2
 0 <= s_k1_AXP_s005 <= 0.2
 0 <= s_k1_AXP_s006 <= 0.2
 0 <= s_k1_AXP_s007 <= 0.2
 0 <= s_k1_AXP_s010 <= 0.2
 0 <= s_k1_AXP_s011 <= 0.2
 0 <= s_k1_AXP_s012 <= 0.2
 0 <= s_k1_AXP_s013 <= 0.2
 0 <= s_k1_AXP_s014 <= 0.2
 0 <= s_k1_AXP_s015 <= 0.2
 0 <= s_k1_AXP_s016 <= 0.2
 0 <= s_k1_AXP_s017 <= 0.2
 0 <= s_k1_AXP_s100 <= 0.2
 0 <= s_k1_AXP_s101 <= 0.2
 0 <= s_k1_AXP_s102 <= 0.2
 0 <= s_k1_AXP_s103 <= 0.2
 0 <= s_k1_AXP_s104 <= 0.2
 0 <= s_k1_AXP_s105 <= 0.2
 0 <= s_k1_AXP_s106 <= 0.2
 0 <= s_k1_AXP_s107 <= 0.2
 0 <= s_k1_AXP_s110 <= 0.2
 0 <= s_k1_AXP_s111 <= 0.2
 0 <= s_k1_AXP_s112 <= 0.2
 0 <= s_k1_AXP_s113 <= 0.2
 0 <= s_k1_AXP_s114 <= 0.2
 0 <= s_k1_AXP_s115 <= 0.2
 0 <= s_k1_AXP_s116 <= 0.2
 0 <= s_k1_AXP_s117 <= 0.2
 0 <= s_k1_DIS_s0 <= 0.2
 0 <= s_k1_DIS_s1 <= 0.2
 0 <= s_k1_DIS_s00 <= 0.2
 0 <= s_k1_DIS_s01 <= 0.2
 0 <= s_k1_DIS_s10 <= 0.2
 0 <= s_k1_DIS_s11 <= 0.2
 0 <= s_k1_DIS_s000 <= 0.2
 0 <= s_k1_DIS_s001 <= 0.2
 0 <= s_k1_DIS_s002 <= 0.2
 0 <= s_k1_DIS_s003 <= 0.2
 0 <= s_k1_DIS_s004 <= 0.2
 0 <= s_k1_DIS_s005 <= 0.2
 0 <= s_k1_DIS_s006 <= 0.2
 0 <= s_k1_DIS_s007 <= 0.2
 0 <= s_k1_DIS_s010 <= 0.2
 0 <= s_k1_DIS_s011 <= 0.2
 0 <= s_k1_DIS_s012 <= 0.2
 0 <= s_k1_DIS_s013 <= 0.2
 0 <= s_k1_DIS_s014 <= 0.2
 0 <= s_k1_DIS_s015 <= 0.2
 0 <= s_k1_DIS_s016 <= 0.2
 0 <= s_k1_DIS_s017 <= 0.2
 0 <= s_k1_DIS_s100 <= 0.2
 0 <= s_k1_DIS_s101 <= 0.2
 0 <= s_k1_DIS_s102 <= 0.2
 0 <= s_k1_DIS_s103 <= 0.2
 0 <= s_k1_DIS_s104 <= 0.2
 0 <= s_k1_DIS_s105 <= 0.2
 0 <= s_k1_DIS_s106 <= 0.2
 0 <= s_k1_DIS_s107 <= 0.2
 0 <= s_k1_DIS_s110 <= 0.2
 0 <= s_k1_DIS_s111 <= 0.2
 0 <= s_k1_DIS_s112 <= 0.2
 0 <= s_k1_DIS_s113 <= 0.2
 0 <= s_k1_DIS_s114 <= 0.2
 0 <= s_k1_DIS_s115 <= 0.2
 0 <= s_k1_DIS_s116 <= 0.2
 0 <= s_k1_DIS_s117 <= 0.2
 0 <= s_k1_FDX_s0 <= 0.2
 0 <= s_k1_FDX_s1 <= 0.2
 0 <= s_k1_FDX_s00 <= 0.2
 0 <= s_k1_FDX_s01 <= 0.2
 0 <= s_k1_FDX_s10 <= 0.2
 0 <= s_k1_FDX_s11 <= 0.2
 0 <= s_k1_FDX_s000 <= 0.2
 0 <= s_k1_FDX_s001 <= 0.2
 0 <= s_k1_FDX_s002 <= 0.2
 0 <= s_k1_FDX_s003 <= 0.2
 0 <= s_k1_FDX_s004 <= 0.2
 0 <= s_k1_FDX_s005 <= 0.2
 0 <= s_k1_FDX_s006 <= 0.2
 0 <= s_k1_FDX_s007 <= 0.2
 0 <= s_k1_FDX_s010 <= 0.2
 0 <= s_k1_FDX_s011 <= 0.2
 0 <= s_k1_FDX_s012 <= 0.2
 0 <= s_k1_FDX_s013 <= 0.2
 0 <= s_k1_FDX_s014 <= 0.2
 0 <= s_k1_FDX_s015 <= 0.2
 0 <= s_k1_FDX_s016 <= 0.2
 0 <= s_k1_FDX_s017 <= 0.2
 0 <= s_k1_FDX_s100 <= 0.2
 0 <= s_k1_FDX_s101 <= 0.2
 0 <= s_k1_FDX_s102 <= 0.2
 0 <= s_k1_FDX_s103 <= 0.2
 0 <= s_k1_FDX_s104 <= 0.2
 0 <= s_k1_FDX_s105 <= 0.2
 0 <= s_k1_FDX_s106 <= 0.2
 0 <= s_k1_FDX_s107 <= 0.2
 0 <= s_k1_FDX_s110 <= 0.2
 0 <= s_k1_FDX_s111 <= 0.2
 0 <= s_k1_FDX_s112 <= 0.2
 0 <= s_k1_FDX_s113 <= 0.2
 0 <= s_k1_FDX_s114 <= 0.2
 0 <= s_k1_FDX_s115 <= 0.2
 0 <= s_k1_FDX_s116 <= 0.2
 0 <= s_k1_FDX_s117 <= 0.2
 0 <= s_k1_IBM_s0 <= 0.2
 0 <= s_k1_IBM_s1 <= 0.2
 0 <= s_k1_IBM_s00 <= 0.2
 0 <= s_k1_IBM_s01 <= 0.2
 0 <= s_k1_IBM_s10 <= 0.2
 0 <= s_k1_IBM_s11 <= 0.2
 0 <= s_k1_IBM_s000 <= 0.2
 0 <= s_k1_IBM_s001 <= 0.2
 0 <= s_k1_IBM_s002 <= 0.2
 0 <= s_k1_IBM_s003 <= 0.2
 0 <= s_k1_IBM_s004 <= 0.2
 0 <= s_k1_IBM_s005 <= 0.2
 0 <= s_k1_IBM_s006 <= 0.2
 0 <= s_k1_IBM_s007 <= 0.2
 0 <= s_k1_IBM_s010 <= 0.2
 0 <= s_k1_IBM_s011 <= 0.2
 0 <= s_k1_IBM_s012 <= 0.2
 0 <= s_k1_IBM_s013 <= 0.2
 0 <= s_k1_IBM_s014 <= 0.2
 0 <= s_k1_IBM_s015 <= 0.2
 0 <= s_k1_IBM_s016 <= 0.2
 0 <= s_k1_IBM_s017 <= 0.2
 0 <= s_k1_IBM_s100 <= 0.2
 0 <= s_k1_IBM_s101 <= 0.2
 0 <= s_k1_IBM_s102 <= 0.2
 0 <= s_k1_IBM_s103 <= 0.2
 0 <= s_k1_IBM_s104 <= 0.2
 0 <= s_k1_IBM_s105 <= 0.2
 0 <= s_k1_IBM_s106 <= 0.2
 0 <= s_k1_IBM_s107 <= 0.2
 0 <= s_k1_IBM_s110 <= 0.2
 0 <= s_k1_IBM_s111 <= 0.2
 0 <= s_k1_IBM_s112 <= 0.2
 0 <= s_k1_IBM_s113 <= 0.2
 0 <= s_k1_IBM_s114 <= 0.2
 0 <= s_k1_IBM_s115 <= 0.2
 0 <= s_k1_IBM_s116 <= 0.2
 0 <= s_k1_IBM_s117 <= 0.2
 0 <= s_k1_INTC_s0 <= 0.2
 0 <= s_k1_INTC_s1 <= 0.2
 0 <= s_k1_INTC_s00 <= 0.2
 0 <= s_k1_INTC_s01 <= 0.2
 0 <= s_k1_INTC_s10 <= 0.2
 0 <= s_k1_INTC_s11 <= 0.2
 0 <= s_k1_INTC_s000 <= 0.2
 0 <= s_k1_INTC_s001 <= 0.2
 0 <= s_k1_INTC_s002 <= 0.2
 0 <= s_k1_INTC_s003 <= 0.2
 0 <= s_k1_INTC_s004 <= 0.2
 0 <= s_k1_INTC_s005 <= 0.2
 0 <= s_k1_INTC_s006 <= 0.2
 0 <= s_k1_INTC_s007 <= 0.2
 0 <= s_k1_INTC_s010 <= 0.2
 0 <= s_k1_INTC_s011 <= 0.2
 0 <= s_k1_INTC_s012 <= 0.2
 0 <= s_k1_INTC_s013 <= 0.2
 0 <= s_k1_INTC_s014 <= 0.2
 0 <= s_k1_INTC_s015 <= 0.2
 0 <= s_k1_INTC_s016 <= 0.2
 0 <= s_k1_INTC_s017 <= 0.2
 0 <= s_k1_INTC_s100 <= 0.2
 0 <= s_k1_INTC_s101 <= 0.2
 0 <= s_k1_INTC_s102 <= 0.2
 0 <= s_k1_INTC_s103 <= 0.2
 0 <= s_k1_INTC_s104 <= 0.2
 0 <= s_k1_INTC_s105 <= 0.2
 0 <= s_k1_INTC_s106 <= 0.2
 0 <= s_k1_INTC_s107 <= 0.2
 0 <= s_k1_INTC_s110 <= 0.2
 0 <= s_k1_INTC_s111 <= 0.2
 0 <= s_k1_INTC_s112 <= 0.2
 0 <= s_k1_INTC_s113 <= 0.2
 0 <= s_k1_INTC_s114 <= 0.2
 0 <= s_k1_INTC_s115 <= 0.2
 0 <= s_k1_INTC_s116 <= 0.2
 0 <= s_k1_INTC_s117 <= 0.2
 0 <= s_k1_KO_s0 <= 0.2
 0 <= s_k1_KO_s1 <= 0.2
 0 <= s_k1_KO_s00 <= 0.2
 0 <= s_k1_KO_s01 <= 0.2
 0 <= s_k1_KO_s10 <= 0.2
 0 <= s_k1_KO_s11 <= 0.2
 0 <= s_k1_KO_s000 <= 0.2
 0 <= s_k1_KO_s001 <= 0.2
 0 <= s_k1_KO_s002 <= 0.2
 0 <= s_k1_KO_s003 <= 0.2
 0 <= s_k1_KO_s004 <= 0.2
 0 <= s_k1_KO_s005 <= 0.2
 0 <= s_k1_KO_s006 <= 0.2
 0 <= s_k1_KO_s007 <= 0.2
 0 <= s_k1_KO_s010 <= 0.2
 0 <= s_k1_KO_s011 <= 0.2
 0 <= s_k1_KO_s012 <= 0.2
 0 <= s_k1_KO_s013 <= 0.2
 0 <= s_k1_KO_s014 <= 0.2
 0 <= s_k1_KO_s015 <= 0.2
 0 <= s_k1_KO_s016 <= 0.2
 0 <= s_k1_KO_s017 <= 0.2
 0 <= s_k1_KO_s100 <= 0.2
 0 <= s_k1_KO_s101 <= 0.2
 0 <= s_k1_KO_s102 <= 0.2
 0 <= s_k1_KO_s103 <= 0.2
 0 <= s_k1_KO_s104 <= 0.2
 0 <= s_k1_KO_s105 <= 0.2
 0 <= s_k1_KO_s106 <= 0.2
 0 <= s_k1_KO_s107 <= 0.2
 0 <= s_k1_KO_s110 <= 0.2
 0 <= s_k1_KO_s111 <= 0.2
 0 <= s_k1_KO_s112 <= 0.2
 0 <= s_k1_KO_s113 <= 0.2
 0 <= s_k1_KO_s114 <= 0.2
 0 <= s_k1_KO_s115 <= 0.2
 0 <= s_k1_KO_s116 <= 0.2
 0 <= s_k1_KO_s117 <= 0.2
 0 <= s_k1_MAT_s0 <= 0.2
 0 <= s_k1_MAT_s1 <= 0.2
 0 <= s_k1_MAT_s00 <= 0.2
 0 <= s_k1_MAT_s01 <= 0.2
 0 <= s_k1_MAT_s10 <= 0.2
 0 <= s_k1_MAT_s11 <= 0.2
 0 <= s_k1_MAT_s000 <= 0.2
 0 <= s_k1_MAT_s001 <= 0.2
 0 <= s_k1_MAT_s002 <= 0.2
 0 <= s_k1_MAT_s003 <= 0.2
 0 <= s_k1_MAT_s004 <= 0.2
 0 <= s_k1_MAT_s005 <= 0.2
 0 <= s_k1_MAT_s006 <= 0.2
 0 <= s_k1_MAT_s007 <= 0.2
 0 <= s_k1_MAT_s010 <= 0.2
 0 <= s_k1_MAT_s011 <= 0.2
 0 <= s_k1_MAT_s012 <= 0.2
 0 <= s_k1_MAT_s013 <= 0.2
 0 <= s_k1_MAT_s014 <= 0.2
 0 <= s_k1_MAT_s015 <= 0.2
 0 <= s_k1_MAT_s016 <= 0.2
 0 <= s_k1_MAT_s017 <= 0.2
 0 <= s_k1_MAT_s100 <= 0.2
 0 <= s_k1_MAT_s101 <= 0.2
 0 <= s_k1_MAT_s102 <= 0.2
 0 <= s_k1_MAT_s103 <= 0.2
 0 <= s_k1_MAT_s104 <= 0.2
 0 <= s_k1_MAT_s105 <= 0.2
 0 <= s_k1_MAT_s106 <= 0.2
 0 <= s_k1_MAT_s107 <= 0.2
 0 <= s_k1_MAT_s110 <= 0.2
 0 <= s_k1_MAT_s111 <= 0.2
 0 <= s_k1_MAT_s112 <= 0.2
 0 <= s_k1_MAT_s113 <= 0.2
 0 <= s_k1_MAT_s114 <= 0.2
 0 <= s_k1_MAT_s115 <= 0.2
 0 <= s_k1_MAT_s116 <= 0.2
 0 <= s_k1_MAT_s117 <= 0.2
 0 <= s_k1_MSFT_s0 <= 0.2
 0 <= s_k1_MSFT_s1 <= 0.2
 0 <= s_k1_MSFT_s00 <= 0.2
 0 <= s_k1_MSFT_s01 <= 0.2
 0 <= s_k1_MSFT_s10 <= 0.2
 0 <= s_k1_MSFT_s11 <= 0.2
 0 <= s_k1_MSFT_s000 <= 0.2
 0 <= s_k1_MSFT_s001 <= 0.2
 0 <= s_k1_MSFT_s002 <= 0.2
 0 <= s_k1_MSFT_s003 <= 0.2
 0 <= s_k1_MSFT_s004 <= 0.2
 0 <= s_k1_MSFT_s005 <= 0.2
 0 <= s_k1_MSFT_s006 <= 0.2
 0 <= s_k1_MSFT_s007 <= 0.2
 0 <= s_k1_MSFT_s010 <= 0.2
 0 <= s_k1_MSFT_s011 <= 0.2
 0 <= s_k1_MSFT_s012 <= 0.2
 0 <= s_k1_MSFT_s013 <= 0.2
 0 <= s_k1_MSFT_s014 <= 0.2
 0 <= s_k1_MSFT_s015 <= 0.2
 0 <= s_k1_MSFT_s016 <= 0.2
 0 <= s_k1_MSFT_s017 <= 0.2
 0 <= s_k1_MSFT_s100 <= 0.2
 0 <= s_k1_MSFT_s101 <= 0.2
 0 <= s_k1_MSFT_s102 <= 0.2
 0 <= s_k1_MSFT_s103 <= 0.2
 0 <= s_k1_MSFT_s104 <= 0.2
 0 <= s_k1_MSFT_s105 <= 0.2
 0 <= s_k1_MSFT_s106 <= 0.2
 0 <= s_k1_MSFT_s107 <= 0.2
 0 <= s_k1_MSFT_s110 <= 0.2
 0 <= s_k1_MSFT_s111 <= 0.2
 0 <= s_k1_MSFT_s112 <= 0.2
 0 <= s_k1_MSFT_s113 <= 0.2
 0 <= s_k1_MSFT_s114 <= 0.2
 0 <= s_k1_MSFT_s115 <= 0.2
 0 <= s_k1_MSFT_s116 <= 0.2
 0 <= s_k1_MSFT_s117 <= 0.2
 0 <= s_k1_PG_s0 <= 0.2
 0 <= s_k1_PG_s1 <= 0.2
 0 <= s_k1_PG_s00 <= 0.2
 0 <= s_k1_PG_s01 <= 0.2
 0 <= s_k1_PG_s10 <= 0.2
 0 <= s_k1_PG_s11 <= 0.2
 0 <= s_k1_PG_s000 <= 0.2
 0 <= s_k1_PG_s001 <= 0.2
 0 <= s_k1_PG_s002 <= 0.2
 0 <= s_k1_PG_s003 <= 0.2
 0 <= s_k1_PG_s004 <= 0.2
 0 <= s_k1_PG_s005 <= 0.2
 0 <= s_k1_PG_s006 <= 0.2
 0 <= s_k1_PG_s007 <= 0.2
 0 <= s_k1_PG_s010 <= 0.2
 0 <= s_k1_PG_s011 <= 0.2
 0 <= s_k1_PG_s012 <= 0.2
 0 <= s_k1_PG_s013 <= 0.2
 0 <= s_k1_PG_s014 <= 0.2
 0 <= s_k1_PG_s015 <= 0.2
 0 <= s_k1_PG_s016 <= 0.2
 0 <= s_k1_PG_s017 <= 0.2
 0 <= s_k1_PG_s100 <= 0.2
 0 <= s_k1_PG_s101 <= 0.2
 0 <= s_k1_PG_s102 <= 0.2
 0 <= s_k1_PG_s103 <= 0.2
 0 <= s_k1_PG_s104 <= 0.2
 0 <= s_k1_PG_s105 <= 0.2
 0 <= s_k1_PG_s106 <= 0.2
 0 <= s_k1_PG_s107 <= 0.2
 0 <= s_k1_PG_s110 <= 0.2
 0 <= s_k1_PG_s111 <= 0.2
 0 <= s_k1_PG_s112 <= 0.2
 0 <= s_k1_PG_s113 <= 0.2
 0 <= s_k1_PG_s114 <= 0.2
 0 <= s_k1_PG_s115 <= 0.2
 0 <= s_k1_PG_s116 <= 0.2
 0 <= s_k1_PG_s117 <= 0.2
End
