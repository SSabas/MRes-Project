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
 w_k1_ADM_s0_return_constraint:  1.10592 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.92369 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.22877 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.92295 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.968728 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.11794 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.02717 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.831592 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.06003 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.08024 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.959364 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.791931 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.76335 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.940083 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.730276 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.22123 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.13788 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.20096 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.85389 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.912553 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.00089 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.897889 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.725767 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.00944 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.996303 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.12595 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.86104 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.39762 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.922738 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.807743 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.902185 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.917005 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.15552 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.02746 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.20738 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.798695 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.08032 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.00685 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.18656 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.879634 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.15552 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.856396 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.898541 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.28292 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.21215 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.871826 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.68173 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.864155 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.18465 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.73358 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.76339 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.19531 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.693008 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.57947 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.30537 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.02123 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.64121 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.892881 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.969354 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.818304 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.749109 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.29169 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.983508 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.17377 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.09989 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.55098 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.07359 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.702266 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.02414 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.905887 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.44295 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.13384 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.46224 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.799958 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.24755 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.03311 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.2158 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.849254 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.2118 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.837708 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.866311 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.42252 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.46863 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.937733 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.89124 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.933842 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.144 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.639977 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.720835 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.11642 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.619635 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.78925 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.23169 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.14746 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.58573 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.916045 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.792785 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.686312 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.711936 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.21808 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.692845 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.10049 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.827286 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.69136 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.12502 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.609691 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.903689 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.776642 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.43569 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.944036 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.63526 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.784359 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.08813 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.35488 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.15782 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.862815 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.1595 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.849463 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.923843 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.25522 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.23587 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.911916 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.37681 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.0998 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.18493 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.817272 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.783978 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.978425 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.748195 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.26883 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.08599 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.15794 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.802378 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.949857 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.989406 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.824183 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.821763 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.07496 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.989737 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.20562 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.921415 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.56998 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.09386 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.703435 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.892137 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.942507 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.18799 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.05163 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.46473 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.735702 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.197 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.11532 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.11278 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.962321 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.975968 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.03915 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.930234 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.31661 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.07843 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.877022 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.73409 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.617777 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.826058 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.60916 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.10646 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.53142 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.699297 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.82815 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.34613 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.780731 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.432445 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.983012 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.582117 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.894404 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.699413 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.55955 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.676929 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.661488 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.11892 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.957458 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.03895 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.1848 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.35749 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.618064 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.86779 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.994813 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.957125 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.00928 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.756884 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.11787 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.08088 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.910783 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.05871 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.87445 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.979809 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.31682 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.4772 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.921293 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.18452 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.09157 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.24281 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.877493 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.991008 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.09242 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.955368 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.16498 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.11378 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.10658 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.881542 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.10901 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.912124 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.841015 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.851139 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.12253 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.899597 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.03059 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.896594 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.14465 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.07094 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.694343 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.975269 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.846513 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.26333 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.871675 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.24685 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.793459 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.915655 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.909564 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.14506 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.902727 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.02851 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.965672 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.924032 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.23679 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.14765 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.993042 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.54106 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.809695 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.00785 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.644564 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.994778 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.2786 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.674292 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.5873 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.23466 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.840982 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.58116 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.918132 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.675023 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.746189 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.756313 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.36661 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.740588 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.822901 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.976639 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.30918 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.09957 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.926919 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.1589 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.778176 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.50894 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.04812 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.34972 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.01441 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.00984 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.38424 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.24052 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.929253 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.07434 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.98126 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.959329 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.44457 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.9028 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.866197 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.61452 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.405772 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.912423 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.431409 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.918736 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.50992 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.603098 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 2.0506 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.13529 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.75837 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.400502 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.829438 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.66529 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.809885 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.523708 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.52975 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.718265 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.82203 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.17256 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.996889 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.845953 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.954171 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.17712 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.524683 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.70887 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.13303 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.97025 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.885904 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.980764 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.01251 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.09573 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.898696 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.16549 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.903477 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.950287 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.16842 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.21941 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.979508 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.3819 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.926885 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.05652 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.775901 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.853626 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.905745 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.806075 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.26129 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.10856 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.05835 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.869797 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.996158 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.833315 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.742772 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.76676 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.01665 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.937002 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.094 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.860375 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.38357 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.09634 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.739318 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.976017 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.993454 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.25794 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.934158 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.25427 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.868083 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.19638 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.24926 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.08063 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.91489 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.10464 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.918475 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.902334 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.10794 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.14763 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.871272 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.2009 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.896633 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.12724 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.754758 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.895375 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.09622 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.80225 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.26457 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.0665 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.10232 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.849984 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.973123 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.918326 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.902249 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.890849 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.18675 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.976771 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.0168 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.01432 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.26183 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.06584 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.936723 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.10404 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.971815 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.40017 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.04982 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.30441 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.04992 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.15429 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.17021 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.616000375962899 z_k1_s0000
                                 - 0.243000328497238 z_k1_s0001
                                 - 0.357000486251501 z_k1_s0002
                                 - 0.488999301670824 z_k1_s0003
                                 - 0.403001549444598 z_k1_s0004
                                 - 0.200000271582913 z_k1_s0005
                                 - 0.403001549444598 z_k1_s0006
                                 - 0.24100034644543 z_k1_s0007
                                 - 0.464999831976658 z_k1_s0010
                                 - 0.511999395446941 z_k1_s0011
                                 - 1.49100149993684 z_k1_s0012
                                 - 1.45899869023651 z_k1_s0013
                                 - 0.447999617040939 z_k1_s0014
                                 - 0.448999595325425 z_k1_s0015
                                 - 0.436999855911599 z_k1_s0016
                                 - 0.579999671090349 z_k1_s0017
                                 - 0.285999793966736 z_k1_s0020
                                 - 0.206999993559768 z_k1_s0021
                                 - 0.457998699822752 z_k1_s0022
                                 - 0.241999825664161 z_k1_s0023
                                 - 0.469999090686886 z_k1_s0024
                                 - 0.441999225003371 z_k1_s0025
                                 - 0.350999661531947 z_k1_s0026
                                 - 0.682998965778881 z_k1_s0027
                                 - 0.0950000053791972 z_k1_s0030
                                 - 0.123999964652833 z_k1_s0031
                                 - 0.245000078346371 z_k1_s0032
                                 - 0.132000092211122 z_k1_s0033
                                 - 0.160000101164993 z_k1_s0034
                                 - 0.273000087300242 z_k1_s0035
                                 - 0.413000132069597 z_k1_s0036
                                 - 0.308000098492581 z_k1_s0037
                                 - 0.378000071889818 z_k1_s0040
                                 - 0.212999510259544 z_k1_s0041
                                 - 0.479997979276752 z_k1_s0042
                                 - 0.363997972194134 z_k1_s0043
                                 - 0.229999633323147 z_k1_s0044
                                 - 0.403000585923902 z_k1_s0045
                                 - 0.546999130408523 z_k1_s0046
                                 - 0.215999515323741 z_k1_s0047
                                 - 1.217998011481 z_k1_s0050
                                 - 0.478997862896174 z_k1_s0051
                                 - 0.310999884912438 z_k1_s0052
                                 - 0.528001442691073 z_k1_s0053
                                 - 0.345999842709017 z_k1_s0054
                                 - 0.722001382043526 z_k1_s0055
                                 - 0.359000173593431 z_k1_s0056
                                 - 0.369000161535311 z_k1_s0057
                                 - 0.241999283523624 z_k1_s0060
                                 - 0.264998960840368 z_k1_s0061
                                 - 0.566996976991721 z_k1_s0062
                                 - 0.343999029124329 z_k1_s0063
                                 - 0.533997990735397 z_k1_s0064
                                 - 0.332998407375471 z_k1_s0065
                                 - 0.308000135026522 z_k1_s0066
                                 - 0.285998891528806 z_k1_s0067
                                 - 0.196999520463135 z_k1_s0070
                                 - 0.318999126333932 z_k1_s0071
                                 - 0.651998747123907 z_k1_s0072
                                 - 0.397000951505486 z_k1_s0073
                                 - 0.670997066691179 z_k1_s0074
                                 - 0.791999863693285 z_k1_s0075
                                 - 0.419999521717085 z_k1_s0076
                                 - 0.379998929269331 z_k1_s0077
                                 - 0.28100032896863 z_k1_s0100
                                 - 0.224000205767825 z_k1_s0101
                                 - 0.937000097530787 z_k1_s0102
                                 - 0.202000273774636 z_k1_s0103
                                 - 0.287000310421318 z_k1_s0104
                                 - 0.41100172351183 z_k1_s0105
                                 - 0.377001828613266 z_k1_s0106
                                 - 0.275000347515942 z_k1_s0107
                                 - 0.444000708200794 z_k1_s0110
                                 - 0.237999851017612 z_k1_s0111
                                 - 0.159999897871504 z_k1_s0112
                                 - 0.297001076336939 z_k1_s0113
                                 - 0.209999909962593 z_k1_s0114
                                 - 0.245000951106221 z_k1_s0115
                                 - 0.455001095768784 z_k1_s0116
                                 - 0.298000898206784 z_k1_s0117
                                 - 1.23499762953863 z_k1_s0120
                                 - 0.620999307406394 z_k1_s0121
                                 - 0.753001868591625 z_k1_s0122
                                 - 0.555999469051694 z_k1_s0123
                                 - 0.347999425677192 z_k1_s0124
                                 - 1.20700069643526 z_k1_s0125
                                 - 1.51899655670106 z_k1_s0126
                                 - 0.769001117221157 z_k1_s0127
                                 - 0.42899946058846 z_k1_s0130
                                 - 0.352999226876652 z_k1_s0131
                                 - 0.230999347348549 z_k1_s0132
                                 - 0.768999362354705 z_k1_s0133
                                 - 0.586997944836493 z_k1_s0134
                                 - 0.466997766448946 z_k1_s0135
                                 - 0.622996911755507 z_k1_s0136
                                 - 0.162999729194383 z_k1_s0137
                                 - 0.0820000295392777 z_k1_s0140
                                 - 0.0889999237487621 z_k1_s0141
                                 - 0.0949999550711392 z_k1_s0142
                                 - 0.356999216282263 z_k1_s0143
                                 - 0.28799945387474 z_k1_s0144
                                 - 0.187000321480956 z_k1_s0145
                                 - 0.283999319126524 z_k1_s0146
                                 - 0.326000733982789 z_k1_s0147
                                 - 0.274000711318121 z_k1_s0150
                                 - 0.355001875760911 z_k1_s0151
                                 - 0.39300019665526 z_k1_s0152
                                 - 0.821002467826181 z_k1_s0153
                                 - 0.479000379740424 z_k1_s0154
                                 - 0.200000305779196 z_k1_s0155
                                 - 0.616002111402414 z_k1_s0156
                                 - 0.302000650926989 z_k1_s0157
                                 - 0.211000625035958 z_k1_s0160
                                 - 0.141000026216533 z_k1_s0161
                                 - 0.284000768787751 z_k1_s0162
                                 - 0.209000673481102 z_k1_s0163
                                 - 0.124000055600348 z_k1_s0164
                                 - 0.32699972721714 z_k1_s0165
                                 - 0.490999578714423 z_k1_s0166
                                 - 0.125000031377775 z_k1_s0167
                                 - 0.229000085325 z_k1_s0170
                                 - 0.534000243113325 z_k1_s0171
                                 - 0.189999940458279 z_k1_s0172
                                 - 0.492999399576423 z_k1_s0173
                                 - 0.252999804166009 z_k1_s0174
                                 - 0.277998909541979 z_k1_s0175
                                 - 0.142000021376197 z_k1_s0176
                                 - 0.287999995892561 z_k1_s0177
                                 - 0.763004297112091 z_k1_s1000
                                 - 1.18699986094534 z_k1_s1001
                                 - 0.980003319359609 z_k1_s1002
                                 - 0.283000322384165 z_k1_s1003
                                 - 0.932998511263371 z_k1_s1004
                                 - 0.593001176642412 z_k1_s1005
                                 - 0.921002385483382 z_k1_s1006
                                 - 1.19100085321006 z_k1_s1007
                                 - 1.51800063857422 z_k1_s1010
                                 - 0.737999693364979 z_k1_s1011
                                 - 0.421999616434865 z_k1_s1012
                                 - 1.46099520295966 z_k1_s1013
                                 - 0.757999822395911 z_k1_s1014
                                 - 0.757999822395911 z_k1_s1015
                                 - 1.46500046012214 z_k1_s1016
                                 - 1.05399568329799 z_k1_s1017
                                 - 0.260001063270159 z_k1_s1020
                                 - 0.275999942593859 z_k1_s1021
                                 - 0.430000036715136 z_k1_s1022
                                 - 0.568001251512713 z_k1_s1023
                                 - 0.301999054120358 z_k1_s1024
                                 - 0.191999981691392 z_k1_s1025
                                 - 0.20000016745363 z_k1_s1026
                                 - 0.259001288750008 z_k1_s1027
                                 - 1.19800303105983 z_k1_s1030
                                 - 0.294000868470835 z_k1_s1031
                                 - 1.22700507425529 z_k1_s1032
                                 - 0.358000893715672 z_k1_s1033
                                 - 0.513001933108587 z_k1_s1034
                                 - 0.403001257617037 z_k1_s1035
                                 - 0.288001076804794 z_k1_s1036
                                 - 0.53500149028508 z_k1_s1037
                                 - 0.639000301075255 z_k1_s1040
                                 - 0.575999868109936 z_k1_s1041
                                 - 0.266000671186464 z_k1_s1042
                                 - 0.137000106914742 z_k1_s1043
                                 - 0.159000150683543 z_k1_s1044
                                 - 0.182000007132672 z_k1_s1045
                                 - 0.273000140849057 z_k1_s1046
                                 - 0.371000525128333 z_k1_s1047
                                 - 0.24900054150953 z_k1_s1050
                                 - 0.482999579782936 z_k1_s1051
                                 - 0.32800005434963 z_k1_s1052
                                 - 0.208000034465619 z_k1_s1053
                                 - 0.24900054150953 z_k1_s1054
                                 - 0.180000029826016 z_k1_s1055
                                 - 0.169000128053357 z_k1_s1056
                                 - 0.135000122419554 z_k1_s1057
                                 - 1.20099750144999 z_k1_s1060
                                 - 0.831999677395337 z_k1_s1061
                                 - 0.912999842308408 z_k1_s1062
                                 - 0.505998909592333 z_k1_s1063
                                 - 0.852000758902526 z_k1_s1064
                                 - 0.407998879195912 z_k1_s1065
                                 - 0.24799971131825 z_k1_s1066
                                 - 0.308999321623013 z_k1_s1067
                                 - 0.2920001524466 z_k1_s1070
                                 - 0.314000675781881 z_k1_s1071
                                 - 0.309001236206031 z_k1_s1072
                                 - 0.336001199117163 z_k1_s1073
                                 - 0.597999732106094 z_k1_s1074
                                 - 0.202000026976032 z_k1_s1075
                                 - 0.316000949812478 z_k1_s1076
                                 - 0.124000051088275 z_k1_s1077
                                 - 0.140000308550697 z_k1_s1100
                                 - 0.146000369574409 z_k1_s1101
                                 - 0.170000613669255 z_k1_s1102
                                 - 0.137000278038842 z_k1_s1103
                                 - 0.21500107134709 z_k1_s1104
                                 - 0.284000769317459 z_k1_s1105
                                 - 0.143000339062553 z_k1_s1106
                                 - 0.438000774122455 z_k1_s1107
                                 - 0.393999801796885 z_k1_s1110
                                 - 0.206999349294927 z_k1_s1111
                                 - 0.433998217780956 z_k1_s1112
                                 - 0.413997453793589 z_k1_s1113
                                 - 0.280999064057523 z_k1_s1114
                                 - 0.379000006804025 z_k1_s1115
                                 - 0.244999244875593 z_k1_s1116
                                 - 0.757996901617387 z_k1_s1117
                                 - 0.0639998925541196 z_k1_s1120
                                 - 0.31799944781581 z_k1_s1121
                                 - 0.0969998463085736 z_k1_s1122
                                 - 0.240999165056012 z_k1_s1123
                                 - 0.203999181391981 z_k1_s1124
                                 - 0.0689998713411797 z_k1_s1125
                                 - 0.0669998563863913 z_k1_s1126
                                 - 0.111999782669754 z_k1_s1127
                                 - 0.41399985482721 z_k1_s1130
                                 - 0.233000644350393 z_k1_s1131
                                 - 0.145000109597142 z_k1_s1132
                                 - 0.245999426317094 z_k1_s1133
                                 - 0.186000294250292 z_k1_s1134
                                 - 0.553001060178756 z_k1_s1135
                                 - 0.133000199103691 z_k1_s1136
                                 - 0.192000249497017 z_k1_s1137
                                 - 0.114000007393885 z_k1_s1140
                                 - 0.0419999156609849 z_k1_s1141
                                 - 0.138999541454377 z_k1_s1142
                                 - 0.0519999130850129 z_k1_s1143
                                 - 0.203999432810643 z_k1_s1144
                                 - 0.074999888780294 z_k1_s1145
                                 - 0.174999495420911 z_k1_s1146
                                 - 0.117999491723968 z_k1_s1147
                                 - 0.118999953843036 z_k1_s1150
                                 - 0.261999742666203 z_k1_s1151
                                 - 0.239000505943492 z_k1_s1152
                                 - 0.0580000185352556 z_k1_s1153
                                 - 0.217999460936047 z_k1_s1154
                                 - 0.184000153780797 z_k1_s1155
                                 - 0.0689999637678123 z_k1_s1156
                                 - 0.103000022122955 z_k1_s1157
                                 - 0.184999782935663 z_k1_s1160
                                 - 0.163999829348346 z_k1_s1161
                                 - 0.0579999101744493 z_k1_s1162
                                 - 0.127999621198995 z_k1_s1163
                                 - 0.0459999079245874 z_k1_s1164
                                 - 0.216000241897279 z_k1_s1165
                                 - 0.080999864136743 z_k1_s1166
                                 - 0.128999571036542 z_k1_s1167
                                 - 0.119999549643267 z_k1_s1170
                                 - 0.0909999107126038 z_k1_s1171
                                 - 0.183000178005273 z_k1_s1172
                                 - 0.0889999542518531 z_k1_s1173
                                 - 0.0969998881947671 z_k1_s1174
                                 - 0.138000346889048 z_k1_s1175
                                 - 0.27899938532024 z_k1_s1176
                                 - 0.0839999009501094 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.31513 w_k1_ADM_s000
                                 + 1.39974 w_k1_AXP_s000
                                 + 1.34687 w_k1_DIS_s000
                                 + 1.29374 w_k1_FDX_s000
                                 + 0.964246 w_k1_IBM_s000
                                 + 0.943074 w_k1_INTC_s000
                                 + 1.17505 w_k1_KO_s000
                                 + 0.964585 w_k1_MAT_s000
                                 + 1.05594 w_k1_MSFT_s000 + 1.0223 w_k1_PG_s000
                                 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.816812 w_k1_ADM_s000
                                 + 0.972074 w_k1_AXP_s000
                                 + 0.976388 w_k1_DIS_s000
                                 + 0.89825 w_k1_FDX_s000
                                 + 1.57894 w_k1_IBM_s000
                                 + 0.781692 w_k1_INTC_s000
                                 + 1.20276 w_k1_KO_s000 + 1.46198 w_k1_MAT_s000
                                 + 0.985979 w_k1_MSFT_s000
                                 + 0.99403 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.966636 w_k1_ADM_s000
                                 + 0.723898 w_k1_AXP_s000
                                 + 0.901658 w_k1_DIS_s000
                                 + 0.832786 w_k1_FDX_s000
                                 + 1.05165 w_k1_IBM_s000
                                 + 0.823422 w_k1_INTC_s000
                                 + 1.16597 w_k1_KO_s000 + 0.68393 w_k1_MAT_s000
                                 + 0.8964 w_k1_MSFT_s000
                                 + 0.928161 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.780865 w_k1_ADM_s000
                                 + 0.727737 w_k1_AXP_s000
                                 + 0.670034 w_k1_DIS_s000
                                 + 0.748427 w_k1_FDX_s000
                                 + 0.838145 w_k1_IBM_s000
                                 + 0.741633 w_k1_INTC_s000
                                 + 0.796328 w_k1_KO_s000
                                 + 0.903659 w_k1_MAT_s000
                                 + 0.813813 w_k1_MSFT_s000
                                 + 0.876264 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.05838 w_k1_ADM_s000
                                 + 0.922407 w_k1_AXP_s000
                                 + 1.09818 w_k1_DIS_s000
                                 + 1.08922 w_k1_FDX_s000
                                 + 0.893985 w_k1_IBM_s000
                                 + 0.904515 w_k1_INTC_s000
                                 + 0.922473 w_k1_KO_s000
                                 + 0.900118 w_k1_MAT_s000
                                 + 1.08251 w_k1_MSFT_s000
                                 + 0.937969 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.23516 w_k1_ADM_s000
                                 + 0.800814 w_k1_AXP_s000
                                 + 0.783272 w_k1_DIS_s000
                                 + 0.96248 w_k1_FDX_s000
                                 + 0.626818 w_k1_IBM_s000
                                 + 0.836554 w_k1_INTC_s000
                                 + 0.840122 w_k1_KO_s000
                                 + 0.57046 w_k1_MAT_s000
                                 + 0.942746 w_k1_MSFT_s000
                                 + 0.915131 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.998683 w_k1_ADM_s000
                                 + 1.0419 w_k1_AXP_s000 + 0.92873 w_k1_DIS_s000
                                 + 0.94565 w_k1_FDX_s000
                                 + 1.15418 w_k1_IBM_s000
                                 + 0.746151 w_k1_INTC_s000 + 1.132 w_k1_KO_s000
                                 + 1.21904 w_k1_MAT_s000
                                 + 0.991271 w_k1_MSFT_s000
                                 + 0.921908 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.08719 w_k1_ADM_s000
                                 + 1.39295 w_k1_AXP_s000 + 1.7752 w_k1_DIS_s000
                                 + 1.19663 w_k1_FDX_s000
                                 + 1.46131 w_k1_IBM_s000
                                 + 0.9152 w_k1_INTC_s000 + 1.47685 w_k1_KO_s000
                                 + 1.58901 w_k1_MAT_s000
                                 + 1.31712 w_k1_MSFT_s000
                                 + 1.29841 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.930824 w_k1_ADM_s001
                                 + 0.742477 w_k1_AXP_s001
                                 + 1.07104 w_k1_DIS_s001
                                 + 0.889971 w_k1_FDX_s001
                                 + 1.03619 w_k1_IBM_s001
                                 + 0.886894 w_k1_INTC_s001
                                 + 1.00394 w_k1_KO_s001
                                 + 0.710779 w_k1_MAT_s001
                                 + 1.0008 w_k1_MSFT_s001
                                 + 0.967818 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.15474 w_k1_ADM_s001
                                 + 1.33252 w_k1_AXP_s001
                                 + 1.62369 w_k1_DIS_s001
                                 + 1.33147 w_k1_FDX_s001
                                 + 1.29728 w_k1_IBM_s001
                                 + 1.14359 w_k1_INTC_s001
                                 + 1.30693 w_k1_KO_s001 + 1.32434 w_k1_MAT_s001
                                 + 1.26848 w_k1_MSFT_s001
                                 + 1.41715 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.0092 w_k1_ADM_s001
                                 + 1.04104 w_k1_AXP_s001
                                 + 0.852041 w_k1_DIS_s001
                                 + 0.959447 w_k1_FDX_s001
                                 + 0.875549 w_k1_IBM_s001
                                 + 0.988783 w_k1_INTC_s001
                                 + 0.810612 w_k1_KO_s001
                                 + 0.913829 w_k1_MAT_s001
                                 + 0.947236 w_k1_MSFT_s001
                                 + 0.999933 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.892001 w_k1_ADM_s001
                                 + 0.857427 w_k1_AXP_s001
                                 + 0.704171 w_k1_DIS_s001
                                 + 0.78998 w_k1_FDX_s001 + 0.97 w_k1_IBM_s001
                                 + 1.06626 w_k1_INTC_s001
                                 + 0.914739 w_k1_KO_s001
                                 + 0.800556 w_k1_MAT_s001
                                 + 0.872358 w_k1_MSFT_s001
                                 + 0.953201 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.944315 w_k1_ADM_s001
                                 + 1.10522 w_k1_AXP_s001
                                 + 1.20987 w_k1_DIS_s001
                                 + 0.903793 w_k1_FDX_s001
                                 + 1.61053 w_k1_IBM_s001
                                 + 1.09669 w_k1_INTC_s001
                                 + 1.37007 w_k1_KO_s001 + 1.55082 w_k1_MAT_s001
                                 + 0.965441 w_k1_MSFT_s001
                                 + 1.16364 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.03256 w_k1_ADM_s001
                                 + 1.46086 w_k1_AXP_s001
                                 + 1.12751 w_k1_DIS_s001
                                 + 1.19962 w_k1_FDX_s001
                                 + 0.977775 w_k1_IBM_s001
                                 + 1.44267 w_k1_INTC_s001
                                 + 0.971134 w_k1_KO_s001
                                 + 0.990442 w_k1_MAT_s001
                                 + 1.08567 w_k1_MSFT_s001
                                 + 1.08468 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.04624 w_k1_ADM_s001
                                 + 0.823617 w_k1_AXP_s001
                                 + 1.2769 w_k1_DIS_s001 + 1.07829 w_k1_FDX_s001
                                 + 0.907753 w_k1_IBM_s001
                                 + 1.02556 w_k1_INTC_s001
                                 + 1.04956 w_k1_KO_s001
                                 + 0.656792 w_k1_MAT_s001
                                 + 1.13552 w_k1_MSFT_s001
                                 + 1.13222 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.39546 w_k1_ADM_s001
                                 + 1.29333 w_k1_AXP_s001
                                 + 1.24512 w_k1_DIS_s001
                                 + 1.18531 w_k1_FDX_s001
                                 + 1.39768 w_k1_IBM_s001
                                 + 1.41006 w_k1_INTC_s001
                                 + 1.27799 w_k1_KO_s001 + 1.2897 w_k1_MAT_s001
                                 + 1.08797 w_k1_MSFT_s001
                                 + 1.11622 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.56596 w_k1_ADM_s002
                                 + 1.41902 w_k1_AXP_s002
                                 + 1.81962 w_k1_DIS_s002
                                 + 1.51724 w_k1_FDX_s002
                                 + 1.09142 w_k1_IBM_s002
                                 + 1.70549 w_k1_INTC_s002
                                 + 1.12114 w_k1_KO_s002 + 0.93999 w_k1_MAT_s002
                                 + 1.49492 w_k1_MSFT_s002
                                 + 1.14432 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.38521 w_k1_ADM_s002
                                 + 0.786797 w_k1_AXP_s002
                                 + 0.636659 w_k1_DIS_s002
                                 + 0.868132 w_k1_FDX_s002
                                 + 0.61719 w_k1_IBM_s002
                                 + 0.942086 w_k1_INTC_s002
                                 + 0.744635 w_k1_KO_s002
                                 + 0.526083 w_k1_MAT_s002
                                 + 0.778166 w_k1_MSFT_s002
                                 + 0.964014 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.28927 w_k1_ADM_s002
                                 + 1.14998 w_k1_AXP_s002
                                 + 1.12958 w_k1_DIS_s002
                                 + 1.21524 w_k1_FDX_s002
                                 + 0.861211 w_k1_IBM_s002
                                 + 1.49193 w_k1_INTC_s002
                                 + 1.08072 w_k1_KO_s002
                                 + 0.741413 w_k1_MAT_s002
                                 + 1.11425 w_k1_MSFT_s002
                                 + 0.996837 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.20006 w_k1_ADM_s002
                                 + 0.81487 w_k1_AXP_s002
                                 + 0.81614 w_k1_DIS_s002
                                 + 0.915063 w_k1_FDX_s002
                                 + 0.731769 w_k1_IBM_s002
                                 + 0.982613 w_k1_INTC_s002
                                 + 0.880625 w_k1_KO_s002
                                 + 0.682911 w_k1_MAT_s002
                                 + 0.847717 w_k1_MSFT_s002
                                 + 1.00576 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.31498 w_k1_ADM_s002
                                 + 1.05537 w_k1_AXP_s002
                                 + 0.886808 w_k1_DIS_s002
                                 + 1.05293 w_k1_FDX_s002
                                 + 0.836405 w_k1_IBM_s002
                                 + 1.26597 w_k1_INTC_s002
                                 + 0.922512 w_k1_KO_s002
                                 + 0.865409 w_k1_MAT_s002
                                 + 0.984893 w_k1_MSFT_s002
                                 + 1.04116 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.868169 w_k1_ADM_s002
                                 + 0.683334 w_k1_AXP_s002
                                 + 0.577121 w_k1_DIS_s002
                                 + 0.721881 w_k1_FDX_s002
                                 + 0.999698 w_k1_IBM_s002
                                 + 0.926151 w_k1_INTC_s002
                                 + 0.881666 w_k1_KO_s002
                                 + 1.06859 w_k1_MAT_s002
                                 + 0.735105 w_k1_MSFT_s002
                                 + 0.905511 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.922792 w_k1_ADM_s002
                                 + 0.587631 w_k1_AXP_s002
                                 + 0.829886 w_k1_DIS_s002
                                 + 0.825847 w_k1_FDX_s002
                                 + 1.00383 w_k1_IBM_s002
                                 + 1.21658 w_k1_INTC_s002
                                 + 1.00094 w_k1_KO_s002
                                 + 0.835541 w_k1_MAT_s002
                                 + 0.912343 w_k1_MSFT_s002
                                 + 0.956097 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.31233 w_k1_ADM_s002
                                 + 1.18255 w_k1_AXP_s002
                                 + 1.31716 w_k1_DIS_s002
                                 + 1.05751 w_k1_FDX_s002
                                 + 1.49037 w_k1_IBM_s002
                                 + 1.53993 w_k1_INTC_s002
                                 + 1.36787 w_k1_KO_s002 + 1.19091 w_k1_MAT_s002
                                 + 1.11448 w_k1_MSFT_s002
                                 + 1.26312 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.722049 w_k1_ADM_s003
                                 + 0.660309 w_k1_AXP_s003
                                 + 0.625053 w_k1_DIS_s003
                                 + 0.731611 w_k1_FDX_s003
                                 + 0.50255 w_k1_IBM_s003
                                 + 0.642503 w_k1_INTC_s003
                                 + 0.662992 w_k1_KO_s003
                                 + 1.07322 w_k1_MAT_s003
                                 + 0.84326 w_k1_MSFT_s003
                                 + 0.834798 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.84964 w_k1_ADM_s003
                                 + 0.751215 w_k1_AXP_s003
                                 + 0.903545 w_k1_DIS_s003
                                 + 0.783677 w_k1_FDX_s003
                                 + 0.844074 w_k1_IBM_s003
                                 + 0.779183 w_k1_INTC_s003
                                 + 0.954825 w_k1_KO_s003
                                 + 1.43192 w_k1_MAT_s003
                                 + 0.996058 w_k1_MSFT_s003
                                 + 0.949547 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.857715 w_k1_ADM_s003
                                 + 0.956737 w_k1_AXP_s003
                                 + 0.74083 w_k1_DIS_s003
                                 + 0.866664 w_k1_FDX_s003
                                 + 0.914959 w_k1_IBM_s003
                                 + 0.904613 w_k1_INTC_s003
                                 + 0.841376 w_k1_KO_s003
                                 + 1.26379 w_k1_MAT_s003
                                 + 0.892334 w_k1_MSFT_s003
                                 + 1.00639 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.871012 w_k1_ADM_s003
                                 + 0.890828 w_k1_AXP_s003
                                 + 0.873693 w_k1_DIS_s003
                                 + 0.961854 w_k1_FDX_s003
                                 + 0.623713 w_k1_IBM_s003
                                 + 0.873332 w_k1_INTC_s003
                                 + 0.893773 w_k1_KO_s003
                                 + 0.945248 w_k1_MAT_s003
                                 + 1.16793 w_k1_MSFT_s003
                                 + 0.989759 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.01129 w_k1_ADM_s003
                                 + 1.5811 w_k1_AXP_s003 + 1.35361 w_k1_DIS_s003
                                 + 1.22527 w_k1_FDX_s003
                                 + 1.38492 w_k1_IBM_s003
                                 + 1.20627 w_k1_INTC_s003
                                 + 1.13343 w_k1_KO_s003 + 1.90657 w_k1_MAT_s003
                                 + 1.31009 w_k1_MSFT_s003
                                 + 1.17909 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.05176 w_k1_ADM_s003
                                 + 1.16795 w_k1_AXP_s003
                                 + 1.50475 w_k1_DIS_s003
                                 + 1.11222 w_k1_FDX_s003
                                 + 1.33493 w_k1_IBM_s003
                                 + 1.07649 w_k1_INTC_s003
                                 + 1.34834 w_k1_KO_s003 + 1.39352 w_k1_MAT_s003
                                 + 1.25477 w_k1_MSFT_s003 + 1.3429 w_k1_PG_s003
                                 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.16673 w_k1_ADM_s003
                                 + 1.38293 w_k1_AXP_s003
                                 + 1.10156 w_k1_DIS_s003
                                 + 1.11721 w_k1_FDX_s003
                                 + 0.871404 w_k1_IBM_s003
                                 + 0.990839 w_k1_INTC_s003
                                 + 1.01027 w_k1_KO_s003 + 1.65517 w_k1_MAT_s003
                                 + 1.16448 w_k1_MSFT_s003
                                 + 1.09941 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.721335 w_k1_ADM_s003
                                 + 1.0732 w_k1_AXP_s003 + 1.00582 w_k1_DIS_s003
                                 + 0.873453 w_k1_FDX_s003
                                 + 1.08318 w_k1_IBM_s003
                                 + 0.885245 w_k1_INTC_s003
                                 + 1.09645 w_k1_KO_s003 + 1.50277 w_k1_MAT_s003
                                 + 1.07317 w_k1_MSFT_s003
                                 + 0.945728 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.11286 w_k1_ADM_s004
                                 + 1.10737 w_k1_AXP_s004
                                 + 1.22619 w_k1_DIS_s004
                                 + 1.02622 w_k1_FDX_s004
                                 + 1.32892 w_k1_IBM_s004
                                 + 1.07576 w_k1_INTC_s004
                                 + 1.25344 w_k1_KO_s004 + 0.87036 w_k1_MAT_s004
                                 + 1.02638 w_k1_MSFT_s004
                                 + 0.974001 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.932609 w_k1_ADM_s004
                                 + 0.554636 w_k1_AXP_s004
                                 + 0.684132 w_k1_DIS_s004
                                 + 0.739413 w_k1_FDX_s004
                                 + 0.826525 w_k1_IBM_s004
                                 + 0.680569 w_k1_INTC_s004
                                 + 0.809761 w_k1_KO_s004
                                 + 0.722451 w_k1_MAT_s004
                                 + 0.852989 w_k1_MSFT_s004
                                 + 0.699231 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.04224 w_k1_ADM_s004
                                 + 0.753177 w_k1_AXP_s004
                                 + 1.03333 w_k1_DIS_s004
                                 + 0.857547 w_k1_FDX_s004
                                 + 1.02852 w_k1_IBM_s004
                                 + 0.85005 w_k1_INTC_s004
                                 + 1.08326 w_k1_KO_s004
                                 + 0.642245 w_k1_MAT_s004
                                 + 0.953811 w_k1_MSFT_s004
                                 + 0.916967 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.762902 w_k1_ADM_s004
                                 + 0.909398 w_k1_AXP_s004
                                 + 0.705215 w_k1_DIS_s004
                                 + 0.847583 w_k1_FDX_s004
                                 + 0.947032 w_k1_IBM_s004
                                 + 0.838192 w_k1_INTC_s004
                                 + 0.841207 w_k1_KO_s004
                                 + 0.843714 w_k1_MAT_s004
                                 + 0.764862 w_k1_MSFT_s004
                                 + 0.799119 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.26364 w_k1_ADM_s004
                                 + 1.46452 w_k1_AXP_s004
                                 + 1.71443 w_k1_DIS_s004
                                 + 1.41891 w_k1_FDX_s004
                                 + 1.33126 w_k1_IBM_s004
                                 + 1.24813 w_k1_INTC_s004
                                 + 1.45346 w_k1_KO_s004
                                 + 0.920796 w_k1_MAT_s004
                                 + 1.43124 w_k1_MSFT_s004
                                 + 1.13447 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.904435 w_k1_ADM_s004
                                 + 1.23002 w_k1_AXP_s004
                                 + 0.89045 w_k1_DIS_s004
                                 + 0.935146 w_k1_FDX_s004
                                 + 1.14303 w_k1_IBM_s004
                                 + 0.857448 w_k1_INTC_s004
                                 + 1.07284 w_k1_KO_s004 + 1.00318 w_k1_MAT_s004
                                 + 0.913883 w_k1_MSFT_s004
                                 + 0.922842 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.39445 w_k1_ADM_s004
                                 + 1.35878 w_k1_AXP_s004
                                 + 1.32937 w_k1_DIS_s004
                                 + 1.18713 w_k1_FDX_s004
                                 + 1.08724 w_k1_IBM_s004
                                 + 1.02085 w_k1_INTC_s004
                                 + 1.09147 w_k1_KO_s004 + 1.09274 w_k1_MAT_s004
                                 + 1.23122 w_k1_MSFT_s004
                                 + 1.01443 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.857033 w_k1_ADM_s004
                                 + 0.850549 w_k1_AXP_s004
                                 + 1.28429 w_k1_DIS_s004
                                 + 0.992486 w_k1_FDX_s004
                                 + 0.945715 w_k1_IBM_s004
                                 + 1.05352 w_k1_INTC_s004
                                 + 1.01839 w_k1_KO_s004 + 0.71828 w_k1_MAT_s004
                                 + 1.25007 w_k1_MSFT_s004
                                 + 0.942991 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.769195 w_k1_ADM_s005
                                 + 0.674543 w_k1_AXP_s005
                                 + 0.719441 w_k1_DIS_s005
                                 + 0.764047 w_k1_FDX_s005
                                 + 0.849473 w_k1_IBM_s005
                                 + 0.912983 w_k1_INTC_s005
                                 + 0.919607 w_k1_KO_s005
                                 + 0.560344 w_k1_MAT_s005
                                 + 0.891592 w_k1_MSFT_s005
                                 + 0.908258 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.13846 w_k1_ADM_s005
                                 + 1.30705 w_k1_AXP_s005
                                 + 1.07546 w_k1_DIS_s005
                                 + 1.06672 w_k1_FDX_s005
                                 + 0.974265 w_k1_IBM_s005
                                 + 1.03008 w_k1_INTC_s005
                                 + 1.03847 w_k1_KO_s005 + 1.09398 w_k1_MAT_s005
                                 + 1.04646 w_k1_MSFT_s005
                                 + 1.11635 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.795949 w_k1_ADM_s005
                                 + 0.849124 w_k1_AXP_s005
                                 + 1.2442 w_k1_DIS_s005 + 1.04832 w_k1_FDX_s005
                                 + 0.748041 w_k1_IBM_s005
                                 + 1.15716 w_k1_INTC_s005
                                 + 1.15357 w_k1_KO_s005
                                 + 0.875378 w_k1_MAT_s005
                                 + 1.30277 w_k1_MSFT_s005
                                 + 1.03824 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.0096 w_k1_ADM_s005
                                 + 1.03089 w_k1_AXP_s005
                                 + 1.19543 w_k1_DIS_s005
                                 + 1.06703 w_k1_FDX_s005
                                 + 1.26296 w_k1_IBM_s005
                                 + 1.06931 w_k1_INTC_s005 + 1.2076 w_k1_KO_s005
                                 + 1.23628 w_k1_MAT_s005
                                 + 0.943114 w_k1_MSFT_s005
                                 + 1.18081 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.05736 w_k1_ADM_s005
                                 + 1.41807 w_k1_AXP_s005
                                 + 1.84862 w_k1_DIS_s005 + 1.3633 w_k1_FDX_s005
                                 + 1.38733 w_k1_IBM_s005
                                 + 1.47794 w_k1_INTC_s005
                                 + 1.53936 w_k1_KO_s005 + 1.59439 w_k1_MAT_s005
                                 + 1.44371 w_k1_MSFT_s005
                                 + 1.38078 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.976727 w_k1_ADM_s005
                                 + 0.965632 w_k1_AXP_s005
                                 + 0.815604 w_k1_DIS_s005
                                 + 0.983413 w_k1_FDX_s005
                                 + 0.788053 w_k1_IBM_s005
                                 + 0.878516 w_k1_INTC_s005
                                 + 0.845667 w_k1_KO_s005
                                 + 0.665389 w_k1_MAT_s005
                                 + 0.907789 w_k1_MSFT_s005
                                 + 0.937187 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.33227 w_k1_ADM_s005
                                 + 1.26492 w_k1_AXP_s005 + 1.3407 w_k1_DIS_s005
                                 + 1.29874 w_k1_FDX_s005
                                 + 0.936605 w_k1_IBM_s005
                                 + 1.06487 w_k1_INTC_s005
                                 + 1.14753 w_k1_KO_s005 + 1.35913 w_k1_MAT_s005
                                 + 1.24225 w_k1_MSFT_s005
                                 + 1.28597 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.795648 w_k1_ADM_s005
                                 + 1.03019 w_k1_AXP_s005
                                 + 0.952102 w_k1_DIS_s005
                                 + 0.887712 w_k1_FDX_s005
                                 + 1.38281 w_k1_IBM_s005
                                 + 0.80423 w_k1_INTC_s005
                                 + 1.16692 w_k1_KO_s005 + 1.42045 w_k1_MAT_s005
                                 + 0.908681 w_k1_MSFT_s005
                                 + 1.07444 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.37723 w_k1_ADM_s006
                                 + 1.6236 w_k1_AXP_s006 + 1.18747 w_k1_DIS_s006
                                 + 1.2312 w_k1_FDX_s006 + 1.23756 w_k1_IBM_s006
                                 + 1.16669 w_k1_INTC_s006
                                 + 1.16238 w_k1_KO_s006 + 1.32433 w_k1_MAT_s006
                                 + 1.10864 w_k1_MSFT_s006 + 1.0884 w_k1_PG_s006
                                 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.881822 w_k1_ADM_s006
                                 + 0.890451 w_k1_AXP_s006
                                 + 0.555442 w_k1_DIS_s006
                                 + 0.736592 w_k1_FDX_s006
                                 + 0.63365 w_k1_IBM_s006
                                 + 0.749857 w_k1_INTC_s006
                                 + 0.595823 w_k1_KO_s006
                                 + 0.752716 w_k1_MAT_s006
                                 + 0.783592 w_k1_MSFT_s006
                                 + 0.863345 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.956745 w_k1_ADM_s006
                                 + 0.962574 w_k1_AXP_s006
                                 + 0.968766 w_k1_DIS_s006
                                 + 0.910704 w_k1_FDX_s006 + 1.094 w_k1_IBM_s006
                                 + 0.736633 w_k1_INTC_s006
                                 + 0.967535 w_k1_KO_s006
                                 + 1.17868 w_k1_MAT_s006
                                 + 0.795824 w_k1_MSFT_s006
                                 + 0.996517 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.19109 w_k1_ADM_s006
                                 + 1.49768 w_k1_AXP_s006
                                 + 1.06966 w_k1_DIS_s006
                                 + 1.24027 w_k1_FDX_s006
                                 + 0.967115 w_k1_IBM_s006
                                 + 0.961142 w_k1_INTC_s006
                                 + 0.766407 w_k1_KO_s006
                                 + 1.39097 w_k1_MAT_s006
                                 + 1.0551 w_k1_MSFT_s006 + 1.09256 w_k1_PG_s006
                                 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.35299 w_k1_ADM_s006
                                 + 1.33008 w_k1_AXP_s006
                                 + 1.45694 w_k1_DIS_s006
                                 + 1.20112 w_k1_FDX_s006
                                 + 1.53281 w_k1_IBM_s006
                                 + 0.97232 w_k1_INTC_s006
                                 + 1.33605 w_k1_KO_s006 + 1.49288 w_k1_MAT_s006
                                 + 1.18277 w_k1_MSFT_s006
                                 + 1.33891 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.819468 w_k1_ADM_s006
                                 + 0.986391 w_k1_AXP_s006
                                 + 0.891242 w_k1_DIS_s006
                                 + 0.950226 w_k1_FDX_s006
                                 + 0.822287 w_k1_IBM_s006
                                 + 0.852198 w_k1_INTC_s006
                                 + 0.883557 w_k1_KO_s006
                                 + 0.944628 w_k1_MAT_s006
                                 + 0.86172 w_k1_MSFT_s006
                                 + 0.840307 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.1675 w_k1_ADM_s006
                                 + 1.04134 w_k1_AXP_s006
                                 + 1.02825 w_k1_DIS_s006
                                 + 1.11435 w_k1_FDX_s006
                                 + 0.707113 w_k1_IBM_s006
                                 + 0.985565 w_k1_INTC_s006
                                 + 0.913252 w_k1_KO_s006
                                 + 0.904988 w_k1_MAT_s006
                                 + 0.924982 w_k1_MSFT_s006
                                 + 0.954934 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.892638 w_k1_ADM_s006
                                 + 0.749557 w_k1_AXP_s006
                                 + 0.744148 w_k1_DIS_s006
                                 + 0.776756 w_k1_FDX_s006
                                 + 0.920179 w_k1_IBM_s006
                                 + 0.860284 w_k1_INTC_s006
                                 + 0.878857 w_k1_KO_s006
                                 + 0.767011 w_k1_MAT_s006
                                 + 0.806012 w_k1_MSFT_s006
                                 + 0.863242 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.893897 w_k1_ADM_s007
                                 + 0.819871 w_k1_AXP_s007
                                 + 0.618708 w_k1_DIS_s007
                                 + 0.747577 w_k1_FDX_s007
                                 + 1.04344 w_k1_IBM_s007
                                 + 0.686766 w_k1_INTC_s007
                                 + 0.786151 w_k1_KO_s007
                                 + 1.02129 w_k1_MAT_s007
                                 + 0.747044 w_k1_MSFT_s007
                                 + 0.830423 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.97309 w_k1_ADM_s007
                                 + 1.29666 w_k1_AXP_s007
                                 + 0.963909 w_k1_DIS_s007
                                 + 0.975731 w_k1_FDX_s007
                                 + 1.13626 w_k1_IBM_s007
                                 + 0.881836 w_k1_INTC_s007
                                 + 1.00041 w_k1_KO_s007 + 1.20293 w_k1_MAT_s007
                                 + 0.916265 w_k1_MSFT_s007
                                 + 0.947216 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.956056 w_k1_ADM_s007
                                 + 0.896467 w_k1_AXP_s007
                                 + 1.20558 w_k1_DIS_s007
                                 + 0.951982 w_k1_FDX_s007
                                 + 1.26049 w_k1_IBM_s007
                                 + 0.842509 w_k1_INTC_s007
                                 + 1.13721 w_k1_KO_s007
                                 + 0.926216 w_k1_MAT_s007
                                 + 1.09413 w_k1_MSFT_s007
                                 + 0.952785 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.970893 w_k1_ADM_s007
                                 + 0.931385 w_k1_AXP_s007
                                 + 1.14589 w_k1_DIS_s007
                                 + 1.07006 w_k1_FDX_s007
                                 + 0.82366 w_k1_IBM_s007
                                 + 1.12869 w_k1_INTC_s007
                                 + 0.782929 w_k1_KO_s007
                                 + 0.684505 w_k1_MAT_s007
                                 + 1.09692 w_k1_MSFT_s007
                                 + 0.878395 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.13609 w_k1_ADM_s007
                                 + 1.38817 w_k1_AXP_s007
                                 + 1.37097 w_k1_DIS_s007
                                 + 1.33564 w_k1_FDX_s007
                                 + 1.03878 w_k1_IBM_s007
                                 + 1.4125 w_k1_INTC_s007 + 1.07755 w_k1_KO_s007
                                 + 0.775056 w_k1_MAT_s007
                                 + 1.42278 w_k1_MSFT_s007
                                 + 1.03734 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.911331 w_k1_ADM_s007
                                 + 0.996188 w_k1_AXP_s007
                                 + 0.896379 w_k1_DIS_s007
                                 + 0.948143 w_k1_FDX_s007
                                 + 0.848044 w_k1_IBM_s007
                                 + 1.12679 w_k1_INTC_s007
                                 + 0.857183 w_k1_KO_s007
                                 + 0.912271 w_k1_MAT_s007
                                 + 0.99241 w_k1_MSFT_s007
                                 + 0.981729 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.892494 w_k1_ADM_s007
                                 + 1.40803 w_k1_AXP_s007
                                 + 1.52048 w_k1_DIS_s007
                                 + 1.12148 w_k1_FDX_s007
                                 + 1.68619 w_k1_IBM_s007
                                 + 1.21121 w_k1_INTC_s007
                                 + 1.30946 w_k1_KO_s007 + 1.41287 w_k1_MAT_s007
                                 + 1.25472 w_k1_MSFT_s007
                                 + 0.997026 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.930506 w_k1_ADM_s007
                                 + 0.654299 w_k1_AXP_s007
                                 + 0.686008 w_k1_DIS_s007
                                 + 0.767629 w_k1_FDX_s007
                                 + 0.682903 w_k1_IBM_s007
                                 + 0.858137 w_k1_INTC_s007
                                 + 0.700423 w_k1_KO_s007
                                 + 0.433858 w_k1_MAT_s007
                                 + 0.827558 w_k1_MSFT_s007
                                 + 0.833683 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.04803 w_k1_ADM_s010
                                 + 1.58206 w_k1_AXP_s010 + 1.6412 w_k1_DIS_s010
                                 + 1.29541 w_k1_FDX_s010
                                 + 1.45037 w_k1_IBM_s010
                                 + 1.40371 w_k1_INTC_s010
                                 + 1.34701 w_k1_KO_s010 + 1.42617 w_k1_MAT_s010
                                 + 1.27476 w_k1_MSFT_s010
                                 + 1.20014 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.847194 w_k1_ADM_s010
                                 + 0.848582 w_k1_AXP_s010
                                 + 0.790221 w_k1_DIS_s010
                                 + 0.941812 w_k1_FDX_s010
                                 + 0.577494 w_k1_IBM_s010
                                 + 0.816601 w_k1_INTC_s010
                                 + 0.672797 w_k1_KO_s010
                                 + 0.705902 w_k1_MAT_s010
                                 + 0.972627 w_k1_MSFT_s010
                                 + 0.904569 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.09595 w_k1_ADM_s010
                                 + 1.08689 w_k1_AXP_s010
                                 + 1.31394 w_k1_DIS_s010 + 1.2022 w_k1_FDX_s010
                                 + 0.968901 w_k1_IBM_s010
                                 + 1.08041 w_k1_INTC_s010
                                 + 1.06043 w_k1_KO_s010
                                 + 0.780657 w_k1_MAT_s010
                                 + 1.0779 w_k1_MSFT_s010 + 1.03411 w_k1_PG_s010
                                 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.805485 w_k1_ADM_s010
                                 + 1.06486 w_k1_AXP_s010
                                 + 0.963194 w_k1_DIS_s010
                                 + 0.859071 w_k1_FDX_s010
                                 + 1.48717 w_k1_IBM_s010
                                 + 0.864393 w_k1_INTC_s010
                                 + 1.05622 w_k1_KO_s010 + 1.19411 w_k1_MAT_s010
                                 + 0.846829 w_k1_MSFT_s010
                                 + 1.01328 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.825579 w_k1_ADM_s010
                                 + 0.594796 w_k1_AXP_s010
                                 + 0.654888 w_k1_DIS_s010
                                 + 0.718758 w_k1_FDX_s010
                                 + 0.805282 w_k1_IBM_s010
                                 + 0.579362 w_k1_INTC_s010
                                 + 0.78603 w_k1_KO_s010
                                 + 0.711812 w_k1_MAT_s010
                                 + 0.729258 w_k1_MSFT_s010
                                 + 0.763305 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.858806 w_k1_ADM_s010
                                 + 0.855293 w_k1_AXP_s010
                                 + 1.06425 w_k1_DIS_s010
                                 + 1.01687 w_k1_FDX_s010
                                 + 0.887489 w_k1_IBM_s010
                                 + 0.881977 w_k1_INTC_s010
                                 + 1.00513 w_k1_KO_s010 + 1.04902 w_k1_MAT_s010
                                 + 1.061 w_k1_MSFT_s010 + 1.10779 w_k1_PG_s010
                                 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.21539 w_k1_ADM_s010
                                 + 1.04357 w_k1_AXP_s010
                                 + 0.797383 w_k1_DIS_s010
                                 + 0.955833 w_k1_FDX_s010
                                 + 0.937314 w_k1_IBM_s010
                                 + 0.936293 w_k1_INTC_s010
                                 + 0.863191 w_k1_KO_s010
                                 + 0.781694 w_k1_MAT_s010
                                 + 0.906465 w_k1_MSFT_s010
                                 + 0.905745 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.677364 w_k1_ADM_s010
                                 + 0.93029 w_k1_AXP_s010
                                 + 0.828245 w_k1_DIS_s010
                                 + 0.819608 w_k1_FDX_s010
                                 + 1.11882 w_k1_IBM_s010
                                 + 0.721045 w_k1_INTC_s010
                                 + 0.80211 w_k1_KO_s010
                                 + 0.977009 w_k1_MAT_s010
                                 + 0.864935 w_k1_MSFT_s010
                                 + 0.902092 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.915961 w_k1_ADM_s011
                                 + 0.728584 w_k1_AXP_s011
                                 + 0.706646 w_k1_DIS_s011
                                 + 0.782176 w_k1_FDX_s011
                                 + 1.01345 w_k1_IBM_s011
                                 + 0.86607 w_k1_INTC_s011
                                 + 0.982742 w_k1_KO_s011
                                 + 0.839278 w_k1_MAT_s011
                                 + 0.763053 w_k1_MSFT_s011
                                 + 0.964337 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.55994 w_k1_ADM_s011
                                 + 1.34346 w_k1_AXP_s011
                                 + 1.70101 w_k1_DIS_s011
                                 + 1.42474 w_k1_FDX_s011
                                 + 1.17949 w_k1_IBM_s011
                                 + 1.19268 w_k1_INTC_s011
                                 + 1.34926 w_k1_KO_s011 + 1.38373 w_k1_MAT_s011
                                 + 1.36763 w_k1_MSFT_s011
                                 + 1.43114 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.992987 w_k1_ADM_s011
                                 + 1.13493 w_k1_AXP_s011
                                 + 1.32628 w_k1_DIS_s011
                                 + 1.13523 w_k1_FDX_s011
                                 + 1.94937 w_k1_IBM_s011
                                 + 1.27574 w_k1_INTC_s011
                                 + 1.47666 w_k1_KO_s011 + 1.42709 w_k1_MAT_s011
                                 + 1.17534 w_k1_MSFT_s011
                                 + 1.08784 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.872023 w_k1_ADM_s011
                                 + 1.11225 w_k1_AXP_s011
                                 + 1.10345 w_k1_DIS_s011
                                 + 1.03942 w_k1_FDX_s011 + 1.4637 w_k1_IBM_s011
                                 + 1.22557 w_k1_INTC_s011
                                 + 1.17641 w_k1_KO_s011 + 1.09622 w_k1_MAT_s011
                                 + 0.985745 w_k1_MSFT_s011
                                 + 1.12983 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.10975 w_k1_ADM_s011
                                 + 0.893095 w_k1_AXP_s011
                                 + 1.29694 w_k1_DIS_s011
                                 + 1.10877 w_k1_FDX_s011
                                 + 0.735168 w_k1_IBM_s011
                                 + 1.16809 w_k1_INTC_s011
                                 + 0.909025 w_k1_KO_s011
                                 + 0.733421 w_k1_MAT_s011
                                 + 1.05526 w_k1_MSFT_s011
                                 + 1.09604 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.27573 w_k1_ADM_s011
                                 + 0.788125 w_k1_AXP_s011
                                 + 1.19072 w_k1_DIS_s011
                                 + 1.01964 w_k1_FDX_s011
                                 + 0.886238 w_k1_IBM_s011
                                 + 1.2641 w_k1_INTC_s011 + 1.05409 w_k1_KO_s011
                                 + 0.67938 w_k1_MAT_s011
                                 + 1.15385 w_k1_MSFT_s011
                                 + 1.09748 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.01489 w_k1_ADM_s011
                                 + 1.09601 w_k1_AXP_s011
                                 + 0.904487 w_k1_DIS_s011
                                 + 0.97084 w_k1_FDX_s011
                                 + 0.962857 w_k1_IBM_s011
                                 + 1.14413 w_k1_INTC_s011
                                 + 0.901328 w_k1_KO_s011
                                 + 0.825633 w_k1_MAT_s011
                                 + 0.962821 w_k1_MSFT_s011
                                 + 1.06815 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.07354 w_k1_ADM_s011
                                 + 0.953797 w_k1_AXP_s011
                                 + 1.16429 w_k1_DIS_s011
                                 + 1.23125 w_k1_FDX_s011
                                 + 0.611639 w_k1_IBM_s011
                                 + 1.36256 w_k1_INTC_s011
                                 + 0.826063 w_k1_KO_s011
                                 + 0.419742 w_k1_MAT_s011
                                 + 1.14696 w_k1_MSFT_s011
                                 + 0.975053 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.872105 w_k1_ADM_s012
                                 + 0.834137 w_k1_AXP_s012
                                 + 0.889559 w_k1_DIS_s012
                                 + 0.846177 w_k1_FDX_s012
                                 + 1.01459 w_k1_IBM_s012
                                 + 0.816626 w_k1_INTC_s012
                                 + 0.871043 w_k1_KO_s012
                                 + 1.30101 w_k1_MAT_s012
                                 + 0.857837 w_k1_MSFT_s012
                                 + 0.979774 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.844338 w_k1_ADM_s012
                                 + 0.587014 w_k1_AXP_s012
                                 + 0.902252 w_k1_DIS_s012
                                 + 0.917919 w_k1_FDX_s012
                                 + 0.661997 w_k1_IBM_s012
                                 + 0.869196 w_k1_INTC_s012
                                 + 0.784014 w_k1_KO_s012
                                 + 0.70076 w_k1_MAT_s012
                                 + 0.923965 w_k1_MSFT_s012
                                 + 0.897814 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.06278 w_k1_ADM_s012
                                 + 1.3089 w_k1_AXP_s012 + 1.31408 w_k1_DIS_s012
                                 + 1.15863 w_k1_FDX_s012
                                 + 1.63737 w_k1_IBM_s012
                                 + 1.11359 w_k1_INTC_s012
                                 + 1.10701 w_k1_KO_s012 + 1.68157 w_k1_MAT_s012
                                 + 0.984564 w_k1_MSFT_s012
                                 + 1.32361 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.739155 w_k1_ADM_s012
                                 + 0.817842 w_k1_AXP_s012
                                 + 0.751157 w_k1_DIS_s012
                                 + 0.821117 w_k1_FDX_s012
                                 + 0.83806 w_k1_IBM_s012
                                 + 0.824473 w_k1_INTC_s012
                                 + 0.797843 w_k1_KO_s012
                                 + 1.04346 w_k1_MAT_s012
                                 + 0.856295 w_k1_MSFT_s012
                                 + 0.945259 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.573381 w_k1_ADM_s012
                                 + 0.620965 w_k1_AXP_s012
                                 + 0.521862 w_k1_DIS_s012
                                 + 0.692975 w_k1_FDX_s012
                                 + 0.706477 w_k1_IBM_s012
                                 + 0.822842 w_k1_INTC_s012
                                 + 0.543954 w_k1_KO_s012
                                 + 0.975009 w_k1_MAT_s012
                                 + 0.654664 w_k1_MSFT_s012
                                 + 0.81858 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.02248 w_k1_ADM_s012
                                 + 1.10622 w_k1_AXP_s012
                                 + 1.01895 w_k1_DIS_s012
                                 + 0.953254 w_k1_FDX_s012 + 1.271 w_k1_IBM_s012
                                 + 0.960196 w_k1_INTC_s012
                                 + 0.991257 w_k1_KO_s012
                                 + 1.62829 w_k1_MAT_s012
                                 + 0.862926 w_k1_MSFT_s012
                                 + 0.996833 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.2564 w_k1_ADM_s012
                                 + 1.33843 w_k1_AXP_s012 + 1.4287 w_k1_DIS_s012
                                 + 1.26778 w_k1_FDX_s012
                                 + 0.968081 w_k1_IBM_s012
                                 + 1.25853 w_k1_INTC_s012
                                 + 1.00948 w_k1_KO_s012 + 1.02005 w_k1_MAT_s012
                                 + 1.22177 w_k1_MSFT_s012
                                 + 1.18701 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.03916 w_k1_ADM_s012
                                 + 1.10391 w_k1_AXP_s012
                                 + 0.872604 w_k1_DIS_s012
                                 + 1.07648 w_k1_FDX_s012
                                 + 0.660621 w_k1_IBM_s012
                                 + 0.999628 w_k1_INTC_s012
                                 + 0.76829 w_k1_KO_s012 + 1.16413 w_k1_MAT_s012
                                 + 1.10116 w_k1_MSFT_s012
                                 + 1.13682 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.863799 w_k1_ADM_s013
                                 + 0.855048 w_k1_AXP_s013
                                 + 0.879675 w_k1_DIS_s013
                                 + 0.826187 w_k1_FDX_s013
                                 + 1.1863 w_k1_IBM_s013
                                 + 0.913485 w_k1_INTC_s013
                                 + 1.15045 w_k1_KO_s013 + 1.03654 w_k1_MAT_s013
                                 + 0.993864 w_k1_MSFT_s013
                                 + 0.92249 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.816959 w_k1_ADM_s013
                                 + 0.74598 w_k1_AXP_s013
                                 + 0.656981 w_k1_DIS_s013
                                 + 0.765717 w_k1_FDX_s013
                                 + 0.812879 w_k1_IBM_s013
                                 + 0.890193 w_k1_INTC_s013
                                 + 0.904053 w_k1_KO_s013
                                 + 0.544212 w_k1_MAT_s013
                                 + 0.824564 w_k1_MSFT_s013
                                 + 0.746108 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.05701 w_k1_ADM_s013
                                 + 1.29127 w_k1_AXP_s013
                                 + 1.84976 w_k1_DIS_s013
                                 + 1.29008 w_k1_FDX_s013
                                 + 1.55775 w_k1_IBM_s013
                                 + 1.30839 w_k1_INTC_s013
                                 + 1.72639 w_k1_KO_s013 + 1.88291 w_k1_MAT_s013
                                 + 1.61606 w_k1_MSFT_s013
                                 + 1.18859 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.08647 w_k1_ADM_s013
                                 + 1.16476 w_k1_AXP_s013
                                 + 1.32615 w_k1_DIS_s013
                                 + 1.22485 w_k1_FDX_s013
                                 + 0.968746 w_k1_IBM_s013
                                 + 0.979311 w_k1_INTC_s013
                                 + 1.05569 w_k1_KO_s013
                                 + 0.950695 w_k1_MAT_s013
                                 + 1.10593 w_k1_MSFT_s013
                                 + 0.981371 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.899903 w_k1_ADM_s013
                                 + 0.958212 w_k1_AXP_s013
                                 + 0.916346 w_k1_DIS_s013
                                 + 0.962901 w_k1_FDX_s013
                                 + 0.676816 w_k1_IBM_s013
                                 + 1.05962 w_k1_INTC_s013
                                 + 0.953143 w_k1_KO_s013
                                 + 0.496566 w_k1_MAT_s013
                                 + 1.10178 w_k1_MSFT_s013
                                 + 0.840165 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.00065 w_k1_ADM_s013
                                 + 1.31506 w_k1_AXP_s013
                                 + 1.16568 w_k1_DIS_s013
                                 + 1.08624 w_k1_FDX_s013
                                 + 1.38947 w_k1_IBM_s013
                                 + 1.53807 w_k1_INTC_s013
                                 + 1.40163 w_k1_KO_s013 + 1.28688 w_k1_MAT_s013
                                 + 1.15435 w_k1_MSFT_s013
                                 + 1.19594 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.943373 w_k1_ADM_s013
                                 + 1.14436 w_k1_AXP_s013
                                 + 0.933234 w_k1_DIS_s013
                                 + 0.961005 w_k1_FDX_s013
                                 + 0.991554 w_k1_IBM_s013
                                 + 1.01164 w_k1_INTC_s013
                                 + 0.941194 w_k1_KO_s013
                                 + 0.826641 w_k1_MAT_s013
                                 + 0.93895 w_k1_MSFT_s013
                                 + 0.85719 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.74111 w_k1_ADM_s013
                                 + 0.578529 w_k1_AXP_s013
                                 + 0.74255 w_k1_DIS_s013
                                 + 0.827356 w_k1_FDX_s013
                                 + 0.533073 w_k1_IBM_s013
                                 + 0.846923 w_k1_INTC_s013
                                 + 0.890968 w_k1_KO_s013
                                 + 0.593338 w_k1_MAT_s013
                                 + 0.926348 w_k1_MSFT_s013
                                 + 0.785333 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.45202 w_k1_ADM_s014
                                 + 1.60573 w_k1_AXP_s014
                                 + 2.49528 w_k1_DIS_s014
                                 + 1.46664 w_k1_FDX_s014
                                 + 1.50188 w_k1_IBM_s014
                                 + 1.27459 w_k1_INTC_s014
                                 + 1.63026 w_k1_KO_s014 + 1.57208 w_k1_MAT_s014
                                 + 1.40894 w_k1_MSFT_s014
                                 + 1.19453 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.848237 w_k1_ADM_s014
                                 + 1.14015 w_k1_AXP_s014
                                 + 1.10806 w_k1_DIS_s014
                                 + 0.999991 w_k1_FDX_s014
                                 + 1.80763 w_k1_IBM_s014
                                 + 0.930399 w_k1_INTC_s014
                                 + 1.07073 w_k1_KO_s014 + 1.58928 w_k1_MAT_s014
                                 + 0.904228 w_k1_MSFT_s014
                                 + 1.07706 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.01224 w_k1_ADM_s014
                                 + 1.33184 w_k1_AXP_s014
                                 + 1.22633 w_k1_DIS_s014
                                 + 1.19581 w_k1_FDX_s014 + 1.2519 w_k1_IBM_s014
                                 + 1.10845 w_k1_INTC_s014
                                 + 1.26395 w_k1_KO_s014 + 1.30602 w_k1_MAT_s014
                                 + 1.09543 w_k1_MSFT_s014
                                 + 1.08321 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.81773 w_k1_ADM_s014
                                 + 0.775264 w_k1_AXP_s014
                                 + 0.823135 w_k1_DIS_s014
                                 + 0.853054 w_k1_FDX_s014
                                 + 0.76203 w_k1_IBM_s014
                                 + 0.818956 w_k1_INTC_s014
                                 + 0.858014 w_k1_KO_s014
                                 + 0.941837 w_k1_MAT_s014
                                 + 0.911197 w_k1_MSFT_s014
                                 + 0.865051 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.11135 w_k1_ADM_s014
                                 + 0.884846 w_k1_AXP_s014
                                 + 1.10396 w_k1_DIS_s014
                                 + 0.967333 w_k1_FDX_s014
                                 + 1.1672 w_k1_IBM_s014
                                 + 0.878694 w_k1_INTC_s014
                                 + 1.0997 w_k1_KO_s014 + 0.929993 w_k1_MAT_s014
                                 + 0.958158 w_k1_MSFT_s014
                                 + 0.935778 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.781055 w_k1_ADM_s014
                                 + 0.696133 w_k1_AXP_s014
                                 + 0.6514 w_k1_DIS_s014
                                 + 0.707501 w_k1_FDX_s014
                                 + 0.921241 w_k1_IBM_s014
                                 + 0.795815 w_k1_INTC_s014
                                 + 0.720098 w_k1_KO_s014
                                 + 0.691698 w_k1_MAT_s014
                                 + 0.649848 w_k1_MSFT_s014
                                 + 0.728169 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.12273 w_k1_ADM_s014
                                 + 1.2989 w_k1_AXP_s014 + 1.74721 w_k1_DIS_s014
                                 + 1.27184 w_k1_FDX_s014
                                 + 1.25509 w_k1_IBM_s014
                                 + 1.17935 w_k1_INTC_s014
                                 + 1.29786 w_k1_KO_s014 + 1.19697 w_k1_MAT_s014
                                 + 1.17119 w_k1_MSFT_s014
                                 + 1.18921 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.916857 w_k1_ADM_s014
                                 + 0.818709 w_k1_AXP_s014
                                 + 1.14811 w_k1_DIS_s014
                                 + 1.12964 w_k1_FDX_s014 + 0.8356 w_k1_IBM_s014
                                 + 1.05478 w_k1_INTC_s014
                                 + 0.831636 w_k1_KO_s014
                                 + 0.61458 w_k1_MAT_s014
                                 + 1.08062 w_k1_MSFT_s014
                                 + 0.87346 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.28533 w_k1_ADM_s015
                                 + 1.18705 w_k1_AXP_s015
                                 + 1.38651 w_k1_DIS_s015
                                 + 1.28597 w_k1_FDX_s015
                                 + 0.768485 w_k1_IBM_s015
                                 + 0.997249 w_k1_INTC_s015
                                 + 1.08016 w_k1_KO_s015 + 1.02313 w_k1_MAT_s015
                                 + 1.22654 w_k1_MSFT_s015
                                 + 1.06756 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.907237 w_k1_ADM_s015
                                 + 1.14054 w_k1_AXP_s015
                                 + 1.07016 w_k1_DIS_s015
                                 + 0.944588 w_k1_FDX_s015
                                 + 1.60321 w_k1_IBM_s015
                                 + 0.788872 w_k1_INTC_s015
                                 + 1.20996 w_k1_KO_s015 + 1.93894 w_k1_MAT_s015
                                 + 0.897002 w_k1_MSFT_s015
                                 + 1.06332 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.900106 w_k1_ADM_s015
                                 + 0.805875 w_k1_AXP_s015
                                 + 0.947837 w_k1_DIS_s015
                                 + 0.968266 w_k1_FDX_s015
                                 + 0.722474 w_k1_IBM_s015
                                 + 0.888076 w_k1_INTC_s015
                                 + 0.764885 w_k1_KO_s015
                                 + 0.633655 w_k1_MAT_s015
                                 + 0.872296 w_k1_MSFT_s015
                                 + 0.867313 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.07569 w_k1_ADM_s015
                                 + 1.27768 w_k1_AXP_s015 + 1.4706 w_k1_DIS_s015
                                 + 1.19577 w_k1_FDX_s015
                                 + 1.24186 w_k1_IBM_s015
                                 + 1.20777 w_k1_INTC_s015
                                 + 1.28313 w_k1_KO_s015 + 1.2192 w_k1_MAT_s015
                                 + 1.23952 w_k1_MSFT_s015
                                 + 1.15515 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.748163 w_k1_ADM_s015
                                 + 0.67004 w_k1_AXP_s015
                                 + 0.706167 w_k1_DIS_s015
                                 + 0.711582 w_k1_FDX_s015
                                 + 0.814661 w_k1_IBM_s015
                                 + 0.821334 w_k1_INTC_s015
                                 + 0.829297 w_k1_KO_s015
                                 + 0.788109 w_k1_MAT_s015
                                 + 0.702248 w_k1_MSFT_s015
                                 + 0.906671 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.859624 w_k1_ADM_s015
                                 + 0.885943 w_k1_AXP_s015
                                 + 0.990579 w_k1_DIS_s015
                                 + 0.787643 w_k1_FDX_s015
                                 + 1.18488 w_k1_IBM_s015
                                 + 0.729157 w_k1_INTC_s015
                                 + 1.10284 w_k1_KO_s015 + 1.2674 w_k1_MAT_s015
                                 + 0.90892 w_k1_MSFT_s015
                                 + 0.996611 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.04234 w_k1_ADM_s015
                                 + 1.01091 w_k1_AXP_s015
                                 + 0.910033 w_k1_DIS_s015
                                 + 0.936632 w_k1_FDX_s015
                                 + 0.988677 w_k1_IBM_s015
                                 + 0.91093 w_k1_INTC_s015
                                 + 1.03538 w_k1_KO_s015 + 1.08119 w_k1_MAT_s015
                                 + 0.928634 w_k1_MSFT_s015
                                 + 0.987233 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.03912 w_k1_ADM_s015
                                 + 0.930721 w_k1_AXP_s015
                                 + 1.06403 w_k1_DIS_s015
                                 + 0.982414 w_k1_FDX_s015
                                 + 0.751423 w_k1_IBM_s015
                                 + 1.00604 w_k1_INTC_s015
                                 + 1.01652 w_k1_KO_s015
                                 + 0.850044 w_k1_MAT_s015
                                 + 0.919818 w_k1_MSFT_s015
                                 + 1.02627 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.18021 w_k1_ADM_s016
                                 + 1.16773 w_k1_AXP_s016
                                 + 0.924263 w_k1_DIS_s016
                                 + 1.07558 w_k1_FDX_s016
                                 + 0.995174 w_k1_IBM_s016
                                 + 1.27433 w_k1_INTC_s016
                                 + 1.13947 w_k1_KO_s016
                                 + 0.994027 w_k1_MAT_s016
                                 + 1.20346 w_k1_MSFT_s016
                                 + 1.06397 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.04373 w_k1_ADM_s016
                                 + 1.23982 w_k1_AXP_s016
                                 + 1.51334 w_k1_DIS_s016
                                 + 1.45686 w_k1_FDX_s016
                                 + 0.744774 w_k1_IBM_s016
                                 + 1.1912 w_k1_INTC_s016 + 1.01846 w_k1_KO_s016
                                 + 0.692493 w_k1_MAT_s016
                                 + 1.34786 w_k1_MSFT_s016
                                 + 1.23772 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.779232 w_k1_ADM_s016
                                 + 0.7184 w_k1_AXP_s016
                                 + 0.719113 w_k1_DIS_s016
                                 + 0.73568 w_k1_FDX_s016
                                 + 0.91906 w_k1_IBM_s016
                                 + 1.10911 w_k1_INTC_s016
                                 + 0.850791 w_k1_KO_s016
                                 + 0.796514 w_k1_MAT_s016
                                 + 0.933569 w_k1_MSFT_s016
                                 + 0.845091 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.04945 w_k1_ADM_s016
                                 + 1.11274 w_k1_AXP_s016
                                 + 1.02506 w_k1_DIS_s016
                                 + 1.01363 w_k1_FDX_s016 + 1.4025 w_k1_IBM_s016
                                 + 1.12416 w_k1_INTC_s016
                                 + 1.26034 w_k1_KO_s016 + 1.28687 w_k1_MAT_s016
                                 + 1.06325 w_k1_MSFT_s016
                                 + 1.06003 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.35041 w_k1_ADM_s016
                                 + 1.47271 w_k1_AXP_s016
                                 + 1.04958 w_k1_DIS_s016
                                 + 1.26412 w_k1_FDX_s016
                                 + 0.976378 w_k1_IBM_s016
                                 + 1.10282 w_k1_INTC_s016 + 1.0645 w_k1_KO_s016
                                 + 1.22686 w_k1_MAT_s016
                                 + 1.20579 w_k1_MSFT_s016 + 1.0225 w_k1_PG_s016
                                 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.963094 w_k1_ADM_s016
                                 + 1.10243 w_k1_AXP_s016 + 1.234 w_k1_DIS_s016
                                 + 1.1286 w_k1_FDX_s016 + 1.05929 w_k1_IBM_s016
                                 + 1.44171 w_k1_INTC_s016
                                 + 1.03974 w_k1_KO_s016
                                 + 0.669363 w_k1_MAT_s016
                                 + 1.30757 w_k1_MSFT_s016
                                 + 0.947671 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.825012 w_k1_ADM_s016
                                 + 0.844941 w_k1_AXP_s016
                                 + 0.895078 w_k1_DIS_s016
                                 + 0.97892 w_k1_FDX_s016
                                 + 0.75329 w_k1_IBM_s016
                                 + 0.859196 w_k1_INTC_s016
                                 + 0.881929 w_k1_KO_s016
                                 + 0.774364 w_k1_MAT_s016
                                 + 1.09009 w_k1_MSFT_s016
                                 + 0.938135 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.36672 w_k1_ADM_s016
                                 + 1.62135 w_k1_AXP_s016
                                 + 1.76091 w_k1_DIS_s016
                                 + 1.39855 w_k1_FDX_s016 + 1.4893 w_k1_IBM_s016
                                 + 1.41311 w_k1_INTC_s016
                                 + 1.47516 w_k1_KO_s016 + 1.27067 w_k1_MAT_s016
                                 + 1.47867 w_k1_MSFT_s016
                                 + 1.14792 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.843011 w_k1_ADM_s017
                                 + 0.751263 w_k1_AXP_s017
                                 + 0.840565 w_k1_DIS_s017
                                 + 0.835886 w_k1_FDX_s017
                                 + 0.673929 w_k1_IBM_s017
                                 + 0.94391 w_k1_INTC_s017
                                 + 0.877658 w_k1_KO_s017
                                 + 0.626835 w_k1_MAT_s017
                                 + 0.979932 w_k1_MSFT_s017
                                 + 0.895128 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.900692 w_k1_ADM_s017
                                 + 1.01325 w_k1_AXP_s017
                                 + 1.16998 w_k1_DIS_s017
                                 + 0.933414 w_k1_FDX_s017
                                 + 1.27738 w_k1_IBM_s017
                                 + 1.05529 w_k1_INTC_s017
                                 + 1.23911 w_k1_KO_s017 + 1.1995 w_k1_MAT_s017
                                 + 1.0827 w_k1_MSFT_s017 + 1.00435 w_k1_PG_s017
                                 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.635194 w_k1_ADM_s017
                                 + 0.778111 w_k1_AXP_s017
                                 + 0.85927 w_k1_DIS_s017
                                 + 0.775497 w_k1_FDX_s017
                                 + 1.07406 w_k1_IBM_s017
                                 + 0.958514 w_k1_INTC_s017
                                 + 0.97394 w_k1_KO_s017 + 1.03747 w_k1_MAT_s017
                                 + 0.888262 w_k1_MSFT_s017
                                 + 0.892438 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.1563 w_k1_ADM_s017
                                 + 1.27868 w_k1_AXP_s017
                                 + 1.33567 w_k1_DIS_s017
                                 + 1.22334 w_k1_FDX_s017
                                 + 0.909071 w_k1_IBM_s017
                                 + 1.09426 w_k1_INTC_s017
                                 + 1.22576 w_k1_KO_s017 + 0.97025 w_k1_MAT_s017
                                 + 1.1995 w_k1_MSFT_s017 + 1.05499 w_k1_PG_s017
                                 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.801173 w_k1_ADM_s017
                                 + 0.63678 w_k1_AXP_s017
                                 + 0.679265 w_k1_DIS_s017
                                 + 0.799374 w_k1_FDX_s017
                                 + 0.819365 w_k1_IBM_s017
                                 + 0.918035 w_k1_INTC_s017
                                 + 0.815837 w_k1_KO_s017
                                 + 0.737849 w_k1_MAT_s017
                                 + 1.10041 w_k1_MSFT_s017
                                 + 0.877139 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.900078 w_k1_ADM_s017
                                 + 1.40443 w_k1_AXP_s017
                                 + 1.56325 w_k1_DIS_s017
                                 + 1.22814 w_k1_FDX_s017
                                 + 1.46465 w_k1_IBM_s017
                                 + 1.37404 w_k1_INTC_s017 + 1.5717 w_k1_KO_s017
                                 + 1.66656 w_k1_MAT_s017
                                 + 1.43272 w_k1_MSFT_s017
                                 + 1.13393 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.977133 w_k1_ADM_s017
                                 + 0.931499 w_k1_AXP_s017
                                 + 1.0372 w_k1_DIS_s017 + 1.22404 w_k1_FDX_s017
                                 + 0.611033 w_k1_IBM_s017
                                 + 1.22624 w_k1_INTC_s017
                                 + 0.842389 w_k1_KO_s017
                                 + 0.61665 w_k1_MAT_s017
                                 + 1.19965 w_k1_MSFT_s017
                                 + 0.899905 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.1793 w_k1_ADM_s017
                                 + 1.18638 w_k1_AXP_s017
                                 + 0.933175 w_k1_DIS_s017
                                 + 1.03044 w_k1_FDX_s017
                                 + 0.840453 w_k1_IBM_s017
                                 + 0.960069 w_k1_INTC_s017
                                 + 0.849752 w_k1_KO_s017
                                 + 1.01033 w_k1_MAT_s017
                                 + 0.947356 w_k1_MSFT_s017
                                 + 0.977395 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.950246 w_k1_ADM_s100
                                 + 0.911576 w_k1_AXP_s100
                                 + 0.9853 w_k1_DIS_s100 + 1.03944 w_k1_FDX_s100
                                 + 1.0045 w_k1_IBM_s100
                                 + 1.02043 w_k1_INTC_s100
                                 + 1.02117 w_k1_KO_s100
                                 + 0.928521 w_k1_MAT_s100
                                 + 1.05014 w_k1_MSFT_s100
                                 + 1.02522 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.941776 w_k1_ADM_s100
                                 + 0.598154 w_k1_AXP_s100
                                 + 0.564673 w_k1_DIS_s100
                                 + 0.737442 w_k1_FDX_s100
                                 + 0.629852 w_k1_IBM_s100
                                 + 0.816826 w_k1_INTC_s100
                                 + 0.696039 w_k1_KO_s100
                                 + 0.490328 w_k1_MAT_s100
                                 + 0.840883 w_k1_MSFT_s100
                                 + 0.75865 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.27079 w_k1_ADM_s100
                                 + 0.962761 w_k1_AXP_s100
                                 + 1.19778 w_k1_DIS_s100
                                 + 1.11263 w_k1_FDX_s100
                                 + 0.876084 w_k1_IBM_s100
                                 + 0.958297 w_k1_INTC_s100
                                 + 1.03099 w_k1_KO_s100 + 1.15869 w_k1_MAT_s100
                                 + 1.29734 w_k1_MSFT_s100 + 1.0748 w_k1_PG_s100
                                 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.0829 w_k1_ADM_s100
                                 + 0.913404 w_k1_AXP_s100
                                 + 0.890793 w_k1_DIS_s100
                                 + 1.06417 w_k1_FDX_s100
                                 + 0.579048 w_k1_IBM_s100
                                 + 0.990093 w_k1_INTC_s100
                                 + 0.703563 w_k1_KO_s100 + 0.5548 w_k1_MAT_s100
                                 + 1.01305 w_k1_MSFT_s100
                                 + 0.799758 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.45377 w_k1_ADM_s100
                                 + 1.4793 w_k1_AXP_s100 + 1.74499 w_k1_DIS_s100
                                 + 1.43681 w_k1_FDX_s100
                                 + 1.20699 w_k1_IBM_s100
                                 + 1.30003 w_k1_INTC_s100
                                 + 1.35584 w_k1_KO_s100 + 1.38775 w_k1_MAT_s100
                                 + 1.50166 w_k1_MSFT_s100
                                 + 1.22491 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.846038 w_k1_ADM_s100
                                 + 0.744852 w_k1_AXP_s100
                                 + 0.969997 w_k1_DIS_s100
                                 + 0.807282 w_k1_FDX_s100
                                 + 1.15448 w_k1_IBM_s100
                                 + 1.02812 w_k1_INTC_s100
                                 + 1.15374 w_k1_KO_s100 + 1.11013 w_k1_MAT_s100
                                 + 0.968353 w_k1_MSFT_s100
                                 + 1.03166 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.882316 w_k1_ADM_s100
                                 + 0.849276 w_k1_AXP_s100
                                 + 0.678818 w_k1_DIS_s100
                                 + 0.807653 w_k1_FDX_s100
                                 + 0.946883 w_k1_IBM_s100
                                 + 0.963177 w_k1_INTC_s100
                                 + 0.864618 w_k1_KO_s100
                                 + 1.06145 w_k1_MAT_s100
                                 + 0.83233 w_k1_MSFT_s100
                                 + 0.890188 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.13219 w_k1_ADM_s100
                                 + 1.05401 w_k1_AXP_s100
                                 + 1.01383 w_k1_DIS_s100
                                 + 1.04504 w_k1_FDX_s100
                                 + 1.41333 w_k1_IBM_s100
                                 + 1.18049 w_k1_INTC_s100
                                 + 1.25984 w_k1_KO_s100 + 1.93466 w_k1_MAT_s100
                                 + 1.16442 w_k1_MSFT_s100
                                 + 1.12133 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.837168 w_k1_ADM_s101
                                 + 0.812098 w_k1_AXP_s101
                                 + 1.16508 w_k1_DIS_s101
                                 + 0.996244 w_k1_FDX_s101
                                 + 1.03728 w_k1_IBM_s101
                                 + 0.847456 w_k1_INTC_s101
                                 + 1.06504 w_k1_KO_s101 + 0.79888 w_k1_MAT_s101
                                 + 1.09897 w_k1_MSFT_s101
                                 + 0.933432 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.847575 w_k1_ADM_s101
                                 + 0.816614 w_k1_AXP_s101
                                 + 1.09374 w_k1_DIS_s101
                                 + 1.11168 w_k1_FDX_s101
                                 + 0.701409 w_k1_IBM_s101
                                 + 0.984081 w_k1_INTC_s101
                                 + 0.914318 w_k1_KO_s101
                                 + 0.814952 w_k1_MAT_s101
                                 + 1.18423 w_k1_MSFT_s101 + 1.0929 w_k1_PG_s101
                                 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.03192 w_k1_ADM_s101
                                 + 1.04025 w_k1_AXP_s101
                                 + 0.898272 w_k1_DIS_s101
                                 + 1.10537 w_k1_FDX_s101
                                 + 0.658832 w_k1_IBM_s101
                                 + 0.734167 w_k1_INTC_s101
                                 + 0.680864 w_k1_KO_s101
                                 + 0.753873 w_k1_MAT_s101
                                 + 0.905362 w_k1_MSFT_s101
                                 + 0.919241 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.999285 w_k1_ADM_s101
                                 + 1.14282 w_k1_AXP_s101
                                 + 1.02942 w_k1_DIS_s101 + 1.0508 w_k1_FDX_s101
                                 + 1.49657 w_k1_IBM_s101
                                 + 1.21774 w_k1_INTC_s101
                                 + 1.07647 w_k1_KO_s101 + 1.57827 w_k1_MAT_s101
                                 + 1.13677 w_k1_MSFT_s101
                                 + 1.09396 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.17638 w_k1_ADM_s101
                                 + 1.37195 w_k1_AXP_s101
                                 + 1.51311 w_k1_DIS_s101
                                 + 1.30251 w_k1_FDX_s101
                                 + 0.97124 w_k1_IBM_s101
                                 + 1.22431 w_k1_INTC_s101 + 1.1999 w_k1_KO_s101
                                 + 0.995334 w_k1_MAT_s101
                                 + 1.3037 w_k1_MSFT_s101 + 1.11072 w_k1_PG_s101
                                 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.32099 w_k1_ADM_s101
                                 + 1.59604 w_k1_AXP_s101 + 2.0426 w_k1_DIS_s101
                                 + 1.45584 w_k1_FDX_s101
                                 + 1.36597 w_k1_IBM_s101
                                 + 1.01983 w_k1_INTC_s101
                                 + 1.45689 w_k1_KO_s101 + 1.87658 w_k1_MAT_s101
                                 + 1.1933 w_k1_MSFT_s101 + 1.34713 w_k1_PG_s101
                                 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.641135 w_k1_ADM_s101
                                 + 0.664699 w_k1_AXP_s101
                                 + 0.70532 w_k1_DIS_s101
                                 + 0.697886 w_k1_FDX_s101
                                 + 0.960199 w_k1_IBM_s101
                                 + 0.717597 w_k1_INTC_s101
                                 + 0.826375 w_k1_KO_s101
                                 + 0.793987 w_k1_MAT_s101
                                 + 0.847835 w_k1_MSFT_s101
                                 + 0.814574 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.972788 w_k1_ADM_s101
                                 + 0.861449 w_k1_AXP_s101
                                 + 0.865522 w_k1_DIS_s101
                                 + 0.898915 w_k1_FDX_s101
                                 + 0.809995 w_k1_IBM_s101
                                 + 0.905652 w_k1_INTC_s101
                                 + 0.775098 w_k1_KO_s101
                                 + 0.744975 w_k1_MAT_s101
                                 + 0.9569 w_k1_MSFT_s101
                                 + 0.862603 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.1516 w_k1_ADM_s102
                                 + 1.08058 w_k1_AXP_s102
                                 + 1.28747 w_k1_DIS_s102
                                 + 0.959926 w_k1_FDX_s102
                                 + 1.38325 w_k1_IBM_s102
                                 + 1.01059 w_k1_INTC_s102
                                 + 1.09186 w_k1_KO_s102 + 1.20616 w_k1_MAT_s102
                                 + 0.8981 w_k1_MSFT_s102 + 1.05801 w_k1_PG_s102
                                 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.10091 w_k1_ADM_s102
                                 + 1.01866 w_k1_AXP_s102
                                 + 1.36658 w_k1_DIS_s102
                                 + 1.12051 w_k1_FDX_s102
                                 + 0.865442 w_k1_IBM_s102
                                 + 0.943189 w_k1_INTC_s102
                                 + 1.0939 w_k1_KO_s102 + 0.897281 w_k1_MAT_s102
                                 + 1.11146 w_k1_MSFT_s102
                                 + 1.01332 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.813812 w_k1_ADM_s102
                                 + 0.938164 w_k1_AXP_s102
                                 + 0.941757 w_k1_DIS_s102
                                 + 0.823875 w_k1_FDX_s102
                                 + 0.899168 w_k1_IBM_s102
                                 + 0.870158 w_k1_INTC_s102
                                 + 0.913617 w_k1_KO_s102
                                 + 0.795203 w_k1_MAT_s102
                                 + 0.740658 w_k1_MSFT_s102
                                 + 0.870979 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.00403 w_k1_ADM_s102
                                 + 1.53455 w_k1_AXP_s102
                                 + 1.56354 w_k1_DIS_s102
                                 + 1.21156 w_k1_FDX_s102
                                 + 1.44161 w_k1_IBM_s102
                                 + 1.31372 w_k1_INTC_s102
                                 + 1.43423 w_k1_KO_s102 + 1.43458 w_k1_MAT_s102
                                 + 1.03323 w_k1_MSFT_s102 + 1.1262 w_k1_PG_s102
                                 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.935953 w_k1_ADM_s102
                                 + 1.04563 w_k1_AXP_s102
                                 + 1.07977 w_k1_DIS_s102
                                 + 0.998873 w_k1_FDX_s102
                                 + 0.960891 w_k1_IBM_s102
                                 + 0.924235 w_k1_INTC_s102
                                 + 0.963466 w_k1_KO_s102
                                 + 0.999409 w_k1_MAT_s102
                                 + 0.867357 w_k1_MSFT_s102
                                 + 1.00353 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.82838 w_k1_ADM_s102
                                 + 0.941705 w_k1_AXP_s102
                                 + 1.02748 w_k1_DIS_s102
                                 + 0.865305 w_k1_FDX_s102
                                 + 1.28903 w_k1_IBM_s102
                                 + 1.00177 w_k1_INTC_s102
                                 + 1.02699 w_k1_KO_s102 + 1.14127 w_k1_MAT_s102
                                 + 0.898942 w_k1_MSFT_s102
                                 + 0.967919 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.835926 w_k1_ADM_s102
                                 + 0.632301 w_k1_AXP_s102
                                 + 0.597839 w_k1_DIS_s102
                                 + 0.75684 w_k1_FDX_s102
                                 + 0.552426 w_k1_IBM_s102
                                 + 0.737721 w_k1_INTC_s102
                                 + 0.67272 w_k1_KO_s102
                                 + 0.558464 w_k1_MAT_s102
                                 + 0.623283 w_k1_MSFT_s102
                                 + 0.816348 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.829968 w_k1_ADM_s102
                                 + 0.704249 w_k1_AXP_s102
                                 + 0.844012 w_k1_DIS_s102
                                 + 0.878332 w_k1_FDX_s102
                                 + 0.705662 w_k1_IBM_s102
                                 + 0.840236 w_k1_INTC_s102
                                 + 0.860795 w_k1_KO_s102
                                 + 0.72337 w_k1_MAT_s102
                                 + 0.961875 w_k1_MSFT_s102
                                 + 0.927424 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.00672 w_k1_ADM_s103
                                 + 1.29216 w_k1_AXP_s103
                                 + 1.05364 w_k1_DIS_s103
                                 + 1.10098 w_k1_FDX_s103
                                 + 1.38068 w_k1_IBM_s103
                                 + 1.06764 w_k1_INTC_s103 + 1.2496 w_k1_KO_s103
                                 + 1.34327 w_k1_MAT_s103
                                 + 1.04514 w_k1_MSFT_s103
                                 + 1.18091 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.797375 w_k1_ADM_s103
                                 + 0.568964 w_k1_AXP_s103
                                 + 0.918873 w_k1_DIS_s103
                                 + 0.943422 w_k1_FDX_s103
                                 + 0.672114 w_k1_IBM_s103
                                 + 0.980499 w_k1_INTC_s103
                                 + 0.843979 w_k1_KO_s103
                                 + 0.600405 w_k1_MAT_s103
                                 + 0.94842 w_k1_MSFT_s103
                                 + 0.910565 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.912374 w_k1_ADM_s103
                                 + 0.849074 w_k1_AXP_s103
                                 + 0.899097 w_k1_DIS_s103
                                 + 0.865334 w_k1_FDX_s103
                                 + 1.14637 w_k1_IBM_s103
                                 + 0.942389 w_k1_INTC_s103
                                 + 1.04314 w_k1_KO_s103
                                 + 0.929323 w_k1_MAT_s103
                                 + 0.817266 w_k1_MSFT_s103
                                 + 0.992263 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.07401 w_k1_ADM_s103
                                 + 1.37812 w_k1_AXP_s103
                                 + 1.59692 w_k1_DIS_s103
                                 + 1.37479 w_k1_FDX_s103
                                 + 1.70102 w_k1_IBM_s103
                                 + 1.32022 w_k1_INTC_s103
                                 + 1.63957 w_k1_KO_s103 + 1.59405 w_k1_MAT_s103
                                 + 1.36993 w_k1_MSFT_s103
                                 + 1.26183 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.800327 w_k1_ADM_s103
                                 + 0.548551 w_k1_AXP_s103
                                 + 0.634153 w_k1_DIS_s103
                                 + 0.777526 w_k1_FDX_s103
                                 + 0.614726 w_k1_IBM_s103
                                 + 0.726065 w_k1_INTC_s103
                                 + 0.745957 w_k1_KO_s103
                                 + 0.482158 w_k1_MAT_s103
                                 + 0.721017 w_k1_MSFT_s103
                                 + 0.801973 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.02378 w_k1_ADM_s103
                                 + 0.971349 w_k1_AXP_s103
                                 + 1.57877 w_k1_DIS_s103
                                 + 1.23809 w_k1_FDX_s103
                                 + 1.03294 w_k1_IBM_s103
                                 + 1.13715 w_k1_INTC_s103 + 1.4013 w_k1_KO_s103
                                 + 0.905548 w_k1_MAT_s103
                                 + 1.39294 w_k1_MSFT_s103
                                 + 1.17616 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.05786 w_k1_ADM_s103
                                 + 0.823461 w_k1_AXP_s103
                                 + 0.767853 w_k1_DIS_s103
                                 + 1.03651 w_k1_FDX_s103
                                 + 0.683864 w_k1_IBM_s103
                                 + 0.809177 w_k1_INTC_s103
                                 + 0.706668 w_k1_KO_s103
                                 + 0.601777 w_k1_MAT_s103
                                 + 0.873115 w_k1_MSFT_s103
                                 + 0.969355 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.973124 w_k1_ADM_s103
                                 + 0.944868 w_k1_AXP_s103
                                 + 1.12208 w_k1_DIS_s103
                                 + 1.07317 w_k1_FDX_s103
                                 + 0.80049 w_k1_IBM_s103
                                 + 1.10627 w_k1_INTC_s103
                                 + 0.951997 w_k1_KO_s103
                                 + 0.524964 w_k1_MAT_s103
                                 + 1.08433 w_k1_MSFT_s103
                                 + 1.04116 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.06588 w_k1_ADM_s104
                                 + 0.918204 w_k1_AXP_s104
                                 + 1.09195 w_k1_DIS_s104
                                 + 0.966058 w_k1_FDX_s104
                                 + 1.06989 w_k1_IBM_s104
                                 + 0.94568 w_k1_INTC_s104
                                 + 1.00195 w_k1_KO_s104 + 1.09451 w_k1_MAT_s104
                                 + 1.00953 w_k1_MSFT_s104
                                 + 1.02147 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.905291 w_k1_ADM_s104
                                 + 0.763456 w_k1_AXP_s104
                                 + 0.924691 w_k1_DIS_s104
                                 + 0.926325 w_k1_FDX_s104
                                 + 0.693984 w_k1_IBM_s104
                                 + 0.888271 w_k1_INTC_s104
                                 + 0.82688 w_k1_KO_s104
                                 + 0.707769 w_k1_MAT_s104
                                 + 0.973408 w_k1_MSFT_s104
                                 + 0.940044 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.934573 w_k1_ADM_s104
                                 + 1.01921 w_k1_AXP_s104
                                 + 1.03221 w_k1_DIS_s104
                                 + 0.955871 w_k1_FDX_s104
                                 + 1.27942 w_k1_IBM_s104
                                 + 1.10482 w_k1_INTC_s104
                                 + 1.10114 w_k1_KO_s104 + 1.02273 w_k1_MAT_s104
                                 + 1.00716 w_k1_MSFT_s104 + 1.0653 w_k1_PG_s104
                                 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.44745 w_k1_ADM_s104
                                 + 1.3608 w_k1_AXP_s104 + 1.70798 w_k1_DIS_s104
                                 + 1.48712 w_k1_FDX_s104
                                 + 1.11218 w_k1_IBM_s104
                                 + 1.32864 w_k1_INTC_s104
                                 + 1.47575 w_k1_KO_s104 + 1.00771 w_k1_MAT_s104
                                 + 1.59933 w_k1_MSFT_s104
                                 + 1.19402 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.09618 w_k1_ADM_s104
                                 + 1.36944 w_k1_AXP_s104
                                 + 1.33221 w_k1_DIS_s104
                                 + 1.21748 w_k1_FDX_s104
                                 + 1.55707 w_k1_IBM_s104
                                 + 1.34757 w_k1_INTC_s104
                                 + 1.40277 w_k1_KO_s104 + 1.82823 w_k1_MAT_s104
                                 + 1.52423 w_k1_MSFT_s104
                                 + 1.24645 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.24972 w_k1_ADM_s104
                                 + 1.24797 w_k1_AXP_s104
                                 + 1.61705 w_k1_DIS_s104
                                 + 1.22282 w_k1_FDX_s104
                                 + 1.26481 w_k1_IBM_s104
                                 + 1.10818 w_k1_INTC_s104
                                 + 1.26008 w_k1_KO_s104 + 1.01243 w_k1_MAT_s104
                                 + 1.13016 w_k1_MSFT_s104
                                 + 1.29005 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.34315 w_k1_ADM_s104
                                 + 1.28451 w_k1_AXP_s104
                                 + 1.37923 w_k1_DIS_s104
                                 + 1.26361 w_k1_FDX_s104
                                 + 0.851388 w_k1_IBM_s104
                                 + 1.07262 w_k1_INTC_s104
                                 + 1.18993 w_k1_KO_s104 + 0.9433 w_k1_MAT_s104
                                 + 1.15679 w_k1_MSFT_s104
                                 + 1.04751 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.869296 w_k1_ADM_s104
                                 + 0.743918 w_k1_AXP_s104
                                 + 0.768458 w_k1_DIS_s104
                                 + 0.768957 w_k1_FDX_s104
                                 + 0.869069 w_k1_IBM_s104
                                 + 0.789525 w_k1_INTC_s104
                                 + 0.835315 w_k1_KO_s104
                                 + 0.808443 w_k1_MAT_s104
                                 + 0.809395 w_k1_MSFT_s104
                                 + 0.871903 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.09106 w_k1_ADM_s105
                                 + 1.32549 w_k1_AXP_s105
                                 + 1.52491 w_k1_DIS_s105
                                 + 1.17665 w_k1_FDX_s105
                                 + 1.36902 w_k1_IBM_s105
                                 + 1.31883 w_k1_INTC_s105
                                 + 1.30346 w_k1_KO_s105 + 1.56893 w_k1_MAT_s105
                                 + 1.2306 w_k1_MSFT_s105 + 1.20007 w_k1_PG_s105
                                 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.860428 w_k1_ADM_s105
                                 + 0.959627 w_k1_AXP_s105
                                 + 1.20657 w_k1_DIS_s105 + 1.0365 w_k1_FDX_s105
                                 + 1.11264 w_k1_IBM_s105
                                 + 1.11051 w_k1_INTC_s105
                                 + 1.00446 w_k1_KO_s105
                                 + 0.768192 w_k1_MAT_s105
                                 + 1.051 w_k1_MSFT_s105 + 1.01661 w_k1_PG_s105
                                 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.929024 w_k1_ADM_s105
                                 + 0.946836 w_k1_AXP_s105
                                 + 0.879448 w_k1_DIS_s105
                                 + 0.909638 w_k1_FDX_s105
                                 + 1.19569 w_k1_IBM_s105
                                 + 0.960043 w_k1_INTC_s105
                                 + 0.917716 w_k1_KO_s105
                                 + 1.95228 w_k1_MAT_s105
                                 + 0.947264 w_k1_MSFT_s105
                                 + 1.07632 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.827401 w_k1_ADM_s105
                                 + 0.700758 w_k1_AXP_s105
                                 + 0.789485 w_k1_DIS_s105
                                 + 0.818281 w_k1_FDX_s105
                                 + 0.795505 w_k1_IBM_s105
                                 + 1.18032 w_k1_INTC_s105
                                 + 0.747551 w_k1_KO_s105
                                 + 0.63645 w_k1_MAT_s105
                                 + 0.905647 w_k1_MSFT_s105
                                 + 0.854171 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.39479 w_k1_ADM_s105
                                 + 1.20933 w_k1_AXP_s105
                                 + 1.10444 w_k1_DIS_s105 + 1.2073 w_k1_FDX_s105
                                 + 0.856831 w_k1_IBM_s105
                                 + 1.23897 w_k1_INTC_s105
                                 + 0.883827 w_k1_KO_s105
                                 + 0.717594 w_k1_MAT_s105
                                 + 1.00472 w_k1_MSFT_s105
                                 + 1.11814 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.819673 w_k1_ADM_s105
                                 + 0.673006 w_k1_AXP_s105
                                 + 0.67236 w_k1_DIS_s105
                                 + 0.88866 w_k1_FDX_s105
                                 + 0.57604 w_k1_IBM_s105
                                 + 0.992952 w_k1_INTC_s105
                                 + 0.640379 w_k1_KO_s105
                                 + 0.571038 w_k1_MAT_s105
                                 + 0.835944 w_k1_MSFT_s105
                                 + 0.851613 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.553664 w_k1_ADM_s105
                                 + 0.671036 w_k1_AXP_s105
                                 + 0.705839 w_k1_DIS_s105
                                 + 0.765419 w_k1_FDX_s105
                                 + 1.13105 w_k1_IBM_s105
                                 + 1.02056 w_k1_INTC_s105
                                 + 0.757276 w_k1_KO_s105 + 1.0134 w_k1_MAT_s105
                                 + 0.781791 w_k1_MSFT_s105
                                 + 0.859394 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.25206 w_k1_ADM_s105
                                 + 1.63022 w_k1_AXP_s105
                                 + 1.97509 w_k1_DIS_s105
                                 + 1.40717 w_k1_FDX_s105
                                 + 1.81862 w_k1_IBM_s105
                                 + 1.33886 w_k1_INTC_s105
                                 + 1.69386 w_k1_KO_s105 + 2.17377 w_k1_MAT_s105
                                 + 1.59255 w_k1_MSFT_s105
                                 + 1.54403 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.846391 w_k1_ADM_s106
                                 + 0.628571 w_k1_AXP_s106
                                 + 0.76195 w_k1_DIS_s106
                                 + 0.770437 w_k1_FDX_s106
                                 + 0.885715 w_k1_IBM_s106
                                 + 0.692785 w_k1_INTC_s106
                                 + 0.906966 w_k1_KO_s106 + 1.0035 w_k1_MAT_s106
                                 + 0.764222 w_k1_MSFT_s106
                                 + 0.953658 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.07148 w_k1_ADM_s106
                                 + 0.842632 w_k1_AXP_s106
                                 + 0.857578 w_k1_DIS_s106
                                 + 0.919526 w_k1_FDX_s106
                                 + 0.680339 w_k1_IBM_s106
                                 + 0.775556 w_k1_INTC_s106
                                 + 0.816236 w_k1_KO_s106
                                 + 0.919612 w_k1_MAT_s106
                                 + 0.836099 w_k1_MSFT_s106
                                 + 0.958564 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.34706 w_k1_ADM_s106
                                 + 0.911762 w_k1_AXP_s106
                                 + 1.19857 w_k1_DIS_s106
                                 + 1.10835 w_k1_FDX_s106
                                 + 0.915639 w_k1_IBM_s106
                                 + 0.884726 w_k1_INTC_s106
                                 + 0.971397 w_k1_KO_s106
                                 + 1.19709 w_k1_MAT_s106
                                 + 1.1386 w_k1_MSFT_s106 + 1.08738 w_k1_PG_s106
                                 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.27948 w_k1_ADM_s106
                                 + 1.32147 w_k1_AXP_s106
                                 + 1.15868 w_k1_DIS_s106
                                 + 1.15123 w_k1_FDX_s106
                                 + 1.16418 w_k1_IBM_s106
                                 + 1.11426 w_k1_INTC_s106
                                 + 1.07099 w_k1_KO_s106 + 1.37727 w_k1_MAT_s106
                                 + 1.07975 w_k1_MSFT_s106 + 1.1397 w_k1_PG_s106
                                 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.31034 w_k1_ADM_s106
                                 + 0.867685 w_k1_AXP_s106
                                 + 1.08555 w_k1_DIS_s106
                                 + 0.954737 w_k1_FDX_s106
                                 + 1.31596 w_k1_IBM_s106
                                 + 0.746473 w_k1_INTC_s106
                                 + 1.09991 w_k1_KO_s106 + 1.4786 w_k1_MAT_s106
                                 + 1.04174 w_k1_MSFT_s106
                                 + 1.01929 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.17795 w_k1_ADM_s106
                                 + 1.23189 w_k1_AXP_s106
                                 + 0.919065 w_k1_DIS_s106
                                 + 0.970601 w_k1_FDX_s106
                                 + 1.07146 w_k1_IBM_s106
                                 + 0.849197 w_k1_INTC_s106
                                 + 0.962978 w_k1_KO_s106
                                 + 1.30767 w_k1_MAT_s106
                                 + 0.891344 w_k1_MSFT_s106
                                 + 1.02986 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.34759 w_k1_ADM_s106
                                 + 1.12578 w_k1_AXP_s106
                                 + 1.95809 w_k1_DIS_s106
                                 + 1.28083 w_k1_FDX_s106
                                 + 1.09708 w_k1_IBM_s106
                                 + 1.00584 w_k1_INTC_s106
                                 + 1.34049 w_k1_KO_s106 + 1.2006 w_k1_MAT_s106
                                 + 1.26179 w_k1_MSFT_s106
                                 + 1.16398 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.29796 w_k1_ADM_s106
                                 + 1.57227 w_k1_AXP_s106
                                 + 1.57083 w_k1_DIS_s106
                                 + 1.37833 w_k1_FDX_s106
                                 + 1.52921 w_k1_IBM_s106
                                 + 1.23368 w_k1_INTC_s106
                                 + 1.43323 w_k1_KO_s106 + 1.81119 w_k1_MAT_s106
                                 + 1.22195 w_k1_MSFT_s106
                                 + 1.35942 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.788383 w_k1_ADM_s107
                                 + 0.777429 w_k1_AXP_s107
                                 + 0.85212 w_k1_DIS_s107
                                 + 0.951432 w_k1_FDX_s107
                                 + 0.607155 w_k1_IBM_s107
                                 + 1.00373 w_k1_INTC_s107
                                 + 0.712099 w_k1_KO_s107
                                 + 0.65106 w_k1_MAT_s107
                                 + 0.820769 w_k1_MSFT_s107
                                 + 0.873812 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.665813 w_k1_ADM_s107
                                 + 1.13767 w_k1_AXP_s107
                                 + 0.999455 w_k1_DIS_s107
                                 + 0.902293 w_k1_FDX_s107
                                 + 1.40546 w_k1_IBM_s107
                                 + 0.945637 w_k1_INTC_s107
                                 + 1.02828 w_k1_KO_s107 + 1.51179 w_k1_MAT_s107
                                 + 0.938032 w_k1_MSFT_s107
                                 + 0.928892 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.2099 w_k1_ADM_s107
                                 + 1.52429 w_k1_AXP_s107
                                 + 1.40552 w_k1_DIS_s107
                                 + 1.25562 w_k1_FDX_s107
                                 + 1.06056 w_k1_IBM_s107
                                 + 1.24312 w_k1_INTC_s107
                                 + 1.09691 w_k1_KO_s107 + 1.41413 w_k1_MAT_s107
                                 + 1.17021 w_k1_MSFT_s107
                                 + 1.16254 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.02976 w_k1_ADM_s107
                                 + 1.45957 w_k1_AXP_s107
                                 + 1.43082 w_k1_DIS_s107
                                 + 1.27476 w_k1_FDX_s107
                                 + 1.36342 w_k1_IBM_s107
                                 + 1.32041 w_k1_INTC_s107 + 1.2453 w_k1_KO_s107
                                 + 1.50791 w_k1_MAT_s107
                                 + 1.20888 w_k1_MSFT_s107
                                 + 1.16359 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.872987 w_k1_ADM_s107
                                 + 0.922943 w_k1_AXP_s107
                                 + 0.93277 w_k1_DIS_s107
                                 + 0.965975 w_k1_FDX_s107
                                 + 0.94767 w_k1_IBM_s107
                                 + 1.08558 w_k1_INTC_s107
                                 + 0.962519 w_k1_KO_s107
                                 + 0.935625 w_k1_MAT_s107
                                 + 0.965063 w_k1_MSFT_s107
                                 + 0.89842 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.84979 w_k1_ADM_s107
                                 + 1.16888 w_k1_AXP_s107
                                 + 1.22146 w_k1_DIS_s107
                                 + 1.22312 w_k1_FDX_s107
                                 + 0.756044 w_k1_IBM_s107
                                 + 1.20591 w_k1_INTC_s107
                                 + 0.887421 w_k1_KO_s107
                                 + 0.776783 w_k1_MAT_s107
                                 + 1.12684 w_k1_MSFT_s107
                                 + 0.901813 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.606171 w_k1_ADM_s107
                                 + 0.526029 w_k1_AXP_s107
                                 + 0.733828 w_k1_DIS_s107
                                 + 0.71787 w_k1_FDX_s107
                                 + 0.715471 w_k1_IBM_s107
                                 + 0.899377 w_k1_INTC_s107
                                 + 0.71253 w_k1_KO_s107
                                 + 0.671466 w_k1_MAT_s107
                                 + 0.813366 w_k1_MSFT_s107
                                 + 0.777797 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.76225 w_k1_ADM_s107
                                 + 1.05559 w_k1_AXP_s107
                                 + 0.844791 w_k1_DIS_s107
                                 + 0.878568 w_k1_FDX_s107
                                 + 0.823891 w_k1_IBM_s107
                                 + 0.890559 w_k1_INTC_s107
                                 + 0.783793 w_k1_KO_s107
                                 + 1.04272 w_k1_MAT_s107
                                 + 0.780862 w_k1_MSFT_s107
                                 + 0.868282 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.59678 w_k1_ADM_s110
                                 + 1.11466 w_k1_AXP_s110
                                 + 1.23886 w_k1_DIS_s110
                                 + 0.96097 w_k1_FDX_s110 + 1.68 w_k1_IBM_s110
                                 + 0.857072 w_k1_INTC_s110
                                 + 1.34039 w_k1_KO_s110 + 1.87137 w_k1_MAT_s110
                                 + 1.0211 w_k1_MSFT_s110 + 1.08535 w_k1_PG_s110
                                 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.31972 w_k1_ADM_s110
                                 + 1.22711 w_k1_AXP_s110
                                 + 1.18489 w_k1_DIS_s110
                                 + 1.11312 w_k1_FDX_s110
                                 + 1.34649 w_k1_IBM_s110
                                 + 0.90936 w_k1_INTC_s110 + 1.1888 w_k1_KO_s110
                                 + 1.49463 w_k1_MAT_s110
                                 + 0.971257 w_k1_MSFT_s110
                                 + 0.998898 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.09845 w_k1_ADM_s110
                                 + 0.693483 w_k1_AXP_s110
                                 + 0.73591 w_k1_DIS_s110
                                 + 0.80643 w_k1_FDX_s110
                                 + 1.17007 w_k1_IBM_s110
                                 + 0.758468 w_k1_INTC_s110
                                 + 0.946116 w_k1_KO_s110
                                 + 0.995451 w_k1_MAT_s110
                                 + 0.820139 w_k1_MSFT_s110
                                 + 0.939487 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.44534 w_k1_ADM_s110
                                 + 1.61149 w_k1_AXP_s110
                                 + 1.57925 w_k1_DIS_s110 + 1.2953 w_k1_FDX_s110
                                 + 1.20599 w_k1_IBM_s110 + 1.141 w_k1_INTC_s110
                                 + 1.14397 w_k1_KO_s110 + 1.51848 w_k1_MAT_s110
                                 + 1.18491 w_k1_MSFT_s110
                                 + 1.13587 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.955638 w_k1_ADM_s110
                                 + 1.01121 w_k1_AXP_s110
                                 + 0.896826 w_k1_DIS_s110
                                 + 0.965103 w_k1_FDX_s110
                                 + 1.04974 w_k1_IBM_s110
                                 + 0.932524 w_k1_INTC_s110
                                 + 0.933448 w_k1_KO_s110
                                 + 1.12727 w_k1_MAT_s110
                                 + 1.00902 w_k1_MSFT_s110
                                 + 0.956804 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.768144 w_k1_ADM_s110
                                 + 0.790013 w_k1_AXP_s110
                                 + 0.570462 w_k1_DIS_s110
                                 + 0.750177 w_k1_FDX_s110
                                 + 0.818488 w_k1_IBM_s110
                                 + 0.693058 w_k1_INTC_s110
                                 + 0.725333 w_k1_KO_s110
                                 + 0.904706 w_k1_MAT_s110
                                 + 0.825628 w_k1_MSFT_s110
                                 + 0.737075 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.27789 w_k1_ADM_s110
                                 + 1.54298 w_k1_AXP_s110
                                 + 1.13291 w_k1_DIS_s110
                                 + 1.16068 w_k1_FDX_s110
                                 + 1.07251 w_k1_IBM_s110
                                 + 0.896833 w_k1_INTC_s110
                                 + 0.889839 w_k1_KO_s110
                                 + 1.62523 w_k1_MAT_s110
                                 + 1.0011 w_k1_MSFT_s110 + 1.11509 w_k1_PG_s110
                                 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.11754 w_k1_ADM_s110
                                 + 0.941838 w_k1_AXP_s110
                                 + 0.99676 w_k1_DIS_s110
                                 + 1.03094 w_k1_FDX_s110
                                 + 0.842395 w_k1_IBM_s110
                                 + 0.859026 w_k1_INTC_s110
                                 + 0.765611 w_k1_KO_s110
                                 + 0.973723 w_k1_MAT_s110
                                 + 0.935926 w_k1_MSFT_s110
                                 + 0.905276 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.01865 w_k1_ADM_s111
                                 + 1.17963 w_k1_AXP_s111
                                 + 1.05667 w_k1_DIS_s111
                                 + 1.07728 w_k1_FDX_s111
                                 + 0.974544 w_k1_IBM_s111
                                 + 0.976897 w_k1_INTC_s111
                                 + 1.0695 w_k1_KO_s111 + 1.94677 w_k1_MAT_s111
                                 + 1.01848 w_k1_MSFT_s111
                                 + 1.08863 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.963481 w_k1_ADM_s111
                                 + 0.666457 w_k1_AXP_s111
                                 + 1.31221 w_k1_DIS_s111
                                 + 1.08021 w_k1_FDX_s111
                                 + 0.895008 w_k1_IBM_s111
                                 + 1.19775 w_k1_INTC_s111
                                 + 1.10339 w_k1_KO_s111
                                 + 0.816219 w_k1_MAT_s111
                                 + 1.18385 w_k1_MSFT_s111
                                 + 0.972487 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.723556 w_k1_ADM_s111
                                 + 0.616369 w_k1_AXP_s111
                                 + 0.539091 w_k1_DIS_s111
                                 + 0.745963 w_k1_FDX_s111
                                 + 0.732029 w_k1_IBM_s111
                                 + 0.7808 w_k1_INTC_s111
                                 + 0.667857 w_k1_KO_s111
                                 + 0.91116 w_k1_MAT_s111
                                 + 0.731885 w_k1_MSFT_s111
                                 + 0.732536 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.13882 w_k1_ADM_s111
                                 + 0.994273 w_k1_AXP_s111
                                 + 1.0725 w_k1_DIS_s111 + 1.00472 w_k1_FDX_s111
                                 + 1.61353 w_k1_IBM_s111
                                 + 1.05671 w_k1_INTC_s111
                                 + 1.22623 w_k1_KO_s111 + 1.81679 w_k1_MAT_s111
                                 + 0.97592 w_k1_MSFT_s111
                                 + 1.03251 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.39006 w_k1_ADM_s111
                                 + 1.41092 w_k1_AXP_s111 + 1.5806 w_k1_DIS_s111
                                 + 1.32174 w_k1_FDX_s111
                                 + 1.42599 w_k1_IBM_s111
                                 + 1.26481 w_k1_INTC_s111
                                 + 1.34122 w_k1_KO_s111 + 1.38804 w_k1_MAT_s111
                                 + 1.31828 w_k1_MSFT_s111
                                 + 1.30909 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.59038 w_k1_ADM_s111
                                 + 1.18052 w_k1_AXP_s111
                                 + 1.42511 w_k1_DIS_s111
                                 + 1.40817 w_k1_FDX_s111
                                 + 0.875217 w_k1_IBM_s111
                                 + 1.06651 w_k1_INTC_s111
                                 + 1.10912 w_k1_KO_s111 + 1.19016 w_k1_MAT_s111
                                 + 1.1671 w_k1_MSFT_s111 + 1.10725 w_k1_PG_s111
                                 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.18795 w_k1_ADM_s111
                                 + 0.849989 w_k1_AXP_s111
                                 + 0.974588 w_k1_DIS_s111
                                 + 1.04335 w_k1_FDX_s111
                                 + 1.12663 w_k1_IBM_s111
                                 + 0.990045 w_k1_INTC_s111
                                 + 1.07337 w_k1_KO_s111 + 1.1942 w_k1_MAT_s111
                                 + 1.01465 w_k1_MSFT_s111
                                 + 1.04162 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.06571 w_k1_ADM_s111
                                 + 0.787625 w_k1_AXP_s111
                                 + 0.809821 w_k1_DIS_s111
                                 + 0.851331 w_k1_FDX_s111
                                 + 0.880008 w_k1_IBM_s111
                                 + 0.795755 w_k1_INTC_s111
                                 + 0.871732 w_k1_KO_s111
                                 + 1.07874 w_k1_MAT_s111
                                 + 0.791099 w_k1_MSFT_s111
                                 + 0.877246 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.30141 w_k1_ADM_s112
                                 + 1.01041 w_k1_AXP_s112
                                 + 0.953609 w_k1_DIS_s112
                                 + 1.03238 w_k1_FDX_s112
                                 + 0.644057 w_k1_IBM_s112
                                 + 0.901926 w_k1_INTC_s112
                                 + 0.963721 w_k1_KO_s112
                                 + 0.802627 w_k1_MAT_s112
                                 + 1.00085 w_k1_MSFT_s112
                                 + 0.906712 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.961267 w_k1_ADM_s112
                                 + 1.09214 w_k1_AXP_s112
                                 + 0.962009 w_k1_DIS_s112
                                 + 0.914658 w_k1_FDX_s112
                                 + 1.3477 w_k1_IBM_s112
                                 + 0.773422 w_k1_INTC_s112
                                 + 1.12274 w_k1_KO_s112 + 1.25446 w_k1_MAT_s112
                                 + 0.798603 w_k1_MSFT_s112
                                 + 0.947578 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.946579 w_k1_ADM_s112
                                 + 0.617168 w_k1_AXP_s112
                                 + 0.628928 w_k1_DIS_s112
                                 + 0.680282 w_k1_FDX_s112
                                 + 0.717789 w_k1_IBM_s112
                                 + 0.563417 w_k1_INTC_s112
                                 + 0.707911 w_k1_KO_s112
                                 + 0.567549 w_k1_MAT_s112
                                 + 0.631291 w_k1_MSFT_s112
                                 + 0.651139 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.03363 w_k1_ADM_s112
                                 + 0.84652 w_k1_AXP_s112
                                 + 0.753615 w_k1_DIS_s112
                                 + 0.786256 w_k1_FDX_s112
                                 + 0.850409 w_k1_IBM_s112
                                 + 0.689798 w_k1_INTC_s112
                                 + 0.920335 w_k1_KO_s112
                                 + 1.02872 w_k1_MAT_s112
                                 + 0.953807 w_k1_MSFT_s112
                                 + 0.913355 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.934841 w_k1_ADM_s112
                                 + 0.964815 w_k1_AXP_s112
                                 + 1.24614 w_k1_DIS_s112
                                 + 1.11832 w_k1_FDX_s112
                                 + 0.935149 w_k1_IBM_s112
                                 + 0.785429 w_k1_INTC_s112
                                 + 1.02608 w_k1_KO_s112
                                 + 0.840952 w_k1_MAT_s112
                                 + 0.939636 w_k1_MSFT_s112
                                 + 1.01252 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.53299 w_k1_ADM_s112
                                 + 1.64941 w_k1_AXP_s112
                                 + 1.60009 w_k1_DIS_s112
                                 + 1.46847 w_k1_FDX_s112
                                 + 0.814147 w_k1_IBM_s112
                                 + 0.960883 w_k1_INTC_s112
                                 + 1.04421 w_k1_KO_s112
                                 + 0.902379 w_k1_MAT_s112
                                 + 1.13399 w_k1_MSFT_s112
                                 + 1.10974 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.798222 w_k1_ADM_s112
                                 + 0.813188 w_k1_AXP_s112
                                 + 0.821676 w_k1_DIS_s112
                                 + 0.949397 w_k1_FDX_s112
                                 + 0.576088 w_k1_IBM_s112
                                 + 0.603109 w_k1_INTC_s112
                                 + 0.691447 w_k1_KO_s112 + 0.7075 w_k1_MAT_s112
                                 + 0.787366 w_k1_MSFT_s112
                                 + 0.87091 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.23019 w_k1_ADM_s112
                                 + 1.66084 w_k1_AXP_s112
                                 + 1.86857 w_k1_DIS_s112
                                 + 1.29476 w_k1_FDX_s112
                                 + 1.31561 w_k1_IBM_s112
                                 + 1.1469 w_k1_INTC_s112 + 1.62804 w_k1_KO_s112
                                 + 1.44699 w_k1_MAT_s112
                                 + 1.33091 w_k1_MSFT_s112
                                 + 1.18722 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.916976 w_k1_ADM_s113
                                 + 0.766171 w_k1_AXP_s113
                                 + 0.813949 w_k1_DIS_s113
                                 + 0.783292 w_k1_FDX_s113
                                 + 0.693227 w_k1_IBM_s113
                                 + 0.977078 w_k1_INTC_s113
                                 + 0.77845 w_k1_KO_s113
                                 + 0.499675 w_k1_MAT_s113
                                 + 0.989551 w_k1_MSFT_s113
                                 + 0.926417 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.959738 w_k1_ADM_s113
                                 + 0.941074 w_k1_AXP_s113
                                 + 1.04632 w_k1_DIS_s113
                                 + 0.925282 w_k1_FDX_s113
                                 + 1.20662 w_k1_IBM_s113
                                 + 0.97968 w_k1_INTC_s113
                                 + 1.19626 w_k1_KO_s113 + 1.24919 w_k1_MAT_s113
                                 + 1.15068 w_k1_MSFT_s113
                                 + 1.14351 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.12503 w_k1_ADM_s113
                                 + 1.34263 w_k1_AXP_s113
                                 + 1.31347 w_k1_DIS_s113
                                 + 1.29002 w_k1_FDX_s113
                                 + 1.00624 w_k1_IBM_s113
                                 + 1.33307 w_k1_INTC_s113
                                 + 1.29371 w_k1_KO_s113
                                 + 0.981747 w_k1_MAT_s113
                                 + 1.49697 w_k1_MSFT_s113
                                 + 1.13293 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.824464 w_k1_ADM_s113
                                 + 1.02235 w_k1_AXP_s113
                                 + 0.827978 w_k1_DIS_s113
                                 + 0.807192 w_k1_FDX_s113
                                 + 1.19426 w_k1_IBM_s113
                                 + 0.866399 w_k1_INTC_s113
                                 + 0.969182 w_k1_KO_s113
                                 + 1.17386 w_k1_MAT_s113
                                 + 0.85225 w_k1_MSFT_s113
                                 + 1.06996 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.18974 w_k1_ADM_s113
                                 + 1.35812 w_k1_AXP_s113
                                 + 1.27343 w_k1_DIS_s113
                                 + 1.15863 w_k1_FDX_s113
                                 + 1.39341 w_k1_IBM_s113
                                 + 1.07745 w_k1_INTC_s113 + 1.2702 w_k1_KO_s113
                                 + 1.85301 w_k1_MAT_s113
                                 + 1.2146 w_k1_MSFT_s113 + 1.27224 w_k1_PG_s113
                                 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.02197 w_k1_ADM_s113
                                 + 0.871115 w_k1_AXP_s113
                                 + 0.940186 w_k1_DIS_s113
                                 + 0.937237 w_k1_FDX_s113
                                 + 0.895541 w_k1_IBM_s113
                                 + 0.974988 w_k1_INTC_s113
                                 + 0.840595 w_k1_KO_s113
                                 + 0.735865 w_k1_MAT_s113
                                 + 1.03925 w_k1_MSFT_s113
                                 + 0.96119 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.27611 w_k1_ADM_s113
                                 + 1.3586 w_k1_AXP_s113 + 1.70721 w_k1_DIS_s113
                                 + 1.3944 w_k1_FDX_s113 + 1.44584 w_k1_IBM_s113
                                 + 1.11084 w_k1_INTC_s113
                                 + 1.35166 w_k1_KO_s113 + 1.16454 w_k1_MAT_s113
                                 + 1.35905 w_k1_MSFT_s113
                                 + 1.36344 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.22314 w_k1_ADM_s113
                                 + 0.973503 w_k1_AXP_s113
                                 + 1.41313 w_k1_DIS_s113
                                 + 1.12768 w_k1_FDX_s113
                                 + 0.916391 w_k1_IBM_s113
                                 + 1.17808 w_k1_INTC_s113
                                 + 1.19797 w_k1_KO_s113
                                 + 0.951292 w_k1_MAT_s113
                                 + 1.2553 w_k1_MSFT_s113 + 1.06848 w_k1_PG_s113
                                 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.44068 w_k1_ADM_s114
                                 + 1.52353 w_k1_AXP_s114
                                 + 1.30519 w_k1_DIS_s114
                                 + 1.22469 w_k1_FDX_s114
                                 + 1.45333 w_k1_IBM_s114
                                 + 1.03439 w_k1_INTC_s114
                                 + 1.29317 w_k1_KO_s114 + 1.91587 w_k1_MAT_s114
                                 + 1.22346 w_k1_MSFT_s114
                                 + 1.24962 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.59247 w_k1_ADM_s114
                                 + 1.0129 w_k1_AXP_s114 + 1.49873 w_k1_DIS_s114
                                 + 1.3746 w_k1_FDX_s114
                                 + 0.864249 w_k1_IBM_s114
                                 + 0.99398 w_k1_INTC_s114
                                 + 0.983188 w_k1_KO_s114
                                 + 0.864221 w_k1_MAT_s114
                                 + 1.50386 w_k1_MSFT_s114
                                 + 0.961246 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.2133 w_k1_ADM_s114
                                 + 1.07851 w_k1_AXP_s114 + 1.0628 w_k1_DIS_s114
                                 + 0.969378 w_k1_FDX_s114
                                 + 1.00673 w_k1_IBM_s114
                                 + 0.948859 w_k1_INTC_s114
                                 + 0.87564 w_k1_KO_s114 + 1.17405 w_k1_MAT_s114
                                 + 1.1417 w_k1_MSFT_s114 + 1.028 w_k1_PG_s114
                                 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.71535 w_k1_ADM_s114
                                 + 1.8737 w_k1_AXP_s114 + 2.00458 w_k1_DIS_s114
                                 + 1.51125 w_k1_FDX_s114 + 1.5133 w_k1_IBM_s114
                                 + 1.34283 w_k1_INTC_s114
                                 + 1.34402 w_k1_KO_s114 + 1.78535 w_k1_MAT_s114
                                 + 1.61335 w_k1_MSFT_s114
                                 + 1.21793 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.800404 w_k1_ADM_s114
                                 + 0.747527 w_k1_AXP_s114
                                 + 0.685578 w_k1_DIS_s114
                                 + 0.769447 w_k1_FDX_s114
                                 + 0.856995 w_k1_IBM_s114
                                 + 0.923246 w_k1_INTC_s114
                                 + 0.7214 w_k1_KO_s114 + 0.504379 w_k1_MAT_s114
                                 + 0.911154 w_k1_MSFT_s114
                                 + 0.762928 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.05711 w_k1_ADM_s114
                                 + 0.930125 w_k1_AXP_s114
                                 + 0.955286 w_k1_DIS_s114
                                 + 0.90796 w_k1_FDX_s114
                                 + 0.732475 w_k1_IBM_s114
                                 + 0.991269 w_k1_INTC_s114
                                 + 0.845113 w_k1_KO_s114
                                 + 0.605506 w_k1_MAT_s114
                                 + 1.07216 w_k1_MSFT_s114
                                 + 0.841534 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.11063 w_k1_ADM_s114
                                 + 1.05809 w_k1_AXP_s114
                                 + 0.930658 w_k1_DIS_s114
                                 + 0.952658 w_k1_FDX_s114
                                 + 1.27312 w_k1_IBM_s114
                                 + 1.04736 w_k1_INTC_s114
                                 + 1.10012 w_k1_KO_s114 + 1.62038 w_k1_MAT_s114
                                 + 1.30111 w_k1_MSFT_s114
                                 + 1.09259 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.26723 w_k1_ADM_s114
                                 + 0.913966 w_k1_AXP_s114
                                 + 0.969844 w_k1_DIS_s114
                                 + 1.14855 w_k1_FDX_s114
                                 + 0.684448 w_k1_IBM_s114
                                 + 1.10228 w_k1_INTC_s114
                                 + 0.810191 w_k1_KO_s114
                                 + 0.488336 w_k1_MAT_s114 + 1.21 w_k1_MSFT_s114
                                 + 0.928223 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.809082 w_k1_ADM_s115
                                 + 0.881665 w_k1_AXP_s115
                                 + 0.669944 w_k1_DIS_s115
                                 + 0.808131 w_k1_FDX_s115
                                 + 1.05235 w_k1_IBM_s115
                                 + 0.872747 w_k1_INTC_s115
                                 + 0.785575 w_k1_KO_s115
                                 + 1.14211 w_k1_MAT_s115
                                 + 0.747187 w_k1_MSFT_s115
                                 + 0.829643 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.39485 w_k1_ADM_s115
                                 + 1.22783 w_k1_AXP_s115 + 1.6072 w_k1_DIS_s115
                                 + 1.38509 w_k1_FDX_s115 + 1.3131 w_k1_IBM_s115
                                 + 1.21653 w_k1_INTC_s115
                                 + 1.28456 w_k1_KO_s115 + 1.14139 w_k1_MAT_s115
                                 + 1.22166 w_k1_MSFT_s115
                                 + 1.11038 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.03279 w_k1_ADM_s115
                                 + 0.85208 w_k1_AXP_s115
                                 + 0.954137 w_k1_DIS_s115
                                 + 1.03699 w_k1_FDX_s115
                                 + 1.08478 w_k1_IBM_s115
                                 + 0.727048 w_k1_INTC_s115
                                 + 1.03037 w_k1_KO_s115 + 1.10939 w_k1_MAT_s115
                                 + 0.805697 w_k1_MSFT_s115
                                 + 0.905234 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.05298 w_k1_ADM_s115
                                 + 0.890392 w_k1_AXP_s115
                                 + 1.10999 w_k1_DIS_s115
                                 + 1.06363 w_k1_FDX_s115
                                 + 1.59151 w_k1_IBM_s115
                                 + 1.16069 w_k1_INTC_s115
                                 + 1.22695 w_k1_KO_s115 + 1.37095 w_k1_MAT_s115
                                 + 1.13604 w_k1_MSFT_s115
                                 + 0.960488 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.00686 w_k1_ADM_s115
                                 + 0.985581 w_k1_AXP_s115
                                 + 1.15726 w_k1_DIS_s115
                                 + 1.20599 w_k1_FDX_s115
                                 + 0.92528 w_k1_IBM_s115
                                 + 1.20729 w_k1_INTC_s115
                                 + 1.05618 w_k1_KO_s115
                                 + 0.953566 w_k1_MAT_s115
                                 + 1.09237 w_k1_MSFT_s115
                                 + 1.00367 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.06335 w_k1_ADM_s115
                                 + 0.825926 w_k1_AXP_s115
                                 + 0.838628 w_k1_DIS_s115
                                 + 0.918075 w_k1_FDX_s115
                                 + 0.813185 w_k1_IBM_s115
                                 + 0.721403 w_k1_INTC_s115
                                 + 0.741941 w_k1_KO_s115
                                 + 0.606642 w_k1_MAT_s115
                                 + 0.932774 w_k1_MSFT_s115
                                 + 0.832924 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.0343 w_k1_ADM_s115
                                 + 1.05251 w_k1_AXP_s115
                                 + 1.02516 w_k1_DIS_s115
                                 + 1.01626 w_k1_FDX_s115
                                 + 1.51769 w_k1_IBM_s115
                                 + 0.791051 w_k1_INTC_s115
                                 + 1.14667 w_k1_KO_s115 + 1.57897 w_k1_MAT_s115
                                 + 0.846577 w_k1_MSFT_s115
                                 + 0.954326 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.883756 w_k1_ADM_s115
                                 + 0.707823 w_k1_AXP_s115
                                 + 0.636965 w_k1_DIS_s115
                                 + 0.812537 w_k1_FDX_s115
                                 + 0.655055 w_k1_IBM_s115
                                 + 0.853626 w_k1_INTC_s115
                                 + 0.739807 w_k1_KO_s115
                                 + 0.568542 w_k1_MAT_s115
                                 + 0.810135 w_k1_MSFT_s115
                                 + 0.802658 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.800325 w_k1_ADM_s116
                                 + 1.15209 w_k1_AXP_s116
                                 + 0.917021 w_k1_DIS_s116
                                 + 0.960866 w_k1_FDX_s116
                                 + 0.886052 w_k1_IBM_s116
                                 + 1.07405 w_k1_INTC_s116
                                 + 0.820719 w_k1_KO_s116
                                 + 0.919434 w_k1_MAT_s116
                                 + 0.955764 w_k1_MSFT_s116
                                 + 0.970246 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.937874 w_k1_ADM_s116
                                 + 0.9191 w_k1_AXP_s116
                                 + 0.948381 w_k1_DIS_s116
                                 + 0.851883 w_k1_FDX_s116
                                 + 1.25525 w_k1_IBM_s116
                                 + 0.986331 w_k1_INTC_s116
                                 + 0.912272 w_k1_KO_s116
                                 + 0.74384 w_k1_MAT_s116
                                 + 0.841501 w_k1_MSFT_s116
                                 + 0.988632 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.14694 w_k1_ADM_s116
                                 + 0.961604 w_k1_AXP_s116
                                 + 1.07084 w_k1_DIS_s116
                                 + 0.992512 w_k1_FDX_s116
                                 + 1.14131 w_k1_IBM_s116
                                 + 0.94852 w_k1_INTC_s116
                                 + 1.16179 w_k1_KO_s116 + 1.14875 w_k1_MAT_s116
                                 + 1.02953 w_k1_MSFT_s116
                                 + 1.04241 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.951985 w_k1_ADM_s116
                                 + 0.72978 w_k1_AXP_s116
                                 + 0.55895 w_k1_DIS_s116
                                 + 0.734212 w_k1_FDX_s116
                                 + 0.868637 w_k1_IBM_s116
                                 + 0.808924 w_k1_INTC_s116
                                 + 0.684849 w_k1_KO_s116
                                 + 0.699196 w_k1_MAT_s116
                                 + 0.635812 w_k1_MSFT_s116
                                 + 0.782834 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.21321 w_k1_ADM_s116
                                 + 0.966412 w_k1_AXP_s116
                                 + 1.63712 w_k1_DIS_s116
                                 + 1.09436 w_k1_FDX_s116 + 1.2334 w_k1_IBM_s116
                                 + 1.10173 w_k1_INTC_s116
                                 + 1.19496 w_k1_KO_s116 + 1.03009 w_k1_MAT_s116
                                 + 1.14011 w_k1_MSFT_s116
                                 + 0.970306 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.07263 w_k1_ADM_s116
                                 + 0.918715 w_k1_AXP_s116
                                 + 1.04749 w_k1_DIS_s116
                                 + 1.07125 w_k1_FDX_s116
                                 + 0.920991 w_k1_IBM_s116
                                 + 0.873492 w_k1_INTC_s116
                                 + 0.933244 w_k1_KO_s116
                                 + 0.889288 w_k1_MAT_s116
                                 + 0.90184 w_k1_MSFT_s116
                                 + 0.933776 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.58719 w_k1_ADM_s116
                                 + 1.40615 w_k1_AXP_s116
                                 + 1.62298 w_k1_DIS_s116
                                 + 1.29078 w_k1_FDX_s116
                                 + 1.36821 w_k1_IBM_s116
                                 + 1.42644 w_k1_INTC_s116
                                 + 1.50582 w_k1_KO_s116 + 1.13554 w_k1_MAT_s116
                                 + 1.336 w_k1_MSFT_s116 + 1.21134 w_k1_PG_s116
                                 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.41835 w_k1_ADM_s116
                                 + 1.20413 w_k1_AXP_s116
                                 + 1.34153 w_k1_DIS_s116
                                 + 1.34175 w_k1_FDX_s116
                                 + 0.824701 w_k1_IBM_s116
                                 + 1.34703 w_k1_INTC_s116
                                 + 1.08589 w_k1_KO_s116
                                 + 0.566907 w_k1_MAT_s116
                                 + 1.04626 w_k1_MSFT_s116
                                 + 1.01358 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.1852 w_k1_ADM_s117
                                 + 0.91471 w_k1_AXP_s117
                                 + 1.02465 w_k1_DIS_s117
                                 + 1.11994 w_k1_FDX_s117
                                 + 0.701336 w_k1_IBM_s117
                                 + 1.26153 w_k1_INTC_s117
                                 + 0.830174 w_k1_KO_s117
                                 + 0.609567 w_k1_MAT_s117
                                 + 1.00312 w_k1_MSFT_s117
                                 + 0.854201 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.16383 w_k1_ADM_s117
                                 + 1.25038 w_k1_AXP_s117
                                 + 1.25785 w_k1_DIS_s117
                                 + 1.27141 w_k1_FDX_s117
                                 + 1.03476 w_k1_IBM_s117
                                 + 1.45001 w_k1_INTC_s117
                                 + 1.08823 w_k1_KO_s117 + 1.10811 w_k1_MAT_s117
                                 + 1.3143 w_k1_MSFT_s117 + 1.17642 w_k1_PG_s117
                                 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.788151 w_k1_ADM_s117
                                 + 0.725802 w_k1_AXP_s117
                                 + 0.637918 w_k1_DIS_s117
                                 + 0.729133 w_k1_FDX_s117
                                 + 0.799994 w_k1_IBM_s117
                                 + 0.857134 w_k1_INTC_s117
                                 + 0.659646 w_k1_KO_s117
                                 + 0.813001 w_k1_MAT_s117
                                 + 0.765042 w_k1_MSFT_s117
                                 + 0.801279 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.31849 w_k1_ADM_s117
                                 + 1.46331 w_k1_AXP_s117
                                 + 1.37267 w_k1_DIS_s117
                                 + 1.12332 w_k1_FDX_s117
                                 + 1.51108 w_k1_IBM_s117
                                 + 1.03086 w_k1_INTC_s117
                                 + 1.16384 w_k1_KO_s117 + 1.52111 w_k1_MAT_s117
                                 + 1.00903 w_k1_MSFT_s117 + 1.1944 w_k1_PG_s117
                                 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.04012 w_k1_ADM_s117
                                 + 1.06189 w_k1_AXP_s117
                                 + 1.00286 w_k1_DIS_s117
                                 + 0.922289 w_k1_FDX_s117
                                 + 1.17606 w_k1_IBM_s117
                                 + 1.07045 w_k1_INTC_s117
                                 + 1.09591 w_k1_KO_s117 + 1.35814 w_k1_MAT_s117
                                 + 1.07032 w_k1_MSFT_s117
                                 + 1.08644 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.20629 w_k1_ADM_s117
                                 + 1.23026 w_k1_AXP_s117
                                 + 1.55128 w_k1_DIS_s117
                                 + 1.25728 w_k1_FDX_s117
                                 + 1.15959 w_k1_IBM_s117
                                 + 1.42889 w_k1_INTC_s117 + 1.2496 w_k1_KO_s117
                                 + 1.10175 w_k1_MAT_s117
                                 + 1.33508 w_k1_MSFT_s117 + 1.0809 w_k1_PG_s117
                                 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.02927 w_k1_ADM_s117
                                 + 1.0758 w_k1_AXP_s117
                                 + 0.985563 w_k1_DIS_s117
                                 + 0.953535 w_k1_FDX_s117
                                 + 0.9356 w_k1_IBM_s117
                                 + 0.956665 w_k1_INTC_s117
                                 + 0.864215 w_k1_KO_s117
                                 + 0.817983 w_k1_MAT_s117
                                 + 0.780559 w_k1_MSFT_s117
                                 + 0.97094 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.974902 w_k1_ADM_s117
                                 + 1.14892 w_k1_AXP_s117
                                 + 1.04096 w_k1_DIS_s117
                                 + 0.976362 w_k1_FDX_s117
                                 + 1.40403 w_k1_IBM_s117
                                 + 1.05346 w_k1_INTC_s117
                                 + 1.09126 w_k1_KO_s117 + 1.48953 w_k1_MAT_s117
                                 + 0.934319 w_k1_MSFT_s117
                                 + 1.04448 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0307358319889348 w_k1_ADM_s000
                                 + 0.0598345104070808 w_k1_ADM_s001
                                 + 0.0383749013481447 w_k1_ADM_s002
                                 + 0.0165203186462929 w_k1_ADM_s003
                                 + 0.0300028051676836 w_k1_ADM_s004
                                 + 0.0410573223073559 w_k1_ADM_s005
                                 + 0.0312943502980526 w_k1_ADM_s006
                                 + 0.0370783032126874 w_k1_ADM_s007
                                 + 0.0290827361581718 w_k1_ADM_s010
                                 + 0.0252311681824044 w_k1_ADM_s011
                                 + 0.0695387883997049 w_k1_ADM_s012
                                 + 0.0344268027929129 w_k1_ADM_s013
                                 + 0.0166652677454239 w_k1_ADM_s014
                                 + 0.0339733213181807 w_k1_ADM_s015
                                 + 0.0189513222657214 w_k1_ADM_s016
                                 + 0.0229606998245912 w_k1_ADM_s017
                                 + 0.0741389040162364 w_k1_ADM_s100
                                 + 0.076493355235001 w_k1_ADM_s101
                                 + 0.0234735179483319 w_k1_ADM_s102
                                 + 0.0459290815638672 w_k1_ADM_s103
                                 + 0.027403742333012 w_k1_ADM_s104
                                 + 0.0192152242217668 w_k1_ADM_s105
                                 + 0.0611754918764881 w_k1_ADM_s106
                                 + 0.0213890812436612 w_k1_ADM_s107
                                 + 0.0189681722553201 w_k1_ADM_s110
                                 + 0.0347848975372977 w_k1_ADM_s111
                                 + 0.0121763914337412 w_k1_ADM_s112
                                 + 0.0216020320040698 w_k1_ADM_s113
                                 + 0.0107542461084644 w_k1_ADM_s114
                                 + 0.0134718630460104 w_k1_ADM_s115
                                 + 0.0108927186728919 w_k1_ADM_s116
                                 + 0.0114612654622097 w_k1_ADM_s117
                                 + 0.0300023270502387 w_k1_AXP_s000
                                 + 0.0609179335664587 w_k1_AXP_s001
                                 + 0.0310458889217282 w_k1_AXP_s002
                                 + 0.0198139260035927 w_k1_AXP_s003
                                 + 0.0298877309580419 w_k1_AXP_s004
                                 + 0.0427814147437876 w_k1_AXP_s005
                                 + 0.0326368765179099 w_k1_AXP_s006
                                 + 0.0408984729731654 w_k1_AXP_s007
                                 + 0.030396513759174 w_k1_AXP_s010
                                 + 0.0231872258093807 w_k1_AXP_s011
                                 + 0.0726829872718934 w_k1_AXP_s012
                                 + 0.0380795909232136 w_k1_AXP_s013
                                 + 0.0165723526115656 w_k1_AXP_s014
                                 + 0.0349777317793569 w_k1_AXP_s015
                                 + 0.0201844088115558 w_k1_AXP_s016
                                 + 0.025168300902202 w_k1_AXP_s017
                                 + 0.0646694256354719 w_k1_AXP_s100
                                 + 0.0807554956412206 w_k1_AXP_s101
                                 + 0.0264258724362535 w_k1_AXP_s102
                                 + 0.0466599983963173 w_k1_AXP_s103
                                 + 0.0255556057215366 w_k1_AXP_s104
                                 + 0.020056171075621 w_k1_AXP_s105
                                 + 0.0496398419012576 w_k1_AXP_s106
                                 + 0.0263081563570323 w_k1_AXP_s107
                                 + 0.0174882634045595 w_k1_AXP_s110
                                 + 0.0293100794967914 w_k1_AXP_s111
                                 + 0.0122696960897088 w_k1_AXP_s112
                                 + 0.0208458966935111 w_k1_AXP_s113
                                 + 0.00978835873564479 w_k1_AXP_s114
                                 + 0.0119425511619561 w_k1_AXP_s115
                                 + 0.0102518088993839 w_k1_AXP_s116
                                 + 0.0115604189762986 w_k1_AXP_s117
                                 + 0.0311779983196015 w_k1_DIS_s000
                                 + 0.0595558783101997 w_k1_DIS_s001
                                 + 0.032298458742833 w_k1_DIS_s002
                                 + 0.0186036225388146 w_k1_DIS_s003
                                 + 0.0311965322561084 w_k1_DIS_s004
                                 + 0.0447068233941625 w_k1_DIS_s005
                                 + 0.0295611967810095 w_k1_DIS_s006
                                 + 0.041994594506401 w_k1_DIS_s007
                                 + 0.032176589218252 w_k1_DIS_s010
                                 + 0.0258110881086962 w_k1_DIS_s011
                                 + 0.0731875357946005 w_k1_DIS_s012
                                 + 0.0384109626671583 w_k1_DIS_s013
                                 + 0.02023833173591 w_k1_DIS_s014
                                 + 0.0375797025630662 w_k1_DIS_s015
                                 + 0.0202013029348889 w_k1_DIS_s016
                                 + 0.0266147734368552 w_k1_DIS_s017
                                 + 0.0688392581529234 w_k1_DIS_s100
                                 + 0.0909959821472013 w_k1_DIS_s101
                                 + 0.0286650154753187 w_k1_DIS_s102
                                 + 0.0499033371552196 w_k1_DIS_s103
                                 + 0.0290669327629352 w_k1_DIS_s104
                                 + 0.0219710336903305 w_k1_DIS_s105
                                 + 0.0558004777598826 w_k1_DIS_s106
                                 + 0.0261888900377415 w_k1_DIS_s107
                                 + 0.0164131636427428 w_k1_DIS_s110
                                 + 0.032028101423937 w_k1_DIS_s111
                                 + 0.012385248613788 w_k1_DIS_s112
                                 + 0.022300662009451 w_k1_DIS_s113
                                 + 0.00952514548040482 w_k1_DIS_s114
                                 + 0.0133616110687939 w_k1_DIS_s115
                                 + 0.0106492043686083 w_k1_DIS_s116
                                 + 0.0115009494507193 w_k1_DIS_s117
                                 + 0.0297944492764436 w_k1_FDX_s000
                                 + 0.0578088469914236 w_k1_FDX_s001
                                 + 0.0321775431113105 w_k1_FDX_s002
                                 + 0.0173608750245796 w_k1_FDX_s003
                                 + 0.0283249389934578 w_k1_FDX_s004
                                 + 0.0430652599326704 w_k1_FDX_s005
                                 + 0.0295935443995793 w_k1_FDX_s006
                                 + 0.0391389446710719 w_k1_FDX_s007
                                 + 0.0308493317776826 w_k1_FDX_s010
                                 + 0.0246802138819175 w_k1_FDX_s011
                                 + 0.0708934574466997 w_k1_FDX_s012
                                 + 0.0367070230639654 w_k1_FDX_s013
                                 + 0.0176776460513031 w_k1_FDX_s014
                                 + 0.0342197990207028 w_k1_FDX_s015
                                 + 0.020344193326137 w_k1_FDX_s016
                                 + 0.0245455500570389 w_k1_FDX_s017
                                 + 0.068677369224681 w_k1_FDX_s100
                                 + 0.0839508265345471 w_k1_FDX_s101
                                 + 0.0244793084489775 w_k1_FDX_s102
                                 + 0.0492077723560119 w_k1_FDX_s103
                                 + 0.0265525426820138 w_k1_FDX_s104
                                 + 0.0204208055280249 w_k1_FDX_s105
                                 + 0.0523777202121292 w_k1_FDX_s106
                                 + 0.0253796106452996 w_k1_FDX_s107
                                 + 0.0164968125861512 w_k1_FDX_s110
                                 + 0.0319377937278805 w_k1_FDX_s111
                                 + 0.0115049014384793 w_k1_FDX_s112
                                 + 0.0206126478135235 w_k1_FDX_s113
                                 + 0.00937982355391559 w_k1_FDX_s114
                                 + 0.0135423681287262 w_k1_FDX_s115
                                 + 0.0102838203021935 w_k1_FDX_s116
                                 + 0.0109451438569901 w_k1_FDX_s117
                                 + 0.0306590437327886 w_k1_IBM_s000
                                 + 0.0623458762646406 w_k1_IBM_s001
                                 + 0.0321666619940269 w_k1_IBM_s002
                                 + 0.0173843545948099 w_k1_IBM_s003
                                 + 0.0308261091276789 w_k1_IBM_s004
                                 + 0.0429630243254241 w_k1_IBM_s005
                                 + 0.0299368533380629 w_k1_IBM_s006
                                 + 0.0405322529906179 w_k1_IBM_s007
                                 + 0.0300210297397011 w_k1_IBM_s010
                                 + 0.0246919476651181 w_k1_IBM_s011
                                 + 0.0712150097615888 w_k1_IBM_s012
                                 + 0.0365146683031616 w_k1_IBM_s013
                                 + 0.0181228393755012 w_k1_IBM_s014
                                 + 0.0354636718793678 w_k1_IBM_s015
                                 + 0.0189261876737575 w_k1_IBM_s016
                                 + 0.0243198126928264 w_k1_IBM_s017
                                 + 0.0690259466073993 w_k1_IBM_s100
                                 + 0.0858876774669578 w_k1_IBM_s101
                                 + 0.0263492153973315 w_k1_IBM_s102
                                 + 0.0522407806855876 w_k1_IBM_s103
                                 + 0.0260871397516189 w_k1_IBM_s104
                                 + 0.0218964294382247 w_k1_IBM_s105
                                 + 0.0535778917739765 w_k1_IBM_s106
                                 + 0.0245210920448861 w_k1_IBM_s107
                                 + 0.0177640754141933 w_k1_IBM_s110
                                 + 0.0323040875040273 w_k1_IBM_s111
                                 + 0.0117725223233461 w_k1_IBM_s112
                                 + 0.021304854009566 w_k1_IBM_s113
                                 + 0.0095392640387342 w_k1_IBM_s114
                                 + 0.013443599826278 w_k1_IBM_s115
                                 + 0.010200427624632 w_k1_IBM_s116
                                 + 0.0111227988362763 w_k1_IBM_s117
                                 + 0.0248060092470687 w_k1_INTC_s000
                                 + 0.0643295068383102 w_k1_INTC_s001
                                 + 0.0408702829813836 w_k1_INTC_s002
                                 + 0.0166332398437822 w_k1_INTC_s003
                                 + 0.0268330605202787 w_k1_INTC_s004
                                 + 0.0435459773508402 w_k1_INTC_s005
                                 + 0.025819419954738 w_k1_INTC_s006
                                 + 0.0408900294761313 w_k1_INTC_s007
                                 + 0.0284435660146543 w_k1_INTC_s010
                                 + 0.0271813623918416 w_k1_INTC_s011
                                 + 0.0697096675876039 w_k1_INTC_s012
                                 + 0.0387001561605759 w_k1_INTC_s013
                                 + 0.0166566591223927 w_k1_INTC_s014
                                 + 0.0329810462736094 w_k1_INTC_s015
                                 + 0.0219347413391867 w_k1_INTC_s016
                                 + 0.0256613798415663 w_k1_INTC_s017
                                 + 0.0708315103861728 w_k1_INTC_s100
                                 + 0.078085116714451 w_k1_INTC_s101
                                 + 0.0248006160830273 w_k1_INTC_s102
                                 + 0.0485190511308423 w_k1_INTC_s103
                                 + 0.0259353266879853 w_k1_INTC_s104
                                 + 0.0226562262948085 w_k1_INTC_s105
                                 + 0.0446198573878156 w_k1_INTC_s106
                                 + 0.0270520867958533 w_k1_INTC_s107
                                 + 0.0143983872352946 w_k1_INTC_s110
                                 + 0.0301452957814212 w_k1_INTC_s111
                                 + 0.00919951484030041 w_k1_INTC_s112
                                 + 0.0215271418427576 w_k1_INTC_s113
                                 + 0.00937428229367263 w_k1_INTC_s114
                                 + 0.0120210510037589 w_k1_INTC_s115
                                 + 0.0104767487314614 w_k1_INTC_s116
                                 + 0.0118835630840496 w_k1_INTC_s117
                                 + 0.0317365980183693 w_k1_KO_s000
                                 + 0.0592892696873455 w_k1_KO_s001
                                 + 0.032917235869535 w_k1_KO_s002
                                 + 0.0180989201794256 w_k1_KO_s003
                                 + 0.0305609466269369 w_k1_KO_s004
                                 + 0.0459962728578159 w_k1_KO_s005
                                 + 0.0279172524354781 w_k1_KO_s006
                                 + 0.0374433861128191 w_k1_KO_s007
                                 + 0.0292090279269727 w_k1_KO_s010
                                 + 0.0244854327269494 w_k1_KO_s011
                                 + 0.0634974515550893 w_k1_KO_s012
                                 + 0.0396893189920962 w_k1_KO_s013
                                 + 0.017464386349355 w_k1_KO_s014
                                 + 0.0364214341282078 w_k1_KO_s015
                                 + 0.0197848354510968 w_k1_KO_s016
                                 + 0.0265970160004034 w_k1_KO_s017
                                 + 0.070607891068294 w_k1_KO_s100
                                 + 0.0819297570622903 w_k1_KO_s101
                                 + 0.0263893988551957 w_k1_KO_s102
                                 + 0.0527230474905664 w_k1_KO_s103
                                 + 0.0269873958106497 w_k1_KO_s104
                                 + 0.0195821022442113 w_k1_KO_s105
                                 + 0.0530250339875126 w_k1_KO_s106
                                 + 0.0236537816961607 w_k1_KO_s107
                                 + 0.0154805770104461 w_k1_KO_s110
                                 + 0.031682716922927 w_k1_KO_s111
                                 + 0.0121921277683717 w_k1_KO_s112
                                 + 0.0213790293353274 w_k1_KO_s113
                                 + 0.00878988827242237 w_k1_KO_s114
                                 + 0.0126954410679844 w_k1_KO_s115
                                 + 0.00975088956720542 w_k1_KO_s116
                                 + 0.010344759477767 w_k1_KO_s117
                                 + 0.0298656637836547 w_k1_MAT_s000
                                 + 0.0571362236854735 w_k1_MAT_s001
                                 + 0.0286828895836891 w_k1_MAT_s002
                                 + 0.0254583661847659 w_k1_MAT_s003
                                 + 0.0246720517085085 w_k1_MAT_s004
                                 + 0.0417565669282373 w_k1_MAT_s005
                                 + 0.0327661145729755 w_k1_MAT_s006
                                 + 0.0346141717704714 w_k1_MAT_s007
                                 + 0.0273037983937446 w_k1_MAT_s010
                                 + 0.0207709483722667 w_k1_MAT_s011
                                 + 0.0863762374309071 w_k1_MAT_s012
                                 + 0.0330697837665994 w_k1_MAT_s013
                                 + 0.0166814073802227 w_k1_MAT_s014
                                 + 0.037723724951092 w_k1_MAT_s015
                                 + 0.0171260754875572 w_k1_MAT_s016
                                 + 0.0248804732998753 w_k1_MAT_s017
                                 + 0.0781787017871844 w_k1_MAT_s100
                                 + 0.0856341574849934 w_k1_MAT_s101
                                 + 0.0253802398909982 w_k1_MAT_s102
                                 + 0.0456317039945433 w_k1_MAT_s103
                                 + 0.0254957523746851 w_k1_MAT_s104
                                 + 0.022806224983215 w_k1_MAT_s105
                                 + 0.0641085913068763 w_k1_MAT_s106
                                 + 0.0266633510055281 w_k1_MAT_s107
                                 + 0.0201566969940264 w_k1_MAT_s110
                                 + 0.0403494031107069 w_k1_MAT_s111
                                 + 0.0119654097432667 w_k1_MAT_s112
                                 + 0.0201817664515601 w_k1_MAT_s113
                                 + 0.0100023185450707 w_k1_MAT_s114
                                 + 0.0126662262689013 w_k1_MAT_s115
                                 + 0.00852788757252728 w_k1_MAT_s116
                                 + 0.0109526216167011 w_k1_MAT_s117
                                 + 0.029497346710048 w_k1_MSFT_s000
                                 + 0.0584715920866377 w_k1_MSFT_s001
                                 + 0.031733350127775 w_k1_MSFT_s002
                                 + 0.0194004334050416 w_k1_MSFT_s003
                                 + 0.0294686436578898 w_k1_MSFT_s004
                                 + 0.0442655600630008 w_k1_MSFT_s005
                                 + 0.0272408563981671 w_k1_MSFT_s006
                                 + 0.0417042746224316 w_k1_MSFT_s007
                                 + 0.0298209521163604 w_k1_MSFT_s010
                                 + 0.024292919872575 w_k1_MSFT_s011
                                 + 0.0682271905600148 w_k1_MSFT_s012
                                 + 0.0386298473528853 w_k1_MSFT_s013
                                 + 0.0170774260169664 w_k1_MSFT_s014
                                 + 0.0338295917953814 w_k1_MSFT_s015
                                 + 0.0222849397485502 w_k1_MSFT_s016
                                 + 0.0268257604366882 w_k1_MSFT_s017
                                 + 0.0748616446140307 w_k1_MSFT_s100
                                 + 0.0872844916322876 w_k1_MSFT_s101
                                 + 0.0225395016152585 w_k1_MSFT_s102
                                 + 0.0478695879762569 w_k1_MSFT_s103
                                 + 0.0275691820125224 w_k1_MSFT_s104
                                 + 0.0206089286849918 w_k1_MSFT_s105
                                 + 0.0514109265426697 w_k1_MSFT_s106
                                 + 0.0246056930353913 w_k1_MSFT_s107
                                 + 0.0159102864889547 w_k1_MSFT_s110
                                 + 0.0302900770731611 w_k1_MSFT_s111
                                 + 0.0108085660467256 w_k1_MSFT_s112
                                 + 0.0232689111990027 w_k1_MSFT_s113
                                 + 0.0108198603689715 w_k1_MSFT_s114
                                 + 0.0121906649166634 w_k1_MSFT_s115
                                 + 0.00946344018465784 w_k1_MSFT_s116
                                 + 0.0105410218502723 w_k1_MSFT_s117
                                 + 0.0287661023442047 w_k1_PG_s000
                                 + 0.0620775412916589 w_k1_PG_s001
                                 + 0.0331464876115666 w_k1_PG_s002
                                 + 0.018748706459045 w_k1_PG_s003
                                 + 0.0263953978873058 w_k1_PG_s004
                                 + 0.0459987956505632 w_k1_PG_s005
                                 + 0.029688414523593 w_k1_PG_s006
                                 + 0.0364482270814191 w_k1_PG_s007
                                 + 0.0297742289119152 w_k1_PG_s010
                                 + 0.0255402046808798 w_k1_PG_s011
                                 + 0.0745512819301581 w_k1_PG_s012
                                 + 0.0340207267330877 w_k1_PG_s013
                                 + 0.0163369200497872 w_k1_PG_s014
                                 + 0.0351091805269547 w_k1_PG_s015
                                 + 0.019013625062134 w_k1_PG_s016
                                 + 0.0237740371991476 w_k1_PG_s017
                                 + 0.0687238910805371 w_k1_PG_s100
                                 + 0.0817528088526437 w_k1_PG_s101
                                 + 0.0246134328718968 w_k1_PG_s102
                                 + 0.0507329375781211 w_k1_PG_s103
                                 + 0.0268355793565293 w_k1_PG_s104
                                 + 0.0210593066454104 w_k1_PG_s105
                                 + 0.0550967468084371 w_k1_PG_s106
                                 + 0.0236989265812417 w_k1_PG_s107
                                 + 0.0158413027339649 w_k1_PG_s110
                                 + 0.0308324669560911 w_k1_PG_s111
                                 + 0.0111708132193702 w_k1_PG_s112
                                 + 0.0223212191211743 w_k1_PG_s113
                                 + 0.0090853709951636 w_k1_PG_s114
                                 + 0.0118228637299329 w_k1_PG_s115
                                 + 0.00977492509680319 w_k1_PG_s116
                                 + 0.0107567060846555 w_k1_PG_s117 >= 1.09
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
