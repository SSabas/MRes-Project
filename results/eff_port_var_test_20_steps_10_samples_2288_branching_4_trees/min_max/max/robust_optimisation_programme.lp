\ENCODING=ISO-8859-1
\Problem name: 

Maximize
 obj: opt_return
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
 w_k4_ADM_s0_return_constraint:  1.0931 w_0_ADM - w_k4_ADM_s0
                                 + 0.99 b_k4_ADM_s0 - 1.01 s_k4_ADM_s0  = 0
 w_k4_ADM_s1_return_constraint:  0.912942 w_0_ADM - w_k4_ADM_s1
                                 + 0.99 b_k4_ADM_s1 - 1.01 s_k4_ADM_s1  = 0
 w_k4_ADM_s00_return_constraint: 0.930068 w_k4_ADM_s0 - w_k4_ADM_s00
                                 + 0.99 b_k4_ADM_s00 - 1.01 s_k4_ADM_s00  = 0
 w_k4_ADM_s01_return_constraint: 1.20392 w_k4_ADM_s0 - w_k4_ADM_s01
                                 + 0.99 b_k4_ADM_s01 - 1.01 s_k4_ADM_s01  = 0
 w_k4_ADM_s10_return_constraint: 0.983543 w_k4_ADM_s1 - w_k4_ADM_s10
                                 + 0.99 b_k4_ADM_s10 - 1.01 s_k4_ADM_s10  = 0
 w_k4_ADM_s11_return_constraint: 1.22524 w_k4_ADM_s1 - w_k4_ADM_s11
                                 + 0.99 b_k4_ADM_s11 - 1.01 s_k4_ADM_s11  = 0
 w_k4_ADM_s000_return_constraint: 0.790546 w_k4_ADM_s00 - w_k4_ADM_s000
                                 + 0.99 b_k4_ADM_s000 - 1.01 s_k4_ADM_s000  = 0
 w_k4_ADM_s001_return_constraint: 0.947155 w_k4_ADM_s00 - w_k4_ADM_s001
                                 + 0.99 b_k4_ADM_s001 - 1.01 s_k4_ADM_s001  = 0
 w_k4_ADM_s002_return_constraint: 0.842894 w_k4_ADM_s00 - w_k4_ADM_s002
                                 + 0.99 b_k4_ADM_s002 - 1.01 s_k4_ADM_s002  = 0
 w_k4_ADM_s003_return_constraint: 1.10738 w_k4_ADM_s00 - w_k4_ADM_s003
                                 + 0.99 b_k4_ADM_s003 - 1.01 s_k4_ADM_s003  = 0
 w_k4_ADM_s004_return_constraint: 1.12669 w_k4_ADM_s00 - w_k4_ADM_s004
                                 + 0.99 b_k4_ADM_s004 - 1.01 s_k4_ADM_s004  = 0
 w_k4_ADM_s005_return_constraint: 1.33531 w_k4_ADM_s00 - w_k4_ADM_s005
                                 + 0.99 b_k4_ADM_s005 - 1.01 s_k4_ADM_s005  = 0
 w_k4_ADM_s006_return_constraint: 0.97137 w_k4_ADM_s00 - w_k4_ADM_s006
                                 + 0.99 b_k4_ADM_s006 - 1.01 s_k4_ADM_s006  = 0
 w_k4_ADM_s007_return_constraint: 1.40163 w_k4_ADM_s00 - w_k4_ADM_s007
                                 + 0.99 b_k4_ADM_s007 - 1.01 s_k4_ADM_s007  = 0
 w_k4_ADM_s010_return_constraint: 0.823318 w_k4_ADM_s01 - w_k4_ADM_s010
                                 + 0.99 b_k4_ADM_s010 - 1.01 s_k4_ADM_s010  = 0
 w_k4_ADM_s011_return_constraint: 0.997841 w_k4_ADM_s01 - w_k4_ADM_s011
                                 + 0.99 b_k4_ADM_s011 - 1.01 s_k4_ADM_s011  = 0
 w_k4_ADM_s012_return_constraint: 0.939248 w_k4_ADM_s01 - w_k4_ADM_s012
                                 + 0.99 b_k4_ADM_s012 - 1.01 s_k4_ADM_s012  = 0
 w_k4_ADM_s013_return_constraint: 1.11145 w_k4_ADM_s01 - w_k4_ADM_s013
                                 + 0.99 b_k4_ADM_s013 - 1.01 s_k4_ADM_s013  = 0
 w_k4_ADM_s014_return_constraint: 0.807243 w_k4_ADM_s01 - w_k4_ADM_s014
                                 + 0.99 b_k4_ADM_s014 - 1.01 s_k4_ADM_s014  = 0
 w_k4_ADM_s015_return_constraint: 0.886785 w_k4_ADM_s01 - w_k4_ADM_s015
                                 + 0.99 b_k4_ADM_s015 - 1.01 s_k4_ADM_s015  = 0
 w_k4_ADM_s016_return_constraint: 1.14889 w_k4_ADM_s01 - w_k4_ADM_s016
                                 + 0.99 b_k4_ADM_s016 - 1.01 s_k4_ADM_s016  = 0
 w_k4_ADM_s017_return_constraint: 0.909416 w_k4_ADM_s01 - w_k4_ADM_s017
                                 + 0.99 b_k4_ADM_s017 - 1.01 s_k4_ADM_s017  = 0
 w_k4_ADM_s100_return_constraint: 1.04091 w_k4_ADM_s10 - w_k4_ADM_s100
                                 + 0.99 b_k4_ADM_s100 - 1.01 s_k4_ADM_s100  = 0
 w_k4_ADM_s101_return_constraint: 0.804257 w_k4_ADM_s10 - w_k4_ADM_s101
                                 + 0.99 b_k4_ADM_s101 - 1.01 s_k4_ADM_s101  = 0
 w_k4_ADM_s102_return_constraint: 0.991049 w_k4_ADM_s10 - w_k4_ADM_s102
                                 + 0.99 b_k4_ADM_s102 - 1.01 s_k4_ADM_s102  = 0
 w_k4_ADM_s103_return_constraint: 0.939527 w_k4_ADM_s10 - w_k4_ADM_s103
                                 + 0.99 b_k4_ADM_s103 - 1.01 s_k4_ADM_s103  = 0
 w_k4_ADM_s104_return_constraint: 0.903461 w_k4_ADM_s10 - w_k4_ADM_s104
                                 + 0.99 b_k4_ADM_s104 - 1.01 s_k4_ADM_s104  = 0
 w_k4_ADM_s105_return_constraint: 1.03608 w_k4_ADM_s10 - w_k4_ADM_s105
                                 + 0.99 b_k4_ADM_s105 - 1.01 s_k4_ADM_s105  = 0
 w_k4_ADM_s106_return_constraint: 0.859989 w_k4_ADM_s10 - w_k4_ADM_s106
                                 + 0.99 b_k4_ADM_s106 - 1.01 s_k4_ADM_s106  = 0
 w_k4_ADM_s107_return_constraint: 1.35401 w_k4_ADM_s10 - w_k4_ADM_s107
                                 + 0.99 b_k4_ADM_s107 - 1.01 s_k4_ADM_s107  = 0
 w_k4_ADM_s110_return_constraint: 0.942154 w_k4_ADM_s11 - w_k4_ADM_s110
                                 + 0.99 b_k4_ADM_s110 - 1.01 s_k4_ADM_s110  = 0
 w_k4_ADM_s111_return_constraint: 0.866553 w_k4_ADM_s11 - w_k4_ADM_s111
                                 + 0.99 b_k4_ADM_s111 - 1.01 s_k4_ADM_s111  = 0
 w_k4_ADM_s112_return_constraint: 0.678636 w_k4_ADM_s11 - w_k4_ADM_s112
                                 + 0.99 b_k4_ADM_s112 - 1.01 s_k4_ADM_s112  = 0
 w_k4_ADM_s113_return_constraint: 1.19333 w_k4_ADM_s11 - w_k4_ADM_s113
                                 + 0.99 b_k4_ADM_s113 - 1.01 s_k4_ADM_s113  = 0
 w_k4_ADM_s114_return_constraint: 1.02971 w_k4_ADM_s11 - w_k4_ADM_s114
                                 + 0.99 b_k4_ADM_s114 - 1.01 s_k4_ADM_s114  = 0
 w_k4_ADM_s115_return_constraint: 0.940517 w_k4_ADM_s11 - w_k4_ADM_s115
                                 + 0.99 b_k4_ADM_s115 - 1.01 s_k4_ADM_s115  = 0
 w_k4_ADM_s116_return_constraint: 0.957604 w_k4_ADM_s11 - w_k4_ADM_s116
                                 + 0.99 b_k4_ADM_s116 - 1.01 s_k4_ADM_s116  = 0
 w_k4_ADM_s117_return_constraint: 0.823336 w_k4_ADM_s11 - w_k4_ADM_s117
                                 + 0.99 b_k4_ADM_s117 - 1.01 s_k4_ADM_s117  = 0
 w_k4_AXP_s0_return_constraint:  1.18882 w_0_AXP - w_k4_AXP_s0
                                 + 0.99 b_k4_AXP_s0 - 1.01 s_k4_AXP_s0  = 0
 w_k4_AXP_s1_return_constraint:  0.868345 w_0_AXP - w_k4_AXP_s1
                                 + 0.99 b_k4_AXP_s1 - 1.01 s_k4_AXP_s1  = 0
 w_k4_AXP_s00_return_constraint: 0.813575 w_k4_AXP_s0 - w_k4_AXP_s00
                                 + 0.99 b_k4_AXP_s00 - 1.01 s_k4_AXP_s00  = 0
 w_k4_AXP_s01_return_constraint: 1.22445 w_k4_AXP_s0 - w_k4_AXP_s01
                                 + 0.99 b_k4_AXP_s01 - 1.01 s_k4_AXP_s01  = 0
 w_k4_AXP_s10_return_constraint: 0.87371 w_k4_AXP_s1 - w_k4_AXP_s10
                                 + 0.99 b_k4_AXP_s10 - 1.01 s_k4_AXP_s10  = 0
 w_k4_AXP_s11_return_constraint: 1.28307 w_k4_AXP_s1 - w_k4_AXP_s11
                                 + 0.99 b_k4_AXP_s11 - 1.01 s_k4_AXP_s11  = 0
 w_k4_AXP_s000_return_constraint: 0.738222 w_k4_AXP_s00 - w_k4_AXP_s000
                                 + 0.99 b_k4_AXP_s000 - 1.01 s_k4_AXP_s000  = 0
 w_k4_AXP_s001_return_constraint: 1.11754 w_k4_AXP_s00 - w_k4_AXP_s001
                                 + 0.99 b_k4_AXP_s001 - 1.01 s_k4_AXP_s001  = 0
 w_k4_AXP_s002_return_constraint: 0.724778 w_k4_AXP_s00 - w_k4_AXP_s002
                                 + 0.99 b_k4_AXP_s002 - 1.01 s_k4_AXP_s002  = 0
 w_k4_AXP_s003_return_constraint: 1.23142 w_k4_AXP_s00 - w_k4_AXP_s003
                                 + 0.99 b_k4_AXP_s003 - 1.01 s_k4_AXP_s003  = 0
 w_k4_AXP_s004_return_constraint: 1.01271 w_k4_AXP_s00 - w_k4_AXP_s004
                                 + 0.99 b_k4_AXP_s004 - 1.01 s_k4_AXP_s004  = 0
 w_k4_AXP_s005_return_constraint: 1.27395 w_k4_AXP_s00 - w_k4_AXP_s005
                                 + 0.99 b_k4_AXP_s005 - 1.01 s_k4_AXP_s005  = 0
 w_k4_AXP_s006_return_constraint: 0.8874 w_k4_AXP_s00 - w_k4_AXP_s006
                                 + 0.99 b_k4_AXP_s006 - 1.01 s_k4_AXP_s006  = 0
 w_k4_AXP_s007_return_constraint: 1.42004 w_k4_AXP_s00 - w_k4_AXP_s007
                                 + 0.99 b_k4_AXP_s007 - 1.01 s_k4_AXP_s007  = 0
 w_k4_AXP_s010_return_constraint: 0.907161 w_k4_AXP_s01 - w_k4_AXP_s010
                                 + 0.99 b_k4_AXP_s010 - 1.01 s_k4_AXP_s010  = 0
 w_k4_AXP_s011_return_constraint: 0.979884 w_k4_AXP_s01 - w_k4_AXP_s011
                                 + 0.99 b_k4_AXP_s011 - 1.01 s_k4_AXP_s011  = 0
 w_k4_AXP_s012_return_constraint: 0.855938 w_k4_AXP_s01 - w_k4_AXP_s012
                                 + 0.99 b_k4_AXP_s012 - 1.01 s_k4_AXP_s012  = 0
 w_k4_AXP_s013_return_constraint: 1.23936 w_k4_AXP_s01 - w_k4_AXP_s013
                                 + 0.99 b_k4_AXP_s013 - 1.01 s_k4_AXP_s013  = 0
 w_k4_AXP_s014_return_constraint: 0.659264 w_k4_AXP_s01 - w_k4_AXP_s014
                                 + 0.99 b_k4_AXP_s014 - 1.01 s_k4_AXP_s014  = 0
 w_k4_AXP_s015_return_constraint: 0.718119 w_k4_AXP_s01 - w_k4_AXP_s015
                                 + 0.99 b_k4_AXP_s015 - 1.01 s_k4_AXP_s015  = 0
 w_k4_AXP_s016_return_constraint: 1.48202 w_k4_AXP_s01 - w_k4_AXP_s016
                                 + 0.99 b_k4_AXP_s016 - 1.01 s_k4_AXP_s016  = 0
 w_k4_AXP_s017_return_constraint: 1.14663 w_k4_AXP_s01 - w_k4_AXP_s017
                                 + 0.99 b_k4_AXP_s017 - 1.01 s_k4_AXP_s017  = 0
 w_k4_AXP_s100_return_constraint: 0.965828 w_k4_AXP_s10 - w_k4_AXP_s100
                                 + 0.99 b_k4_AXP_s100 - 1.01 s_k4_AXP_s100  = 0
 w_k4_AXP_s101_return_constraint: 0.7138 w_k4_AXP_s10 - w_k4_AXP_s101
                                 + 0.99 b_k4_AXP_s101 - 1.01 s_k4_AXP_s101  = 0
 w_k4_AXP_s102_return_constraint: 1.1561 w_k4_AXP_s10 - w_k4_AXP_s102
                                 + 0.99 b_k4_AXP_s102 - 1.01 s_k4_AXP_s102  = 0
 w_k4_AXP_s103_return_constraint: 0.856194 w_k4_AXP_s10 - w_k4_AXP_s103
                                 + 0.99 b_k4_AXP_s103 - 1.01 s_k4_AXP_s103  = 0
 w_k4_AXP_s104_return_constraint: 1.20585 w_k4_AXP_s10 - w_k4_AXP_s104
                                 + 0.99 b_k4_AXP_s104 - 1.01 s_k4_AXP_s104  = 0
 w_k4_AXP_s105_return_constraint: 1.11148 w_k4_AXP_s10 - w_k4_AXP_s105
                                 + 0.99 b_k4_AXP_s105 - 1.01 s_k4_AXP_s105  = 0
 w_k4_AXP_s106_return_constraint: 1.37229 w_k4_AXP_s10 - w_k4_AXP_s106
                                 + 0.99 b_k4_AXP_s106 - 1.01 s_k4_AXP_s106  = 0
 w_k4_AXP_s107_return_constraint: 1.46911 w_k4_AXP_s10 - w_k4_AXP_s107
                                 + 0.99 b_k4_AXP_s107 - 1.01 s_k4_AXP_s107  = 0
 w_k4_AXP_s110_return_constraint: 0.82038 w_k4_AXP_s11 - w_k4_AXP_s110
                                 + 0.99 b_k4_AXP_s110 - 1.01 s_k4_AXP_s110  = 0
 w_k4_AXP_s111_return_constraint: 0.914119 w_k4_AXP_s11 - w_k4_AXP_s111
                                 + 0.99 b_k4_AXP_s111 - 1.01 s_k4_AXP_s111  = 0
 w_k4_AXP_s112_return_constraint: 0.630397 w_k4_AXP_s11 - w_k4_AXP_s112
                                 + 0.99 b_k4_AXP_s112 - 1.01 s_k4_AXP_s112  = 0
 w_k4_AXP_s113_return_constraint: 1.06088 w_k4_AXP_s11 - w_k4_AXP_s113
                                 + 0.99 b_k4_AXP_s113 - 1.01 s_k4_AXP_s113  = 0
 w_k4_AXP_s114_return_constraint: 0.87182 w_k4_AXP_s11 - w_k4_AXP_s114
                                 + 0.99 b_k4_AXP_s114 - 1.01 s_k4_AXP_s114  = 0
 w_k4_AXP_s115_return_constraint: 1.22979 w_k4_AXP_s11 - w_k4_AXP_s115
                                 + 0.99 b_k4_AXP_s115 - 1.01 s_k4_AXP_s115  = 0
 w_k4_AXP_s116_return_constraint: 1.27181 w_k4_AXP_s11 - w_k4_AXP_s116
                                 + 0.99 b_k4_AXP_s116 - 1.01 s_k4_AXP_s116  = 0
 w_k4_AXP_s117_return_constraint: 0.618427 w_k4_AXP_s11 - w_k4_AXP_s117
                                 + 0.99 b_k4_AXP_s117 - 1.01 s_k4_AXP_s117  = 0
 w_k4_DIS_s0_return_constraint:  1.25055 w_0_DIS - w_k4_DIS_s0
                                 + 0.99 b_k4_DIS_s0 - 1.01 s_k4_DIS_s0  = 0
 w_k4_DIS_s1_return_constraint:  0.835509 w_0_DIS - w_k4_DIS_s1
                                 + 0.99 b_k4_DIS_s1 - 1.01 s_k4_DIS_s1  = 0
 w_k4_DIS_s00_return_constraint: 0.791846 w_k4_DIS_s0 - w_k4_DIS_s00
                                 + 0.99 b_k4_DIS_s00 - 1.01 s_k4_DIS_s00  = 0
 w_k4_DIS_s01_return_constraint: 1.29395 w_k4_DIS_s0 - w_k4_DIS_s01
                                 + 0.99 b_k4_DIS_s01 - 1.01 s_k4_DIS_s01  = 0
 w_k4_DIS_s10_return_constraint: 0.836807 w_k4_DIS_s1 - w_k4_DIS_s10
                                 + 0.99 b_k4_DIS_s10 - 1.01 s_k4_DIS_s10  = 0
 w_k4_DIS_s11_return_constraint: 1.37685 w_k4_DIS_s1 - w_k4_DIS_s11
                                 + 0.99 b_k4_DIS_s11 - 1.01 s_k4_DIS_s11  = 0
 w_k4_DIS_s000_return_constraint: 0.560367 w_k4_DIS_s00 - w_k4_DIS_s000
                                 + 0.99 b_k4_DIS_s000 - 1.01 s_k4_DIS_s000  = 0
 w_k4_DIS_s001_return_constraint: 0.978475 w_k4_DIS_s00 - w_k4_DIS_s001
                                 + 0.99 b_k4_DIS_s001 - 1.01 s_k4_DIS_s001  = 0
 w_k4_DIS_s002_return_constraint: 0.707832 w_k4_DIS_s00 - w_k4_DIS_s002
                                 + 0.99 b_k4_DIS_s002 - 1.01 s_k4_DIS_s002  = 0
 w_k4_DIS_s003_return_constraint: 1.14838 w_k4_DIS_s00 - w_k4_DIS_s003
                                 + 0.99 b_k4_DIS_s003 - 1.01 s_k4_DIS_s003  = 0
 w_k4_DIS_s004_return_constraint: 1.00594 w_k4_DIS_s00 - w_k4_DIS_s004
                                 + 0.99 b_k4_DIS_s004 - 1.01 s_k4_DIS_s004  = 0
 w_k4_DIS_s005_return_constraint: 1.41547 w_k4_DIS_s00 - w_k4_DIS_s005
                                 + 0.99 b_k4_DIS_s005 - 1.01 s_k4_DIS_s005  = 0
 w_k4_DIS_s006_return_constraint: 0.92648 w_k4_DIS_s00 - w_k4_DIS_s006
                                 + 0.99 b_k4_DIS_s006 - 1.01 s_k4_DIS_s006  = 0
 w_k4_DIS_s007_return_constraint: 1.88619 w_k4_DIS_s00 - w_k4_DIS_s007
                                 + 0.99 b_k4_DIS_s007 - 1.01 s_k4_DIS_s007  = 0
 w_k4_DIS_s010_return_constraint: 0.790824 w_k4_DIS_s01 - w_k4_DIS_s010
                                 + 0.99 b_k4_DIS_s010 - 1.01 s_k4_DIS_s010  = 0
 w_k4_DIS_s011_return_constraint: 0.992482 w_k4_DIS_s01 - w_k4_DIS_s011
                                 + 0.99 b_k4_DIS_s011 - 1.01 s_k4_DIS_s011  = 0
 w_k4_DIS_s012_return_constraint: 1.05724 w_k4_DIS_s01 - w_k4_DIS_s012
                                 + 0.99 b_k4_DIS_s012 - 1.01 s_k4_DIS_s012  = 0
 w_k4_DIS_s013_return_constraint: 1.22053 w_k4_DIS_s01 - w_k4_DIS_s013
                                 + 0.99 b_k4_DIS_s013 - 1.01 s_k4_DIS_s013  = 0
 w_k4_DIS_s014_return_constraint: 0.622774 w_k4_DIS_s01 - w_k4_DIS_s014
                                 + 0.99 b_k4_DIS_s014 - 1.01 s_k4_DIS_s014  = 0
 w_k4_DIS_s015_return_constraint: 0.668858 w_k4_DIS_s01 - w_k4_DIS_s015
                                 + 0.99 b_k4_DIS_s015 - 1.01 s_k4_DIS_s015  = 0
 w_k4_DIS_s016_return_constraint: 1.75211 w_k4_DIS_s01 - w_k4_DIS_s016
                                 + 0.99 b_k4_DIS_s016 - 1.01 s_k4_DIS_s016  = 0
 w_k4_DIS_s017_return_constraint: 0.731204 w_k4_DIS_s01 - w_k4_DIS_s017
                                 + 0.99 b_k4_DIS_s017 - 1.01 s_k4_DIS_s017  = 0
 w_k4_DIS_s100_return_constraint: 1.17942 w_k4_DIS_s10 - w_k4_DIS_s100
                                 + 0.99 b_k4_DIS_s100 - 1.01 s_k4_DIS_s100  = 0
 w_k4_DIS_s101_return_constraint: 0.681527 w_k4_DIS_s10 - w_k4_DIS_s101
                                 + 0.99 b_k4_DIS_s101 - 1.01 s_k4_DIS_s101  = 0
 w_k4_DIS_s102_return_constraint: 0.918349 w_k4_DIS_s10 - w_k4_DIS_s102
                                 + 0.99 b_k4_DIS_s102 - 1.01 s_k4_DIS_s102  = 0
 w_k4_DIS_s103_return_constraint: 0.851939 w_k4_DIS_s10 - w_k4_DIS_s103
                                 + 0.99 b_k4_DIS_s103 - 1.01 s_k4_DIS_s103  = 0
 w_k4_DIS_s104_return_constraint: 1.22036 w_k4_DIS_s10 - w_k4_DIS_s104
                                 + 0.99 b_k4_DIS_s104 - 1.01 s_k4_DIS_s104  = 0
 w_k4_DIS_s105_return_constraint: 1.18578 w_k4_DIS_s10 - w_k4_DIS_s105
                                 + 0.99 b_k4_DIS_s105 - 1.01 s_k4_DIS_s105  = 0
 w_k4_DIS_s106_return_constraint: 1.03932 w_k4_DIS_s10 - w_k4_DIS_s106
                                 + 0.99 b_k4_DIS_s106 - 1.01 s_k4_DIS_s106  = 0
 w_k4_DIS_s107_return_constraint: 1.54957 w_k4_DIS_s10 - w_k4_DIS_s107
                                 + 0.99 b_k4_DIS_s107 - 1.01 s_k4_DIS_s107  = 0
 w_k4_DIS_s110_return_constraint: 1.00397 w_k4_DIS_s11 - w_k4_DIS_s110
                                 + 0.99 b_k4_DIS_s110 - 1.01 s_k4_DIS_s110  = 0
 w_k4_DIS_s111_return_constraint: 0.806885 w_k4_DIS_s11 - w_k4_DIS_s111
                                 + 0.99 b_k4_DIS_s111 - 1.01 s_k4_DIS_s111  = 0
 w_k4_DIS_s112_return_constraint: 0.635297 w_k4_DIS_s11 - w_k4_DIS_s112
                                 + 0.99 b_k4_DIS_s112 - 1.01 s_k4_DIS_s112  = 0
 w_k4_DIS_s113_return_constraint: 0.982564 w_k4_DIS_s11 - w_k4_DIS_s113
                                 + 0.99 b_k4_DIS_s113 - 1.01 s_k4_DIS_s113  = 0
 w_k4_DIS_s114_return_constraint: 0.827952 w_k4_DIS_s11 - w_k4_DIS_s114
                                 + 0.99 b_k4_DIS_s114 - 1.01 s_k4_DIS_s114  = 0
 w_k4_DIS_s115_return_constraint: 1.19545 w_k4_DIS_s11 - w_k4_DIS_s115
                                 + 0.99 b_k4_DIS_s115 - 1.01 s_k4_DIS_s115  = 0
 w_k4_DIS_s116_return_constraint: 1.46346 w_k4_DIS_s11 - w_k4_DIS_s116
                                 + 0.99 b_k4_DIS_s116 - 1.01 s_k4_DIS_s116  = 0
 w_k4_DIS_s117_return_constraint: 0.660005 w_k4_DIS_s11 - w_k4_DIS_s117
                                 + 0.99 b_k4_DIS_s117 - 1.01 s_k4_DIS_s117  = 0
 w_k4_FDX_s0_return_constraint:  1.13327 w_0_FDX - w_k4_FDX_s0
                                 + 0.99 b_k4_FDX_s0 - 1.01 s_k4_FDX_s0  = 0
 w_k4_FDX_s1_return_constraint:  0.895008 w_0_FDX - w_k4_FDX_s1
                                 + 0.99 b_k4_FDX_s1 - 1.01 s_k4_FDX_s1  = 0
 w_k4_FDX_s00_return_constraint: 0.891357 w_k4_FDX_s0 - w_k4_FDX_s00
                                 + 0.99 b_k4_FDX_s00 - 1.01 s_k4_FDX_s00  = 0
 w_k4_FDX_s01_return_constraint: 1.16947 w_k4_FDX_s0 - w_k4_FDX_s01
                                 + 0.99 b_k4_FDX_s01 - 1.01 s_k4_FDX_s01  = 0
 w_k4_FDX_s10_return_constraint: 0.882463 w_k4_FDX_s1 - w_k4_FDX_s10
                                 + 0.99 b_k4_FDX_s10 - 1.01 s_k4_FDX_s10  = 0
 w_k4_FDX_s11_return_constraint: 1.18892 w_k4_FDX_s1 - w_k4_FDX_s11
                                 + 0.99 b_k4_FDX_s11 - 1.01 s_k4_FDX_s11  = 0
 w_k4_FDX_s000_return_constraint: 0.757874 w_k4_FDX_s00 - w_k4_FDX_s000
                                 + 0.99 b_k4_FDX_s000 - 1.01 s_k4_FDX_s000  = 0
 w_k4_FDX_s001_return_constraint: 0.950439 w_k4_FDX_s00 - w_k4_FDX_s001
                                 + 0.99 b_k4_FDX_s001 - 1.01 s_k4_FDX_s001  = 0
 w_k4_FDX_s002_return_constraint: 0.736164 w_k4_FDX_s00 - w_k4_FDX_s002
                                 + 0.99 b_k4_FDX_s002 - 1.01 s_k4_FDX_s002  = 0
 w_k4_FDX_s003_return_constraint: 1.0191 w_k4_FDX_s00 - w_k4_FDX_s003
                                 + 0.99 b_k4_FDX_s003 - 1.01 s_k4_FDX_s003  = 0
 w_k4_FDX_s004_return_constraint: 1.13967 w_k4_FDX_s00 - w_k4_FDX_s004
                                 + 0.99 b_k4_FDX_s004 - 1.01 s_k4_FDX_s004  = 0
 w_k4_FDX_s005_return_constraint: 1.30525 w_k4_FDX_s00 - w_k4_FDX_s005
                                 + 0.99 b_k4_FDX_s005 - 1.01 s_k4_FDX_s005  = 0
 w_k4_FDX_s006_return_constraint: 0.956902 w_k4_FDX_s00 - w_k4_FDX_s006
                                 + 0.99 b_k4_FDX_s006 - 1.01 s_k4_FDX_s006  = 0
 w_k4_FDX_s007_return_constraint: 1.44455 w_k4_FDX_s00 - w_k4_FDX_s007
                                 + 0.99 b_k4_FDX_s007 - 1.01 s_k4_FDX_s007  = 0
 w_k4_FDX_s010_return_constraint: 0.795396 w_k4_FDX_s01 - w_k4_FDX_s010
                                 + 0.99 b_k4_FDX_s010 - 1.01 s_k4_FDX_s010  = 0
 w_k4_FDX_s011_return_constraint: 1.10924 w_k4_FDX_s01 - w_k4_FDX_s011
                                 + 0.99 b_k4_FDX_s011 - 1.01 s_k4_FDX_s011  = 0
 w_k4_FDX_s012_return_constraint: 0.95949 w_k4_FDX_s01 - w_k4_FDX_s012
                                 + 0.99 b_k4_FDX_s012 - 1.01 s_k4_FDX_s012  = 0
 w_k4_FDX_s013_return_constraint: 1.11163 w_k4_FDX_s01 - w_k4_FDX_s013
                                 + 0.99 b_k4_FDX_s013 - 1.01 s_k4_FDX_s013  = 0
 w_k4_FDX_s014_return_constraint: 0.748803 w_k4_FDX_s01 - w_k4_FDX_s014
                                 + 0.99 b_k4_FDX_s014 - 1.01 s_k4_FDX_s014  = 0
 w_k4_FDX_s015_return_constraint: 0.842555 w_k4_FDX_s01 - w_k4_FDX_s015
                                 + 0.99 b_k4_FDX_s015 - 1.01 s_k4_FDX_s015  = 0
 w_k4_FDX_s016_return_constraint: 1.51004 w_k4_FDX_s01 - w_k4_FDX_s016
                                 + 0.99 b_k4_FDX_s016 - 1.01 s_k4_FDX_s016  = 0
 w_k4_FDX_s017_return_constraint: 0.852742 w_k4_FDX_s01 - w_k4_FDX_s017
                                 + 0.99 b_k4_FDX_s017 - 1.01 s_k4_FDX_s017  = 0
 w_k4_FDX_s100_return_constraint: 1.09548 w_k4_FDX_s10 - w_k4_FDX_s100
                                 + 0.99 b_k4_FDX_s100 - 1.01 s_k4_FDX_s100  = 0
 w_k4_FDX_s101_return_constraint: 0.774602 w_k4_FDX_s10 - w_k4_FDX_s101
                                 + 0.99 b_k4_FDX_s101 - 1.01 s_k4_FDX_s101  = 0
 w_k4_FDX_s102_return_constraint: 0.992669 w_k4_FDX_s10 - w_k4_FDX_s102
                                 + 0.99 b_k4_FDX_s102 - 1.01 s_k4_FDX_s102  = 0
 w_k4_FDX_s103_return_constraint: 1.01777 w_k4_FDX_s10 - w_k4_FDX_s103
                                 + 0.99 b_k4_FDX_s103 - 1.01 s_k4_FDX_s103  = 0
 w_k4_FDX_s104_return_constraint: 1.01426 w_k4_FDX_s10 - w_k4_FDX_s104
                                 + 0.99 b_k4_FDX_s104 - 1.01 s_k4_FDX_s104  = 0
 w_k4_FDX_s105_return_constraint: 1.10593 w_k4_FDX_s10 - w_k4_FDX_s105
                                 + 0.99 b_k4_FDX_s105 - 1.01 s_k4_FDX_s105  = 0
 w_k4_FDX_s106_return_constraint: 1.07845 w_k4_FDX_s10 - w_k4_FDX_s106
                                 + 0.99 b_k4_FDX_s106 - 1.01 s_k4_FDX_s106  = 0
 w_k4_FDX_s107_return_constraint: 1.46916 w_k4_FDX_s10 - w_k4_FDX_s107
                                 + 0.99 b_k4_FDX_s107 - 1.01 s_k4_FDX_s107  = 0
 w_k4_FDX_s110_return_constraint: 0.962267 w_k4_FDX_s11 - w_k4_FDX_s110
                                 + 0.99 b_k4_FDX_s110 - 1.01 s_k4_FDX_s110  = 0
 w_k4_FDX_s111_return_constraint: 1.02017 w_k4_FDX_s11 - w_k4_FDX_s111
                                 + 0.99 b_k4_FDX_s111 - 1.01 s_k4_FDX_s111  = 0
 w_k4_FDX_s112_return_constraint: 0.710912 w_k4_FDX_s11 - w_k4_FDX_s112
                                 + 0.99 b_k4_FDX_s112 - 1.01 s_k4_FDX_s112  = 0
 w_k4_FDX_s113_return_constraint: 1.26909 w_k4_FDX_s11 - w_k4_FDX_s113
                                 + 0.99 b_k4_FDX_s113 - 1.01 s_k4_FDX_s113  = 0
 w_k4_FDX_s114_return_constraint: 0.821871 w_k4_FDX_s11 - w_k4_FDX_s114
                                 + 0.99 b_k4_FDX_s114 - 1.01 s_k4_FDX_s114  = 0
 w_k4_FDX_s115_return_constraint: 1.09139 w_k4_FDX_s11 - w_k4_FDX_s115
                                 + 0.99 b_k4_FDX_s115 - 1.01 s_k4_FDX_s115  = 0
 w_k4_FDX_s116_return_constraint: 1.29305 w_k4_FDX_s11 - w_k4_FDX_s116
                                 + 0.99 b_k4_FDX_s116 - 1.01 s_k4_FDX_s116  = 0
 w_k4_FDX_s117_return_constraint: 0.848929 w_k4_FDX_s11 - w_k4_FDX_s117
                                 + 0.99 b_k4_FDX_s117 - 1.01 s_k4_FDX_s117  = 0
 w_k4_IBM_s0_return_constraint:  1.14411 w_0_IBM - w_k4_IBM_s0
                                 + 0.99 b_k4_IBM_s0 - 1.01 s_k4_IBM_s0  = 0
 w_k4_IBM_s1_return_constraint:  0.927032 w_0_IBM - w_k4_IBM_s1
                                 + 0.99 b_k4_IBM_s1 - 1.01 s_k4_IBM_s1  = 0
 w_k4_IBM_s00_return_constraint: 0.838777 w_k4_IBM_s0 - w_k4_IBM_s00
                                 + 0.99 b_k4_IBM_s00 - 1.01 s_k4_IBM_s00  = 0
 w_k4_IBM_s01_return_constraint: 1.26589 w_k4_IBM_s0 - w_k4_IBM_s01
                                 + 0.99 b_k4_IBM_s01 - 1.01 s_k4_IBM_s01  = 0
 w_k4_IBM_s10_return_constraint: 0.94909 w_k4_IBM_s1 - w_k4_IBM_s10
                                 + 0.99 b_k4_IBM_s10 - 1.01 s_k4_IBM_s10  = 0
 w_k4_IBM_s11_return_constraint: 1.14104 w_k4_IBM_s1 - w_k4_IBM_s11
                                 + 0.99 b_k4_IBM_s11 - 1.01 s_k4_IBM_s11  = 0
 w_k4_IBM_s000_return_constraint: 0.616044 w_k4_IBM_s00 - w_k4_IBM_s000
                                 + 0.99 b_k4_IBM_s000 - 1.01 s_k4_IBM_s000  = 0
 w_k4_IBM_s001_return_constraint: 1.2097 w_k4_IBM_s00 - w_k4_IBM_s001
                                 + 0.99 b_k4_IBM_s001 - 1.01 s_k4_IBM_s001  = 0
 w_k4_IBM_s002_return_constraint: 1.1515 w_k4_IBM_s00 - w_k4_IBM_s002
                                 + 0.99 b_k4_IBM_s002 - 1.01 s_k4_IBM_s002  = 0
 w_k4_IBM_s003_return_constraint: 1.65086 w_k4_IBM_s00 - w_k4_IBM_s003
                                 + 0.99 b_k4_IBM_s003 - 1.01 s_k4_IBM_s003  = 0
 w_k4_IBM_s004_return_constraint: 0.716045 w_k4_IBM_s00 - w_k4_IBM_s004
                                 + 0.99 b_k4_IBM_s004 - 1.01 s_k4_IBM_s004  = 0
 w_k4_IBM_s005_return_constraint: 0.901203 w_k4_IBM_s00 - w_k4_IBM_s005
                                 + 0.99 b_k4_IBM_s005 - 1.01 s_k4_IBM_s005  = 0
 w_k4_IBM_s006_return_constraint: 0.922034 w_k4_IBM_s00 - w_k4_IBM_s006
                                 + 0.99 b_k4_IBM_s006 - 1.01 s_k4_IBM_s006  = 0
 w_k4_IBM_s007_return_constraint: 1.29163 w_k4_IBM_s00 - w_k4_IBM_s007
                                 + 0.99 b_k4_IBM_s007 - 1.01 s_k4_IBM_s007  = 0
 w_k4_IBM_s010_return_constraint: 1.32863 w_k4_IBM_s01 - w_k4_IBM_s010
                                 + 0.99 b_k4_IBM_s010 - 1.01 s_k4_IBM_s010  = 0
 w_k4_IBM_s011_return_constraint: 0.66985 w_k4_IBM_s01 - w_k4_IBM_s011
                                 + 0.99 b_k4_IBM_s011 - 1.01 s_k4_IBM_s011  = 0
 w_k4_IBM_s012_return_constraint: 0.996786 w_k4_IBM_s01 - w_k4_IBM_s012
                                 + 0.99 b_k4_IBM_s012 - 1.01 s_k4_IBM_s012  = 0
 w_k4_IBM_s013_return_constraint: 1.43869 w_k4_IBM_s01 - w_k4_IBM_s013
                                 + 0.99 b_k4_IBM_s013 - 1.01 s_k4_IBM_s013  = 0
 w_k4_IBM_s014_return_constraint: 0.887795 w_k4_IBM_s01 - w_k4_IBM_s014
                                 + 0.99 b_k4_IBM_s014 - 1.01 s_k4_IBM_s014  = 0
 w_k4_IBM_s015_return_constraint: 0.580762 w_k4_IBM_s01 - w_k4_IBM_s015
                                 + 0.99 b_k4_IBM_s015 - 1.01 s_k4_IBM_s015  = 0
 w_k4_IBM_s016_return_constraint: 0.948224 w_k4_IBM_s01 - w_k4_IBM_s016
                                 + 0.99 b_k4_IBM_s016 - 1.01 s_k4_IBM_s016  = 0
 w_k4_IBM_s017_return_constraint: 0.927604 w_k4_IBM_s01 - w_k4_IBM_s017
                                 + 0.99 b_k4_IBM_s017 - 1.01 s_k4_IBM_s017  = 0
 w_k4_IBM_s100_return_constraint: 0.830507 w_k4_IBM_s10 - w_k4_IBM_s100
                                 + 0.99 b_k4_IBM_s100 - 1.01 s_k4_IBM_s100  = 0
 w_k4_IBM_s101_return_constraint: 0.735139 w_k4_IBM_s10 - w_k4_IBM_s101
                                 + 0.99 b_k4_IBM_s101 - 1.01 s_k4_IBM_s101  = 0
 w_k4_IBM_s102_return_constraint: 1.11121 w_k4_IBM_s10 - w_k4_IBM_s102
                                 + 0.99 b_k4_IBM_s102 - 1.01 s_k4_IBM_s102  = 0
 w_k4_IBM_s103_return_constraint: 0.586347 w_k4_IBM_s10 - w_k4_IBM_s103
                                 + 0.99 b_k4_IBM_s103 - 1.01 s_k4_IBM_s103  = 0
 w_k4_IBM_s104_return_constraint: 1.63209 w_k4_IBM_s10 - w_k4_IBM_s104
                                 + 0.99 b_k4_IBM_s104 - 1.01 s_k4_IBM_s104  = 0
 w_k4_IBM_s105_return_constraint: 1.08766 w_k4_IBM_s10 - w_k4_IBM_s105
                                 + 0.99 b_k4_IBM_s105 - 1.01 s_k4_IBM_s105  = 0
 w_k4_IBM_s106_return_constraint: 1.21918 w_k4_IBM_s10 - w_k4_IBM_s106
                                 + 0.99 b_k4_IBM_s106 - 1.01 s_k4_IBM_s106  = 0
 w_k4_IBM_s107_return_constraint: 0.928947 w_k4_IBM_s10 - w_k4_IBM_s107
                                 + 0.99 b_k4_IBM_s107 - 1.01 s_k4_IBM_s107  = 0
 w_k4_IBM_s110_return_constraint: 0.897041 w_k4_IBM_s11 - w_k4_IBM_s110
                                 + 0.99 b_k4_IBM_s110 - 1.01 s_k4_IBM_s110  = 0
 w_k4_IBM_s111_return_constraint: 0.625413 w_k4_IBM_s11 - w_k4_IBM_s111
                                 + 0.99 b_k4_IBM_s111 - 1.01 s_k4_IBM_s111  = 0
 w_k4_IBM_s112_return_constraint: 0.714927 w_k4_IBM_s11 - w_k4_IBM_s112
                                 + 0.99 b_k4_IBM_s112 - 1.01 s_k4_IBM_s112  = 0
 w_k4_IBM_s113_return_constraint: 0.656961 w_k4_IBM_s11 - w_k4_IBM_s113
                                 + 0.99 b_k4_IBM_s113 - 1.01 s_k4_IBM_s113  = 0
 w_k4_IBM_s114_return_constraint: 1.2179 w_k4_IBM_s11 - w_k4_IBM_s114
                                 + 0.99 b_k4_IBM_s114 - 1.01 s_k4_IBM_s114  = 0
 w_k4_IBM_s115_return_constraint: 1.45458 w_k4_IBM_s11 - w_k4_IBM_s115
                                 + 0.99 b_k4_IBM_s115 - 1.01 s_k4_IBM_s115  = 0
 w_k4_IBM_s116_return_constraint: 0.989848 w_k4_IBM_s11 - w_k4_IBM_s116
                                 + 0.99 b_k4_IBM_s116 - 1.01 s_k4_IBM_s116  = 0
 w_k4_IBM_s117_return_constraint: 0.523262 w_k4_IBM_s11 - w_k4_IBM_s117
                                 + 0.99 b_k4_IBM_s117 - 1.01 s_k4_IBM_s117  = 0
 w_k4_INTC_s0_return_constraint: 1.10988 w_0_INTC - w_k4_INTC_s0
                                 + 0.99 b_k4_INTC_s0 - 1.01 s_k4_INTC_s0  = 0
 w_k4_INTC_s1_return_constraint: 0.916302 w_0_INTC - w_k4_INTC_s1
                                 + 0.99 b_k4_INTC_s1 - 1.01 s_k4_INTC_s1  = 0
 w_k4_INTC_s00_return_constraint: 0.897972 w_k4_INTC_s0 - w_k4_INTC_s00
                                 + 0.99 b_k4_INTC_s00 - 1.01 s_k4_INTC_s00  = 0
 w_k4_INTC_s01_return_constraint: 1.09085 w_k4_INTC_s0 - w_k4_INTC_s01
                                 + 0.99 b_k4_INTC_s01 - 1.01 s_k4_INTC_s01  = 0
 w_k4_INTC_s10_return_constraint: 0.865535 w_k4_INTC_s1 - w_k4_INTC_s10
                                 + 0.99 b_k4_INTC_s10 - 1.01 s_k4_INTC_s10  = 0
 w_k4_INTC_s11_return_constraint: 1.08362 w_k4_INTC_s1 - w_k4_INTC_s11
                                 + 0.99 b_k4_INTC_s11 - 1.01 s_k4_INTC_s11  = 0
 w_k4_INTC_s000_return_constraint: 0.855573 w_k4_INTC_s00 - w_k4_INTC_s000
                                 + 0.99 b_k4_INTC_s000 - 1.01 s_k4_INTC_s000
                                  = 0
 w_k4_INTC_s001_return_constraint: 0.888546 w_k4_INTC_s00 - w_k4_INTC_s001
                                 + 0.99 b_k4_INTC_s001 - 1.01 s_k4_INTC_s001
                                  = 0
 w_k4_INTC_s002_return_constraint: 0.73129 w_k4_INTC_s00 - w_k4_INTC_s002
                                 + 0.99 b_k4_INTC_s002 - 1.01 s_k4_INTC_s002
                                  = 0
 w_k4_INTC_s003_return_constraint: 0.964551 w_k4_INTC_s00 - w_k4_INTC_s003
                                 + 0.99 b_k4_INTC_s003 - 1.01 s_k4_INTC_s003
                                  = 0
 w_k4_INTC_s004_return_constraint: 1.14465 w_k4_INTC_s00 - w_k4_INTC_s004
                                 + 0.99 b_k4_INTC_s004 - 1.01 s_k4_INTC_s004
                                  = 0
 w_k4_INTC_s005_return_constraint: 1.19657 w_k4_INTC_s00 - w_k4_INTC_s005
                                 + 0.99 b_k4_INTC_s005 - 1.01 s_k4_INTC_s005
                                  = 0
 w_k4_INTC_s006_return_constraint: 0.920756 w_k4_INTC_s00 - w_k4_INTC_s006
                                 + 0.99 b_k4_INTC_s006 - 1.01 s_k4_INTC_s006
                                  = 0
 w_k4_INTC_s007_return_constraint: 1.32334 w_k4_INTC_s00 - w_k4_INTC_s007
                                 + 0.99 b_k4_INTC_s007 - 1.01 s_k4_INTC_s007
                                  = 0
 w_k4_INTC_s010_return_constraint: 0.961227 w_k4_INTC_s01 - w_k4_INTC_s010
                                 + 0.99 b_k4_INTC_s010 - 1.01 s_k4_INTC_s010
                                  = 0
 w_k4_INTC_s011_return_constraint: 0.99579 w_k4_INTC_s01 - w_k4_INTC_s011
                                 + 0.99 b_k4_INTC_s011 - 1.01 s_k4_INTC_s011
                                  = 0
 w_k4_INTC_s012_return_constraint: 0.993925 w_k4_INTC_s01 - w_k4_INTC_s012
                                 + 0.99 b_k4_INTC_s012 - 1.01 s_k4_INTC_s012
                                  = 0
 w_k4_INTC_s013_return_constraint: 1.04199 w_k4_INTC_s01 - w_k4_INTC_s013
                                 + 0.99 b_k4_INTC_s013 - 1.01 s_k4_INTC_s013
                                  = 0
 w_k4_INTC_s014_return_constraint: 0.763928 w_k4_INTC_s01 - w_k4_INTC_s014
                                 + 0.99 b_k4_INTC_s014 - 1.01 s_k4_INTC_s014
                                  = 0
 w_k4_INTC_s015_return_constraint: 0.991034 w_k4_INTC_s01 - w_k4_INTC_s015
                                 + 0.99 b_k4_INTC_s015 - 1.01 s_k4_INTC_s015
                                  = 0
 w_k4_INTC_s016_return_constraint: 1.34666 w_k4_INTC_s01 - w_k4_INTC_s016
                                 + 0.99 b_k4_INTC_s016 - 1.01 s_k4_INTC_s016
                                  = 0
 w_k4_INTC_s017_return_constraint: 0.88034 w_k4_INTC_s01 - w_k4_INTC_s017
                                 + 0.99 b_k4_INTC_s017 - 1.01 s_k4_INTC_s017
                                  = 0
 w_k4_INTC_s100_return_constraint: 1.24513 w_k4_INTC_s10 - w_k4_INTC_s100
                                 + 0.99 b_k4_INTC_s100 - 1.01 s_k4_INTC_s100
                                  = 0
 w_k4_INTC_s101_return_constraint: 0.905966 w_k4_INTC_s10 - w_k4_INTC_s101
                                 + 0.99 b_k4_INTC_s101 - 1.01 s_k4_INTC_s101
                                  = 0
 w_k4_INTC_s102_return_constraint: 0.946878 w_k4_INTC_s10 - w_k4_INTC_s102
                                 + 0.99 b_k4_INTC_s102 - 1.01 s_k4_INTC_s102
                                  = 0
 w_k4_INTC_s103_return_constraint: 0.981763 w_k4_INTC_s10 - w_k4_INTC_s103
                                 + 0.99 b_k4_INTC_s103 - 1.01 s_k4_INTC_s103
                                  = 0
 w_k4_INTC_s104_return_constraint: 1.13152 w_k4_INTC_s10 - w_k4_INTC_s104
                                 + 0.99 b_k4_INTC_s104 - 1.01 s_k4_INTC_s104
                                  = 0
 w_k4_INTC_s105_return_constraint: 1.11394 w_k4_INTC_s10 - w_k4_INTC_s105
                                 + 0.99 b_k4_INTC_s105 - 1.01 s_k4_INTC_s105
                                  = 0
 w_k4_INTC_s106_return_constraint: 1.05345 w_k4_INTC_s10 - w_k4_INTC_s106
                                 + 0.99 b_k4_INTC_s106 - 1.01 s_k4_INTC_s106
                                  = 0
 w_k4_INTC_s107_return_constraint: 1.21659 w_k4_INTC_s10 - w_k4_INTC_s107
                                 + 0.99 b_k4_INTC_s107 - 1.01 s_k4_INTC_s107
                                  = 0
 w_k4_INTC_s110_return_constraint: 1.18043 w_k4_INTC_s11 - w_k4_INTC_s110
                                 + 0.99 b_k4_INTC_s110 - 1.01 s_k4_INTC_s110
                                  = 0
 w_k4_INTC_s111_return_constraint: 1.11474 w_k4_INTC_s11 - w_k4_INTC_s111
                                 + 0.99 b_k4_INTC_s111 - 1.01 s_k4_INTC_s111
                                  = 0
 w_k4_INTC_s112_return_constraint: 0.81451 w_k4_INTC_s11 - w_k4_INTC_s112
                                 + 0.99 b_k4_INTC_s112 - 1.01 s_k4_INTC_s112
                                  = 0
 w_k4_INTC_s113_return_constraint: 1.20625 w_k4_INTC_s11 - w_k4_INTC_s113
                                 + 0.99 b_k4_INTC_s113 - 1.01 s_k4_INTC_s113
                                  = 0
 w_k4_INTC_s114_return_constraint: 0.88419 w_k4_INTC_s11 - w_k4_INTC_s114
                                 + 0.99 b_k4_INTC_s114 - 1.01 s_k4_INTC_s114
                                  = 0
 w_k4_INTC_s115_return_constraint: 1.26998 w_k4_INTC_s11 - w_k4_INTC_s115
                                 + 0.99 b_k4_INTC_s115 - 1.01 s_k4_INTC_s115
                                  = 0
 w_k4_INTC_s116_return_constraint: 1.29471 w_k4_INTC_s11 - w_k4_INTC_s116
                                 + 0.99 b_k4_INTC_s116 - 1.01 s_k4_INTC_s116
                                  = 0
 w_k4_INTC_s117_return_constraint: 0.842914 w_k4_INTC_s11 - w_k4_INTC_s117
                                 + 0.99 b_k4_INTC_s117 - 1.01 s_k4_INTC_s117
                                  = 0
 w_k4_KO_s0_return_constraint:   1.19247 w_0_KO - w_k4_KO_s0 + 0.99 b_k4_KO_s0
                                 - 1.01 s_k4_KO_s0  = 0
 w_k4_KO_s1_return_constraint:   0.879718 w_0_KO - w_k4_KO_s1 + 0.99 b_k4_KO_s1
                                 - 1.01 s_k4_KO_s1  = 0
 w_k4_KO_s00_return_constraint:  0.862875 w_k4_KO_s0 - w_k4_KO_s00
                                 + 0.99 b_k4_KO_s00 - 1.01 s_k4_KO_s00  = 0
 w_k4_KO_s01_return_constraint:  1.13244 w_k4_KO_s0 - w_k4_KO_s01
                                 + 0.99 b_k4_KO_s01 - 1.01 s_k4_KO_s01  = 0
 w_k4_KO_s10_return_constraint:  0.945624 w_k4_KO_s1 - w_k4_KO_s10
                                 + 0.99 b_k4_KO_s10 - 1.01 s_k4_KO_s10  = 0
 w_k4_KO_s11_return_constraint:  1.16013 w_k4_KO_s1 - w_k4_KO_s11
                                 + 0.99 b_k4_KO_s11 - 1.01 s_k4_KO_s11  = 0
 w_k4_KO_s000_return_constraint: 0.663977 w_k4_KO_s00 - w_k4_KO_s000
                                 + 0.99 b_k4_KO_s000 - 1.01 s_k4_KO_s000  = 0
 w_k4_KO_s001_return_constraint: 1.02969 w_k4_KO_s00 - w_k4_KO_s001
                                 + 0.99 b_k4_KO_s001 - 1.01 s_k4_KO_s001  = 0
 w_k4_KO_s002_return_constraint: 0.847311 w_k4_KO_s00 - w_k4_KO_s002
                                 + 0.99 b_k4_KO_s002 - 1.01 s_k4_KO_s002  = 0
 w_k4_KO_s003_return_constraint: 1.14161 w_k4_KO_s00 - w_k4_KO_s003
                                 + 0.99 b_k4_KO_s003 - 1.01 s_k4_KO_s003  = 0
 w_k4_KO_s004_return_constraint: 0.859085 w_k4_KO_s00 - w_k4_KO_s004
                                 + 0.99 b_k4_KO_s004 - 1.01 s_k4_KO_s004  = 0
 w_k4_KO_s005_return_constraint: 1.11501 w_k4_KO_s00 - w_k4_KO_s005
                                 + 0.99 b_k4_KO_s005 - 1.01 s_k4_KO_s005  = 0
 w_k4_KO_s006_return_constraint: 0.926823 w_k4_KO_s00 - w_k4_KO_s006
                                 + 0.99 b_k4_KO_s006 - 1.01 s_k4_KO_s006  = 0
 w_k4_KO_s007_return_constraint: 1.32452 w_k4_KO_s00 - w_k4_KO_s007
                                 + 0.99 b_k4_KO_s007 - 1.01 s_k4_KO_s007  = 0
 w_k4_KO_s010_return_constraint: 1.03949 w_k4_KO_s01 - w_k4_KO_s010
                                 + 0.99 b_k4_KO_s010 - 1.01 s_k4_KO_s010  = 0
 w_k4_KO_s011_return_constraint: 0.914319 w_k4_KO_s01 - w_k4_KO_s011
                                 + 0.99 b_k4_KO_s011 - 1.01 s_k4_KO_s011  = 0
 w_k4_KO_s012_return_constraint: 0.977516 w_k4_KO_s01 - w_k4_KO_s012
                                 + 0.99 b_k4_KO_s012 - 1.01 s_k4_KO_s012  = 0
 w_k4_KO_s013_return_constraint: 1.41876 w_k4_KO_s01 - w_k4_KO_s013
                                 + 0.99 b_k4_KO_s013 - 1.01 s_k4_KO_s013  = 0
 w_k4_KO_s014_return_constraint: 0.780156 w_k4_KO_s01 - w_k4_KO_s014
                                 + 0.99 b_k4_KO_s014 - 1.01 s_k4_KO_s014  = 0
 w_k4_KO_s015_return_constraint: 0.744842 w_k4_KO_s01 - w_k4_KO_s015
                                 + 0.99 b_k4_KO_s015 - 1.01 s_k4_KO_s015  = 0
 w_k4_KO_s016_return_constraint: 1.38588 w_k4_KO_s01 - w_k4_KO_s016
                                 + 0.99 b_k4_KO_s016 - 1.01 s_k4_KO_s016  = 0
 w_k4_KO_s017_return_constraint: 0.940748 w_k4_KO_s01 - w_k4_KO_s017
                                 + 0.99 b_k4_KO_s017 - 1.01 s_k4_KO_s017  = 0
 w_k4_KO_s100_return_constraint: 1.12036 w_k4_KO_s10 - w_k4_KO_s100
                                 + 0.99 b_k4_KO_s100 - 1.01 s_k4_KO_s100  = 0
 w_k4_KO_s101_return_constraint: 0.746858 w_k4_KO_s10 - w_k4_KO_s101
                                 + 0.99 b_k4_KO_s101 - 1.01 s_k4_KO_s101  = 0
 w_k4_KO_s102_return_constraint: 0.947018 w_k4_KO_s10 - w_k4_KO_s102
                                 + 0.99 b_k4_KO_s102 - 1.01 s_k4_KO_s102  = 0
 w_k4_KO_s103_return_constraint: 0.684527 w_k4_KO_s10 - w_k4_KO_s103
                                 + 0.99 b_k4_KO_s103 - 1.01 s_k4_KO_s103  = 0
 w_k4_KO_s104_return_constraint: 1.28325 w_k4_KO_s10 - w_k4_KO_s104
                                 + 0.99 b_k4_KO_s104 - 1.01 s_k4_KO_s104  = 0
 w_k4_KO_s105_return_constraint: 1.11128 w_k4_KO_s10 - w_k4_KO_s105
                                 + 0.99 b_k4_KO_s105 - 1.01 s_k4_KO_s105  = 0
 w_k4_KO_s106_return_constraint: 1.06169 w_k4_KO_s10 - w_k4_KO_s106
                                 + 0.99 b_k4_KO_s106 - 1.01 s_k4_KO_s106  = 0
 w_k4_KO_s107_return_constraint: 1.06752 w_k4_KO_s10 - w_k4_KO_s107
                                 + 0.99 b_k4_KO_s107 - 1.01 s_k4_KO_s107  = 0
 w_k4_KO_s110_return_constraint: 1.01943 w_k4_KO_s11 - w_k4_KO_s110
                                 + 0.99 b_k4_KO_s110 - 1.01 s_k4_KO_s110  = 0
 w_k4_KO_s111_return_constraint: 0.757399 w_k4_KO_s11 - w_k4_KO_s111
                                 + 0.99 b_k4_KO_s111 - 1.01 s_k4_KO_s111  = 0
 w_k4_KO_s112_return_constraint: 0.758197 w_k4_KO_s11 - w_k4_KO_s112
                                 + 0.99 b_k4_KO_s112 - 1.01 s_k4_KO_s112  = 0
 w_k4_KO_s113_return_constraint: 0.893546 w_k4_KO_s11 - w_k4_KO_s113
                                 + 0.99 b_k4_KO_s113 - 1.01 s_k4_KO_s113  = 0
 w_k4_KO_s114_return_constraint: 1.14145 w_k4_KO_s11 - w_k4_KO_s114
                                 + 0.99 b_k4_KO_s114 - 1.01 s_k4_KO_s114  = 0
 w_k4_KO_s115_return_constraint: 1.46338 w_k4_KO_s11 - w_k4_KO_s115
                                 + 0.99 b_k4_KO_s115 - 1.01 s_k4_KO_s115  = 0
 w_k4_KO_s116_return_constraint: 1.28485 w_k4_KO_s11 - w_k4_KO_s116
                                 + 0.99 b_k4_KO_s116 - 1.01 s_k4_KO_s116  = 0
 w_k4_KO_s117_return_constraint: 0.728279 w_k4_KO_s11 - w_k4_KO_s117
                                 + 0.99 b_k4_KO_s117 - 1.01 s_k4_KO_s117  = 0
 w_k4_MAT_s0_return_constraint:  1.26706 w_0_MAT - w_k4_MAT_s0
                                 + 0.99 b_k4_MAT_s0 - 1.01 s_k4_MAT_s0  = 0
 w_k4_MAT_s1_return_constraint:  0.867621 w_0_MAT - w_k4_MAT_s1
                                 + 0.99 b_k4_MAT_s1 - 1.01 s_k4_MAT_s1  = 0
 w_k4_MAT_s00_return_constraint: 0.745852 w_k4_MAT_s0 - w_k4_MAT_s00
                                 + 0.99 b_k4_MAT_s00 - 1.01 s_k4_MAT_s00  = 0
 w_k4_MAT_s01_return_constraint: 1.10665 w_k4_MAT_s0 - w_k4_MAT_s01
                                 + 0.99 b_k4_MAT_s01 - 1.01 s_k4_MAT_s01  = 0
 w_k4_MAT_s10_return_constraint: 0.92072 w_k4_MAT_s1 - w_k4_MAT_s10
                                 + 0.99 b_k4_MAT_s10 - 1.01 s_k4_MAT_s10  = 0
 w_k4_MAT_s11_return_constraint: 1.19438 w_k4_MAT_s1 - w_k4_MAT_s11
                                 + 0.99 b_k4_MAT_s11 - 1.01 s_k4_MAT_s11  = 0
 w_k4_MAT_s000_return_constraint: 0.638348 w_k4_MAT_s00 - w_k4_MAT_s000
                                 + 0.99 b_k4_MAT_s000 - 1.01 s_k4_MAT_s000  = 0
 w_k4_MAT_s001_return_constraint: 1.47348 w_k4_MAT_s00 - w_k4_MAT_s001
                                 + 0.99 b_k4_MAT_s001 - 1.01 s_k4_MAT_s001  = 0
 w_k4_MAT_s002_return_constraint: 1.21976 w_k4_MAT_s00 - w_k4_MAT_s002
                                 + 0.99 b_k4_MAT_s002 - 1.01 s_k4_MAT_s002  = 0
 w_k4_MAT_s003_return_constraint: 1.48317 w_k4_MAT_s00 - w_k4_MAT_s003
                                 + 0.99 b_k4_MAT_s003 - 1.01 s_k4_MAT_s003  = 0
 w_k4_MAT_s004_return_constraint: 0.727355 w_k4_MAT_s00 - w_k4_MAT_s004
                                 + 0.99 b_k4_MAT_s004 - 1.01 s_k4_MAT_s004  = 0
 w_k4_MAT_s005_return_constraint: 1.00872 w_k4_MAT_s00 - w_k4_MAT_s005
                                 + 0.99 b_k4_MAT_s005 - 1.01 s_k4_MAT_s005  = 0
 w_k4_MAT_s006_return_constraint: 0.938264 w_k4_MAT_s00 - w_k4_MAT_s006
                                 + 0.99 b_k4_MAT_s006 - 1.01 s_k4_MAT_s006  = 0
 w_k4_MAT_s007_return_constraint: 1.59622 w_k4_MAT_s00 - w_k4_MAT_s007
                                 + 0.99 b_k4_MAT_s007 - 1.01 s_k4_MAT_s007  = 0
 w_k4_MAT_s010_return_constraint: 1.38627 w_k4_MAT_s01 - w_k4_MAT_s010
                                 + 0.99 b_k4_MAT_s010 - 1.01 s_k4_MAT_s010  = 0
 w_k4_MAT_s011_return_constraint: 0.69472 w_k4_MAT_s01 - w_k4_MAT_s011
                                 + 0.99 b_k4_MAT_s011 - 1.01 s_k4_MAT_s011  = 0
 w_k4_MAT_s012_return_constraint: 1.17448 w_k4_MAT_s01 - w_k4_MAT_s012
                                 + 0.99 b_k4_MAT_s012 - 1.01 s_k4_MAT_s012  = 0
 w_k4_MAT_s013_return_constraint: 1.68895 w_k4_MAT_s01 - w_k4_MAT_s013
                                 + 0.99 b_k4_MAT_s013 - 1.01 s_k4_MAT_s013  = 0
 w_k4_MAT_s014_return_constraint: 0.833676 w_k4_MAT_s01 - w_k4_MAT_s014
                                 + 0.99 b_k4_MAT_s014 - 1.01 s_k4_MAT_s014  = 0
 w_k4_MAT_s015_return_constraint: 0.596705 w_k4_MAT_s01 - w_k4_MAT_s015
                                 + 0.99 b_k4_MAT_s015 - 1.01 s_k4_MAT_s015  = 0
 w_k4_MAT_s016_return_constraint: 1.37039 w_k4_MAT_s01 - w_k4_MAT_s016
                                 + 0.99 b_k4_MAT_s016 - 1.01 s_k4_MAT_s016  = 0
 w_k4_MAT_s017_return_constraint: 1.18145 w_k4_MAT_s01 - w_k4_MAT_s017
                                 + 0.99 b_k4_MAT_s017 - 1.01 s_k4_MAT_s017  = 0
 w_k4_MAT_s100_return_constraint: 0.961174 w_k4_MAT_s10 - w_k4_MAT_s100
                                 + 0.99 b_k4_MAT_s100 - 1.01 s_k4_MAT_s100  = 0
 w_k4_MAT_s101_return_constraint: 0.702861 w_k4_MAT_s10 - w_k4_MAT_s101
                                 + 0.99 b_k4_MAT_s101 - 1.01 s_k4_MAT_s101  = 0
 w_k4_MAT_s102_return_constraint: 1.18083 w_k4_MAT_s10 - w_k4_MAT_s102
                                 + 0.99 b_k4_MAT_s102 - 1.01 s_k4_MAT_s102  = 0
 w_k4_MAT_s103_return_constraint: 0.533948 w_k4_MAT_s10 - w_k4_MAT_s103
                                 + 0.99 b_k4_MAT_s103 - 1.01 s_k4_MAT_s103  = 0
 w_k4_MAT_s104_return_constraint: 1.67112 w_k4_MAT_s10 - w_k4_MAT_s104
                                 + 0.99 b_k4_MAT_s104 - 1.01 s_k4_MAT_s104  = 0
 w_k4_MAT_s105_return_constraint: 0.980664 w_k4_MAT_s10 - w_k4_MAT_s105
                                 + 0.99 b_k4_MAT_s105 - 1.01 s_k4_MAT_s105  = 0
 w_k4_MAT_s106_return_constraint: 1.67417 w_k4_MAT_s10 - w_k4_MAT_s106
                                 + 0.99 b_k4_MAT_s106 - 1.01 s_k4_MAT_s106  = 0
 w_k4_MAT_s107_return_constraint: 1.2731 w_k4_MAT_s10 - w_k4_MAT_s107
                                 + 0.99 b_k4_MAT_s107 - 1.01 s_k4_MAT_s107  = 0
 w_k4_MAT_s110_return_constraint: 0.820064 w_k4_MAT_s11 - w_k4_MAT_s110
                                 + 0.99 b_k4_MAT_s110 - 1.01 s_k4_MAT_s110  = 0
 w_k4_MAT_s111_return_constraint: 0.529552 w_k4_MAT_s11 - w_k4_MAT_s111
                                 + 0.99 b_k4_MAT_s111 - 1.01 s_k4_MAT_s111  = 0
 w_k4_MAT_s112_return_constraint: 0.757642 w_k4_MAT_s11 - w_k4_MAT_s112
                                 + 0.99 b_k4_MAT_s112 - 1.01 s_k4_MAT_s112  = 0
 w_k4_MAT_s113_return_constraint: 0.653527 w_k4_MAT_s11 - w_k4_MAT_s113
                                 + 0.99 b_k4_MAT_s113 - 1.01 s_k4_MAT_s113  = 0
 w_k4_MAT_s114_return_constraint: 1.11705 w_k4_MAT_s11 - w_k4_MAT_s114
                                 + 0.99 b_k4_MAT_s114 - 1.01 s_k4_MAT_s114  = 0
 w_k4_MAT_s115_return_constraint: 1.34013 w_k4_MAT_s11 - w_k4_MAT_s115
                                 + 0.99 b_k4_MAT_s115 - 1.01 s_k4_MAT_s115  = 0
 w_k4_MAT_s116_return_constraint: 1.28349 w_k4_MAT_s11 - w_k4_MAT_s116
                                 + 0.99 b_k4_MAT_s116 - 1.01 s_k4_MAT_s116  = 0
 w_k4_MAT_s117_return_constraint: 0.489305 w_k4_MAT_s11 - w_k4_MAT_s117
                                 + 0.99 b_k4_MAT_s117 - 1.01 s_k4_MAT_s117  = 0
 w_k4_MSFT_s0_return_constraint: 1.13858 w_0_MSFT - w_k4_MSFT_s0
                                 + 0.99 b_k4_MSFT_s0 - 1.01 s_k4_MSFT_s0  = 0
 w_k4_MSFT_s1_return_constraint: 0.886049 w_0_MSFT - w_k4_MSFT_s1
                                 + 0.99 b_k4_MSFT_s1 - 1.01 s_k4_MSFT_s1  = 0
 w_k4_MSFT_s00_return_constraint: 0.881993 w_k4_MSFT_s0 - w_k4_MSFT_s00
                                 + 0.99 b_k4_MSFT_s00 - 1.01 s_k4_MSFT_s00  = 0
 w_k4_MSFT_s01_return_constraint: 1.08514 w_k4_MSFT_s0 - w_k4_MSFT_s01
                                 + 0.99 b_k4_MSFT_s01 - 1.01 s_k4_MSFT_s01  = 0
 w_k4_MSFT_s10_return_constraint: 0.945795 w_k4_MSFT_s1 - w_k4_MSFT_s10
                                 + 0.99 b_k4_MSFT_s10 - 1.01 s_k4_MSFT_s10  = 0
 w_k4_MSFT_s11_return_constraint: 1.22834 w_k4_MSFT_s1 - w_k4_MSFT_s11
                                 + 0.99 b_k4_MSFT_s11 - 1.01 s_k4_MSFT_s11  = 0
 w_k4_MSFT_s000_return_constraint: 0.855423 w_k4_MSFT_s00 - w_k4_MSFT_s000
                                 + 0.99 b_k4_MSFT_s000 - 1.01 s_k4_MSFT_s000
                                  = 0
 w_k4_MSFT_s001_return_constraint: 0.966938 w_k4_MSFT_s00 - w_k4_MSFT_s001
                                 + 0.99 b_k4_MSFT_s001 - 1.01 s_k4_MSFT_s001
                                  = 0
 w_k4_MSFT_s002_return_constraint: 0.779863 w_k4_MSFT_s00 - w_k4_MSFT_s002
                                 + 0.99 b_k4_MSFT_s002 - 1.01 s_k4_MSFT_s002
                                  = 0
 w_k4_MSFT_s003_return_constraint: 0.971588 w_k4_MSFT_s00 - w_k4_MSFT_s003
                                 + 0.99 b_k4_MSFT_s003 - 1.01 s_k4_MSFT_s003
                                  = 0
 w_k4_MSFT_s004_return_constraint: 1.07408 w_k4_MSFT_s00 - w_k4_MSFT_s004
                                 + 0.99 b_k4_MSFT_s004 - 1.01 s_k4_MSFT_s004
                                  = 0
 w_k4_MSFT_s005_return_constraint: 1.1373 w_k4_MSFT_s00 - w_k4_MSFT_s005
                                 + 0.99 b_k4_MSFT_s005 - 1.01 s_k4_MSFT_s005
                                  = 0
 w_k4_MSFT_s006_return_constraint: 0.939166 w_k4_MSFT_s00 - w_k4_MSFT_s006
                                 + 0.99 b_k4_MSFT_s006 - 1.01 s_k4_MSFT_s006
                                  = 0
 w_k4_MSFT_s007_return_constraint: 1.46907 w_k4_MSFT_s00 - w_k4_MSFT_s007
                                 + 0.99 b_k4_MSFT_s007 - 1.01 s_k4_MSFT_s007
                                  = 0
 w_k4_MSFT_s010_return_constraint: 0.990182 w_k4_MSFT_s01 - w_k4_MSFT_s010
                                 + 0.99 b_k4_MSFT_s010 - 1.01 s_k4_MSFT_s010
                                  = 0
 w_k4_MSFT_s011_return_constraint: 1.02574 w_k4_MSFT_s01 - w_k4_MSFT_s011
                                 + 0.99 b_k4_MSFT_s011 - 1.01 s_k4_MSFT_s011
                                  = 0
 w_k4_MSFT_s012_return_constraint: 1.00896 w_k4_MSFT_s01 - w_k4_MSFT_s012
                                 + 0.99 b_k4_MSFT_s012 - 1.01 s_k4_MSFT_s012
                                  = 0
 w_k4_MSFT_s013_return_constraint: 1.05371 w_k4_MSFT_s01 - w_k4_MSFT_s013
                                 + 0.99 b_k4_MSFT_s013 - 1.01 s_k4_MSFT_s013
                                  = 0
 w_k4_MSFT_s014_return_constraint: 0.782355 w_k4_MSFT_s01 - w_k4_MSFT_s014
                                 + 0.99 b_k4_MSFT_s014 - 1.01 s_k4_MSFT_s014
                                  = 0
 w_k4_MSFT_s015_return_constraint: 0.970144 w_k4_MSFT_s01 - w_k4_MSFT_s015
                                 + 0.99 b_k4_MSFT_s015 - 1.01 s_k4_MSFT_s015
                                  = 0
 w_k4_MSFT_s016_return_constraint: 1.3881 w_k4_MSFT_s01 - w_k4_MSFT_s016
                                 + 0.99 b_k4_MSFT_s016 - 1.01 s_k4_MSFT_s016
                                  = 0
 w_k4_MSFT_s017_return_constraint: 0.825813 w_k4_MSFT_s01 - w_k4_MSFT_s017
                                 + 0.99 b_k4_MSFT_s017 - 1.01 s_k4_MSFT_s017
                                  = 0
 w_k4_MSFT_s100_return_constraint: 1.09479 w_k4_MSFT_s10 - w_k4_MSFT_s100
                                 + 0.99 b_k4_MSFT_s100 - 1.01 s_k4_MSFT_s100
                                  = 0
 w_k4_MSFT_s101_return_constraint: 0.794902 w_k4_MSFT_s10 - w_k4_MSFT_s101
                                 + 0.99 b_k4_MSFT_s101 - 1.01 s_k4_MSFT_s101
                                  = 0
 w_k4_MSFT_s102_return_constraint: 0.961785 w_k4_MSFT_s10 - w_k4_MSFT_s102
                                 + 0.99 b_k4_MSFT_s102 - 1.01 s_k4_MSFT_s102
                                  = 0
 w_k4_MSFT_s103_return_constraint: 0.929609 w_k4_MSFT_s10 - w_k4_MSFT_s103
                                 + 0.99 b_k4_MSFT_s103 - 1.01 s_k4_MSFT_s103
                                  = 0
 w_k4_MSFT_s104_return_constraint: 1.03396 w_k4_MSFT_s10 - w_k4_MSFT_s104
                                 + 0.99 b_k4_MSFT_s104 - 1.01 s_k4_MSFT_s104
                                  = 0
 w_k4_MSFT_s105_return_constraint: 1.08357 w_k4_MSFT_s10 - w_k4_MSFT_s105
                                 + 0.99 b_k4_MSFT_s105 - 1.01 s_k4_MSFT_s105
                                  = 0
 w_k4_MSFT_s106_return_constraint: 0.996731 w_k4_MSFT_s10 - w_k4_MSFT_s106
                                 + 0.99 b_k4_MSFT_s106 - 1.01 s_k4_MSFT_s106
                                  = 0
 w_k4_MSFT_s107_return_constraint: 1.24218 w_k4_MSFT_s10 - w_k4_MSFT_s107
                                 + 0.99 b_k4_MSFT_s107 - 1.01 s_k4_MSFT_s107
                                  = 0
 w_k4_MSFT_s110_return_constraint: 1.03468 w_k4_MSFT_s11 - w_k4_MSFT_s110
                                 + 0.99 b_k4_MSFT_s110 - 1.01 s_k4_MSFT_s110
                                  = 0
 w_k4_MSFT_s111_return_constraint: 0.875702 w_k4_MSFT_s11 - w_k4_MSFT_s111
                                 + 0.99 b_k4_MSFT_s111 - 1.01 s_k4_MSFT_s111
                                  = 0
 w_k4_MSFT_s112_return_constraint: 0.789508 w_k4_MSFT_s11 - w_k4_MSFT_s112
                                 + 0.99 b_k4_MSFT_s112 - 1.01 s_k4_MSFT_s112
                                  = 0
 w_k4_MSFT_s113_return_constraint: 1.08102 w_k4_MSFT_s11 - w_k4_MSFT_s113
                                 + 0.99 b_k4_MSFT_s113 - 1.01 s_k4_MSFT_s113
                                  = 0
 w_k4_MSFT_s114_return_constraint: 0.849298 w_k4_MSFT_s11 - w_k4_MSFT_s114
                                 + 0.99 b_k4_MSFT_s114 - 1.01 s_k4_MSFT_s114
                                  = 0
 w_k4_MSFT_s115_return_constraint: 1.0566 w_k4_MSFT_s11 - w_k4_MSFT_s115
                                 + 0.99 b_k4_MSFT_s115 - 1.01 s_k4_MSFT_s115
                                  = 0
 w_k4_MSFT_s116_return_constraint: 1.29806 w_k4_MSFT_s11 - w_k4_MSFT_s116
                                 + 0.99 b_k4_MSFT_s116 - 1.01 s_k4_MSFT_s116
                                  = 0
 w_k4_MSFT_s117_return_constraint: 0.935855 w_k4_MSFT_s11 - w_k4_MSFT_s117
                                 + 0.99 b_k4_MSFT_s117 - 1.01 s_k4_MSFT_s117
                                  = 0
 w_k4_PG_s0_return_constraint:   1.0548 w_0_PG - w_k4_PG_s0 + 0.99 b_k4_PG_s0
                                 - 1.01 s_k4_PG_s0  = 0
 w_k4_PG_s1_return_constraint:   0.919751 w_0_PG - w_k4_PG_s1 + 0.99 b_k4_PG_s1
                                 - 1.01 s_k4_PG_s1  = 0
 w_k4_PG_s00_return_constraint:  0.912104 w_k4_PG_s0 - w_k4_PG_s00
                                 + 0.99 b_k4_PG_s00 - 1.01 s_k4_PG_s00  = 0
 w_k4_PG_s01_return_constraint:  1.22777 w_k4_PG_s0 - w_k4_PG_s01
                                 + 0.99 b_k4_PG_s01 - 1.01 s_k4_PG_s01  = 0
 w_k4_PG_s10_return_constraint:  0.923005 w_k4_PG_s1 - w_k4_PG_s10
                                 + 0.99 b_k4_PG_s10 - 1.01 s_k4_PG_s10  = 0
 w_k4_PG_s11_return_constraint:  1.1451 w_k4_PG_s1 - w_k4_PG_s11
                                 + 0.99 b_k4_PG_s11 - 1.01 s_k4_PG_s11  = 0
 w_k4_PG_s000_return_constraint: 0.840253 w_k4_PG_s00 - w_k4_PG_s000
                                 + 0.99 b_k4_PG_s000 - 1.01 s_k4_PG_s000  = 0
 w_k4_PG_s001_return_constraint: 1.02983 w_k4_PG_s00 - w_k4_PG_s001
                                 + 0.99 b_k4_PG_s001 - 1.01 s_k4_PG_s001  = 0
 w_k4_PG_s002_return_constraint: 0.9435 w_k4_PG_s00 - w_k4_PG_s002
                                 + 0.99 b_k4_PG_s002 - 1.01 s_k4_PG_s002  = 0
 w_k4_PG_s003_return_constraint: 1.13682 w_k4_PG_s00 - w_k4_PG_s003
                                 + 0.99 b_k4_PG_s003 - 1.01 s_k4_PG_s003  = 0
 w_k4_PG_s004_return_constraint: 1.04623 w_k4_PG_s00 - w_k4_PG_s004
                                 + 0.99 b_k4_PG_s004 - 1.01 s_k4_PG_s004  = 0
 w_k4_PG_s005_return_constraint: 1.23541 w_k4_PG_s00 - w_k4_PG_s005
                                 + 0.99 b_k4_PG_s005 - 1.01 s_k4_PG_s005  = 0
 w_k4_PG_s006_return_constraint: 0.967436 w_k4_PG_s00 - w_k4_PG_s006
                                 + 0.99 b_k4_PG_s006 - 1.01 s_k4_PG_s006  = 0
 w_k4_PG_s007_return_constraint: 1.30835 w_k4_PG_s00 - w_k4_PG_s007
                                 + 0.99 b_k4_PG_s007 - 1.01 s_k4_PG_s007  = 0
 w_k4_PG_s010_return_constraint: 0.937364 w_k4_PG_s01 - w_k4_PG_s010
                                 + 0.99 b_k4_PG_s010 - 1.01 s_k4_PG_s010  = 0
 w_k4_PG_s011_return_constraint: 0.876817 w_k4_PG_s01 - w_k4_PG_s011
                                 + 0.99 b_k4_PG_s011 - 1.01 s_k4_PG_s011  = 0
 w_k4_PG_s012_return_constraint: 1.00704 w_k4_PG_s01 - w_k4_PG_s012
                                 + 0.99 b_k4_PG_s012 - 1.01 s_k4_PG_s012  = 0
 w_k4_PG_s013_return_constraint: 1.10213 w_k4_PG_s01 - w_k4_PG_s013
                                 + 0.99 b_k4_PG_s013 - 1.01 s_k4_PG_s013  = 0
 w_k4_PG_s014_return_constraint: 0.777474 w_k4_PG_s01 - w_k4_PG_s014
                                 + 0.99 b_k4_PG_s014 - 1.01 s_k4_PG_s014  = 0
 w_k4_PG_s015_return_constraint: 0.811967 w_k4_PG_s01 - w_k4_PG_s015
                                 + 0.99 b_k4_PG_s015 - 1.01 s_k4_PG_s015  = 0
 w_k4_PG_s016_return_constraint: 1.30593 w_k4_PG_s01 - w_k4_PG_s016
                                 + 0.99 b_k4_PG_s016 - 1.01 s_k4_PG_s016  = 0
 w_k4_PG_s017_return_constraint: 0.913378 w_k4_PG_s01 - w_k4_PG_s017
                                 + 0.99 b_k4_PG_s017 - 1.01 s_k4_PG_s017  = 0
 w_k4_PG_s100_return_constraint: 1.0674 w_k4_PG_s10 - w_k4_PG_s100
                                 + 0.99 b_k4_PG_s100 - 1.01 s_k4_PG_s100  = 0
 w_k4_PG_s101_return_constraint: 0.832276 w_k4_PG_s10 - w_k4_PG_s101
                                 + 0.99 b_k4_PG_s101 - 1.01 s_k4_PG_s101  = 0
 w_k4_PG_s102_return_constraint: 0.985766 w_k4_PG_s10 - w_k4_PG_s102
                                 + 0.99 b_k4_PG_s102 - 1.01 s_k4_PG_s102  = 0
 w_k4_PG_s103_return_constraint: 0.859686 w_k4_PG_s10 - w_k4_PG_s103
                                 + 0.99 b_k4_PG_s103 - 1.01 s_k4_PG_s103  = 0
 w_k4_PG_s104_return_constraint: 1.19201 w_k4_PG_s10 - w_k4_PG_s104
                                 + 0.99 b_k4_PG_s104 - 1.01 s_k4_PG_s104  = 0
 w_k4_PG_s105_return_constraint: 1.15867 w_k4_PG_s10 - w_k4_PG_s105
                                 + 0.99 b_k4_PG_s105 - 1.01 s_k4_PG_s105  = 0
 w_k4_PG_s106_return_constraint: 1.06128 w_k4_PG_s10 - w_k4_PG_s106
                                 + 0.99 b_k4_PG_s106 - 1.01 s_k4_PG_s106  = 0
 w_k4_PG_s107_return_constraint: 1.17031 w_k4_PG_s10 - w_k4_PG_s107
                                 + 0.99 b_k4_PG_s107 - 1.01 s_k4_PG_s107  = 0
 w_k4_PG_s110_return_constraint: 0.999384 w_k4_PG_s11 - w_k4_PG_s110
                                 + 0.99 b_k4_PG_s110 - 1.01 s_k4_PG_s110  = 0
 w_k4_PG_s111_return_constraint: 0.870563 w_k4_PG_s11 - w_k4_PG_s111
                                 + 0.99 b_k4_PG_s111 - 1.01 s_k4_PG_s111  = 0
 w_k4_PG_s112_return_constraint: 0.785311 w_k4_PG_s11 - w_k4_PG_s112
                                 + 0.99 b_k4_PG_s112 - 1.01 s_k4_PG_s112  = 0
 w_k4_PG_s113_return_constraint: 1.01466 w_k4_PG_s11 - w_k4_PG_s113
                                 + 0.99 b_k4_PG_s113 - 1.01 s_k4_PG_s113  = 0
 w_k4_PG_s114_return_constraint: 0.962744 w_k4_PG_s11 - w_k4_PG_s114
                                 + 0.99 b_k4_PG_s114 - 1.01 s_k4_PG_s114  = 0
 w_k4_PG_s115_return_constraint: 1.05788 w_k4_PG_s11 - w_k4_PG_s115
                                 + 0.99 b_k4_PG_s115 - 1.01 s_k4_PG_s115  = 0
 w_k4_PG_s116_return_constraint: 1.11795 w_k4_PG_s11 - w_k4_PG_s116
                                 + 0.99 b_k4_PG_s116 - 1.01 s_k4_PG_s116  = 0
 w_k4_PG_s117_return_constraint: 0.832467 w_k4_PG_s11 - w_k4_PG_s117
                                 + 0.99 b_k4_PG_s117 - 1.01 s_k4_PG_s117  = 0
 k4_s0_balance_constraint:       b_k4_ADM_s0 + b_k4_AXP_s0 + b_k4_DIS_s0
                                 + b_k4_FDX_s0 + b_k4_IBM_s0 + b_k4_INTC_s0
                                 + b_k4_KO_s0 + b_k4_MAT_s0 + b_k4_MSFT_s0
                                 + b_k4_PG_s0 - s_k4_ADM_s0 - s_k4_AXP_s0
                                 - s_k4_DIS_s0 - s_k4_FDX_s0 - s_k4_IBM_s0
                                 - s_k4_INTC_s0 - s_k4_KO_s0 - s_k4_MAT_s0
                                 - s_k4_MSFT_s0 - s_k4_PG_s0  = 0
 k4_s1_balance_constraint:       b_k4_ADM_s1 + b_k4_AXP_s1 + b_k4_DIS_s1
                                 + b_k4_FDX_s1 + b_k4_IBM_s1 + b_k4_INTC_s1
                                 + b_k4_KO_s1 + b_k4_MAT_s1 + b_k4_MSFT_s1
                                 + b_k4_PG_s1 - s_k4_ADM_s1 - s_k4_AXP_s1
                                 - s_k4_DIS_s1 - s_k4_FDX_s1 - s_k4_IBM_s1
                                 - s_k4_INTC_s1 - s_k4_KO_s1 - s_k4_MAT_s1
                                 - s_k4_MSFT_s1 - s_k4_PG_s1  = 0
 k4_s00_balance_constraint:      b_k4_ADM_s00 + b_k4_AXP_s00 + b_k4_DIS_s00
                                 + b_k4_FDX_s00 + b_k4_IBM_s00 + b_k4_INTC_s00
                                 + b_k4_KO_s00 + b_k4_MAT_s00 + b_k4_MSFT_s00
                                 + b_k4_PG_s00 - s_k4_ADM_s00 - s_k4_AXP_s00
                                 - s_k4_DIS_s00 - s_k4_FDX_s00 - s_k4_IBM_s00
                                 - s_k4_INTC_s00 - s_k4_KO_s00 - s_k4_MAT_s00
                                 - s_k4_MSFT_s00 - s_k4_PG_s00  = 0
 k4_s01_balance_constraint:      b_k4_ADM_s01 + b_k4_AXP_s01 + b_k4_DIS_s01
                                 + b_k4_FDX_s01 + b_k4_IBM_s01 + b_k4_INTC_s01
                                 + b_k4_KO_s01 + b_k4_MAT_s01 + b_k4_MSFT_s01
                                 + b_k4_PG_s01 - s_k4_ADM_s01 - s_k4_AXP_s01
                                 - s_k4_DIS_s01 - s_k4_FDX_s01 - s_k4_IBM_s01
                                 - s_k4_INTC_s01 - s_k4_KO_s01 - s_k4_MAT_s01
                                 - s_k4_MSFT_s01 - s_k4_PG_s01  = 0
 k4_s10_balance_constraint:      b_k4_ADM_s10 + b_k4_AXP_s10 + b_k4_DIS_s10
                                 + b_k4_FDX_s10 + b_k4_IBM_s10 + b_k4_INTC_s10
                                 + b_k4_KO_s10 + b_k4_MAT_s10 + b_k4_MSFT_s10
                                 + b_k4_PG_s10 - s_k4_ADM_s10 - s_k4_AXP_s10
                                 - s_k4_DIS_s10 - s_k4_FDX_s10 - s_k4_IBM_s10
                                 - s_k4_INTC_s10 - s_k4_KO_s10 - s_k4_MAT_s10
                                 - s_k4_MSFT_s10 - s_k4_PG_s10  = 0
 k4_s11_balance_constraint:      b_k4_ADM_s11 + b_k4_AXP_s11 + b_k4_DIS_s11
                                 + b_k4_FDX_s11 + b_k4_IBM_s11 + b_k4_INTC_s11
                                 + b_k4_KO_s11 + b_k4_MAT_s11 + b_k4_MSFT_s11
                                 + b_k4_PG_s11 - s_k4_ADM_s11 - s_k4_AXP_s11
                                 - s_k4_DIS_s11 - s_k4_FDX_s11 - s_k4_IBM_s11
                                 - s_k4_INTC_s11 - s_k4_KO_s11 - s_k4_MAT_s11
                                 - s_k4_MSFT_s11 - s_k4_PG_s11  = 0
 k4_s000_balance_constraint:     b_k4_ADM_s000 + b_k4_AXP_s000 + b_k4_DIS_s000
                                 + b_k4_FDX_s000 + b_k4_IBM_s000
                                 + b_k4_INTC_s000 + b_k4_KO_s000
                                 + b_k4_MAT_s000 + b_k4_MSFT_s000
                                 + b_k4_PG_s000 - s_k4_ADM_s000 - s_k4_AXP_s000
                                 - s_k4_DIS_s000 - s_k4_FDX_s000
                                 - s_k4_IBM_s000 - s_k4_INTC_s000
                                 - s_k4_KO_s000 - s_k4_MAT_s000
                                 - s_k4_MSFT_s000 - s_k4_PG_s000  = 0
 k4_s001_balance_constraint:     b_k4_ADM_s001 + b_k4_AXP_s001 + b_k4_DIS_s001
                                 + b_k4_FDX_s001 + b_k4_IBM_s001
                                 + b_k4_INTC_s001 + b_k4_KO_s001
                                 + b_k4_MAT_s001 + b_k4_MSFT_s001
                                 + b_k4_PG_s001 - s_k4_ADM_s001 - s_k4_AXP_s001
                                 - s_k4_DIS_s001 - s_k4_FDX_s001
                                 - s_k4_IBM_s001 - s_k4_INTC_s001
                                 - s_k4_KO_s001 - s_k4_MAT_s001
                                 - s_k4_MSFT_s001 - s_k4_PG_s001  = 0
 k4_s002_balance_constraint:     b_k4_ADM_s002 + b_k4_AXP_s002 + b_k4_DIS_s002
                                 + b_k4_FDX_s002 + b_k4_IBM_s002
                                 + b_k4_INTC_s002 + b_k4_KO_s002
                                 + b_k4_MAT_s002 + b_k4_MSFT_s002
                                 + b_k4_PG_s002 - s_k4_ADM_s002 - s_k4_AXP_s002
                                 - s_k4_DIS_s002 - s_k4_FDX_s002
                                 - s_k4_IBM_s002 - s_k4_INTC_s002
                                 - s_k4_KO_s002 - s_k4_MAT_s002
                                 - s_k4_MSFT_s002 - s_k4_PG_s002  = 0
 k4_s003_balance_constraint:     b_k4_ADM_s003 + b_k4_AXP_s003 + b_k4_DIS_s003
                                 + b_k4_FDX_s003 + b_k4_IBM_s003
                                 + b_k4_INTC_s003 + b_k4_KO_s003
                                 + b_k4_MAT_s003 + b_k4_MSFT_s003
                                 + b_k4_PG_s003 - s_k4_ADM_s003 - s_k4_AXP_s003
                                 - s_k4_DIS_s003 - s_k4_FDX_s003
                                 - s_k4_IBM_s003 - s_k4_INTC_s003
                                 - s_k4_KO_s003 - s_k4_MAT_s003
                                 - s_k4_MSFT_s003 - s_k4_PG_s003  = 0
 k4_s004_balance_constraint:     b_k4_ADM_s004 + b_k4_AXP_s004 + b_k4_DIS_s004
                                 + b_k4_FDX_s004 + b_k4_IBM_s004
                                 + b_k4_INTC_s004 + b_k4_KO_s004
                                 + b_k4_MAT_s004 + b_k4_MSFT_s004
                                 + b_k4_PG_s004 - s_k4_ADM_s004 - s_k4_AXP_s004
                                 - s_k4_DIS_s004 - s_k4_FDX_s004
                                 - s_k4_IBM_s004 - s_k4_INTC_s004
                                 - s_k4_KO_s004 - s_k4_MAT_s004
                                 - s_k4_MSFT_s004 - s_k4_PG_s004  = 0
 k4_s005_balance_constraint:     b_k4_ADM_s005 + b_k4_AXP_s005 + b_k4_DIS_s005
                                 + b_k4_FDX_s005 + b_k4_IBM_s005
                                 + b_k4_INTC_s005 + b_k4_KO_s005
                                 + b_k4_MAT_s005 + b_k4_MSFT_s005
                                 + b_k4_PG_s005 - s_k4_ADM_s005 - s_k4_AXP_s005
                                 - s_k4_DIS_s005 - s_k4_FDX_s005
                                 - s_k4_IBM_s005 - s_k4_INTC_s005
                                 - s_k4_KO_s005 - s_k4_MAT_s005
                                 - s_k4_MSFT_s005 - s_k4_PG_s005  = 0
 k4_s006_balance_constraint:     b_k4_ADM_s006 + b_k4_AXP_s006 + b_k4_DIS_s006
                                 + b_k4_FDX_s006 + b_k4_IBM_s006
                                 + b_k4_INTC_s006 + b_k4_KO_s006
                                 + b_k4_MAT_s006 + b_k4_MSFT_s006
                                 + b_k4_PG_s006 - s_k4_ADM_s006 - s_k4_AXP_s006
                                 - s_k4_DIS_s006 - s_k4_FDX_s006
                                 - s_k4_IBM_s006 - s_k4_INTC_s006
                                 - s_k4_KO_s006 - s_k4_MAT_s006
                                 - s_k4_MSFT_s006 - s_k4_PG_s006  = 0
 k4_s007_balance_constraint:     b_k4_ADM_s007 + b_k4_AXP_s007 + b_k4_DIS_s007
                                 + b_k4_FDX_s007 + b_k4_IBM_s007
                                 + b_k4_INTC_s007 + b_k4_KO_s007
                                 + b_k4_MAT_s007 + b_k4_MSFT_s007
                                 + b_k4_PG_s007 - s_k4_ADM_s007 - s_k4_AXP_s007
                                 - s_k4_DIS_s007 - s_k4_FDX_s007
                                 - s_k4_IBM_s007 - s_k4_INTC_s007
                                 - s_k4_KO_s007 - s_k4_MAT_s007
                                 - s_k4_MSFT_s007 - s_k4_PG_s007  = 0
 k4_s010_balance_constraint:     b_k4_ADM_s010 + b_k4_AXP_s010 + b_k4_DIS_s010
                                 + b_k4_FDX_s010 + b_k4_IBM_s010
                                 + b_k4_INTC_s010 + b_k4_KO_s010
                                 + b_k4_MAT_s010 + b_k4_MSFT_s010
                                 + b_k4_PG_s010 - s_k4_ADM_s010 - s_k4_AXP_s010
                                 - s_k4_DIS_s010 - s_k4_FDX_s010
                                 - s_k4_IBM_s010 - s_k4_INTC_s010
                                 - s_k4_KO_s010 - s_k4_MAT_s010
                                 - s_k4_MSFT_s010 - s_k4_PG_s010  = 0
 k4_s011_balance_constraint:     b_k4_ADM_s011 + b_k4_AXP_s011 + b_k4_DIS_s011
                                 + b_k4_FDX_s011 + b_k4_IBM_s011
                                 + b_k4_INTC_s011 + b_k4_KO_s011
                                 + b_k4_MAT_s011 + b_k4_MSFT_s011
                                 + b_k4_PG_s011 - s_k4_ADM_s011 - s_k4_AXP_s011
                                 - s_k4_DIS_s011 - s_k4_FDX_s011
                                 - s_k4_IBM_s011 - s_k4_INTC_s011
                                 - s_k4_KO_s011 - s_k4_MAT_s011
                                 - s_k4_MSFT_s011 - s_k4_PG_s011  = 0
 k4_s012_balance_constraint:     b_k4_ADM_s012 + b_k4_AXP_s012 + b_k4_DIS_s012
                                 + b_k4_FDX_s012 + b_k4_IBM_s012
                                 + b_k4_INTC_s012 + b_k4_KO_s012
                                 + b_k4_MAT_s012 + b_k4_MSFT_s012
                                 + b_k4_PG_s012 - s_k4_ADM_s012 - s_k4_AXP_s012
                                 - s_k4_DIS_s012 - s_k4_FDX_s012
                                 - s_k4_IBM_s012 - s_k4_INTC_s012
                                 - s_k4_KO_s012 - s_k4_MAT_s012
                                 - s_k4_MSFT_s012 - s_k4_PG_s012  = 0
 k4_s013_balance_constraint:     b_k4_ADM_s013 + b_k4_AXP_s013 + b_k4_DIS_s013
                                 + b_k4_FDX_s013 + b_k4_IBM_s013
                                 + b_k4_INTC_s013 + b_k4_KO_s013
                                 + b_k4_MAT_s013 + b_k4_MSFT_s013
                                 + b_k4_PG_s013 - s_k4_ADM_s013 - s_k4_AXP_s013
                                 - s_k4_DIS_s013 - s_k4_FDX_s013
                                 - s_k4_IBM_s013 - s_k4_INTC_s013
                                 - s_k4_KO_s013 - s_k4_MAT_s013
                                 - s_k4_MSFT_s013 - s_k4_PG_s013  = 0
 k4_s014_balance_constraint:     b_k4_ADM_s014 + b_k4_AXP_s014 + b_k4_DIS_s014
                                 + b_k4_FDX_s014 + b_k4_IBM_s014
                                 + b_k4_INTC_s014 + b_k4_KO_s014
                                 + b_k4_MAT_s014 + b_k4_MSFT_s014
                                 + b_k4_PG_s014 - s_k4_ADM_s014 - s_k4_AXP_s014
                                 - s_k4_DIS_s014 - s_k4_FDX_s014
                                 - s_k4_IBM_s014 - s_k4_INTC_s014
                                 - s_k4_KO_s014 - s_k4_MAT_s014
                                 - s_k4_MSFT_s014 - s_k4_PG_s014  = 0
 k4_s015_balance_constraint:     b_k4_ADM_s015 + b_k4_AXP_s015 + b_k4_DIS_s015
                                 + b_k4_FDX_s015 + b_k4_IBM_s015
                                 + b_k4_INTC_s015 + b_k4_KO_s015
                                 + b_k4_MAT_s015 + b_k4_MSFT_s015
                                 + b_k4_PG_s015 - s_k4_ADM_s015 - s_k4_AXP_s015
                                 - s_k4_DIS_s015 - s_k4_FDX_s015
                                 - s_k4_IBM_s015 - s_k4_INTC_s015
                                 - s_k4_KO_s015 - s_k4_MAT_s015
                                 - s_k4_MSFT_s015 - s_k4_PG_s015  = 0
 k4_s016_balance_constraint:     b_k4_ADM_s016 + b_k4_AXP_s016 + b_k4_DIS_s016
                                 + b_k4_FDX_s016 + b_k4_IBM_s016
                                 + b_k4_INTC_s016 + b_k4_KO_s016
                                 + b_k4_MAT_s016 + b_k4_MSFT_s016
                                 + b_k4_PG_s016 - s_k4_ADM_s016 - s_k4_AXP_s016
                                 - s_k4_DIS_s016 - s_k4_FDX_s016
                                 - s_k4_IBM_s016 - s_k4_INTC_s016
                                 - s_k4_KO_s016 - s_k4_MAT_s016
                                 - s_k4_MSFT_s016 - s_k4_PG_s016  = 0
 k4_s017_balance_constraint:     b_k4_ADM_s017 + b_k4_AXP_s017 + b_k4_DIS_s017
                                 + b_k4_FDX_s017 + b_k4_IBM_s017
                                 + b_k4_INTC_s017 + b_k4_KO_s017
                                 + b_k4_MAT_s017 + b_k4_MSFT_s017
                                 + b_k4_PG_s017 - s_k4_ADM_s017 - s_k4_AXP_s017
                                 - s_k4_DIS_s017 - s_k4_FDX_s017
                                 - s_k4_IBM_s017 - s_k4_INTC_s017
                                 - s_k4_KO_s017 - s_k4_MAT_s017
                                 - s_k4_MSFT_s017 - s_k4_PG_s017  = 0
 k4_s100_balance_constraint:     b_k4_ADM_s100 + b_k4_AXP_s100 + b_k4_DIS_s100
                                 + b_k4_FDX_s100 + b_k4_IBM_s100
                                 + b_k4_INTC_s100 + b_k4_KO_s100
                                 + b_k4_MAT_s100 + b_k4_MSFT_s100
                                 + b_k4_PG_s100 - s_k4_ADM_s100 - s_k4_AXP_s100
                                 - s_k4_DIS_s100 - s_k4_FDX_s100
                                 - s_k4_IBM_s100 - s_k4_INTC_s100
                                 - s_k4_KO_s100 - s_k4_MAT_s100
                                 - s_k4_MSFT_s100 - s_k4_PG_s100  = 0
 k4_s101_balance_constraint:     b_k4_ADM_s101 + b_k4_AXP_s101 + b_k4_DIS_s101
                                 + b_k4_FDX_s101 + b_k4_IBM_s101
                                 + b_k4_INTC_s101 + b_k4_KO_s101
                                 + b_k4_MAT_s101 + b_k4_MSFT_s101
                                 + b_k4_PG_s101 - s_k4_ADM_s101 - s_k4_AXP_s101
                                 - s_k4_DIS_s101 - s_k4_FDX_s101
                                 - s_k4_IBM_s101 - s_k4_INTC_s101
                                 - s_k4_KO_s101 - s_k4_MAT_s101
                                 - s_k4_MSFT_s101 - s_k4_PG_s101  = 0
 k4_s102_balance_constraint:     b_k4_ADM_s102 + b_k4_AXP_s102 + b_k4_DIS_s102
                                 + b_k4_FDX_s102 + b_k4_IBM_s102
                                 + b_k4_INTC_s102 + b_k4_KO_s102
                                 + b_k4_MAT_s102 + b_k4_MSFT_s102
                                 + b_k4_PG_s102 - s_k4_ADM_s102 - s_k4_AXP_s102
                                 - s_k4_DIS_s102 - s_k4_FDX_s102
                                 - s_k4_IBM_s102 - s_k4_INTC_s102
                                 - s_k4_KO_s102 - s_k4_MAT_s102
                                 - s_k4_MSFT_s102 - s_k4_PG_s102  = 0
 k4_s103_balance_constraint:     b_k4_ADM_s103 + b_k4_AXP_s103 + b_k4_DIS_s103
                                 + b_k4_FDX_s103 + b_k4_IBM_s103
                                 + b_k4_INTC_s103 + b_k4_KO_s103
                                 + b_k4_MAT_s103 + b_k4_MSFT_s103
                                 + b_k4_PG_s103 - s_k4_ADM_s103 - s_k4_AXP_s103
                                 - s_k4_DIS_s103 - s_k4_FDX_s103
                                 - s_k4_IBM_s103 - s_k4_INTC_s103
                                 - s_k4_KO_s103 - s_k4_MAT_s103
                                 - s_k4_MSFT_s103 - s_k4_PG_s103  = 0
 k4_s104_balance_constraint:     b_k4_ADM_s104 + b_k4_AXP_s104 + b_k4_DIS_s104
                                 + b_k4_FDX_s104 + b_k4_IBM_s104
                                 + b_k4_INTC_s104 + b_k4_KO_s104
                                 + b_k4_MAT_s104 + b_k4_MSFT_s104
                                 + b_k4_PG_s104 - s_k4_ADM_s104 - s_k4_AXP_s104
                                 - s_k4_DIS_s104 - s_k4_FDX_s104
                                 - s_k4_IBM_s104 - s_k4_INTC_s104
                                 - s_k4_KO_s104 - s_k4_MAT_s104
                                 - s_k4_MSFT_s104 - s_k4_PG_s104  = 0
 k4_s105_balance_constraint:     b_k4_ADM_s105 + b_k4_AXP_s105 + b_k4_DIS_s105
                                 + b_k4_FDX_s105 + b_k4_IBM_s105
                                 + b_k4_INTC_s105 + b_k4_KO_s105
                                 + b_k4_MAT_s105 + b_k4_MSFT_s105
                                 + b_k4_PG_s105 - s_k4_ADM_s105 - s_k4_AXP_s105
                                 - s_k4_DIS_s105 - s_k4_FDX_s105
                                 - s_k4_IBM_s105 - s_k4_INTC_s105
                                 - s_k4_KO_s105 - s_k4_MAT_s105
                                 - s_k4_MSFT_s105 - s_k4_PG_s105  = 0
 k4_s106_balance_constraint:     b_k4_ADM_s106 + b_k4_AXP_s106 + b_k4_DIS_s106
                                 + b_k4_FDX_s106 + b_k4_IBM_s106
                                 + b_k4_INTC_s106 + b_k4_KO_s106
                                 + b_k4_MAT_s106 + b_k4_MSFT_s106
                                 + b_k4_PG_s106 - s_k4_ADM_s106 - s_k4_AXP_s106
                                 - s_k4_DIS_s106 - s_k4_FDX_s106
                                 - s_k4_IBM_s106 - s_k4_INTC_s106
                                 - s_k4_KO_s106 - s_k4_MAT_s106
                                 - s_k4_MSFT_s106 - s_k4_PG_s106  = 0
 k4_s107_balance_constraint:     b_k4_ADM_s107 + b_k4_AXP_s107 + b_k4_DIS_s107
                                 + b_k4_FDX_s107 + b_k4_IBM_s107
                                 + b_k4_INTC_s107 + b_k4_KO_s107
                                 + b_k4_MAT_s107 + b_k4_MSFT_s107
                                 + b_k4_PG_s107 - s_k4_ADM_s107 - s_k4_AXP_s107
                                 - s_k4_DIS_s107 - s_k4_FDX_s107
                                 - s_k4_IBM_s107 - s_k4_INTC_s107
                                 - s_k4_KO_s107 - s_k4_MAT_s107
                                 - s_k4_MSFT_s107 - s_k4_PG_s107  = 0
 k4_s110_balance_constraint:     b_k4_ADM_s110 + b_k4_AXP_s110 + b_k4_DIS_s110
                                 + b_k4_FDX_s110 + b_k4_IBM_s110
                                 + b_k4_INTC_s110 + b_k4_KO_s110
                                 + b_k4_MAT_s110 + b_k4_MSFT_s110
                                 + b_k4_PG_s110 - s_k4_ADM_s110 - s_k4_AXP_s110
                                 - s_k4_DIS_s110 - s_k4_FDX_s110
                                 - s_k4_IBM_s110 - s_k4_INTC_s110
                                 - s_k4_KO_s110 - s_k4_MAT_s110
                                 - s_k4_MSFT_s110 - s_k4_PG_s110  = 0
 k4_s111_balance_constraint:     b_k4_ADM_s111 + b_k4_AXP_s111 + b_k4_DIS_s111
                                 + b_k4_FDX_s111 + b_k4_IBM_s111
                                 + b_k4_INTC_s111 + b_k4_KO_s111
                                 + b_k4_MAT_s111 + b_k4_MSFT_s111
                                 + b_k4_PG_s111 - s_k4_ADM_s111 - s_k4_AXP_s111
                                 - s_k4_DIS_s111 - s_k4_FDX_s111
                                 - s_k4_IBM_s111 - s_k4_INTC_s111
                                 - s_k4_KO_s111 - s_k4_MAT_s111
                                 - s_k4_MSFT_s111 - s_k4_PG_s111  = 0
 k4_s112_balance_constraint:     b_k4_ADM_s112 + b_k4_AXP_s112 + b_k4_DIS_s112
                                 + b_k4_FDX_s112 + b_k4_IBM_s112
                                 + b_k4_INTC_s112 + b_k4_KO_s112
                                 + b_k4_MAT_s112 + b_k4_MSFT_s112
                                 + b_k4_PG_s112 - s_k4_ADM_s112 - s_k4_AXP_s112
                                 - s_k4_DIS_s112 - s_k4_FDX_s112
                                 - s_k4_IBM_s112 - s_k4_INTC_s112
                                 - s_k4_KO_s112 - s_k4_MAT_s112
                                 - s_k4_MSFT_s112 - s_k4_PG_s112  = 0
 k4_s113_balance_constraint:     b_k4_ADM_s113 + b_k4_AXP_s113 + b_k4_DIS_s113
                                 + b_k4_FDX_s113 + b_k4_IBM_s113
                                 + b_k4_INTC_s113 + b_k4_KO_s113
                                 + b_k4_MAT_s113 + b_k4_MSFT_s113
                                 + b_k4_PG_s113 - s_k4_ADM_s113 - s_k4_AXP_s113
                                 - s_k4_DIS_s113 - s_k4_FDX_s113
                                 - s_k4_IBM_s113 - s_k4_INTC_s113
                                 - s_k4_KO_s113 - s_k4_MAT_s113
                                 - s_k4_MSFT_s113 - s_k4_PG_s113  = 0
 k4_s114_balance_constraint:     b_k4_ADM_s114 + b_k4_AXP_s114 + b_k4_DIS_s114
                                 + b_k4_FDX_s114 + b_k4_IBM_s114
                                 + b_k4_INTC_s114 + b_k4_KO_s114
                                 + b_k4_MAT_s114 + b_k4_MSFT_s114
                                 + b_k4_PG_s114 - s_k4_ADM_s114 - s_k4_AXP_s114
                                 - s_k4_DIS_s114 - s_k4_FDX_s114
                                 - s_k4_IBM_s114 - s_k4_INTC_s114
                                 - s_k4_KO_s114 - s_k4_MAT_s114
                                 - s_k4_MSFT_s114 - s_k4_PG_s114  = 0
 k4_s115_balance_constraint:     b_k4_ADM_s115 + b_k4_AXP_s115 + b_k4_DIS_s115
                                 + b_k4_FDX_s115 + b_k4_IBM_s115
                                 + b_k4_INTC_s115 + b_k4_KO_s115
                                 + b_k4_MAT_s115 + b_k4_MSFT_s115
                                 + b_k4_PG_s115 - s_k4_ADM_s115 - s_k4_AXP_s115
                                 - s_k4_DIS_s115 - s_k4_FDX_s115
                                 - s_k4_IBM_s115 - s_k4_INTC_s115
                                 - s_k4_KO_s115 - s_k4_MAT_s115
                                 - s_k4_MSFT_s115 - s_k4_PG_s115  = 0
 k4_s116_balance_constraint:     b_k4_ADM_s116 + b_k4_AXP_s116 + b_k4_DIS_s116
                                 + b_k4_FDX_s116 + b_k4_IBM_s116
                                 + b_k4_INTC_s116 + b_k4_KO_s116
                                 + b_k4_MAT_s116 + b_k4_MSFT_s116
                                 + b_k4_PG_s116 - s_k4_ADM_s116 - s_k4_AXP_s116
                                 - s_k4_DIS_s116 - s_k4_FDX_s116
                                 - s_k4_IBM_s116 - s_k4_INTC_s116
                                 - s_k4_KO_s116 - s_k4_MAT_s116
                                 - s_k4_MSFT_s116 - s_k4_PG_s116  = 0
 k4_s117_balance_constraint:     b_k4_ADM_s117 + b_k4_AXP_s117 + b_k4_DIS_s117
                                 + b_k4_FDX_s117 + b_k4_IBM_s117
                                 + b_k4_INTC_s117 + b_k4_KO_s117
                                 + b_k4_MAT_s117 + b_k4_MSFT_s117
                                 + b_k4_PG_s117 - s_k4_ADM_s117 - s_k4_AXP_s117
                                 - s_k4_DIS_s117 - s_k4_FDX_s117
                                 - s_k4_IBM_s117 - s_k4_INTC_s117
                                 - s_k4_KO_s117 - s_k4_MAT_s117
                                 - s_k4_MSFT_s117 - s_k4_PG_s117  = 0
 k4_return_constraint:           opt_return - 0.0258860520546842 w_k4_ADM_s000
                                 - 0.0223261060365754 w_k4_ADM_s001
                                 - 0.023648030737746 w_k4_ADM_s002
                                 - 0.0191442603160787 w_k4_ADM_s003
                                 - 0.0473341520478767 w_k4_ADM_s004
                                 - 0.0308114428719668 w_k4_ADM_s005
                                 - 0.038949419388604 w_k4_ADM_s006
                                 - 0.0185721308106855 w_k4_ADM_s007
                                 - 0.0252095893127482 w_k4_ADM_s010
                                 - 0.0524485052118821 w_k4_ADM_s011
                                 - 0.0249902945993016 w_k4_ADM_s012
                                 - 0.0553375821303825 w_k4_ADM_s013
                                 - 0.0364641618095995 w_k4_ADM_s014
                                 - 0.0268879233671312 w_k4_ADM_s015
                                 - 0.029653129159584 w_k4_ADM_s016
                                 - 0.016221601201836 w_k4_ADM_s017
                                 - 0.0284736439085825 w_k4_ADM_s100
                                 - 0.0579574661082336 w_k4_ADM_s101
                                 - 0.0171410412755928 w_k4_ADM_s102
                                 - 0.0455874386629408 w_k4_ADM_s103
                                 - 0.0668826103046151 w_k4_ADM_s104
                                 - 0.051890075363285 w_k4_ADM_s105
                                 - 0.0155311577493224 w_k4_ADM_s106
                                 - 0.0196819814375087 w_k4_ADM_s107
                                 - 0.0225700883084227 w_k4_ADM_s110
                                 - 0.022433613450245 w_k4_ADM_s111
                                 - 0.015820018250267 w_k4_ADM_s112
                                 - 0.0137676176875948 w_k4_ADM_s113
                                 - 0.031525412300357 w_k4_ADM_s114
                                 - 0.0488549672876419 w_k4_ADM_s115
                                 - 0.0395485064269226 w_k4_ADM_s116
                                 - 0.010145106590806 w_k4_ADM_s117
                                 - 0.0223914322301091 w_k4_AXP_s000
                                 - 0.0216240432911395 w_k4_AXP_s001
                                 - 0.0284109701910387 w_k4_AXP_s002
                                 - 0.0226712659276211 w_k4_AXP_s003
                                 - 0.0524457770879937 w_k4_AXP_s004
                                 - 0.0326167165440056 w_k4_AXP_s005
                                 - 0.0372584799153317 w_k4_AXP_s006
                                 - 0.0201832027273374 w_k4_AXP_s007
                                 - 0.0251407842912823 w_k4_AXP_s010
                                 - 0.0567300210501726 w_k4_AXP_s011
                                 - 0.0313242387425976 w_k4_AXP_s012
                                 - 0.0615648877056905 w_k4_AXP_s013
                                 - 0.0421584748093222 w_k4_AXP_s014
                                 - 0.0283541272871925 w_k4_AXP_s015
                                 - 0.0257311411349939 w_k4_AXP_s016
                                 - 0.0150720443944994 w_k4_AXP_s017
                                 - 0.0291315343519484 w_k4_AXP_s100
                                 - 0.0611924345108069 w_k4_AXP_s101
                                 - 0.0156045208718653 w_k4_AXP_s102
                                 - 0.0519962344478943 w_k4_AXP_s103
                                 - 0.0621100350643275 w_k4_AXP_s104
                                 - 0.053614724428396 w_k4_AXP_s105
                                 - 0.0148442244907005 w_k4_AXP_s106
                                 - 0.0225212268884673 w_k4_AXP_s107
                                 - 0.0226170996990316 w_k4_AXP_s110
                                 - 0.0199727988840078 w_k4_AXP_s111
                                 - 0.0141153127064781 w_k4_AXP_s112
                                 - 0.0156309926533934 w_k4_AXP_s113
                                 - 0.0368352407578118 w_k4_AXP_s114
                                 - 0.0491068039562665 w_k4_AXP_s115
                                 - 0.0330411700615164 w_k4_AXP_s116
                                 - 0.0107007998234974 w_k4_AXP_s117
                                 - 0.0280850304135978 w_k4_DIS_s000
                                 - 0.0211237599867501 w_k4_DIS_s001
                                 - 0.0277819343132379 w_k4_DIS_s002
                                 - 0.0243879570200557 w_k4_DIS_s003
                                 - 0.0492345111447159 w_k4_DIS_s004
                                 - 0.0322256715419366 w_k4_DIS_s005
                                 - 0.0405256792820414 w_k4_DIS_s006
                                 - 0.0188313662415647 w_k4_DIS_s007
                                 - 0.0246664334805746 w_k4_DIS_s010
                                 - 0.0547827222725405 w_k4_DIS_s011
                                 - 0.0293316265074889 w_k4_DIS_s012
                                 - 0.0595620489713342 w_k4_DIS_s013
                                 - 0.0383073089048465 w_k4_DIS_s014
                                 - 0.0300073012312229 w_k4_DIS_s015
                                 - 0.0253693158781709 w_k4_DIS_s016
                                 - 0.0157357263879291 w_k4_DIS_s017
                                 - 0.0300730640357973 w_k4_DIS_s100
                                 - 0.059848697703341 w_k4_DIS_s101
                                 - 0.0161145246840117 w_k4_DIS_s102
                                 - 0.0524748066104374 w_k4_DIS_s103
                                 - 0.0590838932581833 w_k4_DIS_s104
                                 - 0.0587620633243561 w_k4_DIS_s105
                                 - 0.0145962166876187 w_k4_DIS_s106
                                 - 0.0205417871785512 w_k4_DIS_s107
                                 - 0.0226809647677487 w_k4_DIS_s110
                                 - 0.0202375799810212 w_k4_DIS_s111
                                 - 0.0141094424879577 w_k4_DIS_s112
                                 - 0.0166988824756108 w_k4_DIS_s113
                                 - 0.038135238112975 w_k4_DIS_s114
                                 - 0.0480056791929881 w_k4_DIS_s115
                                 - 0.0337782214889284 w_k4_DIS_s116
                                 - 0.0110199811274328 w_k4_DIS_s117
                                 - 0.0244658301320865 w_k4_FDX_s000
                                 - 0.0216753793671209 w_k4_FDX_s001
                                 - 0.0261681115489351 w_k4_FDX_s002
                                 - 0.0221278438460412 w_k4_FDX_s003
                                 - 0.0482905757417766 w_k4_FDX_s004
                                 - 0.0309678715776778 w_k4_FDX_s005
                                 - 0.0369381524208772 w_k4_FDX_s006
                                 - 0.018027311697064 w_k4_FDX_s007
                                 - 0.0245515758059037 w_k4_FDX_s010
                                 - 0.0518261591782232 w_k4_FDX_s011
                                 - 0.0286403019428685 w_k4_FDX_s012
                                 - 0.0569339828403 w_k4_FDX_s013
                                 - 0.037192359590796 w_k4_FDX_s014
                                 - 0.028183172481189 w_k4_FDX_s015
                                 - 0.0246502997236251 w_k4_FDX_s016
                                 - 0.0160883314281918 w_k4_FDX_s017
                                 - 0.0282783341546766 w_k4_FDX_s100
                                 - 0.0619597207973415 w_k4_FDX_s101
                                 - 0.0155108747140749 w_k4_FDX_s102
                                 - 0.0494575419814967 w_k4_FDX_s103
                                 - 0.0623363046483531 w_k4_FDX_s104
                                 - 0.0549967809627657 w_k4_FDX_s105
                                 - 0.0147566331675326 w_k4_FDX_s106
                                 - 0.0205633700125485 w_k4_FDX_s107
                                 - 0.0230989832436757 w_k4_FDX_s110
                                 - 0.0204914778068588 w_k4_FDX_s111
                                 - 0.015158902138188 w_k4_FDX_s112
                                 - 0.0148346512574055 w_k4_FDX_s113
                                 - 0.0372748294288391 w_k4_FDX_s114
                                 - 0.0463676126729583 w_k4_FDX_s115
                                 - 0.0341965771775864 w_k4_FDX_s116
                                 - 0.0104821947813283 w_k4_FDX_s117
                                 - 0.0253527328291795 w_k4_IBM_s000
                                 - 0.0216698926920393 w_k4_IBM_s001
                                 - 0.0264520300069311 w_k4_IBM_s002
                                 - 0.0233023083768633 w_k4_IBM_s003
                                 - 0.0496090286744652 w_k4_IBM_s004
                                 - 0.0321786415245085 w_k4_IBM_s005
                                 - 0.038675458732216 w_k4_IBM_s006
                                 - 0.0184779449502392 w_k4_IBM_s007
                                 - 0.0258537858481796 w_k4_IBM_s010
                                 - 0.0535008302294692 w_k4_IBM_s011
                                 - 0.029780496830304 w_k4_IBM_s012
                                 - 0.0567835719649831 w_k4_IBM_s013
                                 - 0.037934753675939 w_k4_IBM_s014
                                 - 0.0299109192177609 w_k4_IBM_s015
                                 - 0.0255856924542913 w_k4_IBM_s016
                                 - 0.0157306821545124 w_k4_IBM_s017
                                 - 0.0289070662743734 w_k4_IBM_s100
                                 - 0.0663347437483611 w_k4_IBM_s101
                                 - 0.0156710542852896 w_k4_IBM_s102
                                 - 0.0518552238545163 w_k4_IBM_s103
                                 - 0.0650483948274654 w_k4_IBM_s104
                                 - 0.0549086643766351 w_k4_IBM_s105
                                 - 0.0155682419180598 w_k4_IBM_s106
                                 - 0.0200923987867354 w_k4_IBM_s107
                                 - 0.0230235414955519 w_k4_IBM_s110
                                 - 0.0210302574826915 w_k4_IBM_s111
                                 - 0.0146691918710781 w_k4_IBM_s112
                                 - 0.0148159248706417 w_k4_IBM_s113
                                 - 0.0377233972450248 w_k4_IBM_s114
                                 - 0.0468912193658039 w_k4_IBM_s115
                                 - 0.0346740987539474 w_k4_IBM_s116
                                 - 0.0101439972240373 w_k4_IBM_s117
                                 - 0.0225512427212474 w_k4_INTC_s000
                                 - 0.0226415986645727 w_k4_INTC_s001
                                 - 0.029922990404341 w_k4_INTC_s002
                                 - 0.023369480464194 w_k4_INTC_s003
                                 - 0.0420653172047141 w_k4_INTC_s004
                                 - 0.0316620115277496 w_k4_INTC_s005
                                 - 0.0394519294482659 w_k4_INTC_s006
                                 - 0.0182189981241079 w_k4_INTC_s007
                                 - 0.0230738708524939 w_k4_INTC_s010
                                 - 0.0572708543746592 w_k4_INTC_s011
                                 - 0.0303329741573242 w_k4_INTC_s012
                                 - 0.064295661767153 w_k4_INTC_s013
                                 - 0.0415835345017881 w_k4_INTC_s014
                                 - 0.0250883536386083 w_k4_INTC_s015
                                 - 0.0250976589405446 w_k4_INTC_s016
                                 - 0.0149308833733013 w_k4_INTC_s017
                                 - 0.0244312029118206 w_k4_INTC_s100
                                 - 0.0580989911408916 w_k4_INTC_s101
                                 - 0.0162218986961306 w_k4_INTC_s102
                                 - 0.0528041231307542 w_k4_INTC_s103
                                 - 0.0594259704085897 w_k4_INTC_s104
                                 - 0.0559796515449993 w_k4_INTC_s105
                                 - 0.016455682408964 w_k4_INTC_s106
                                 - 0.0221640042060149 w_k4_INTC_s107
                                 - 0.0200371243879813 w_k4_INTC_s110
                                 - 0.0182884376445525 w_k4_INTC_s111
                                 - 0.0138587150331867 w_k4_INTC_s112
                                 - 0.0141431877305665 w_k4_INTC_s113
                                 - 0.0382275591777586 w_k4_INTC_s114
                                 - 0.0419460870685078 w_k4_INTC_s115
                                 - 0.0341181027866487 w_k4_INTC_s116
                                 - 0.0113538695859044 w_k4_INTC_s117
                                 - 0.0238671585968651 w_k4_KO_s000
                                 - 0.0203367646400671 w_k4_KO_s001
                                 - 0.0277710785825563 w_k4_KO_s002
                                 - 0.0237564371851595 w_k4_KO_s003
                                 - 0.0489677441787456 w_k4_KO_s004
                                 - 0.0293056708676713 w_k4_KO_s005
                                 - 0.0368161559613951 w_k4_KO_s006
                                 - 0.0191599240678477 w_k4_KO_s007
                                 - 0.0257657581814652 w_k4_KO_s010
                                 - 0.0498128451211821 w_k4_KO_s011
                                 - 0.0299132446087278 w_k4_KO_s012
                                 - 0.0526670138749165 w_k4_KO_s013
                                 - 0.0391264936614286 w_k4_KO_s014
                                 - 0.0285863484704252 w_k4_KO_s015
                                 - 0.0232038999192833 w_k4_KO_s016
                                 - 0.0146646981399793 w_k4_KO_s017
                                 - 0.0258221518102704 w_k4_KO_s100
                                 - 0.0635522817346628 w_k4_KO_s101
                                 - 0.0162476738482691 w_k4_KO_s102
                                 - 0.0552109568274062 w_k4_KO_s103
                                 - 0.0615733244730642 w_k4_KO_s104
                                 - 0.0538852624399163 w_k4_KO_s105
                                 - 0.0146072569192982 w_k4_KO_s106
                                 - 0.0212069482917955 w_k4_KO_s107
                                 - 0.0230443686515295 w_k4_KO_s110
                                 - 0.0206550871429816 w_k4_KO_s111
                                 - 0.0145468880301987 w_k4_KO_s112
                                 - 0.0155475277462469 w_k4_KO_s113
                                 - 0.0342330468090695 w_k4_KO_s114
                                 - 0.0403152006288864 w_k4_KO_s115
                                 - 0.033488503323795 w_k4_KO_s116
                                 - 0.0101752533296397 w_k4_KO_s117
                                 - 0.0266400363371181 w_k4_MAT_s000
                                 - 0.0197932677602181 w_k4_MAT_s001
                                 - 0.0230005118856958 w_k4_MAT_s002
                                 - 0.0255948759671046 w_k4_MAT_s003
                                 - 0.0533319825990327 w_k4_MAT_s004
                                 - 0.0306350483418667 w_k4_MAT_s005
                                 - 0.0368695816902136 w_k4_MAT_s006
                                 - 0.0159127036505552 w_k4_MAT_s007
                                 - 0.0263755306416907 w_k4_MAT_s010
                                 - 0.0631611305633763 w_k4_MAT_s011
                                 - 0.0272543146225591 w_k4_MAT_s012
                                 - 0.0595155094654439 w_k4_MAT_s013
                                 - 0.0411921245545165 w_k4_MAT_s014
                                 - 0.032325054399063 w_k4_MAT_s015
                                 - 0.0200932918155347 w_k4_MAT_s016
                                 - 0.0182820073537867 w_k4_MAT_s017
                                 - 0.0272134734816856 w_k4_MAT_s100
                                 - 0.0673983651793459 w_k4_MAT_s101
                                 - 0.0152085028617397 w_k4_MAT_s102
                                 - 0.0662940133307282 w_k4_MAT_s103
                                 - 0.0680004422523751 w_k4_MAT_s104
                                 - 0.0570645752711364 w_k4_MAT_s105
                                 - 0.0136268540187414 w_k4_MAT_s106
                                 - 0.0189651898535062 w_k4_MAT_s107
                                 - 0.0256224290773826 w_k4_MAT_s110
                                 - 0.0267344760785056 w_k4_MAT_s111
                                 - 0.0133367007820854 w_k4_MAT_s112
                                 - 0.0178972721004644 w_k4_MAT_s113
                                 - 0.0388999363186441 w_k4_MAT_s114
                                 - 0.0591900138937408 w_k4_MAT_s115
                                 - 0.0283146829465234 w_k4_MAT_s116
                                 - 0.0113349363011812 w_k4_MAT_s117
                                 - 0.0216640707415119 w_k4_MSFT_s000
                                 - 0.0203215149269972 w_k4_MSFT_s001
                                 - 0.0266343424860356 w_k4_MSFT_s002
                                 - 0.0224645929816842 w_k4_MSFT_s003
                                 - 0.0480212849102247 w_k4_MSFT_s004
                                 - 0.0328043553506751 w_k4_MSFT_s005
                                 - 0.0406331053063848 w_k4_MSFT_s006
                                 - 0.0171073938686651 w_k4_MSFT_s007
                                 - 0.0212809312054139 w_k4_MSFT_s010
                                 - 0.0550032331044643 w_k4_MSFT_s011
                                 - 0.0291477292859126 w_k4_MSFT_s012
                                 - 0.0641447493955406 w_k4_MSFT_s013
                                 - 0.0376930998041619 w_k4_MSFT_s014
                                 - 0.025812263010121 w_k4_MSFT_s015
                                 - 0.0264106325630589 w_k4_MSFT_s016
                                 - 0.0168039246612262 w_k4_MSFT_s017
                                 - 0.0281024180934595 w_k4_MSFT_s100
                                 - 0.0616398679163244 w_k4_MSFT_s101
                                 - 0.0147522030383162 w_k4_MSFT_s102
                                 - 0.0515185157464941 w_k4_MSFT_s103
                                 - 0.0610985630402093 w_k4_MSFT_s104
                                 - 0.0541092037434625 w_k4_MSFT_s105
                                 - 0.0149200408072307 w_k4_MSFT_s106
                                 - 0.0203756741063954 w_k4_MSFT_s107
                                 - 0.0223097880002745 w_k4_MSFT_s110
                                 - 0.0215792285022448 w_k4_MSFT_s111
                                 - 0.013393965811595 w_k4_MSFT_s112
                                 - 0.0157436901458046 w_k4_MSFT_s113
                                 - 0.0367702464641157 w_k4_MSFT_s114
                                 - 0.0433679627448736 w_k4_MSFT_s115
                                 - 0.0323587374092931 w_k4_MSFT_s116
                                 - 0.0097690207037408 w_k4_MSFT_s117
                                 - 0.023506764692255 w_k4_PG_s000
                                 - 0.0211185485245667 w_k4_PG_s001
                                 - 0.025600546752537 w_k4_PG_s002
                                 - 0.0230568990249454 w_k4_PG_s003
                                 - 0.0471399548044012 w_k4_PG_s004
                                 - 0.0293210876034778 w_k4_PG_s005
                                 - 0.0388876886933952 w_k4_PG_s006
                                 - 0.0176756090808779 w_k4_PG_s007
                                 - 0.0232618711013894 w_k4_PG_s010
                                 - 0.0563256523875594 w_k4_PG_s011
                                 - 0.0253774179163701 w_k4_PG_s012
                                 - 0.0550765276152346 w_k4_PG_s013
                                 - 0.0373589680336216 w_k4_PG_s014
                                 - 0.028191026271473 w_k4_PG_s015
                                 - 0.0216896099126284 w_k4_PG_s016
                                 - 0.0157713717734259 w_k4_PG_s017
                                 - 0.0279484901664366 w_k4_PG_s100
                                 - 0.0646432270981755 w_k4_PG_s101
                                 - 0.0159655832607774 w_k4_PG_s102
                                 - 0.0528644504869189 w_k4_PG_s103
                                 - 0.0600086421034387 w_k4_PG_s104
                                 - 0.0526541002234613 w_k4_PG_s105
                                 - 0.014621594525842 w_k4_PG_s106
                                 - 0.0215799419637717 w_k4_PG_s107
                                 - 0.0223974564722153 w_k4_PG_s110
                                 - 0.0196869735276465 w_k4_PG_s111
                                 - 0.0143878550844327 w_k4_PG_s112
                                 - 0.0148672276019942 w_k4_PG_s113
                                 - 0.0367348976244589 w_k4_PG_s114
                                 - 0.0478322891521557 w_k4_PG_s115
                                 - 0.0346874239241957 w_k4_PG_s116
                                 - 0.0102072179507039 w_k4_PG_s117 <= 0
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
 0 <= w_k4_ADM_s0 <= 1
 0 <= w_k4_ADM_s1 <= 1
 0 <= w_k4_ADM_s00 <= 1
 0 <= w_k4_ADM_s01 <= 1
 0 <= w_k4_ADM_s10 <= 1
 0 <= w_k4_ADM_s11 <= 1
 0 <= w_k4_ADM_s000 <= 1
 0 <= w_k4_ADM_s001 <= 1
 0 <= w_k4_ADM_s002 <= 1
 0 <= w_k4_ADM_s003 <= 1
 0 <= w_k4_ADM_s004 <= 1
 0 <= w_k4_ADM_s005 <= 1
 0 <= w_k4_ADM_s006 <= 1
 0 <= w_k4_ADM_s007 <= 1
 0 <= w_k4_ADM_s010 <= 1
 0 <= w_k4_ADM_s011 <= 1
 0 <= w_k4_ADM_s012 <= 1
 0 <= w_k4_ADM_s013 <= 1
 0 <= w_k4_ADM_s014 <= 1
 0 <= w_k4_ADM_s015 <= 1
 0 <= w_k4_ADM_s016 <= 1
 0 <= w_k4_ADM_s017 <= 1
 0 <= w_k4_ADM_s100 <= 1
 0 <= w_k4_ADM_s101 <= 1
 0 <= w_k4_ADM_s102 <= 1
 0 <= w_k4_ADM_s103 <= 1
 0 <= w_k4_ADM_s104 <= 1
 0 <= w_k4_ADM_s105 <= 1
 0 <= w_k4_ADM_s106 <= 1
 0 <= w_k4_ADM_s107 <= 1
 0 <= w_k4_ADM_s110 <= 1
 0 <= w_k4_ADM_s111 <= 1
 0 <= w_k4_ADM_s112 <= 1
 0 <= w_k4_ADM_s113 <= 1
 0 <= w_k4_ADM_s114 <= 1
 0 <= w_k4_ADM_s115 <= 1
 0 <= w_k4_ADM_s116 <= 1
 0 <= w_k4_ADM_s117 <= 1
 0 <= w_k4_AXP_s0 <= 1
 0 <= w_k4_AXP_s1 <= 1
 0 <= w_k4_AXP_s00 <= 1
 0 <= w_k4_AXP_s01 <= 1
 0 <= w_k4_AXP_s10 <= 1
 0 <= w_k4_AXP_s11 <= 1
 0 <= w_k4_AXP_s000 <= 1
 0 <= w_k4_AXP_s001 <= 1
 0 <= w_k4_AXP_s002 <= 1
 0 <= w_k4_AXP_s003 <= 1
 0 <= w_k4_AXP_s004 <= 1
 0 <= w_k4_AXP_s005 <= 1
 0 <= w_k4_AXP_s006 <= 1
 0 <= w_k4_AXP_s007 <= 1
 0 <= w_k4_AXP_s010 <= 1
 0 <= w_k4_AXP_s011 <= 1
 0 <= w_k4_AXP_s012 <= 1
 0 <= w_k4_AXP_s013 <= 1
 0 <= w_k4_AXP_s014 <= 1
 0 <= w_k4_AXP_s015 <= 1
 0 <= w_k4_AXP_s016 <= 1
 0 <= w_k4_AXP_s017 <= 1
 0 <= w_k4_AXP_s100 <= 1
 0 <= w_k4_AXP_s101 <= 1
 0 <= w_k4_AXP_s102 <= 1
 0 <= w_k4_AXP_s103 <= 1
 0 <= w_k4_AXP_s104 <= 1
 0 <= w_k4_AXP_s105 <= 1
 0 <= w_k4_AXP_s106 <= 1
 0 <= w_k4_AXP_s107 <= 1
 0 <= w_k4_AXP_s110 <= 1
 0 <= w_k4_AXP_s111 <= 1
 0 <= w_k4_AXP_s112 <= 1
 0 <= w_k4_AXP_s113 <= 1
 0 <= w_k4_AXP_s114 <= 1
 0 <= w_k4_AXP_s115 <= 1
 0 <= w_k4_AXP_s116 <= 1
 0 <= w_k4_AXP_s117 <= 1
 0 <= w_k4_DIS_s0 <= 1
 0 <= w_k4_DIS_s1 <= 1
 0 <= w_k4_DIS_s00 <= 1
 0 <= w_k4_DIS_s01 <= 1
 0 <= w_k4_DIS_s10 <= 1
 0 <= w_k4_DIS_s11 <= 1
 0 <= w_k4_DIS_s000 <= 1
 0 <= w_k4_DIS_s001 <= 1
 0 <= w_k4_DIS_s002 <= 1
 0 <= w_k4_DIS_s003 <= 1
 0 <= w_k4_DIS_s004 <= 1
 0 <= w_k4_DIS_s005 <= 1
 0 <= w_k4_DIS_s006 <= 1
 0 <= w_k4_DIS_s007 <= 1
 0 <= w_k4_DIS_s010 <= 1
 0 <= w_k4_DIS_s011 <= 1
 0 <= w_k4_DIS_s012 <= 1
 0 <= w_k4_DIS_s013 <= 1
 0 <= w_k4_DIS_s014 <= 1
 0 <= w_k4_DIS_s015 <= 1
 0 <= w_k4_DIS_s016 <= 1
 0 <= w_k4_DIS_s017 <= 1
 0 <= w_k4_DIS_s100 <= 1
 0 <= w_k4_DIS_s101 <= 1
 0 <= w_k4_DIS_s102 <= 1
 0 <= w_k4_DIS_s103 <= 1
 0 <= w_k4_DIS_s104 <= 1
 0 <= w_k4_DIS_s105 <= 1
 0 <= w_k4_DIS_s106 <= 1
 0 <= w_k4_DIS_s107 <= 1
 0 <= w_k4_DIS_s110 <= 1
 0 <= w_k4_DIS_s111 <= 1
 0 <= w_k4_DIS_s112 <= 1
 0 <= w_k4_DIS_s113 <= 1
 0 <= w_k4_DIS_s114 <= 1
 0 <= w_k4_DIS_s115 <= 1
 0 <= w_k4_DIS_s116 <= 1
 0 <= w_k4_DIS_s117 <= 1
 0 <= w_k4_FDX_s0 <= 1
 0 <= w_k4_FDX_s1 <= 1
 0 <= w_k4_FDX_s00 <= 1
 0 <= w_k4_FDX_s01 <= 1
 0 <= w_k4_FDX_s10 <= 1
 0 <= w_k4_FDX_s11 <= 1
 0 <= w_k4_FDX_s000 <= 1
 0 <= w_k4_FDX_s001 <= 1
 0 <= w_k4_FDX_s002 <= 1
 0 <= w_k4_FDX_s003 <= 1
 0 <= w_k4_FDX_s004 <= 1
 0 <= w_k4_FDX_s005 <= 1
 0 <= w_k4_FDX_s006 <= 1
 0 <= w_k4_FDX_s007 <= 1
 0 <= w_k4_FDX_s010 <= 1
 0 <= w_k4_FDX_s011 <= 1
 0 <= w_k4_FDX_s012 <= 1
 0 <= w_k4_FDX_s013 <= 1
 0 <= w_k4_FDX_s014 <= 1
 0 <= w_k4_FDX_s015 <= 1
 0 <= w_k4_FDX_s016 <= 1
 0 <= w_k4_FDX_s017 <= 1
 0 <= w_k4_FDX_s100 <= 1
 0 <= w_k4_FDX_s101 <= 1
 0 <= w_k4_FDX_s102 <= 1
 0 <= w_k4_FDX_s103 <= 1
 0 <= w_k4_FDX_s104 <= 1
 0 <= w_k4_FDX_s105 <= 1
 0 <= w_k4_FDX_s106 <= 1
 0 <= w_k4_FDX_s107 <= 1
 0 <= w_k4_FDX_s110 <= 1
 0 <= w_k4_FDX_s111 <= 1
 0 <= w_k4_FDX_s112 <= 1
 0 <= w_k4_FDX_s113 <= 1
 0 <= w_k4_FDX_s114 <= 1
 0 <= w_k4_FDX_s115 <= 1
 0 <= w_k4_FDX_s116 <= 1
 0 <= w_k4_FDX_s117 <= 1
 0 <= w_k4_IBM_s0 <= 1
 0 <= w_k4_IBM_s1 <= 1
 0 <= w_k4_IBM_s00 <= 1
 0 <= w_k4_IBM_s01 <= 1
 0 <= w_k4_IBM_s10 <= 1
 0 <= w_k4_IBM_s11 <= 1
 0 <= w_k4_IBM_s000 <= 1
 0 <= w_k4_IBM_s001 <= 1
 0 <= w_k4_IBM_s002 <= 1
 0 <= w_k4_IBM_s003 <= 1
 0 <= w_k4_IBM_s004 <= 1
 0 <= w_k4_IBM_s005 <= 1
 0 <= w_k4_IBM_s006 <= 1
 0 <= w_k4_IBM_s007 <= 1
 0 <= w_k4_IBM_s010 <= 1
 0 <= w_k4_IBM_s011 <= 1
 0 <= w_k4_IBM_s012 <= 1
 0 <= w_k4_IBM_s013 <= 1
 0 <= w_k4_IBM_s014 <= 1
 0 <= w_k4_IBM_s015 <= 1
 0 <= w_k4_IBM_s016 <= 1
 0 <= w_k4_IBM_s017 <= 1
 0 <= w_k4_IBM_s100 <= 1
 0 <= w_k4_IBM_s101 <= 1
 0 <= w_k4_IBM_s102 <= 1
 0 <= w_k4_IBM_s103 <= 1
 0 <= w_k4_IBM_s104 <= 1
 0 <= w_k4_IBM_s105 <= 1
 0 <= w_k4_IBM_s106 <= 1
 0 <= w_k4_IBM_s107 <= 1
 0 <= w_k4_IBM_s110 <= 1
 0 <= w_k4_IBM_s111 <= 1
 0 <= w_k4_IBM_s112 <= 1
 0 <= w_k4_IBM_s113 <= 1
 0 <= w_k4_IBM_s114 <= 1
 0 <= w_k4_IBM_s115 <= 1
 0 <= w_k4_IBM_s116 <= 1
 0 <= w_k4_IBM_s117 <= 1
 0 <= w_k4_INTC_s0 <= 1
 0 <= w_k4_INTC_s1 <= 1
 0 <= w_k4_INTC_s00 <= 1
 0 <= w_k4_INTC_s01 <= 1
 0 <= w_k4_INTC_s10 <= 1
 0 <= w_k4_INTC_s11 <= 1
 0 <= w_k4_INTC_s000 <= 1
 0 <= w_k4_INTC_s001 <= 1
 0 <= w_k4_INTC_s002 <= 1
 0 <= w_k4_INTC_s003 <= 1
 0 <= w_k4_INTC_s004 <= 1
 0 <= w_k4_INTC_s005 <= 1
 0 <= w_k4_INTC_s006 <= 1
 0 <= w_k4_INTC_s007 <= 1
 0 <= w_k4_INTC_s010 <= 1
 0 <= w_k4_INTC_s011 <= 1
 0 <= w_k4_INTC_s012 <= 1
 0 <= w_k4_INTC_s013 <= 1
 0 <= w_k4_INTC_s014 <= 1
 0 <= w_k4_INTC_s015 <= 1
 0 <= w_k4_INTC_s016 <= 1
 0 <= w_k4_INTC_s017 <= 1
 0 <= w_k4_INTC_s100 <= 1
 0 <= w_k4_INTC_s101 <= 1
 0 <= w_k4_INTC_s102 <= 1
 0 <= w_k4_INTC_s103 <= 1
 0 <= w_k4_INTC_s104 <= 1
 0 <= w_k4_INTC_s105 <= 1
 0 <= w_k4_INTC_s106 <= 1
 0 <= w_k4_INTC_s107 <= 1
 0 <= w_k4_INTC_s110 <= 1
 0 <= w_k4_INTC_s111 <= 1
 0 <= w_k4_INTC_s112 <= 1
 0 <= w_k4_INTC_s113 <= 1
 0 <= w_k4_INTC_s114 <= 1
 0 <= w_k4_INTC_s115 <= 1
 0 <= w_k4_INTC_s116 <= 1
 0 <= w_k4_INTC_s117 <= 1
 0 <= w_k4_KO_s0 <= 1
 0 <= w_k4_KO_s1 <= 1
 0 <= w_k4_KO_s00 <= 1
 0 <= w_k4_KO_s01 <= 1
 0 <= w_k4_KO_s10 <= 1
 0 <= w_k4_KO_s11 <= 1
 0 <= w_k4_KO_s000 <= 1
 0 <= w_k4_KO_s001 <= 1
 0 <= w_k4_KO_s002 <= 1
 0 <= w_k4_KO_s003 <= 1
 0 <= w_k4_KO_s004 <= 1
 0 <= w_k4_KO_s005 <= 1
 0 <= w_k4_KO_s006 <= 1
 0 <= w_k4_KO_s007 <= 1
 0 <= w_k4_KO_s010 <= 1
 0 <= w_k4_KO_s011 <= 1
 0 <= w_k4_KO_s012 <= 1
 0 <= w_k4_KO_s013 <= 1
 0 <= w_k4_KO_s014 <= 1
 0 <= w_k4_KO_s015 <= 1
 0 <= w_k4_KO_s016 <= 1
 0 <= w_k4_KO_s017 <= 1
 0 <= w_k4_KO_s100 <= 1
 0 <= w_k4_KO_s101 <= 1
 0 <= w_k4_KO_s102 <= 1
 0 <= w_k4_KO_s103 <= 1
 0 <= w_k4_KO_s104 <= 1
 0 <= w_k4_KO_s105 <= 1
 0 <= w_k4_KO_s106 <= 1
 0 <= w_k4_KO_s107 <= 1
 0 <= w_k4_KO_s110 <= 1
 0 <= w_k4_KO_s111 <= 1
 0 <= w_k4_KO_s112 <= 1
 0 <= w_k4_KO_s113 <= 1
 0 <= w_k4_KO_s114 <= 1
 0 <= w_k4_KO_s115 <= 1
 0 <= w_k4_KO_s116 <= 1
 0 <= w_k4_KO_s117 <= 1
 0 <= w_k4_MAT_s0 <= 1
 0 <= w_k4_MAT_s1 <= 1
 0 <= w_k4_MAT_s00 <= 1
 0 <= w_k4_MAT_s01 <= 1
 0 <= w_k4_MAT_s10 <= 1
 0 <= w_k4_MAT_s11 <= 1
 0 <= w_k4_MAT_s000 <= 1
 0 <= w_k4_MAT_s001 <= 1
 0 <= w_k4_MAT_s002 <= 1
 0 <= w_k4_MAT_s003 <= 1
 0 <= w_k4_MAT_s004 <= 1
 0 <= w_k4_MAT_s005 <= 1
 0 <= w_k4_MAT_s006 <= 1
 0 <= w_k4_MAT_s007 <= 1
 0 <= w_k4_MAT_s010 <= 1
 0 <= w_k4_MAT_s011 <= 1
 0 <= w_k4_MAT_s012 <= 1
 0 <= w_k4_MAT_s013 <= 1
 0 <= w_k4_MAT_s014 <= 1
 0 <= w_k4_MAT_s015 <= 1
 0 <= w_k4_MAT_s016 <= 1
 0 <= w_k4_MAT_s017 <= 1
 0 <= w_k4_MAT_s100 <= 1
 0 <= w_k4_MAT_s101 <= 1
 0 <= w_k4_MAT_s102 <= 1
 0 <= w_k4_MAT_s103 <= 1
 0 <= w_k4_MAT_s104 <= 1
 0 <= w_k4_MAT_s105 <= 1
 0 <= w_k4_MAT_s106 <= 1
 0 <= w_k4_MAT_s107 <= 1
 0 <= w_k4_MAT_s110 <= 1
 0 <= w_k4_MAT_s111 <= 1
 0 <= w_k4_MAT_s112 <= 1
 0 <= w_k4_MAT_s113 <= 1
 0 <= w_k4_MAT_s114 <= 1
 0 <= w_k4_MAT_s115 <= 1
 0 <= w_k4_MAT_s116 <= 1
 0 <= w_k4_MAT_s117 <= 1
 0 <= w_k4_MSFT_s0 <= 1
 0 <= w_k4_MSFT_s1 <= 1
 0 <= w_k4_MSFT_s00 <= 1
 0 <= w_k4_MSFT_s01 <= 1
 0 <= w_k4_MSFT_s10 <= 1
 0 <= w_k4_MSFT_s11 <= 1
 0 <= w_k4_MSFT_s000 <= 1
 0 <= w_k4_MSFT_s001 <= 1
 0 <= w_k4_MSFT_s002 <= 1
 0 <= w_k4_MSFT_s003 <= 1
 0 <= w_k4_MSFT_s004 <= 1
 0 <= w_k4_MSFT_s005 <= 1
 0 <= w_k4_MSFT_s006 <= 1
 0 <= w_k4_MSFT_s007 <= 1
 0 <= w_k4_MSFT_s010 <= 1
 0 <= w_k4_MSFT_s011 <= 1
 0 <= w_k4_MSFT_s012 <= 1
 0 <= w_k4_MSFT_s013 <= 1
 0 <= w_k4_MSFT_s014 <= 1
 0 <= w_k4_MSFT_s015 <= 1
 0 <= w_k4_MSFT_s016 <= 1
 0 <= w_k4_MSFT_s017 <= 1
 0 <= w_k4_MSFT_s100 <= 1
 0 <= w_k4_MSFT_s101 <= 1
 0 <= w_k4_MSFT_s102 <= 1
 0 <= w_k4_MSFT_s103 <= 1
 0 <= w_k4_MSFT_s104 <= 1
 0 <= w_k4_MSFT_s105 <= 1
 0 <= w_k4_MSFT_s106 <= 1
 0 <= w_k4_MSFT_s107 <= 1
 0 <= w_k4_MSFT_s110 <= 1
 0 <= w_k4_MSFT_s111 <= 1
 0 <= w_k4_MSFT_s112 <= 1
 0 <= w_k4_MSFT_s113 <= 1
 0 <= w_k4_MSFT_s114 <= 1
 0 <= w_k4_MSFT_s115 <= 1
 0 <= w_k4_MSFT_s116 <= 1
 0 <= w_k4_MSFT_s117 <= 1
 0 <= w_k4_PG_s0 <= 1
 0 <= w_k4_PG_s1 <= 1
 0 <= w_k4_PG_s00 <= 1
 0 <= w_k4_PG_s01 <= 1
 0 <= w_k4_PG_s10 <= 1
 0 <= w_k4_PG_s11 <= 1
 0 <= w_k4_PG_s000 <= 1
 0 <= w_k4_PG_s001 <= 1
 0 <= w_k4_PG_s002 <= 1
 0 <= w_k4_PG_s003 <= 1
 0 <= w_k4_PG_s004 <= 1
 0 <= w_k4_PG_s005 <= 1
 0 <= w_k4_PG_s006 <= 1
 0 <= w_k4_PG_s007 <= 1
 0 <= w_k4_PG_s010 <= 1
 0 <= w_k4_PG_s011 <= 1
 0 <= w_k4_PG_s012 <= 1
 0 <= w_k4_PG_s013 <= 1
 0 <= w_k4_PG_s014 <= 1
 0 <= w_k4_PG_s015 <= 1
 0 <= w_k4_PG_s016 <= 1
 0 <= w_k4_PG_s017 <= 1
 0 <= w_k4_PG_s100 <= 1
 0 <= w_k4_PG_s101 <= 1
 0 <= w_k4_PG_s102 <= 1
 0 <= w_k4_PG_s103 <= 1
 0 <= w_k4_PG_s104 <= 1
 0 <= w_k4_PG_s105 <= 1
 0 <= w_k4_PG_s106 <= 1
 0 <= w_k4_PG_s107 <= 1
 0 <= w_k4_PG_s110 <= 1
 0 <= w_k4_PG_s111 <= 1
 0 <= w_k4_PG_s112 <= 1
 0 <= w_k4_PG_s113 <= 1
 0 <= w_k4_PG_s114 <= 1
 0 <= w_k4_PG_s115 <= 1
 0 <= w_k4_PG_s116 <= 1
 0 <= w_k4_PG_s117 <= 1
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
 0 <= b_k4_ADM_s0 <= 0.2
 0 <= b_k4_ADM_s1 <= 0.2
 0 <= b_k4_ADM_s00 <= 0.2
 0 <= b_k4_ADM_s01 <= 0.2
 0 <= b_k4_ADM_s10 <= 0.2
 0 <= b_k4_ADM_s11 <= 0.2
 0 <= b_k4_ADM_s000 <= 0.2
 0 <= b_k4_ADM_s001 <= 0.2
 0 <= b_k4_ADM_s002 <= 0.2
 0 <= b_k4_ADM_s003 <= 0.2
 0 <= b_k4_ADM_s004 <= 0.2
 0 <= b_k4_ADM_s005 <= 0.2
 0 <= b_k4_ADM_s006 <= 0.2
 0 <= b_k4_ADM_s007 <= 0.2
 0 <= b_k4_ADM_s010 <= 0.2
 0 <= b_k4_ADM_s011 <= 0.2
 0 <= b_k4_ADM_s012 <= 0.2
 0 <= b_k4_ADM_s013 <= 0.2
 0 <= b_k4_ADM_s014 <= 0.2
 0 <= b_k4_ADM_s015 <= 0.2
 0 <= b_k4_ADM_s016 <= 0.2
 0 <= b_k4_ADM_s017 <= 0.2
 0 <= b_k4_ADM_s100 <= 0.2
 0 <= b_k4_ADM_s101 <= 0.2
 0 <= b_k4_ADM_s102 <= 0.2
 0 <= b_k4_ADM_s103 <= 0.2
 0 <= b_k4_ADM_s104 <= 0.2
 0 <= b_k4_ADM_s105 <= 0.2
 0 <= b_k4_ADM_s106 <= 0.2
 0 <= b_k4_ADM_s107 <= 0.2
 0 <= b_k4_ADM_s110 <= 0.2
 0 <= b_k4_ADM_s111 <= 0.2
 0 <= b_k4_ADM_s112 <= 0.2
 0 <= b_k4_ADM_s113 <= 0.2
 0 <= b_k4_ADM_s114 <= 0.2
 0 <= b_k4_ADM_s115 <= 0.2
 0 <= b_k4_ADM_s116 <= 0.2
 0 <= b_k4_ADM_s117 <= 0.2
 0 <= b_k4_AXP_s0 <= 0.2
 0 <= b_k4_AXP_s1 <= 0.2
 0 <= b_k4_AXP_s00 <= 0.2
 0 <= b_k4_AXP_s01 <= 0.2
 0 <= b_k4_AXP_s10 <= 0.2
 0 <= b_k4_AXP_s11 <= 0.2
 0 <= b_k4_AXP_s000 <= 0.2
 0 <= b_k4_AXP_s001 <= 0.2
 0 <= b_k4_AXP_s002 <= 0.2
 0 <= b_k4_AXP_s003 <= 0.2
 0 <= b_k4_AXP_s004 <= 0.2
 0 <= b_k4_AXP_s005 <= 0.2
 0 <= b_k4_AXP_s006 <= 0.2
 0 <= b_k4_AXP_s007 <= 0.2
 0 <= b_k4_AXP_s010 <= 0.2
 0 <= b_k4_AXP_s011 <= 0.2
 0 <= b_k4_AXP_s012 <= 0.2
 0 <= b_k4_AXP_s013 <= 0.2
 0 <= b_k4_AXP_s014 <= 0.2
 0 <= b_k4_AXP_s015 <= 0.2
 0 <= b_k4_AXP_s016 <= 0.2
 0 <= b_k4_AXP_s017 <= 0.2
 0 <= b_k4_AXP_s100 <= 0.2
 0 <= b_k4_AXP_s101 <= 0.2
 0 <= b_k4_AXP_s102 <= 0.2
 0 <= b_k4_AXP_s103 <= 0.2
 0 <= b_k4_AXP_s104 <= 0.2
 0 <= b_k4_AXP_s105 <= 0.2
 0 <= b_k4_AXP_s106 <= 0.2
 0 <= b_k4_AXP_s107 <= 0.2
 0 <= b_k4_AXP_s110 <= 0.2
 0 <= b_k4_AXP_s111 <= 0.2
 0 <= b_k4_AXP_s112 <= 0.2
 0 <= b_k4_AXP_s113 <= 0.2
 0 <= b_k4_AXP_s114 <= 0.2
 0 <= b_k4_AXP_s115 <= 0.2
 0 <= b_k4_AXP_s116 <= 0.2
 0 <= b_k4_AXP_s117 <= 0.2
 0 <= b_k4_DIS_s0 <= 0.2
 0 <= b_k4_DIS_s1 <= 0.2
 0 <= b_k4_DIS_s00 <= 0.2
 0 <= b_k4_DIS_s01 <= 0.2
 0 <= b_k4_DIS_s10 <= 0.2
 0 <= b_k4_DIS_s11 <= 0.2
 0 <= b_k4_DIS_s000 <= 0.2
 0 <= b_k4_DIS_s001 <= 0.2
 0 <= b_k4_DIS_s002 <= 0.2
 0 <= b_k4_DIS_s003 <= 0.2
 0 <= b_k4_DIS_s004 <= 0.2
 0 <= b_k4_DIS_s005 <= 0.2
 0 <= b_k4_DIS_s006 <= 0.2
 0 <= b_k4_DIS_s007 <= 0.2
 0 <= b_k4_DIS_s010 <= 0.2
 0 <= b_k4_DIS_s011 <= 0.2
 0 <= b_k4_DIS_s012 <= 0.2
 0 <= b_k4_DIS_s013 <= 0.2
 0 <= b_k4_DIS_s014 <= 0.2
 0 <= b_k4_DIS_s015 <= 0.2
 0 <= b_k4_DIS_s016 <= 0.2
 0 <= b_k4_DIS_s017 <= 0.2
 0 <= b_k4_DIS_s100 <= 0.2
 0 <= b_k4_DIS_s101 <= 0.2
 0 <= b_k4_DIS_s102 <= 0.2
 0 <= b_k4_DIS_s103 <= 0.2
 0 <= b_k4_DIS_s104 <= 0.2
 0 <= b_k4_DIS_s105 <= 0.2
 0 <= b_k4_DIS_s106 <= 0.2
 0 <= b_k4_DIS_s107 <= 0.2
 0 <= b_k4_DIS_s110 <= 0.2
 0 <= b_k4_DIS_s111 <= 0.2
 0 <= b_k4_DIS_s112 <= 0.2
 0 <= b_k4_DIS_s113 <= 0.2
 0 <= b_k4_DIS_s114 <= 0.2
 0 <= b_k4_DIS_s115 <= 0.2
 0 <= b_k4_DIS_s116 <= 0.2
 0 <= b_k4_DIS_s117 <= 0.2
 0 <= b_k4_FDX_s0 <= 0.2
 0 <= b_k4_FDX_s1 <= 0.2
 0 <= b_k4_FDX_s00 <= 0.2
 0 <= b_k4_FDX_s01 <= 0.2
 0 <= b_k4_FDX_s10 <= 0.2
 0 <= b_k4_FDX_s11 <= 0.2
 0 <= b_k4_FDX_s000 <= 0.2
 0 <= b_k4_FDX_s001 <= 0.2
 0 <= b_k4_FDX_s002 <= 0.2
 0 <= b_k4_FDX_s003 <= 0.2
 0 <= b_k4_FDX_s004 <= 0.2
 0 <= b_k4_FDX_s005 <= 0.2
 0 <= b_k4_FDX_s006 <= 0.2
 0 <= b_k4_FDX_s007 <= 0.2
 0 <= b_k4_FDX_s010 <= 0.2
 0 <= b_k4_FDX_s011 <= 0.2
 0 <= b_k4_FDX_s012 <= 0.2
 0 <= b_k4_FDX_s013 <= 0.2
 0 <= b_k4_FDX_s014 <= 0.2
 0 <= b_k4_FDX_s015 <= 0.2
 0 <= b_k4_FDX_s016 <= 0.2
 0 <= b_k4_FDX_s017 <= 0.2
 0 <= b_k4_FDX_s100 <= 0.2
 0 <= b_k4_FDX_s101 <= 0.2
 0 <= b_k4_FDX_s102 <= 0.2
 0 <= b_k4_FDX_s103 <= 0.2
 0 <= b_k4_FDX_s104 <= 0.2
 0 <= b_k4_FDX_s105 <= 0.2
 0 <= b_k4_FDX_s106 <= 0.2
 0 <= b_k4_FDX_s107 <= 0.2
 0 <= b_k4_FDX_s110 <= 0.2
 0 <= b_k4_FDX_s111 <= 0.2
 0 <= b_k4_FDX_s112 <= 0.2
 0 <= b_k4_FDX_s113 <= 0.2
 0 <= b_k4_FDX_s114 <= 0.2
 0 <= b_k4_FDX_s115 <= 0.2
 0 <= b_k4_FDX_s116 <= 0.2
 0 <= b_k4_FDX_s117 <= 0.2
 0 <= b_k4_IBM_s0 <= 0.2
 0 <= b_k4_IBM_s1 <= 0.2
 0 <= b_k4_IBM_s00 <= 0.2
 0 <= b_k4_IBM_s01 <= 0.2
 0 <= b_k4_IBM_s10 <= 0.2
 0 <= b_k4_IBM_s11 <= 0.2
 0 <= b_k4_IBM_s000 <= 0.2
 0 <= b_k4_IBM_s001 <= 0.2
 0 <= b_k4_IBM_s002 <= 0.2
 0 <= b_k4_IBM_s003 <= 0.2
 0 <= b_k4_IBM_s004 <= 0.2
 0 <= b_k4_IBM_s005 <= 0.2
 0 <= b_k4_IBM_s006 <= 0.2
 0 <= b_k4_IBM_s007 <= 0.2
 0 <= b_k4_IBM_s010 <= 0.2
 0 <= b_k4_IBM_s011 <= 0.2
 0 <= b_k4_IBM_s012 <= 0.2
 0 <= b_k4_IBM_s013 <= 0.2
 0 <= b_k4_IBM_s014 <= 0.2
 0 <= b_k4_IBM_s015 <= 0.2
 0 <= b_k4_IBM_s016 <= 0.2
 0 <= b_k4_IBM_s017 <= 0.2
 0 <= b_k4_IBM_s100 <= 0.2
 0 <= b_k4_IBM_s101 <= 0.2
 0 <= b_k4_IBM_s102 <= 0.2
 0 <= b_k4_IBM_s103 <= 0.2
 0 <= b_k4_IBM_s104 <= 0.2
 0 <= b_k4_IBM_s105 <= 0.2
 0 <= b_k4_IBM_s106 <= 0.2
 0 <= b_k4_IBM_s107 <= 0.2
 0 <= b_k4_IBM_s110 <= 0.2
 0 <= b_k4_IBM_s111 <= 0.2
 0 <= b_k4_IBM_s112 <= 0.2
 0 <= b_k4_IBM_s113 <= 0.2
 0 <= b_k4_IBM_s114 <= 0.2
 0 <= b_k4_IBM_s115 <= 0.2
 0 <= b_k4_IBM_s116 <= 0.2
 0 <= b_k4_IBM_s117 <= 0.2
 0 <= b_k4_INTC_s0 <= 0.2
 0 <= b_k4_INTC_s1 <= 0.2
 0 <= b_k4_INTC_s00 <= 0.2
 0 <= b_k4_INTC_s01 <= 0.2
 0 <= b_k4_INTC_s10 <= 0.2
 0 <= b_k4_INTC_s11 <= 0.2
 0 <= b_k4_INTC_s000 <= 0.2
 0 <= b_k4_INTC_s001 <= 0.2
 0 <= b_k4_INTC_s002 <= 0.2
 0 <= b_k4_INTC_s003 <= 0.2
 0 <= b_k4_INTC_s004 <= 0.2
 0 <= b_k4_INTC_s005 <= 0.2
 0 <= b_k4_INTC_s006 <= 0.2
 0 <= b_k4_INTC_s007 <= 0.2
 0 <= b_k4_INTC_s010 <= 0.2
 0 <= b_k4_INTC_s011 <= 0.2
 0 <= b_k4_INTC_s012 <= 0.2
 0 <= b_k4_INTC_s013 <= 0.2
 0 <= b_k4_INTC_s014 <= 0.2
 0 <= b_k4_INTC_s015 <= 0.2
 0 <= b_k4_INTC_s016 <= 0.2
 0 <= b_k4_INTC_s017 <= 0.2
 0 <= b_k4_INTC_s100 <= 0.2
 0 <= b_k4_INTC_s101 <= 0.2
 0 <= b_k4_INTC_s102 <= 0.2
 0 <= b_k4_INTC_s103 <= 0.2
 0 <= b_k4_INTC_s104 <= 0.2
 0 <= b_k4_INTC_s105 <= 0.2
 0 <= b_k4_INTC_s106 <= 0.2
 0 <= b_k4_INTC_s107 <= 0.2
 0 <= b_k4_INTC_s110 <= 0.2
 0 <= b_k4_INTC_s111 <= 0.2
 0 <= b_k4_INTC_s112 <= 0.2
 0 <= b_k4_INTC_s113 <= 0.2
 0 <= b_k4_INTC_s114 <= 0.2
 0 <= b_k4_INTC_s115 <= 0.2
 0 <= b_k4_INTC_s116 <= 0.2
 0 <= b_k4_INTC_s117 <= 0.2
 0 <= b_k4_KO_s0 <= 0.2
 0 <= b_k4_KO_s1 <= 0.2
 0 <= b_k4_KO_s00 <= 0.2
 0 <= b_k4_KO_s01 <= 0.2
 0 <= b_k4_KO_s10 <= 0.2
 0 <= b_k4_KO_s11 <= 0.2
 0 <= b_k4_KO_s000 <= 0.2
 0 <= b_k4_KO_s001 <= 0.2
 0 <= b_k4_KO_s002 <= 0.2
 0 <= b_k4_KO_s003 <= 0.2
 0 <= b_k4_KO_s004 <= 0.2
 0 <= b_k4_KO_s005 <= 0.2
 0 <= b_k4_KO_s006 <= 0.2
 0 <= b_k4_KO_s007 <= 0.2
 0 <= b_k4_KO_s010 <= 0.2
 0 <= b_k4_KO_s011 <= 0.2
 0 <= b_k4_KO_s012 <= 0.2
 0 <= b_k4_KO_s013 <= 0.2
 0 <= b_k4_KO_s014 <= 0.2
 0 <= b_k4_KO_s015 <= 0.2
 0 <= b_k4_KO_s016 <= 0.2
 0 <= b_k4_KO_s017 <= 0.2
 0 <= b_k4_KO_s100 <= 0.2
 0 <= b_k4_KO_s101 <= 0.2
 0 <= b_k4_KO_s102 <= 0.2
 0 <= b_k4_KO_s103 <= 0.2
 0 <= b_k4_KO_s104 <= 0.2
 0 <= b_k4_KO_s105 <= 0.2
 0 <= b_k4_KO_s106 <= 0.2
 0 <= b_k4_KO_s107 <= 0.2
 0 <= b_k4_KO_s110 <= 0.2
 0 <= b_k4_KO_s111 <= 0.2
 0 <= b_k4_KO_s112 <= 0.2
 0 <= b_k4_KO_s113 <= 0.2
 0 <= b_k4_KO_s114 <= 0.2
 0 <= b_k4_KO_s115 <= 0.2
 0 <= b_k4_KO_s116 <= 0.2
 0 <= b_k4_KO_s117 <= 0.2
 0 <= b_k4_MAT_s0 <= 0.2
 0 <= b_k4_MAT_s1 <= 0.2
 0 <= b_k4_MAT_s00 <= 0.2
 0 <= b_k4_MAT_s01 <= 0.2
 0 <= b_k4_MAT_s10 <= 0.2
 0 <= b_k4_MAT_s11 <= 0.2
 0 <= b_k4_MAT_s000 <= 0.2
 0 <= b_k4_MAT_s001 <= 0.2
 0 <= b_k4_MAT_s002 <= 0.2
 0 <= b_k4_MAT_s003 <= 0.2
 0 <= b_k4_MAT_s004 <= 0.2
 0 <= b_k4_MAT_s005 <= 0.2
 0 <= b_k4_MAT_s006 <= 0.2
 0 <= b_k4_MAT_s007 <= 0.2
 0 <= b_k4_MAT_s010 <= 0.2
 0 <= b_k4_MAT_s011 <= 0.2
 0 <= b_k4_MAT_s012 <= 0.2
 0 <= b_k4_MAT_s013 <= 0.2
 0 <= b_k4_MAT_s014 <= 0.2
 0 <= b_k4_MAT_s015 <= 0.2
 0 <= b_k4_MAT_s016 <= 0.2
 0 <= b_k4_MAT_s017 <= 0.2
 0 <= b_k4_MAT_s100 <= 0.2
 0 <= b_k4_MAT_s101 <= 0.2
 0 <= b_k4_MAT_s102 <= 0.2
 0 <= b_k4_MAT_s103 <= 0.2
 0 <= b_k4_MAT_s104 <= 0.2
 0 <= b_k4_MAT_s105 <= 0.2
 0 <= b_k4_MAT_s106 <= 0.2
 0 <= b_k4_MAT_s107 <= 0.2
 0 <= b_k4_MAT_s110 <= 0.2
 0 <= b_k4_MAT_s111 <= 0.2
 0 <= b_k4_MAT_s112 <= 0.2
 0 <= b_k4_MAT_s113 <= 0.2
 0 <= b_k4_MAT_s114 <= 0.2
 0 <= b_k4_MAT_s115 <= 0.2
 0 <= b_k4_MAT_s116 <= 0.2
 0 <= b_k4_MAT_s117 <= 0.2
 0 <= b_k4_MSFT_s0 <= 0.2
 0 <= b_k4_MSFT_s1 <= 0.2
 0 <= b_k4_MSFT_s00 <= 0.2
 0 <= b_k4_MSFT_s01 <= 0.2
 0 <= b_k4_MSFT_s10 <= 0.2
 0 <= b_k4_MSFT_s11 <= 0.2
 0 <= b_k4_MSFT_s000 <= 0.2
 0 <= b_k4_MSFT_s001 <= 0.2
 0 <= b_k4_MSFT_s002 <= 0.2
 0 <= b_k4_MSFT_s003 <= 0.2
 0 <= b_k4_MSFT_s004 <= 0.2
 0 <= b_k4_MSFT_s005 <= 0.2
 0 <= b_k4_MSFT_s006 <= 0.2
 0 <= b_k4_MSFT_s007 <= 0.2
 0 <= b_k4_MSFT_s010 <= 0.2
 0 <= b_k4_MSFT_s011 <= 0.2
 0 <= b_k4_MSFT_s012 <= 0.2
 0 <= b_k4_MSFT_s013 <= 0.2
 0 <= b_k4_MSFT_s014 <= 0.2
 0 <= b_k4_MSFT_s015 <= 0.2
 0 <= b_k4_MSFT_s016 <= 0.2
 0 <= b_k4_MSFT_s017 <= 0.2
 0 <= b_k4_MSFT_s100 <= 0.2
 0 <= b_k4_MSFT_s101 <= 0.2
 0 <= b_k4_MSFT_s102 <= 0.2
 0 <= b_k4_MSFT_s103 <= 0.2
 0 <= b_k4_MSFT_s104 <= 0.2
 0 <= b_k4_MSFT_s105 <= 0.2
 0 <= b_k4_MSFT_s106 <= 0.2
 0 <= b_k4_MSFT_s107 <= 0.2
 0 <= b_k4_MSFT_s110 <= 0.2
 0 <= b_k4_MSFT_s111 <= 0.2
 0 <= b_k4_MSFT_s112 <= 0.2
 0 <= b_k4_MSFT_s113 <= 0.2
 0 <= b_k4_MSFT_s114 <= 0.2
 0 <= b_k4_MSFT_s115 <= 0.2
 0 <= b_k4_MSFT_s116 <= 0.2
 0 <= b_k4_MSFT_s117 <= 0.2
 0 <= b_k4_PG_s0 <= 0.2
 0 <= b_k4_PG_s1 <= 0.2
 0 <= b_k4_PG_s00 <= 0.2
 0 <= b_k4_PG_s01 <= 0.2
 0 <= b_k4_PG_s10 <= 0.2
 0 <= b_k4_PG_s11 <= 0.2
 0 <= b_k4_PG_s000 <= 0.2
 0 <= b_k4_PG_s001 <= 0.2
 0 <= b_k4_PG_s002 <= 0.2
 0 <= b_k4_PG_s003 <= 0.2
 0 <= b_k4_PG_s004 <= 0.2
 0 <= b_k4_PG_s005 <= 0.2
 0 <= b_k4_PG_s006 <= 0.2
 0 <= b_k4_PG_s007 <= 0.2
 0 <= b_k4_PG_s010 <= 0.2
 0 <= b_k4_PG_s011 <= 0.2
 0 <= b_k4_PG_s012 <= 0.2
 0 <= b_k4_PG_s013 <= 0.2
 0 <= b_k4_PG_s014 <= 0.2
 0 <= b_k4_PG_s015 <= 0.2
 0 <= b_k4_PG_s016 <= 0.2
 0 <= b_k4_PG_s017 <= 0.2
 0 <= b_k4_PG_s100 <= 0.2
 0 <= b_k4_PG_s101 <= 0.2
 0 <= b_k4_PG_s102 <= 0.2
 0 <= b_k4_PG_s103 <= 0.2
 0 <= b_k4_PG_s104 <= 0.2
 0 <= b_k4_PG_s105 <= 0.2
 0 <= b_k4_PG_s106 <= 0.2
 0 <= b_k4_PG_s107 <= 0.2
 0 <= b_k4_PG_s110 <= 0.2
 0 <= b_k4_PG_s111 <= 0.2
 0 <= b_k4_PG_s112 <= 0.2
 0 <= b_k4_PG_s113 <= 0.2
 0 <= b_k4_PG_s114 <= 0.2
 0 <= b_k4_PG_s115 <= 0.2
 0 <= b_k4_PG_s116 <= 0.2
 0 <= b_k4_PG_s117 <= 0.2
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
 0 <= s_k4_ADM_s0 <= 0.2
 0 <= s_k4_ADM_s1 <= 0.2
 0 <= s_k4_ADM_s00 <= 0.2
 0 <= s_k4_ADM_s01 <= 0.2
 0 <= s_k4_ADM_s10 <= 0.2
 0 <= s_k4_ADM_s11 <= 0.2
 0 <= s_k4_ADM_s000 <= 0.2
 0 <= s_k4_ADM_s001 <= 0.2
 0 <= s_k4_ADM_s002 <= 0.2
 0 <= s_k4_ADM_s003 <= 0.2
 0 <= s_k4_ADM_s004 <= 0.2
 0 <= s_k4_ADM_s005 <= 0.2
 0 <= s_k4_ADM_s006 <= 0.2
 0 <= s_k4_ADM_s007 <= 0.2
 0 <= s_k4_ADM_s010 <= 0.2
 0 <= s_k4_ADM_s011 <= 0.2
 0 <= s_k4_ADM_s012 <= 0.2
 0 <= s_k4_ADM_s013 <= 0.2
 0 <= s_k4_ADM_s014 <= 0.2
 0 <= s_k4_ADM_s015 <= 0.2
 0 <= s_k4_ADM_s016 <= 0.2
 0 <= s_k4_ADM_s017 <= 0.2
 0 <= s_k4_ADM_s100 <= 0.2
 0 <= s_k4_ADM_s101 <= 0.2
 0 <= s_k4_ADM_s102 <= 0.2
 0 <= s_k4_ADM_s103 <= 0.2
 0 <= s_k4_ADM_s104 <= 0.2
 0 <= s_k4_ADM_s105 <= 0.2
 0 <= s_k4_ADM_s106 <= 0.2
 0 <= s_k4_ADM_s107 <= 0.2
 0 <= s_k4_ADM_s110 <= 0.2
 0 <= s_k4_ADM_s111 <= 0.2
 0 <= s_k4_ADM_s112 <= 0.2
 0 <= s_k4_ADM_s113 <= 0.2
 0 <= s_k4_ADM_s114 <= 0.2
 0 <= s_k4_ADM_s115 <= 0.2
 0 <= s_k4_ADM_s116 <= 0.2
 0 <= s_k4_ADM_s117 <= 0.2
 0 <= s_k4_AXP_s0 <= 0.2
 0 <= s_k4_AXP_s1 <= 0.2
 0 <= s_k4_AXP_s00 <= 0.2
 0 <= s_k4_AXP_s01 <= 0.2
 0 <= s_k4_AXP_s10 <= 0.2
 0 <= s_k4_AXP_s11 <= 0.2
 0 <= s_k4_AXP_s000 <= 0.2
 0 <= s_k4_AXP_s001 <= 0.2
 0 <= s_k4_AXP_s002 <= 0.2
 0 <= s_k4_AXP_s003 <= 0.2
 0 <= s_k4_AXP_s004 <= 0.2
 0 <= s_k4_AXP_s005 <= 0.2
 0 <= s_k4_AXP_s006 <= 0.2
 0 <= s_k4_AXP_s007 <= 0.2
 0 <= s_k4_AXP_s010 <= 0.2
 0 <= s_k4_AXP_s011 <= 0.2
 0 <= s_k4_AXP_s012 <= 0.2
 0 <= s_k4_AXP_s013 <= 0.2
 0 <= s_k4_AXP_s014 <= 0.2
 0 <= s_k4_AXP_s015 <= 0.2
 0 <= s_k4_AXP_s016 <= 0.2
 0 <= s_k4_AXP_s017 <= 0.2
 0 <= s_k4_AXP_s100 <= 0.2
 0 <= s_k4_AXP_s101 <= 0.2
 0 <= s_k4_AXP_s102 <= 0.2
 0 <= s_k4_AXP_s103 <= 0.2
 0 <= s_k4_AXP_s104 <= 0.2
 0 <= s_k4_AXP_s105 <= 0.2
 0 <= s_k4_AXP_s106 <= 0.2
 0 <= s_k4_AXP_s107 <= 0.2
 0 <= s_k4_AXP_s110 <= 0.2
 0 <= s_k4_AXP_s111 <= 0.2
 0 <= s_k4_AXP_s112 <= 0.2
 0 <= s_k4_AXP_s113 <= 0.2
 0 <= s_k4_AXP_s114 <= 0.2
 0 <= s_k4_AXP_s115 <= 0.2
 0 <= s_k4_AXP_s116 <= 0.2
 0 <= s_k4_AXP_s117 <= 0.2
 0 <= s_k4_DIS_s0 <= 0.2
 0 <= s_k4_DIS_s1 <= 0.2
 0 <= s_k4_DIS_s00 <= 0.2
 0 <= s_k4_DIS_s01 <= 0.2
 0 <= s_k4_DIS_s10 <= 0.2
 0 <= s_k4_DIS_s11 <= 0.2
 0 <= s_k4_DIS_s000 <= 0.2
 0 <= s_k4_DIS_s001 <= 0.2
 0 <= s_k4_DIS_s002 <= 0.2
 0 <= s_k4_DIS_s003 <= 0.2
 0 <= s_k4_DIS_s004 <= 0.2
 0 <= s_k4_DIS_s005 <= 0.2
 0 <= s_k4_DIS_s006 <= 0.2
 0 <= s_k4_DIS_s007 <= 0.2
 0 <= s_k4_DIS_s010 <= 0.2
 0 <= s_k4_DIS_s011 <= 0.2
 0 <= s_k4_DIS_s012 <= 0.2
 0 <= s_k4_DIS_s013 <= 0.2
 0 <= s_k4_DIS_s014 <= 0.2
 0 <= s_k4_DIS_s015 <= 0.2
 0 <= s_k4_DIS_s016 <= 0.2
 0 <= s_k4_DIS_s017 <= 0.2
 0 <= s_k4_DIS_s100 <= 0.2
 0 <= s_k4_DIS_s101 <= 0.2
 0 <= s_k4_DIS_s102 <= 0.2
 0 <= s_k4_DIS_s103 <= 0.2
 0 <= s_k4_DIS_s104 <= 0.2
 0 <= s_k4_DIS_s105 <= 0.2
 0 <= s_k4_DIS_s106 <= 0.2
 0 <= s_k4_DIS_s107 <= 0.2
 0 <= s_k4_DIS_s110 <= 0.2
 0 <= s_k4_DIS_s111 <= 0.2
 0 <= s_k4_DIS_s112 <= 0.2
 0 <= s_k4_DIS_s113 <= 0.2
 0 <= s_k4_DIS_s114 <= 0.2
 0 <= s_k4_DIS_s115 <= 0.2
 0 <= s_k4_DIS_s116 <= 0.2
 0 <= s_k4_DIS_s117 <= 0.2
 0 <= s_k4_FDX_s0 <= 0.2
 0 <= s_k4_FDX_s1 <= 0.2
 0 <= s_k4_FDX_s00 <= 0.2
 0 <= s_k4_FDX_s01 <= 0.2
 0 <= s_k4_FDX_s10 <= 0.2
 0 <= s_k4_FDX_s11 <= 0.2
 0 <= s_k4_FDX_s000 <= 0.2
 0 <= s_k4_FDX_s001 <= 0.2
 0 <= s_k4_FDX_s002 <= 0.2
 0 <= s_k4_FDX_s003 <= 0.2
 0 <= s_k4_FDX_s004 <= 0.2
 0 <= s_k4_FDX_s005 <= 0.2
 0 <= s_k4_FDX_s006 <= 0.2
 0 <= s_k4_FDX_s007 <= 0.2
 0 <= s_k4_FDX_s010 <= 0.2
 0 <= s_k4_FDX_s011 <= 0.2
 0 <= s_k4_FDX_s012 <= 0.2
 0 <= s_k4_FDX_s013 <= 0.2
 0 <= s_k4_FDX_s014 <= 0.2
 0 <= s_k4_FDX_s015 <= 0.2
 0 <= s_k4_FDX_s016 <= 0.2
 0 <= s_k4_FDX_s017 <= 0.2
 0 <= s_k4_FDX_s100 <= 0.2
 0 <= s_k4_FDX_s101 <= 0.2
 0 <= s_k4_FDX_s102 <= 0.2
 0 <= s_k4_FDX_s103 <= 0.2
 0 <= s_k4_FDX_s104 <= 0.2
 0 <= s_k4_FDX_s105 <= 0.2
 0 <= s_k4_FDX_s106 <= 0.2
 0 <= s_k4_FDX_s107 <= 0.2
 0 <= s_k4_FDX_s110 <= 0.2
 0 <= s_k4_FDX_s111 <= 0.2
 0 <= s_k4_FDX_s112 <= 0.2
 0 <= s_k4_FDX_s113 <= 0.2
 0 <= s_k4_FDX_s114 <= 0.2
 0 <= s_k4_FDX_s115 <= 0.2
 0 <= s_k4_FDX_s116 <= 0.2
 0 <= s_k4_FDX_s117 <= 0.2
 0 <= s_k4_IBM_s0 <= 0.2
 0 <= s_k4_IBM_s1 <= 0.2
 0 <= s_k4_IBM_s00 <= 0.2
 0 <= s_k4_IBM_s01 <= 0.2
 0 <= s_k4_IBM_s10 <= 0.2
 0 <= s_k4_IBM_s11 <= 0.2
 0 <= s_k4_IBM_s000 <= 0.2
 0 <= s_k4_IBM_s001 <= 0.2
 0 <= s_k4_IBM_s002 <= 0.2
 0 <= s_k4_IBM_s003 <= 0.2
 0 <= s_k4_IBM_s004 <= 0.2
 0 <= s_k4_IBM_s005 <= 0.2
 0 <= s_k4_IBM_s006 <= 0.2
 0 <= s_k4_IBM_s007 <= 0.2
 0 <= s_k4_IBM_s010 <= 0.2
 0 <= s_k4_IBM_s011 <= 0.2
 0 <= s_k4_IBM_s012 <= 0.2
 0 <= s_k4_IBM_s013 <= 0.2
 0 <= s_k4_IBM_s014 <= 0.2
 0 <= s_k4_IBM_s015 <= 0.2
 0 <= s_k4_IBM_s016 <= 0.2
 0 <= s_k4_IBM_s017 <= 0.2
 0 <= s_k4_IBM_s100 <= 0.2
 0 <= s_k4_IBM_s101 <= 0.2
 0 <= s_k4_IBM_s102 <= 0.2
 0 <= s_k4_IBM_s103 <= 0.2
 0 <= s_k4_IBM_s104 <= 0.2
 0 <= s_k4_IBM_s105 <= 0.2
 0 <= s_k4_IBM_s106 <= 0.2
 0 <= s_k4_IBM_s107 <= 0.2
 0 <= s_k4_IBM_s110 <= 0.2
 0 <= s_k4_IBM_s111 <= 0.2
 0 <= s_k4_IBM_s112 <= 0.2
 0 <= s_k4_IBM_s113 <= 0.2
 0 <= s_k4_IBM_s114 <= 0.2
 0 <= s_k4_IBM_s115 <= 0.2
 0 <= s_k4_IBM_s116 <= 0.2
 0 <= s_k4_IBM_s117 <= 0.2
 0 <= s_k4_INTC_s0 <= 0.2
 0 <= s_k4_INTC_s1 <= 0.2
 0 <= s_k4_INTC_s00 <= 0.2
 0 <= s_k4_INTC_s01 <= 0.2
 0 <= s_k4_INTC_s10 <= 0.2
 0 <= s_k4_INTC_s11 <= 0.2
 0 <= s_k4_INTC_s000 <= 0.2
 0 <= s_k4_INTC_s001 <= 0.2
 0 <= s_k4_INTC_s002 <= 0.2
 0 <= s_k4_INTC_s003 <= 0.2
 0 <= s_k4_INTC_s004 <= 0.2
 0 <= s_k4_INTC_s005 <= 0.2
 0 <= s_k4_INTC_s006 <= 0.2
 0 <= s_k4_INTC_s007 <= 0.2
 0 <= s_k4_INTC_s010 <= 0.2
 0 <= s_k4_INTC_s011 <= 0.2
 0 <= s_k4_INTC_s012 <= 0.2
 0 <= s_k4_INTC_s013 <= 0.2
 0 <= s_k4_INTC_s014 <= 0.2
 0 <= s_k4_INTC_s015 <= 0.2
 0 <= s_k4_INTC_s016 <= 0.2
 0 <= s_k4_INTC_s017 <= 0.2
 0 <= s_k4_INTC_s100 <= 0.2
 0 <= s_k4_INTC_s101 <= 0.2
 0 <= s_k4_INTC_s102 <= 0.2
 0 <= s_k4_INTC_s103 <= 0.2
 0 <= s_k4_INTC_s104 <= 0.2
 0 <= s_k4_INTC_s105 <= 0.2
 0 <= s_k4_INTC_s106 <= 0.2
 0 <= s_k4_INTC_s107 <= 0.2
 0 <= s_k4_INTC_s110 <= 0.2
 0 <= s_k4_INTC_s111 <= 0.2
 0 <= s_k4_INTC_s112 <= 0.2
 0 <= s_k4_INTC_s113 <= 0.2
 0 <= s_k4_INTC_s114 <= 0.2
 0 <= s_k4_INTC_s115 <= 0.2
 0 <= s_k4_INTC_s116 <= 0.2
 0 <= s_k4_INTC_s117 <= 0.2
 0 <= s_k4_KO_s0 <= 0.2
 0 <= s_k4_KO_s1 <= 0.2
 0 <= s_k4_KO_s00 <= 0.2
 0 <= s_k4_KO_s01 <= 0.2
 0 <= s_k4_KO_s10 <= 0.2
 0 <= s_k4_KO_s11 <= 0.2
 0 <= s_k4_KO_s000 <= 0.2
 0 <= s_k4_KO_s001 <= 0.2
 0 <= s_k4_KO_s002 <= 0.2
 0 <= s_k4_KO_s003 <= 0.2
 0 <= s_k4_KO_s004 <= 0.2
 0 <= s_k4_KO_s005 <= 0.2
 0 <= s_k4_KO_s006 <= 0.2
 0 <= s_k4_KO_s007 <= 0.2
 0 <= s_k4_KO_s010 <= 0.2
 0 <= s_k4_KO_s011 <= 0.2
 0 <= s_k4_KO_s012 <= 0.2
 0 <= s_k4_KO_s013 <= 0.2
 0 <= s_k4_KO_s014 <= 0.2
 0 <= s_k4_KO_s015 <= 0.2
 0 <= s_k4_KO_s016 <= 0.2
 0 <= s_k4_KO_s017 <= 0.2
 0 <= s_k4_KO_s100 <= 0.2
 0 <= s_k4_KO_s101 <= 0.2
 0 <= s_k4_KO_s102 <= 0.2
 0 <= s_k4_KO_s103 <= 0.2
 0 <= s_k4_KO_s104 <= 0.2
 0 <= s_k4_KO_s105 <= 0.2
 0 <= s_k4_KO_s106 <= 0.2
 0 <= s_k4_KO_s107 <= 0.2
 0 <= s_k4_KO_s110 <= 0.2
 0 <= s_k4_KO_s111 <= 0.2
 0 <= s_k4_KO_s112 <= 0.2
 0 <= s_k4_KO_s113 <= 0.2
 0 <= s_k4_KO_s114 <= 0.2
 0 <= s_k4_KO_s115 <= 0.2
 0 <= s_k4_KO_s116 <= 0.2
 0 <= s_k4_KO_s117 <= 0.2
 0 <= s_k4_MAT_s0 <= 0.2
 0 <= s_k4_MAT_s1 <= 0.2
 0 <= s_k4_MAT_s00 <= 0.2
 0 <= s_k4_MAT_s01 <= 0.2
 0 <= s_k4_MAT_s10 <= 0.2
 0 <= s_k4_MAT_s11 <= 0.2
 0 <= s_k4_MAT_s000 <= 0.2
 0 <= s_k4_MAT_s001 <= 0.2
 0 <= s_k4_MAT_s002 <= 0.2
 0 <= s_k4_MAT_s003 <= 0.2
 0 <= s_k4_MAT_s004 <= 0.2
 0 <= s_k4_MAT_s005 <= 0.2
 0 <= s_k4_MAT_s006 <= 0.2
 0 <= s_k4_MAT_s007 <= 0.2
 0 <= s_k4_MAT_s010 <= 0.2
 0 <= s_k4_MAT_s011 <= 0.2
 0 <= s_k4_MAT_s012 <= 0.2
 0 <= s_k4_MAT_s013 <= 0.2
 0 <= s_k4_MAT_s014 <= 0.2
 0 <= s_k4_MAT_s015 <= 0.2
 0 <= s_k4_MAT_s016 <= 0.2
 0 <= s_k4_MAT_s017 <= 0.2
 0 <= s_k4_MAT_s100 <= 0.2
 0 <= s_k4_MAT_s101 <= 0.2
 0 <= s_k4_MAT_s102 <= 0.2
 0 <= s_k4_MAT_s103 <= 0.2
 0 <= s_k4_MAT_s104 <= 0.2
 0 <= s_k4_MAT_s105 <= 0.2
 0 <= s_k4_MAT_s106 <= 0.2
 0 <= s_k4_MAT_s107 <= 0.2
 0 <= s_k4_MAT_s110 <= 0.2
 0 <= s_k4_MAT_s111 <= 0.2
 0 <= s_k4_MAT_s112 <= 0.2
 0 <= s_k4_MAT_s113 <= 0.2
 0 <= s_k4_MAT_s114 <= 0.2
 0 <= s_k4_MAT_s115 <= 0.2
 0 <= s_k4_MAT_s116 <= 0.2
 0 <= s_k4_MAT_s117 <= 0.2
 0 <= s_k4_MSFT_s0 <= 0.2
 0 <= s_k4_MSFT_s1 <= 0.2
 0 <= s_k4_MSFT_s00 <= 0.2
 0 <= s_k4_MSFT_s01 <= 0.2
 0 <= s_k4_MSFT_s10 <= 0.2
 0 <= s_k4_MSFT_s11 <= 0.2
 0 <= s_k4_MSFT_s000 <= 0.2
 0 <= s_k4_MSFT_s001 <= 0.2
 0 <= s_k4_MSFT_s002 <= 0.2
 0 <= s_k4_MSFT_s003 <= 0.2
 0 <= s_k4_MSFT_s004 <= 0.2
 0 <= s_k4_MSFT_s005 <= 0.2
 0 <= s_k4_MSFT_s006 <= 0.2
 0 <= s_k4_MSFT_s007 <= 0.2
 0 <= s_k4_MSFT_s010 <= 0.2
 0 <= s_k4_MSFT_s011 <= 0.2
 0 <= s_k4_MSFT_s012 <= 0.2
 0 <= s_k4_MSFT_s013 <= 0.2
 0 <= s_k4_MSFT_s014 <= 0.2
 0 <= s_k4_MSFT_s015 <= 0.2
 0 <= s_k4_MSFT_s016 <= 0.2
 0 <= s_k4_MSFT_s017 <= 0.2
 0 <= s_k4_MSFT_s100 <= 0.2
 0 <= s_k4_MSFT_s101 <= 0.2
 0 <= s_k4_MSFT_s102 <= 0.2
 0 <= s_k4_MSFT_s103 <= 0.2
 0 <= s_k4_MSFT_s104 <= 0.2
 0 <= s_k4_MSFT_s105 <= 0.2
 0 <= s_k4_MSFT_s106 <= 0.2
 0 <= s_k4_MSFT_s107 <= 0.2
 0 <= s_k4_MSFT_s110 <= 0.2
 0 <= s_k4_MSFT_s111 <= 0.2
 0 <= s_k4_MSFT_s112 <= 0.2
 0 <= s_k4_MSFT_s113 <= 0.2
 0 <= s_k4_MSFT_s114 <= 0.2
 0 <= s_k4_MSFT_s115 <= 0.2
 0 <= s_k4_MSFT_s116 <= 0.2
 0 <= s_k4_MSFT_s117 <= 0.2
 0 <= s_k4_PG_s0 <= 0.2
 0 <= s_k4_PG_s1 <= 0.2
 0 <= s_k4_PG_s00 <= 0.2
 0 <= s_k4_PG_s01 <= 0.2
 0 <= s_k4_PG_s10 <= 0.2
 0 <= s_k4_PG_s11 <= 0.2
 0 <= s_k4_PG_s000 <= 0.2
 0 <= s_k4_PG_s001 <= 0.2
 0 <= s_k4_PG_s002 <= 0.2
 0 <= s_k4_PG_s003 <= 0.2
 0 <= s_k4_PG_s004 <= 0.2
 0 <= s_k4_PG_s005 <= 0.2
 0 <= s_k4_PG_s006 <= 0.2
 0 <= s_k4_PG_s007 <= 0.2
 0 <= s_k4_PG_s010 <= 0.2
 0 <= s_k4_PG_s011 <= 0.2
 0 <= s_k4_PG_s012 <= 0.2
 0 <= s_k4_PG_s013 <= 0.2
 0 <= s_k4_PG_s014 <= 0.2
 0 <= s_k4_PG_s015 <= 0.2
 0 <= s_k4_PG_s016 <= 0.2
 0 <= s_k4_PG_s017 <= 0.2
 0 <= s_k4_PG_s100 <= 0.2
 0 <= s_k4_PG_s101 <= 0.2
 0 <= s_k4_PG_s102 <= 0.2
 0 <= s_k4_PG_s103 <= 0.2
 0 <= s_k4_PG_s104 <= 0.2
 0 <= s_k4_PG_s105 <= 0.2
 0 <= s_k4_PG_s106 <= 0.2
 0 <= s_k4_PG_s107 <= 0.2
 0 <= s_k4_PG_s110 <= 0.2
 0 <= s_k4_PG_s111 <= 0.2
 0 <= s_k4_PG_s112 <= 0.2
 0 <= s_k4_PG_s113 <= 0.2
 0 <= s_k4_PG_s114 <= 0.2
 0 <= s_k4_PG_s115 <= 0.2
 0 <= s_k4_PG_s116 <= 0.2
 0 <= s_k4_PG_s117 <= 0.2
End
