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
 w_k1_ADM_s0_return_constraint:  1.20363 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.92218 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.14623 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.822895 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.20029 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.979828 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.24762 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.02049 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.891353 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.06165 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.920791 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.661486 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.902613 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.810302 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.43292 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.04076 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.930223 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.981761 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.24267 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.21783 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.02469 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.04013 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.24136 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.05248 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.893156 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.796719 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.938319 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.87364 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.16756 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.20971 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.25234 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.97401 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.693839 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.01087 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.00524 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.0116 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.913856 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.687552 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.27385 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.912147 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.26678 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.820359 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.25438 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.887397 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.64484 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.08376 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.10705 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.25856 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.827575 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.643619 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.955164 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.733053 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.39079 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.764087 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.88918 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.06419 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.4168 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.41471 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.958819 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.27468 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.59047 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.42613 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.08032 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.670769 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.992741 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.946256 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.09463 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.46109 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.49915 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.0028 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.502724 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.23683 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.988472 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.1289 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.807015 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.698695 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.38346 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.89636 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.24469 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.77771 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.32012 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.808086 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.77803 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.03657 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.22703 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.52504 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.966497 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.558025 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.892633 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.747309 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.2443 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.93493 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.798795 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.12053 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.42181 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.59064 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.706571 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.18597 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.69821 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.7366 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.03619 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.626791 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.04004 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.832152 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.15721 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.40279 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.59394 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.04337 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.463441 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.6833 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.901522 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.15555 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.769941 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.724127 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.20993 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.916614 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.15747 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.84301 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.22875 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.895518 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.42965 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.2003 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.02412 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.33233 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.894895 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.688062 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.815073 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.899204 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.21508 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.00548 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.847608 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.982025 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.17644 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.34669 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.926235 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.29399 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.59238 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.31569 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.829569 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.777772 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.09741 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.878197 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.07372 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.18633 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.2569 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.913698 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.61207 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.40002 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.02443 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.11541 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.946204 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.786679 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.24258 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.954113 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.14577 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.816597 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.1648 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.891793 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.55895 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.737836 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.30788 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.865281 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.11334 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.902021 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.77113 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.703502 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.02503 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.669597 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.23665 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.21021 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.84841 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.22455 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.687626 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.767628 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.86096 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.21352 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.75006 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.696778 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.745271 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.00856 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.18242 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.60477 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.97627 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.59437 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.707286 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.904519 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.767505 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.06899 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.499421 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.907082 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.19262 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.887111 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.13899 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.844841 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.16611 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.01716 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.16143 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.04482 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.0968 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.26554 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.812443 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.664562 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.794681 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.833622 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.19167 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.00321 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.952261 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.95609 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.08317 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.46448 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.893943 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.12906 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.56288 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.19775 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.97461 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.844261 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.08495 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.05306 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.04283 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.28843 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.11091 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.937092 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.712338 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.28116 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.982702 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.999619 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.907785 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.737493 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.24784 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.895181 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.12155 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.885646 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.23229 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.901011 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.48129 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.876446 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.20319 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.25277 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.00276 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.75975 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.1619 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.777771 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.06367 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.809079 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.898396 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.06018 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.43254 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.19459 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.707109 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.831169 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.2136 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.31109 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.25357 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.74565 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.934766 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.905508 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.23511 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.39467 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.63183 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.15937 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.662089 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.24947 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.756015 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.15383 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.676974 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.835279 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.42923 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.947621 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.954123 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.673036 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.32592 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.807033 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.9625 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.945125 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.70834 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.26711 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.1232 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.984698 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.94829 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.684013 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.01598 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.575048 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.11449 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.24043 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 2.13832 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.29229 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.746212 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.06597 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.891115 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.3322 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.51714 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.587662 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.627144 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.790504 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.13519 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.50727 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 2.04854 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.4458 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.609767 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.05558 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.81797 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.11129 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.474246 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.905814 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.19439 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.927332 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.08182 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.844803 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.23374 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.93712 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.27308 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.20706 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.17554 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.41922 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.87827 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.644946 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.908826 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.933438 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.11831 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.01927 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.824791 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.02709 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.14677 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.38244 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.858915 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.13032 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.30771 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.19394 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.836561 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.81002 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.986513 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.806118 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.06407 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.05209 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.25357 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.902945 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.664552 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.42599 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.905546 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.0415 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.836734 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.766518 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.14971 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.922969 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.13108 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.87636 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.16389 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.950254 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.23538 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.967287 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.00811 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.1037 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.974174 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.753447 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.904072 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.805988 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.14435 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.906398 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.949122 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.08382 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.236 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.18323 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.915588 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.11469 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.22066 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.18776 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.03266 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.789229 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.990594 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.939366 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.09601 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.23098 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.31922 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.10841 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.723958 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.1338 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.922015 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.10105 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.822678 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.895141 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.096000112330539 z_k1_s0000
                                 - 0.0760001159075322 z_k1_s0001
                                 - 0.222000737296014 z_k1_s0002
                                 - 0.179999967807061 z_k1_s0003
                                 - 0.126000106965049 z_k1_s0004
                                 - 0.214999961547323 z_k1_s0005
                                 - 0.200999705051006 z_k1_s0006
                                 - 0.179000226986124 z_k1_s0007
                                 - 0.16099975338493 z_k1_s0010
                                 - 0.420998971011857 z_k1_s0011
                                 - 0.350998681366578 z_k1_s0012
                                 - 0.24300084248304 z_k1_s0013
                                 - 0.485999402968048 z_k1_s0014
                                 - 0.114999921646295 z_k1_s0015
                                 - 0.325999392921132 z_k1_s0016
                                 - 0.178999925665061 z_k1_s0017
                                 - 0.311998959513517 z_k1_s0020
                                 - 0.409998300002808 z_k1_s0021
                                 - 0.254998411027848 z_k1_s0022
                                 - 0.123999703580636 z_k1_s0023
                                 - 0.288999029562221 z_k1_s0024
                                 - 0.306999967607865 z_k1_s0025
                                 - 0.248998790343739 z_k1_s0026
                                 - 0.129999531864077 z_k1_s0027
                                 - 0.182000240420227 z_k1_s0030
                                 - 0.247000186570071 z_k1_s0031
                                 - 0.399001204153183 z_k1_s0032
                                 - 0.294000990218309 z_k1_s0033
                                 - 0.213000936954453 z_k1_s0034
                                 - 0.215999490035844 z_k1_s0035
                                 - 0.210999945563544 z_k1_s0036
                                 - 0.194000320755735 z_k1_s0037
                                 - 0.142000091508443 z_k1_s0040
                                 - 0.567999287632603 z_k1_s0041
                                 - 0.21200018977886 z_k1_s0042
                                 - 0.239000389443339 z_k1_s0043
                                 - 0.264000154939624 z_k1_s0044
                                 - 0.47200001558268 z_k1_s0045
                                 - 0.490001227093502 z_k1_s0046
                                 - 0.308999678912879 z_k1_s0047
                                 - 0.191000888720889 z_k1_s0050
                                 - 0.187999835331429 z_k1_s0051
                                 - 0.233000246155949 z_k1_s0052
                                 - 0.0920001388071171 z_k1_s0053
                                 - 0.358001045334787 z_k1_s0054
                                 - 0.174999887648982 z_k1_s0055
                                 - 0.135000155172345 z_k1_s0056
                                 - 0.167000156613898 z_k1_s0057
                                 - 0.147000066068319 z_k1_s0060
                                 - 0.112000050337767 z_k1_s0061
                                 - 0.185999883596558 z_k1_s0062
                                 - 0.25100051281071 z_k1_s0063
                                 - 0.0840000377533249 z_k1_s0064
                                 - 0.319999543821921 z_k1_s0065
                                 - 0.0770000346072145 z_k1_s0066
                                 - 0.223000500226269 z_k1_s0067
                                 - 0.181000323283127 z_k1_s0070
                                 - 0.587999816940746 z_k1_s0071
                                 - 0.62400110981586 z_k1_s0072
                                 - 0.315000449647329 z_k1_s0073
                                 - 0.733998670261763 z_k1_s0074
                                 - 0.355000182173677 z_k1_s0075
                                 - 0.658000200862634 z_k1_s0076
                                 - 0.805000155097821 z_k1_s0077
                                 - 0.120000340901154 z_k1_s0100
                                 - 0.127000271620506 z_k1_s0101
                                 - 0.0809998556087756 z_k1_s0102
                                 - 0.154999994497915 z_k1_s0103
                                 - 0.167999407263034 z_k1_s0104
                                 - 0.160000097868678 z_k1_s0105
                                 - 0.0879998933279859 z_k1_s0106
                                 - 0.170999683285209 z_k1_s0107
                                 - 0.297998765029419 z_k1_s0110
                                 - 0.501997728627221 z_k1_s0111
                                 - 0.349998329162624 z_k1_s0112
                                 - 0.302000246958929 z_k1_s0113
                                 - 0.22599835823214 z_k1_s0114
                                 - 0.10199970069835 z_k1_s0115
                                 - 0.240999537478818 z_k1_s0116
                                 - 0.167999635796958 z_k1_s0117
                                 - 0.747997532804723 z_k1_s0120
                                 - 0.693999594433725 z_k1_s0121
                                 - 0.86499743743552 z_k1_s0122
                                 - 0.257999684014219 z_k1_s0123
                                 - 0.174999633174595 z_k1_s0124
                                 - 0.457998890085731 z_k1_s0125
                                 - 0.182999841225107 z_k1_s0126
                                 - 0.365999307750759 z_k1_s0127
                                 - 0.140999881254326 z_k1_s0130
                                 - 0.155999917864326 z_k1_s0131
                                 - 0.363000759722088 z_k1_s0132
                                 - 0.361999256429147 z_k1_s0133
                                 - 0.195999875224425 z_k1_s0134
                                 - 0.417999280893226 z_k1_s0135
                                 - 0.144999792830396 z_k1_s0136
                                 - 0.32299953996288 z_k1_s0137
                                 - 0.148999979073575 z_k1_s0140
                                 - 0.0730000156695948 z_k1_s0141
                                 - 0.219000274208704 z_k1_s0142
                                 - 0.0629999248217366 z_k1_s0143
                                 - 0.184000126641139 z_k1_s0144
                                 - 0.185999485930945 z_k1_s0145
                                 - 0.0849999202070972 z_k1_s0146
                                 - 0.177000097127627 z_k1_s0147
                                 - 0.134000013595491 z_k1_s0150
                                 - 0.101999966862269 z_k1_s0151
                                 - 0.164000360907677 z_k1_s0152
                                 - 0.102999937972703 z_k1_s0153
                                 - 0.175000407322201 z_k1_s0154
                                 - 0.17999965587479 z_k1_s0155
                                 - 0.105999972703922 z_k1_s0156
                                 - 0.250000061603502 z_k1_s0157
                                 - 0.354000189643881 z_k1_s0160
                                 - 0.168999817614319 z_k1_s0161
                                 - 0.143999984939934 z_k1_s0162
                                 - 0.332999625173785 z_k1_s0163
                                 - 0.261998960155159 z_k1_s0164
                                 - 0.128999954775375 z_k1_s0165
                                 - 0.257999474350989 z_k1_s0166
                                 - 0.527000798661652 z_k1_s0167
                                 - 0.132999843896157 z_k1_s0170
                                 - 0.0839999646922345 z_k1_s0171
                                 - 0.349999518884708 z_k1_s0172
                                 - 0.100999876432426 z_k1_s0173
                                 - 0.132999843896157 z_k1_s0174
                                 - 0.160999832126902 z_k1_s0175
                                 - 0.19200033918175 z_k1_s0176
                                 - 0.182000391099284 z_k1_s0177
                                 - 0.317000363507935 z_k1_s1000
                                 - 0.138999430307842 z_k1_s1001
                                 - 0.0749999863618068 z_k1_s1002
                                 - 0.170000328606789 z_k1_s1003
                                 - 0.128999637566283 z_k1_s1004
                                 - 0.102000068764064 z_k1_s1005
                                 - 0.104000027312376 z_k1_s1006
                                 - 0.247999482391006 z_k1_s1007
                                 - 0.229999966243624 z_k1_s1010
                                 - 0.0639999906069213 z_k1_s1011
                                 - 0.138000143546314 z_k1_s1012
                                 - 0.147000142225412 z_k1_s1013
                                 - 0.305999955089343 z_k1_s1014
                                 - 0.26100078069455 z_k1_s1015
                                 - 0.21599996829836 z_k1_s1016
                                 - 0.275999959492348 z_k1_s1017
                                 - 0.393998579280399 z_k1_s1020
                                 - 0.135999899624834 z_k1_s1021
                                 - 0.611000375483902 z_k1_s1022
                                 - 0.439001044046789 z_k1_s1023
                                 - 0.384000500999789 z_k1_s1024
                                 - 0.368001670750189 z_k1_s1025
                                 - 0.570001587031214 z_k1_s1026
                                 - 0.273000242077938 z_k1_s1027
                                 - 0.672003067766839 z_k1_s1030
                                 - 0.232000897509404 z_k1_s1031
                                 - 0.582999934460616 z_k1_s1032
                                 - 1.1380025691077 z_k1_s1033
                                 - 1.01400563904464 z_k1_s1034
                                 - 0.233000881874461 z_k1_s1035
                                 - 0.833001681745083 z_k1_s1036
                                 - 0.951003230433933 z_k1_s1037
                                 - 1.19399743357327 z_k1_s1040
                                 - 0.838997197624245 z_k1_s1041
                                 - 0.708999475048643 z_k1_s1042
                                 - 0.347999851395542 z_k1_s1043
                                 - 0.734000420311592 z_k1_s1044
                                 - 1.33000113724537 z_k1_s1045
                                 - 0.360999623653102 z_k1_s1046
                                 - 0.478999170219869 z_k1_s1047
                                 - 1.28700260095439 z_k1_s1050
                                 - 0.928003572223664 z_k1_s1051
                                 - 0.370000964089597 z_k1_s1052
                                 - 1.43500128810541 z_k1_s1053
                                 - 0.559001681163568 z_k1_s1054
                                 - 0.470001077102584 z_k1_s1055
                                 - 0.37400089581791 z_k1_s1056
                                 - 0.643003887068455 z_k1_s1057
                                 - 0.202999537020564 z_k1_s1060
                                 - 0.810999167516654 z_k1_s1061
                                 - 0.575998821944854 z_k1_s1062
                                 - 0.825999607978083 z_k1_s1063
                                 - 0.214999496090272 z_k1_s1064
                                 - 0.227999615623887 z_k1_s1065
                                 - 0.852000633644184 z_k1_s1066
                                 - 0.221999832738751 z_k1_s1067
                                 - 0.56700108827569 z_k1_s1070
                                 - 0.180000126271493 z_k1_s1071
                                 - 0.32799949555198 z_k1_s1072
                                 - 0.237000215310834 z_k1_s1073
                                 - 0.357001356025908 z_k1_s1074
                                 - 0.257000405430013 z_k1_s1075
                                 - 0.161000172058433 z_k1_s1076
                                 - 0.177000097753616 z_k1_s1077
                                 - 0.413000907010904 z_k1_s1100
                                 - 0.124999993039101 z_k1_s1101
                                 - 0.205000657320208 z_k1_s1102
                                 - 0.185999998973383 z_k1_s1103
                                 - 0.178000049185287 z_k1_s1104
                                 - 0.386999903799447 z_k1_s1105
                                 - 0.134999930274221 z_k1_s1106
                                 - 0.314999966906532 z_k1_s1107
                                 - 0.516999788699522 z_k1_s1110
                                 - 0.786000049328114 z_k1_s1111
                                 - 0.335000059666819 z_k1_s1112
                                 - 0.89800123050216 z_k1_s1113
                                 - 0.280000032434952 z_k1_s1114
                                 - 0.962999882085199 z_k1_s1115
                                 - 0.692999642201483 z_k1_s1116
                                 - 1.36900197346599 z_k1_s1117
                                 - 0.23999884225272 z_k1_s1120
                                 - 0.208998991795077 z_k1_s1121
                                 - 0.176999944560723 z_k1_s1122
                                 - 0.622998990679373 z_k1_s1123
                                 - 0.12799998133429 z_k1_s1124
                                 - 0.193999862553624 z_k1_s1125
                                 - 0.140999918622979 z_k1_s1126
                                 - 0.284000625997407 z_k1_s1127
                                 - 0.760999697828469 z_k1_s1130
                                 - 0.201999926030811 z_k1_s1131
                                 - 0.479998994628359 z_k1_s1132
                                 - 0.284000382704299 z_k1_s1133
                                 - 0.194999835955067 z_k1_s1134
                                 - 0.285000275872362 z_k1_s1135
                                 - 0.239999776613947 z_k1_s1136
                                 - 0.345999345119551 z_k1_s1137
                                 - 1.09799808007378 z_k1_s1140
                                 - 0.40600058541994 z_k1_s1141
                                 - 0.627000282556896 z_k1_s1142
                                 - 1.05399764480629 z_k1_s1143
                                 - 1.12799770411925 z_k1_s1144
                                 - 0.878003303742291 z_k1_s1145
                                 - 0.842002274886538 z_k1_s1146
                                 - 1.36700309568823 z_k1_s1147
                                 - 1.12300039743305 z_k1_s1150
                                 - 1.01700376245171 z_k1_s1151
                                 - 0.726000967945506 z_k1_s1152
                                 - 0.851000927706713 z_k1_s1153
                                 - 2.64900233761153 z_k1_s1154
                                 - 0.533000481952048 z_k1_s1155
                                 - 0.597000770546898 z_k1_s1156
                                 - 1.28799921996471 z_k1_s1157
                                 - 0.167999904766882 z_k1_s1160
                                 - 0.255000073485252 z_k1_s1161
                                 - 0.246000201479774 z_k1_s1162
                                 - 0.246000201479774 z_k1_s1163
                                 - 0.151999934979541 z_k1_s1164
                                 - 0.203999836788399 z_k1_s1165
                                 - 0.206999720123624 z_k1_s1166
                                 - 0.741998820887735 z_k1_s1167
                                 - 1.10299961315898 z_k1_s1170
                                 - 0.738997583885999 z_k1_s1171
                                 - 1.19400012047723 z_k1_s1172
                                 - 0.354999364704991 z_k1_s1173
                                 - 0.46399866517507 z_k1_s1174
                                 - 0.9019984925989 z_k1_s1175
                                 - 0.866998297476498 z_k1_s1176
                                 - 0.911998548348158 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.23532 w_k1_ADM_s000
                                 + 1.32901 w_k1_AXP_s000
                                 + 1.96543 w_k1_DIS_s000
                                 + 1.54563 w_k1_FDX_s000
                                 + 0.861528 w_k1_IBM_s000
                                 + 1.66284 w_k1_INTC_s000
                                 + 1.52796 w_k1_KO_s000
                                 + 0.899948 w_k1_MAT_s000
                                 + 1.8128 w_k1_MSFT_s000 + 1.36855 w_k1_PG_s000
                                 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.830298 w_k1_ADM_s000
                                 + 1.41932 w_k1_AXP_s000
                                 + 1.70834 w_k1_DIS_s000
                                 + 1.10129 w_k1_FDX_s000
                                 + 1.88277 w_k1_IBM_s000
                                 + 1.48356 w_k1_INTC_s000
                                 + 1.56774 w_k1_KO_s000 + 2.03962 w_k1_MAT_s000
                                 + 1.46512 w_k1_MSFT_s000
                                 + 1.14008 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.864699 w_k1_ADM_s000
                                 + 0.683228 w_k1_AXP_s000
                                 + 0.62225 w_k1_DIS_s000
                                 + 0.764319 w_k1_FDX_s000
                                 + 0.919171 w_k1_IBM_s000
                                 + 0.897881 w_k1_INTC_s000
                                 + 0.848767 w_k1_KO_s000
                                 + 0.66827 w_k1_MAT_s000
                                 + 0.883473 w_k1_MSFT_s000
                                 + 0.965271 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.13972 w_k1_ADM_s000
                                 + 1.06506 w_k1_AXP_s000
                                 + 0.958675 w_k1_DIS_s000
                                 + 0.996536 w_k1_FDX_s000
                                 + 1.29084 w_k1_IBM_s000
                                 + 1.07716 w_k1_INTC_s000
                                 + 1.07472 w_k1_KO_s000
                                 + 0.962325 w_k1_MAT_s000
                                 + 0.985296 w_k1_MSFT_s000
                                 + 1.0643 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.49511 w_k1_ADM_s000
                                 + 1.05462 w_k1_AXP_s000
                                 + 1.23735 w_k1_DIS_s000
                                 + 0.965457 w_k1_FDX_s000
                                 + 1.51643 w_k1_IBM_s000
                                 + 1.00116 w_k1_INTC_s000
                                 + 1.60563 w_k1_KO_s000 + 1.6656 w_k1_MAT_s000
                                 + 1.29192 w_k1_MSFT_s000 + 1.2607 w_k1_PG_s000
                                 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.08482 w_k1_ADM_s000
                                 + 0.969937 w_k1_AXP_s000
                                 + 0.757076 w_k1_DIS_s000
                                 + 0.909763 w_k1_FDX_s000
                                 + 0.713116 w_k1_IBM_s000
                                 + 0.811974 w_k1_INTC_s000
                                 + 0.767986 w_k1_KO_s000 + 0.8044 w_k1_MAT_s000
                                 + 0.84341 w_k1_MSFT_s000
                                 + 0.902068 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.27469 w_k1_ADM_s000
                                 + 1.08672 w_k1_AXP_s000
                                 + 1.05767 w_k1_DIS_s000
                                 + 1.21227 w_k1_FDX_s000
                                 + 0.715787 w_k1_IBM_s000
                                 + 1.33656 w_k1_INTC_s000
                                 + 0.894122 w_k1_KO_s000
                                 + 0.846142 w_k1_MAT_s000
                                 + 1.20229 w_k1_MSFT_s000
                                 + 0.985533 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.924852 w_k1_ADM_s000
                                 + 0.924866 w_k1_AXP_s000
                                 + 1.23229 w_k1_DIS_s000
                                 + 0.993463 w_k1_FDX_s000
                                 + 1.03097 w_k1_IBM_s000
                                 + 1.10306 w_k1_INTC_s000
                                 + 1.11049 w_k1_KO_s000
                                 + 0.887997 w_k1_MAT_s000
                                 + 1.18902 w_k1_MSFT_s000
                                 + 1.06204 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.27474 w_k1_ADM_s001
                                 + 1.54329 w_k1_AXP_s001
                                 + 0.992094 w_k1_DIS_s001
                                 + 1.02745 w_k1_FDX_s001
                                 + 1.21911 w_k1_IBM_s001
                                 + 0.888151 w_k1_INTC_s001
                                 + 1.27547 w_k1_KO_s001 + 1.56547 w_k1_MAT_s001
                                 + 0.896496 w_k1_MSFT_s001
                                 + 1.05249 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.04446 w_k1_ADM_s001
                                 + 0.742856 w_k1_AXP_s001
                                 + 0.683609 w_k1_DIS_s001
                                 + 0.776992 w_k1_FDX_s001
                                 + 0.821543 w_k1_IBM_s001
                                 + 0.733424 w_k1_INTC_s001
                                 + 0.785208 w_k1_KO_s001
                                 + 0.701661 w_k1_MAT_s001
                                 + 0.679242 w_k1_MSFT_s001
                                 + 0.864222 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.823073 w_k1_ADM_s001
                                 + 1.00512 w_k1_AXP_s001
                                 + 0.874335 w_k1_DIS_s001
                                 + 0.915251 w_k1_FDX_s001
                                 + 1.19607 w_k1_IBM_s001
                                 + 0.885259 w_k1_INTC_s001
                                 + 1.10781 w_k1_KO_s001
                                 + 0.992743 w_k1_MAT_s001
                                 + 0.886701 w_k1_MSFT_s001
                                 + 0.941077 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.17285 w_k1_ADM_s001
                                 + 1.15284 w_k1_AXP_s001
                                 + 1.56445 w_k1_DIS_s001
                                 + 1.34943 w_k1_FDX_s001
                                 + 0.828743 w_k1_IBM_s001
                                 + 1.41034 w_k1_INTC_s001
                                 + 1.24741 w_k1_KO_s001 + 1.08551 w_k1_MAT_s001
                                 + 1.32246 w_k1_MSFT_s001
                                 + 1.22401 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.0258 w_k1_ADM_s001
                                 + 0.946357 w_k1_AXP_s001
                                 + 0.980643 w_k1_DIS_s001
                                 + 0.980735 w_k1_FDX_s001
                                 + 0.91966 w_k1_IBM_s001
                                 + 1.02706 w_k1_INTC_s001
                                 + 1.16072 w_k1_KO_s001 + 0.95541 w_k1_MAT_s001
                                 + 0.983613 w_k1_MSFT_s001
                                 + 1.0309 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.879337 w_k1_ADM_s001
                                 + 1.11609 w_k1_AXP_s001
                                 + 0.788755 w_k1_DIS_s001
                                 + 0.761355 w_k1_FDX_s001
                                 + 1.49818 w_k1_IBM_s001
                                 + 0.774681 w_k1_INTC_s001
                                 + 1.12801 w_k1_KO_s001 + 1.46898 w_k1_MAT_s001
                                 + 0.702697 w_k1_MSFT_s001
                                 + 1.02232 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.09054 w_k1_ADM_s001
                                 + 1.21015 w_k1_AXP_s001
                                 + 1.35039 w_k1_DIS_s001 + 1.1501 w_k1_FDX_s001
                                 + 1.26375 w_k1_IBM_s001
                                 + 1.09161 w_k1_INTC_s001
                                 + 1.23397 w_k1_KO_s001 + 1.17845 w_k1_MAT_s001
                                 + 1.04725 w_k1_MSFT_s001
                                 + 1.16968 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.00324 w_k1_ADM_s001
                                 + 0.865755 w_k1_AXP_s001
                                 + 0.965525 w_k1_DIS_s001
                                 + 1.04455 w_k1_FDX_s001
                                 + 0.484973 w_k1_IBM_s001
                                 + 0.90557 w_k1_INTC_s001
                                 + 0.75807 w_k1_KO_s001
                                 + 0.417382 w_k1_MAT_s001
                                 + 0.900777 w_k1_MSFT_s001
                                 + 0.825462 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.1282 w_k1_ADM_s002
                                 + 1.13577 w_k1_AXP_s002
                                 + 1.11596 w_k1_DIS_s002
                                 + 1.11737 w_k1_FDX_s002
                                 + 0.856324 w_k1_IBM_s002
                                 + 0.94277 w_k1_INTC_s002
                                 + 0.929515 w_k1_KO_s002
                                 + 0.977104 w_k1_MAT_s002
                                 + 0.93214 w_k1_MSFT_s002
                                 + 1.16062 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.46595 w_k1_ADM_s002
                                 + 0.937992 w_k1_AXP_s002
                                 + 1.11696 w_k1_DIS_s002
                                 + 1.04038 w_k1_FDX_s002 + 1.1725 w_k1_IBM_s002
                                 + 1.01567 w_k1_INTC_s002
                                 + 1.15789 w_k1_KO_s002 + 1.05324 w_k1_MAT_s002
                                 + 1.03455 w_k1_MSFT_s002
                                 + 1.04548 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.97676 w_k1_ADM_s002
                                 + 0.851539 w_k1_AXP_s002
                                 + 0.738399 w_k1_DIS_s002
                                 + 0.958326 w_k1_FDX_s002
                                 + 0.752967 w_k1_IBM_s002
                                 + 0.808346 w_k1_INTC_s002
                                 + 0.847881 w_k1_KO_s002
                                 + 0.585078 w_k1_MAT_s002
                                 + 0.79779 w_k1_MSFT_s002
                                 + 0.881265 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.958682 w_k1_ADM_s002
                                 + 0.739889 w_k1_AXP_s002
                                 + 0.615949 w_k1_DIS_s002
                                 + 0.774653 w_k1_FDX_s002
                                 + 0.647445 w_k1_IBM_s002
                                 + 0.752084 w_k1_INTC_s002
                                 + 0.820606 w_k1_KO_s002
                                 + 0.498438 w_k1_MAT_s002
                                 + 0.801799 w_k1_MSFT_s002
                                 + 0.915079 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.46378 w_k1_ADM_s002
                                 + 1.41787 w_k1_AXP_s002
                                 + 1.63397 w_k1_DIS_s002
                                 + 1.26887 w_k1_FDX_s002 + 1.3063 w_k1_IBM_s002
                                 + 1.16244 w_k1_INTC_s002
                                 + 1.31707 w_k1_KO_s002
                                 + 0.956773 w_k1_MAT_s002
                                 + 1.13465 w_k1_MSFT_s002 + 1.2721 w_k1_PG_s002
                                 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.956968 w_k1_ADM_s002
                                 + 1.0564 w_k1_AXP_s002
                                 + 0.853426 w_k1_DIS_s002
                                 + 0.854035 w_k1_FDX_s002
                                 + 1.35492 w_k1_IBM_s002
                                 + 0.830155 w_k1_INTC_s002
                                 + 0.999742 w_k1_KO_s002
                                 + 1.54544 w_k1_MAT_s002
                                 + 0.837729 w_k1_MSFT_s002
                                 + 0.998942 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.89533 w_k1_ADM_s002
                                 + 0.879272 w_k1_AXP_s002
                                 + 0.743942 w_k1_DIS_s002
                                 + 0.899688 w_k1_FDX_s002
                                 + 0.976611 w_k1_IBM_s002
                                 + 0.798295 w_k1_INTC_s002
                                 + 0.87837 w_k1_KO_s002 + 1.01541 w_k1_MAT_s002
                                 + 0.767971 w_k1_MSFT_s002
                                 + 0.859933 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.45434 w_k1_ADM_s002
                                 + 1.41319 w_k1_AXP_s002
                                 + 1.31676 w_k1_DIS_s002
                                 + 1.37772 w_k1_FDX_s002
                                 + 0.982173 w_k1_IBM_s002
                                 + 1.13646 w_k1_INTC_s002 + 1.2618 w_k1_KO_s002
                                 + 1.28568 w_k1_MAT_s002
                                 + 1.1574 w_k1_MSFT_s002 + 1.44343 w_k1_PG_s002
                                 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.14782 w_k1_ADM_s003
                                 + 1.07983 w_k1_AXP_s003
                                 + 0.899955 w_k1_DIS_s003
                                 + 0.954594 w_k1_FDX_s003
                                 + 1.22834 w_k1_IBM_s003
                                 + 0.82055 w_k1_INTC_s003
                                 + 0.863308 w_k1_KO_s003
                                 + 0.978836 w_k1_MAT_s003
                                 + 0.759069 w_k1_MSFT_s003
                                 + 0.972257 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.93265 w_k1_ADM_s003
                                 + 0.669239 w_k1_AXP_s003
                                 + 0.6651 w_k1_DIS_s003
                                 + 0.792014 w_k1_FDX_s003
                                 + 0.725875 w_k1_IBM_s003
                                 + 0.638022 w_k1_INTC_s003
                                 + 0.63351 w_k1_KO_s003 + 0.59464 w_k1_MAT_s003
                                 + 0.642858 w_k1_MSFT_s003
                                 + 0.8032 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.18361 w_k1_ADM_s003
                                 + 0.823733 w_k1_AXP_s003
                                 + 0.984306 w_k1_DIS_s003
                                 + 1.04257 w_k1_FDX_s003
                                 + 0.785407 w_k1_IBM_s003
                                 + 0.890171 w_k1_INTC_s003
                                 + 0.838896 w_k1_KO_s003
                                 + 0.624017 w_k1_MAT_s003
                                 + 1.0252 w_k1_MSFT_s003
                                 + 0.826817 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.949982 w_k1_ADM_s003
                                 + 0.712997 w_k1_AXP_s003
                                 + 0.721013 w_k1_DIS_s003
                                 + 0.7386 w_k1_FDX_s003 + 1.11464 w_k1_IBM_s003
                                 + 0.789089 w_k1_INTC_s003
                                 + 0.930952 w_k1_KO_s003
                                 + 0.636634 w_k1_MAT_s003
                                 + 0.790591 w_k1_MSFT_s003
                                 + 0.856412 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.48023 w_k1_ADM_s003
                                 + 1.13604 w_k1_AXP_s003
                                 + 1.32958 w_k1_DIS_s003
                                 + 1.05263 w_k1_FDX_s003
                                 + 1.33939 w_k1_IBM_s003
                                 + 1.03704 w_k1_INTC_s003
                                 + 1.24509 w_k1_KO_s003 + 1.10118 w_k1_MAT_s003
                                 + 1.19008 w_k1_MSFT_s003
                                 + 1.11663 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.56139 w_k1_ADM_s003
                                 + 1.21534 w_k1_AXP_s003
                                 + 1.62938 w_k1_DIS_s003
                                 + 1.35787 w_k1_FDX_s003 + 1.2251 w_k1_IBM_s003
                                 + 0.939144 w_k1_INTC_s003
                                 + 1.12692 w_k1_KO_s003 + 1.12713 w_k1_MAT_s003
                                 + 1.12886 w_k1_MSFT_s003
                                 + 1.11318 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.33818 w_k1_ADM_s003
                                 + 1.37707 w_k1_AXP_s003
                                 + 1.08441 w_k1_DIS_s003
                                 + 1.19449 w_k1_FDX_s003
                                 + 1.16285 w_k1_IBM_s003
                                 + 0.845818 w_k1_INTC_s003
                                 + 0.916653 w_k1_KO_s003
                                 + 1.38154 w_k1_MAT_s003
                                 + 0.914844 w_k1_MSFT_s003
                                 + 1.01733 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.21453 w_k1_ADM_s003
                                 + 1.18425 w_k1_AXP_s003
                                 + 0.956878 w_k1_DIS_s003
                                 + 0.929869 w_k1_FDX_s003
                                 + 1.47629 w_k1_IBM_s003
                                 + 1.02722 w_k1_INTC_s003 + 1.1742 w_k1_KO_s003
                                 + 1.50636 w_k1_MAT_s003
                                 + 0.942711 w_k1_MSFT_s003
                                 + 1.21296 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.84412 w_k1_ADM_s004
                                 + 0.861317 w_k1_AXP_s004
                                 + 0.645037 w_k1_DIS_s004
                                 + 0.861087 w_k1_FDX_s004
                                 + 0.710493 w_k1_IBM_s004
                                 + 1.05755 w_k1_INTC_s004
                                 + 0.829654 w_k1_KO_s004
                                 + 0.857753 w_k1_MAT_s004
                                 + 0.988311 w_k1_MSFT_s004
                                 + 0.822841 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.95444 w_k1_ADM_s004
                                 + 0.827009 w_k1_AXP_s004
                                 + 0.806868 w_k1_DIS_s004
                                 + 0.811267 w_k1_FDX_s004
                                 + 0.952113 w_k1_IBM_s004
                                 + 1.00582 w_k1_INTC_s004
                                 + 0.995714 w_k1_KO_s004
                                 + 1.05641 w_k1_MAT_s004
                                 + 0.961483 w_k1_MSFT_s004
                                 + 0.873324 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.23315 w_k1_ADM_s004
                                 + 1.53291 w_k1_AXP_s004
                                 + 1.42127 w_k1_DIS_s004
                                 + 1.16524 w_k1_FDX_s004
                                 + 1.64485 w_k1_IBM_s004
                                 + 1.25816 w_k1_INTC_s004
                                 + 1.51443 w_k1_KO_s004 + 2.10367 w_k1_MAT_s004
                                 + 1.27279 w_k1_MSFT_s004
                                 + 1.09486 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.44717 w_k1_ADM_s004
                                 + 1.36654 w_k1_AXP_s004
                                 + 1.51799 w_k1_DIS_s004
                                 + 1.38344 w_k1_FDX_s004
                                 + 0.947244 w_k1_IBM_s004
                                 + 1.39834 w_k1_INTC_s004
                                 + 1.30741 w_k1_KO_s004 + 0.78411 w_k1_MAT_s004
                                 + 1.44822 w_k1_MSFT_s004
                                 + 1.18271 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.03209 w_k1_ADM_s004
                                 + 0.817035 w_k1_AXP_s004
                                 + 1.08374 w_k1_DIS_s004
                                 + 1.01831 w_k1_FDX_s004
                                 + 0.781519 w_k1_IBM_s004
                                 + 1.07583 w_k1_INTC_s004
                                 + 1.07184 w_k1_KO_s004
                                 + 0.781638 w_k1_MAT_s004
                                 + 1.16148 w_k1_MSFT_s004
                                 + 1.02448 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.18393 w_k1_ADM_s004
                                 + 1.17894 w_k1_AXP_s004
                                 + 1.11175 w_k1_DIS_s004
                                 + 1.14797 w_k1_FDX_s004
                                 + 1.03393 w_k1_IBM_s004
                                 + 1.36372 w_k1_INTC_s004
                                 + 1.21356 w_k1_KO_s004
                                 + 0.980178 w_k1_MAT_s004
                                 + 1.36312 w_k1_MSFT_s004
                                 + 1.00021 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.999971 w_k1_ADM_s004
                                 + 1.11076 w_k1_AXP_s004
                                 + 1.01672 w_k1_DIS_s004
                                 + 0.997899 w_k1_FDX_s004
                                 + 1.30823 w_k1_IBM_s004
                                 + 1.38142 w_k1_INTC_s004
                                 + 1.18479 w_k1_KO_s004 + 1.53792 w_k1_MAT_s004
                                 + 1.18653 w_k1_MSFT_s004
                                 + 1.03928 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.08511 w_k1_ADM_s004
                                 + 1.01187 w_k1_AXP_s004
                                 + 0.916299 w_k1_DIS_s004
                                 + 1.01968 w_k1_FDX_s004
                                 + 0.628922 w_k1_IBM_s004
                                 + 1.08286 w_k1_INTC_s004
                                 + 0.80163 w_k1_KO_s004
                                 + 0.829385 w_k1_MAT_s004
                                 + 1.14396 w_k1_MSFT_s004
                                 + 0.976817 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.0695 w_k1_ADM_s005
                                 + 0.906206 w_k1_AXP_s005
                                 + 1.27785 w_k1_DIS_s005
                                 + 1.05208 w_k1_FDX_s005
                                 + 0.997869 w_k1_IBM_s005
                                 + 1.12514 w_k1_INTC_s005 + 1.1912 w_k1_KO_s005
                                 + 1.15067 w_k1_MAT_s005
                                 + 1.43082 w_k1_MSFT_s005 + 1.0691 w_k1_PG_s005
                                 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.908712 w_k1_ADM_s005
                                 + 0.759586 w_k1_AXP_s005
                                 + 0.746499 w_k1_DIS_s005
                                 + 0.792184 w_k1_FDX_s005
                                 + 1.07966 w_k1_IBM_s005
                                 + 0.972926 w_k1_INTC_s005
                                 + 1.01969 w_k1_KO_s005 + 1.06065 w_k1_MAT_s005
                                 + 1.06046 w_k1_MSFT_s005
                                 + 0.939455 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.998201 w_k1_ADM_s005
                                 + 1.25644 w_k1_AXP_s005 + 1.0654 w_k1_DIS_s005
                                 + 1.00865 w_k1_FDX_s005
                                 + 1.20435 w_k1_IBM_s005
                                 + 1.05027 w_k1_INTC_s005
                                 + 1.08584 w_k1_KO_s005 + 1.53482 w_k1_MAT_s005
                                 + 1.22722 w_k1_MSFT_s005
                                 + 1.01623 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.681531 w_k1_ADM_s005
                                 + 1.23089 w_k1_AXP_s005
                                 + 1.30907 w_k1_DIS_s005
                                 + 0.938214 w_k1_FDX_s005
                                 + 1.89032 w_k1_IBM_s005
                                 + 1.24453 w_k1_INTC_s005
                                 + 1.67335 w_k1_KO_s005 + 1.93814 w_k1_MAT_s005
                                 + 1.23382 w_k1_MSFT_s005
                                 + 1.27359 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.799076 w_k1_ADM_s005
                                 + 0.815147 w_k1_AXP_s005
                                 + 0.815335 w_k1_DIS_s005
                                 + 0.924624 w_k1_FDX_s005
                                 + 0.84188 w_k1_IBM_s005
                                 + 0.961237 w_k1_INTC_s005
                                 + 0.860885 w_k1_KO_s005
                                 + 0.942458 w_k1_MAT_s005
                                 + 1.15846 w_k1_MSFT_s005
                                 + 0.859329 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.98537 w_k1_ADM_s005
                                 + 0.778025 w_k1_AXP_s005
                                 + 0.84798 w_k1_DIS_s005
                                 + 0.930175 w_k1_FDX_s005
                                 + 0.59033 w_k1_IBM_s005
                                 + 0.935909 w_k1_INTC_s005
                                 + 0.742026 w_k1_KO_s005
                                 + 0.558149 w_k1_MAT_s005
                                 + 1.18914 w_k1_MSFT_s005
                                 + 0.863548 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.11713 w_k1_ADM_s005
                                 + 1.42461 w_k1_AXP_s005
                                 + 1.12668 w_k1_DIS_s005 + 1.1629 w_k1_FDX_s005
                                 + 1.52794 w_k1_IBM_s005
                                 + 1.2046 w_k1_INTC_s005 + 1.30254 w_k1_KO_s005
                                 + 1.67173 w_k1_MAT_s005
                                 + 1.36124 w_k1_MSFT_s005
                                 + 1.05471 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.24327 w_k1_ADM_s005
                                 + 1.55441 w_k1_AXP_s005
                                 + 1.66919 w_k1_DIS_s005
                                 + 1.36631 w_k1_FDX_s005
                                 + 1.11956 w_k1_IBM_s005
                                 + 1.66482 w_k1_INTC_s005 + 1.3662 w_k1_KO_s005
                                 + 1.53765 w_k1_MAT_s005
                                 + 1.62981 w_k1_MSFT_s005
                                 + 1.28913 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.08941 w_k1_ADM_s006
                                 + 1.23262 w_k1_AXP_s006
                                 + 1.41639 w_k1_DIS_s006
                                 + 1.26818 w_k1_FDX_s006
                                 + 0.910068 w_k1_IBM_s006
                                 + 1.47048 w_k1_INTC_s006
                                 + 1.21995 w_k1_KO_s006
                                 + 0.893053 w_k1_MAT_s006
                                 + 1.21952 w_k1_MSFT_s006
                                 + 1.15991 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.43702 w_k1_ADM_s006
                                 + 1.44176 w_k1_AXP_s006
                                 + 1.89606 w_k1_DIS_s006
                                 + 1.45158 w_k1_FDX_s006
                                 + 1.50584 w_k1_IBM_s006
                                 + 1.6175 w_k1_INTC_s006 + 1.84299 w_k1_KO_s006
                                 + 1.77794 w_k1_MAT_s006
                                 + 1.41624 w_k1_MSFT_s006
                                 + 1.55335 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.887957 w_k1_ADM_s006
                                 + 1.29974 w_k1_AXP_s006
                                 + 1.45359 w_k1_DIS_s006 + 1.1441 w_k1_FDX_s006
                                 + 1.44846 w_k1_IBM_s006
                                 + 1.33718 w_k1_INTC_s006
                                 + 1.51906 w_k1_KO_s006 + 1.49579 w_k1_MAT_s006
                                 + 1.10215 w_k1_MSFT_s006
                                 + 1.46642 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.28518 w_k1_ADM_s006
                                 + 0.877045 w_k1_AXP_s006
                                 + 1.07723 w_k1_DIS_s006 + 1.0123 w_k1_FDX_s006
                                 + 0.945585 w_k1_IBM_s006
                                 + 1.10747 w_k1_INTC_s006
                                 + 1.14563 w_k1_KO_s006
                                 + 0.802132 w_k1_MAT_s006
                                 + 0.911282 w_k1_MSFT_s006
                                 + 0.998177 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.00058 w_k1_ADM_s006
                                 + 1.06651 w_k1_AXP_s006
                                 + 0.985979 w_k1_DIS_s006
                                 + 1.05278 w_k1_FDX_s006
                                 + 0.798768 w_k1_IBM_s006
                                 + 1.09704 w_k1_INTC_s006
                                 + 1.10075 w_k1_KO_s006
                                 + 0.913964 w_k1_MAT_s006
                                 + 0.974262 w_k1_MSFT_s006
                                 + 1.14402 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.702609 w_k1_ADM_s006
                                 + 0.67773 w_k1_AXP_s006
                                 + 0.679164 w_k1_DIS_s006
                                 + 0.718459 w_k1_FDX_s006
                                 + 0.816343 w_k1_IBM_s006
                                 + 0.81606 w_k1_INTC_s006
                                 + 0.796448 w_k1_KO_s006
                                 + 0.634988 w_k1_MAT_s006
                                 + 0.783201 w_k1_MSFT_s006
                                 + 0.921704 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.900122 w_k1_ADM_s006
                                 + 0.803228 w_k1_AXP_s006
                                 + 0.761913 w_k1_DIS_s006
                                 + 0.976484 w_k1_FDX_s006
                                 + 0.59408 w_k1_IBM_s006
                                 + 0.980181 w_k1_INTC_s006
                                 + 0.740627 w_k1_KO_s006
                                 + 0.443665 w_k1_MAT_s006
                                 + 0.923813 w_k1_MSFT_s006
                                 + 1.06665 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.682342 w_k1_ADM_s006
                                 + 1.0458 w_k1_AXP_s006
                                 + 0.941202 w_k1_DIS_s006
                                 + 0.857037 w_k1_FDX_s006
                                 + 1.23345 w_k1_IBM_s006
                                 + 1.24561 w_k1_INTC_s006
                                 + 1.02391 w_k1_KO_s006 + 1.28664 w_k1_MAT_s006
                                 + 0.861385 w_k1_MSFT_s006
                                 + 0.982548 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.68583 w_k1_ADM_s007
                                 + 1.41326 w_k1_AXP_s007
                                 + 1.70406 w_k1_DIS_s007
                                 + 1.44309 w_k1_FDX_s007
                                 + 1.49468 w_k1_IBM_s007
                                 + 1.38236 w_k1_INTC_s007
                                 + 1.51411 w_k1_KO_s007 + 1.34955 w_k1_MAT_s007
                                 + 1.29318 w_k1_MSFT_s007
                                 + 1.41748 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.00815 w_k1_ADM_s007
                                 + 1.17825 w_k1_AXP_s007
                                 + 0.969324 w_k1_DIS_s007 + 1.06 w_k1_FDX_s007
                                 + 0.964127 w_k1_IBM_s007
                                 + 1.02127 w_k1_INTC_s007
                                 + 1.04008 w_k1_KO_s007 + 1.46329 w_k1_MAT_s007
                                 + 1.00534 w_k1_MSFT_s007
                                 + 1.09845 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.914095 w_k1_ADM_s007
                                 + 0.95928 w_k1_AXP_s007
                                 + 0.852228 w_k1_DIS_s007
                                 + 0.791091 w_k1_FDX_s007
                                 + 1.15079 w_k1_IBM_s007
                                 + 0.692071 w_k1_INTC_s007
                                 + 0.947683 w_k1_KO_s007
                                 + 1.35312 w_k1_MAT_s007
                                 + 0.728548 w_k1_MSFT_s007
                                 + 0.984421 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.13596 w_k1_ADM_s007
                                 + 0.813705 w_k1_AXP_s007
                                 + 1.04889 w_k1_DIS_s007
                                 + 1.06749 w_k1_FDX_s007
                                 + 0.588403 w_k1_IBM_s007
                                 + 0.945983 w_k1_INTC_s007
                                 + 0.902228 w_k1_KO_s007
                                 + 0.760133 w_k1_MAT_s007
                                 + 1.12004 w_k1_MSFT_s007
                                 + 1.11499 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.856071 w_k1_ADM_s007
                                 + 0.717117 w_k1_AXP_s007
                                 + 0.719513 w_k1_DIS_s007
                                 + 0.8033 w_k1_FDX_s007 + 0.69876 w_k1_IBM_s007
                                 + 0.809304 w_k1_INTC_s007
                                 + 0.712935 w_k1_KO_s007
                                 + 0.671845 w_k1_MAT_s007
                                 + 0.865641 w_k1_MSFT_s007
                                 + 0.812596 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.16601 w_k1_ADM_s007
                                 + 0.907936 w_k1_AXP_s007
                                 + 0.982945 w_k1_DIS_s007
                                 + 0.961178 w_k1_FDX_s007
                                 + 0.991937 w_k1_IBM_s007
                                 + 0.945805 w_k1_INTC_s007
                                 + 0.988549 w_k1_KO_s007
                                 + 0.819226 w_k1_MAT_s007
                                 + 0.992359 w_k1_MSFT_s007
                                 + 0.977209 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.1633 w_k1_ADM_s007
                                 + 1.29382 w_k1_AXP_s007 + 1.3891 w_k1_DIS_s007
                                 + 1.3227 w_k1_FDX_s007
                                 + 0.998065 w_k1_IBM_s007
                                 + 1.12118 w_k1_INTC_s007
                                 + 1.24853 w_k1_KO_s007 + 1.23275 w_k1_MAT_s007
                                 + 1.30188 w_k1_MSFT_s007
                                 + 1.23226 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.13752 w_k1_ADM_s007
                                 + 1.14027 w_k1_AXP_s007 + 1.2221 w_k1_DIS_s007
                                 + 1.00075 w_k1_FDX_s007
                                 + 1.40942 w_k1_IBM_s007
                                 + 1.21232 w_k1_INTC_s007
                                 + 1.31674 w_k1_KO_s007 + 1.7599 w_k1_MAT_s007
                                 + 1.15342 w_k1_MSFT_s007
                                 + 1.19181 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.761093 w_k1_ADM_s010
                                 + 1.08094 w_k1_AXP_s010
                                 + 0.806865 w_k1_DIS_s010
                                 + 0.849595 w_k1_FDX_s010
                                 + 1.11361 w_k1_IBM_s010
                                 + 0.833918 w_k1_INTC_s010
                                 + 0.884162 w_k1_KO_s010
                                 + 1.52555 w_k1_MAT_s010
                                 + 0.858802 w_k1_MSFT_s010
                                 + 0.935516 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.1464 w_k1_ADM_s010
                                 + 1.34028 w_k1_AXP_s010
                                 + 1.70483 w_k1_DIS_s010
                                 + 1.33767 w_k1_FDX_s010
                                 + 1.11463 w_k1_IBM_s010
                                 + 1.06282 w_k1_INTC_s010
                                 + 1.17098 w_k1_KO_s010 + 1.79091 w_k1_MAT_s010
                                 + 1.3689 w_k1_MSFT_s010 + 1.20556 w_k1_PG_s010
                                 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.01461 w_k1_ADM_s010
                                 + 1.32623 w_k1_AXP_s010
                                 + 1.07896 w_k1_DIS_s010
                                 + 1.21625 w_k1_FDX_s010
                                 + 0.893261 w_k1_IBM_s010
                                 + 1.17941 w_k1_INTC_s010
                                 + 1.02467 w_k1_KO_s010
                                 + 0.975584 w_k1_MAT_s010
                                 + 1.17326 w_k1_MSFT_s010
                                 + 1.09475 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.941713 w_k1_ADM_s010
                                 + 0.742005 w_k1_AXP_s010
                                 + 1.23132 w_k1_DIS_s010
                                 + 1.06866 w_k1_FDX_s010
                                 + 0.889439 w_k1_IBM_s010
                                 + 1.04983 w_k1_INTC_s010
                                 + 1.17223 w_k1_KO_s010
                                 + 0.950528 w_k1_MAT_s010
                                 + 1.32349 w_k1_MSFT_s010
                                 + 1.12507 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.854866 w_k1_ADM_s010
                                 + 1.15717 w_k1_AXP_s010
                                 + 0.964226 w_k1_DIS_s010
                                 + 1.01193 w_k1_FDX_s010
                                 + 1.00928 w_k1_IBM_s010
                                 + 1.1411 w_k1_INTC_s010 + 1.01454 w_k1_KO_s010
                                 + 1.38586 w_k1_MAT_s010
                                 + 1.16248 w_k1_MSFT_s010
                                 + 0.973077 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.931014 w_k1_ADM_s010
                                 + 0.881672 w_k1_AXP_s010
                                 + 0.846657 w_k1_DIS_s010
                                 + 0.901253 w_k1_FDX_s010
                                 + 0.988302 w_k1_IBM_s010
                                 + 0.959932 w_k1_INTC_s010
                                 + 0.979838 w_k1_KO_s010
                                 + 0.999322 w_k1_MAT_s010
                                 + 1.01168 w_k1_MSFT_s010
                                 + 0.960062 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.25846 w_k1_ADM_s010
                                 + 1.59472 w_k1_AXP_s010
                                 + 1.44525 w_k1_DIS_s010
                                 + 1.22774 w_k1_FDX_s010
                                 + 1.51843 w_k1_IBM_s010
                                 + 1.42552 w_k1_INTC_s010
                                 + 1.27578 w_k1_KO_s010 + 2.01035 w_k1_MAT_s010
                                 + 1.32235 w_k1_MSFT_s010
                                 + 1.15256 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.822494 w_k1_ADM_s010
                                 + 0.732363 w_k1_AXP_s010
                                 + 0.620307 w_k1_DIS_s010
                                 + 0.77079 w_k1_FDX_s010
                                 + 0.863161 w_k1_IBM_s010
                                 + 0.930808 w_k1_INTC_s010
                                 + 0.695308 w_k1_KO_s010
                                 + 0.646186 w_k1_MAT_s010
                                 + 0.824709 w_k1_MSFT_s010
                                 + 0.787868 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.941171 w_k1_ADM_s011
                                 + 0.882175 w_k1_AXP_s011
                                 + 0.916564 w_k1_DIS_s011
                                 + 1.04388 w_k1_FDX_s011
                                 + 0.588466 w_k1_IBM_s011
                                 + 0.857667 w_k1_INTC_s011
                                 + 0.711841 w_k1_KO_s011
                                 + 0.837774 w_k1_MAT_s011
                                 + 1.10261 w_k1_MSFT_s011
                                 + 0.992256 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.04511 w_k1_ADM_s011
                                 + 1.15348 w_k1_AXP_s011
                                 + 1.00246 w_k1_DIS_s011
                                 + 1.09839 w_k1_FDX_s011 + 1.0736 w_k1_IBM_s011
                                 + 1.09517 w_k1_INTC_s011
                                 + 0.906201 w_k1_KO_s011
                                 + 1.35661 w_k1_MAT_s011
                                 + 1.06894 w_k1_MSFT_s011
                                 + 1.03575 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.935046 w_k1_ADM_s011
                                 + 1.23378 w_k1_AXP_s011
                                 + 1.30057 w_k1_DIS_s011
                                 + 1.14021 w_k1_FDX_s011
                                 + 1.42286 w_k1_IBM_s011
                                 + 0.954836 w_k1_INTC_s011
                                 + 1.21885 w_k1_KO_s011 + 1.5036 w_k1_MAT_s011
                                 + 1.18467 w_k1_MSFT_s011
                                 + 1.11568 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.6858 w_k1_ADM_s011
                                 + 0.86564 w_k1_AXP_s011
                                 + 0.844743 w_k1_DIS_s011
                                 + 0.789698 w_k1_FDX_s011
                                 + 0.939795 w_k1_IBM_s011
                                 + 0.867752 w_k1_INTC_s011
                                 + 0.82142 w_k1_KO_s011 + 0.8511 w_k1_MAT_s011
                                 + 0.856711 w_k1_MSFT_s011
                                 + 0.88229 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.569125 w_k1_ADM_s011
                                 + 0.632074 w_k1_AXP_s011
                                 + 0.563324 w_k1_DIS_s011
                                 + 0.702906 w_k1_FDX_s011
                                 + 0.584334 w_k1_IBM_s011
                                 + 0.811625 w_k1_INTC_s011
                                 + 0.596393 w_k1_KO_s011
                                 + 0.699221 w_k1_MAT_s011
                                 + 0.755725 w_k1_MSFT_s011
                                 + 0.67276 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.06636 w_k1_ADM_s011
                                 + 0.916667 w_k1_AXP_s011
                                 + 0.755631 w_k1_DIS_s011
                                 + 1.02329 w_k1_FDX_s011
                                 + 0.760831 w_k1_IBM_s011
                                 + 0.872508 w_k1_INTC_s011
                                 + 0.779816 w_k1_KO_s011
                                 + 0.824867 w_k1_MAT_s011
                                 + 0.930057 w_k1_MSFT_s011
                                 + 0.970079 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.17016 w_k1_ADM_s011
                                 + 1.46058 w_k1_AXP_s011
                                 + 1.84865 w_k1_DIS_s011
                                 + 1.46434 w_k1_FDX_s011
                                 + 1.04761 w_k1_IBM_s011
                                 + 1.25085 w_k1_INTC_s011
                                 + 1.31821 w_k1_KO_s011 + 1.30019 w_k1_MAT_s011
                                 + 1.29566 w_k1_MSFT_s011
                                 + 1.46502 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.573977 w_k1_ADM_s011
                                 + 0.80011 w_k1_AXP_s011
                                 + 0.719816 w_k1_DIS_s011
                                 + 0.713689 w_k1_FDX_s011
                                 + 1.31911 w_k1_IBM_s011
                                 + 0.847075 w_k1_INTC_s011
                                 + 0.865495 w_k1_KO_s011
                                 + 1.29307 w_k1_MAT_s011
                                 + 0.824714 w_k1_MSFT_s011
                                 + 1.02158 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.07388 w_k1_ADM_s012
                                 + 1.0883 w_k1_AXP_s012 + 1.24287 w_k1_DIS_s012
                                 + 1.04063 w_k1_FDX_s012
                                 + 1.09995 w_k1_IBM_s012
                                 + 0.879054 w_k1_INTC_s012
                                 + 1.2924 w_k1_KO_s012 + 1.20687 w_k1_MAT_s012
                                 + 1.13283 w_k1_MSFT_s012
                                 + 1.10744 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.04362 w_k1_ADM_s012
                                 + 1.04868 w_k1_AXP_s012
                                 + 0.998394 w_k1_DIS_s012
                                 + 1.07234 w_k1_FDX_s012
                                 + 0.815072 w_k1_IBM_s012
                                 + 1.00724 w_k1_INTC_s012 + 1.0248 w_k1_KO_s012
                                 + 0.98584 w_k1_MAT_s012
                                 + 1.07633 w_k1_MSFT_s012
                                 + 0.996233 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.797274 w_k1_ADM_s012
                                 + 0.811432 w_k1_AXP_s012
                                 + 0.739611 w_k1_DIS_s012
                                 + 0.807501 w_k1_FDX_s012
                                 + 0.96048 w_k1_IBM_s012
                                 + 0.823247 w_k1_INTC_s012
                                 + 0.851039 w_k1_KO_s012
                                 + 0.967152 w_k1_MAT_s012
                                 + 0.942454 w_k1_MSFT_s012
                                 + 0.905512 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.30512 w_k1_ADM_s012
                                 + 1.29227 w_k1_AXP_s012
                                 + 1.34237 w_k1_DIS_s012
                                 + 1.00706 w_k1_FDX_s012
                                 + 1.84729 w_k1_IBM_s012
                                 + 0.908344 w_k1_INTC_s012
                                 + 1.62445 w_k1_KO_s012 + 1.94668 w_k1_MAT_s012
                                 + 1.1627 w_k1_MSFT_s012 + 1.10199 w_k1_PG_s012
                                 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.36351 w_k1_ADM_s012
                                 + 1.63224 w_k1_AXP_s012
                                 + 1.46191 w_k1_DIS_s012
                                 + 1.51054 w_k1_FDX_s012
                                 + 1.00249 w_k1_IBM_s012
                                 + 1.32911 w_k1_INTC_s012
                                 + 1.11769 w_k1_KO_s012 + 1.56131 w_k1_MAT_s012
                                 + 1.3771 w_k1_MSFT_s012 + 1.20643 w_k1_PG_s012
                                 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.12313 w_k1_ADM_s012
                                 + 1.16559 w_k1_AXP_s012
                                 + 1.27559 w_k1_DIS_s012
                                 + 1.04975 w_k1_FDX_s012
                                 + 1.42899 w_k1_IBM_s012
                                 + 1.09147 w_k1_INTC_s012
                                 + 1.22356 w_k1_KO_s012 + 1.63434 w_k1_MAT_s012
                                 + 1.16903 w_k1_MSFT_s012
                                 + 1.03454 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.781239 w_k1_ADM_s012
                                 + 0.51115 w_k1_AXP_s012
                                 + 0.601858 w_k1_DIS_s012
                                 + 0.745625 w_k1_FDX_s012
                                 + 0.577804 w_k1_IBM_s012
                                 + 0.604037 w_k1_INTC_s012
                                 + 0.571913 w_k1_KO_s012
                                 + 0.511717 w_k1_MAT_s012
                                 + 0.710459 w_k1_MSFT_s012
                                 + 0.695929 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.773889 w_k1_ADM_s012
                                 + 0.880705 w_k1_AXP_s012
                                 + 0.772475 w_k1_DIS_s012
                                 + 0.927831 w_k1_FDX_s012
                                 + 0.657333 w_k1_IBM_s012
                                 + 0.888016 w_k1_INTC_s012
                                 + 0.67492 w_k1_KO_s012 + 0.80689 w_k1_MAT_s012
                                 + 0.949793 w_k1_MSFT_s012
                                 + 0.874941 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.13513 w_k1_ADM_s013
                                 + 0.966912 w_k1_AXP_s013
                                 + 1.1402 w_k1_DIS_s013
                                 + 0.963513 w_k1_FDX_s013
                                 + 1.29096 w_k1_IBM_s013
                                 + 1.25488 w_k1_INTC_s013
                                 + 1.38334 w_k1_KO_s013 + 1.18512 w_k1_MAT_s013
                                 + 1.11248 w_k1_MSFT_s013
                                 + 1.11232 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.977539 w_k1_ADM_s013
                                 + 0.756511 w_k1_AXP_s013
                                 + 0.817962 w_k1_DIS_s013
                                 + 0.83698 w_k1_FDX_s013 + 1.0309 w_k1_IBM_s013
                                 + 1.0098 w_k1_INTC_s013 + 1.02396 w_k1_KO_s013
                                 + 0.925886 w_k1_MAT_s013
                                 + 0.952756 w_k1_MSFT_s013
                                 + 0.898478 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.27469 w_k1_ADM_s013
                                 + 1.4482 w_k1_AXP_s013 + 1.51275 w_k1_DIS_s013
                                 + 1.43456 w_k1_FDX_s013 + 1.3288 w_k1_IBM_s013
                                 + 1.34926 w_k1_INTC_s013
                                 + 1.36783 w_k1_KO_s013 + 1.62963 w_k1_MAT_s013
                                 + 1.20241 w_k1_MSFT_s013 + 1.1954 w_k1_PG_s013
                                 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.985854 w_k1_ADM_s013
                                 + 0.827553 w_k1_AXP_s013
                                 + 1.08038 w_k1_DIS_s013
                                 + 1.05668 w_k1_FDX_s013
                                 + 0.963384 w_k1_IBM_s013
                                 + 1.0501 w_k1_INTC_s013 + 1.07715 w_k1_KO_s013
                                 + 1.14812 w_k1_MAT_s013
                                 + 1.08493 w_k1_MSFT_s013
                                 + 1.05074 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.18475 w_k1_ADM_s013
                                 + 1.18847 w_k1_AXP_s013
                                 + 1.00623 w_k1_DIS_s013
                                 + 1.17296 w_k1_FDX_s013 + 1.1856 w_k1_IBM_s013
                                 + 0.987291 w_k1_INTC_s013
                                 + 1.04665 w_k1_KO_s013 + 1.14563 w_k1_MAT_s013
                                 + 1.03461 w_k1_MSFT_s013 + 1.1111 w_k1_PG_s013
                                 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.03968 w_k1_ADM_s013
                                 + 0.806381 w_k1_AXP_s013
                                 + 0.876837 w_k1_DIS_s013
                                 + 1.1014 w_k1_FDX_s013
                                 + 0.764232 w_k1_IBM_s013
                                 + 1.12022 w_k1_INTC_s013
                                 + 0.932641 w_k1_KO_s013
                                 + 0.770965 w_k1_MAT_s013
                                 + 1.04563 w_k1_MSFT_s013
                                 + 0.873101 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.02494 w_k1_ADM_s013
                                 + 1.10731 w_k1_AXP_s013
                                 + 1.07889 w_k1_DIS_s013
                                 + 1.07514 w_k1_FDX_s013
                                 + 1.61464 w_k1_IBM_s013
                                 + 1.06388 w_k1_INTC_s013
                                 + 1.19386 w_k1_KO_s013 + 1.58349 w_k1_MAT_s013
                                 + 1.11889 w_k1_MSFT_s013
                                 + 0.99275 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.753743 w_k1_ADM_s013
                                 + 0.704668 w_k1_AXP_s013
                                 + 0.735459 w_k1_DIS_s013
                                 + 0.824388 w_k1_FDX_s013
                                 + 0.810752 w_k1_IBM_s013
                                 + 0.945957 w_k1_INTC_s013
                                 + 0.797064 w_k1_KO_s013
                                 + 0.792886 w_k1_MAT_s013
                                 + 0.913658 w_k1_MSFT_s013
                                 + 0.871216 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.15406 w_k1_ADM_s014
                                 + 0.798183 w_k1_AXP_s014
                                 + 1.04149 w_k1_DIS_s014
                                 + 0.998902 w_k1_FDX_s014
                                 + 0.95933 w_k1_IBM_s014
                                 + 1.09959 w_k1_INTC_s014
                                 + 0.970894 w_k1_KO_s014
                                 + 0.450099 w_k1_MAT_s014
                                 + 1.09952 w_k1_MSFT_s014
                                 + 0.995931 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.25042 w_k1_ADM_s014
                                 + 1.31724 w_k1_AXP_s014
                                 + 2.02538 w_k1_DIS_s014
                                 + 1.62603 w_k1_FDX_s014
                                 + 1.03636 w_k1_IBM_s014
                                 + 1.65948 w_k1_INTC_s014
                                 + 1.31274 w_k1_KO_s014 + 1.02574 w_k1_MAT_s014
                                 + 1.5313 w_k1_MSFT_s014 + 1.08507 w_k1_PG_s014
                                 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.949277 w_k1_ADM_s014
                                 + 0.813417 w_k1_AXP_s014
                                 + 0.743877 w_k1_DIS_s014
                                 + 0.763754 w_k1_FDX_s014
                                 + 1.15237 w_k1_IBM_s014
                                 + 0.747122 w_k1_INTC_s014
                                 + 0.904389 w_k1_KO_s014
                                 + 1.15914 w_k1_MAT_s014
                                 + 0.77108 w_k1_MSFT_s014
                                 + 0.850326 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.03622 w_k1_ADM_s014
                                 + 0.903408 w_k1_AXP_s014
                                 + 1.20666 w_k1_DIS_s014
                                 + 1.03933 w_k1_FDX_s014
                                 + 0.787224 w_k1_IBM_s014
                                 + 1.16788 w_k1_INTC_s014
                                 + 0.982268 w_k1_KO_s014
                                 + 0.647001 w_k1_MAT_s014
                                 + 1.24427 w_k1_MSFT_s014
                                 + 0.991189 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.930011 w_k1_ADM_s014
                                 + 1.39005 w_k1_AXP_s014
                                 + 1.15171 w_k1_DIS_s014
                                 + 1.22903 w_k1_FDX_s014
                                 + 0.846248 w_k1_IBM_s014
                                 + 1.39673 w_k1_INTC_s014
                                 + 1.00881 w_k1_KO_s014
                                 + 0.966395 w_k1_MAT_s014
                                 + 1.11249 w_k1_MSFT_s014 + 1.0306 w_k1_PG_s014
                                 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.93494 w_k1_ADM_s014
                                 + 0.874888 w_k1_AXP_s014
                                 + 0.733901 w_k1_DIS_s014
                                 + 0.937696 w_k1_FDX_s014
                                 + 0.704052 w_k1_IBM_s014
                                 + 0.85438 w_k1_INTC_s014
                                 + 0.704836 w_k1_KO_s014
                                 + 0.663937 w_k1_MAT_s014
                                 + 0.849239 w_k1_MSFT_s014
                                 + 0.895516 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.989348 w_k1_ADM_s014
                                 + 1.062 w_k1_AXP_s014 + 1.25353 w_k1_DIS_s014
                                 + 1.03324 w_k1_FDX_s014 + 1.7531 w_k1_IBM_s014
                                 + 1.212 w_k1_INTC_s014 + 1.24436 w_k1_KO_s014
                                 + 1.91402 w_k1_MAT_s014
                                 + 1.09894 w_k1_MSFT_s014
                                 + 1.12506 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.13664 w_k1_ADM_s014
                                 + 1.1323 w_k1_AXP_s014 + 1.41478 w_k1_DIS_s014
                                 + 1.15601 w_k1_FDX_s014
                                 + 1.16674 w_k1_IBM_s014
                                 + 1.46619 w_k1_INTC_s014
                                 + 1.44331 w_k1_KO_s014 + 1.30611 w_k1_MAT_s014
                                 + 1.3971 w_k1_MSFT_s014 + 1.21586 w_k1_PG_s014
                                 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.02084 w_k1_ADM_s015
                                 + 1.12254 w_k1_AXP_s015
                                 + 1.06496 w_k1_DIS_s015
                                 + 1.02129 w_k1_FDX_s015
                                 + 1.37691 w_k1_IBM_s015
                                 + 0.992951 w_k1_INTC_s015
                                 + 1.17977 w_k1_KO_s015 + 1.49291 w_k1_MAT_s015
                                 + 0.870415 w_k1_MSFT_s015
                                 + 1.10472 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.35349 w_k1_ADM_s015
                                 + 1.19147 w_k1_AXP_s015
                                 + 1.15954 w_k1_DIS_s015
                                 + 1.37309 w_k1_FDX_s015
                                 + 0.676861 w_k1_IBM_s015
                                 + 1.04303 w_k1_INTC_s015
                                 + 0.962943 w_k1_KO_s015
                                 + 1.37553 w_k1_MAT_s015
                                 + 1.12774 w_k1_MSFT_s015
                                 + 1.08038 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.969433 w_k1_ADM_s015
                                 + 0.984681 w_k1_AXP_s015
                                 + 0.764379 w_k1_DIS_s015
                                 + 0.876614 w_k1_FDX_s015
                                 + 1.10939 w_k1_IBM_s015
                                 + 0.712374 w_k1_INTC_s015
                                 + 0.966685 w_k1_KO_s015
                                 + 1.06065 w_k1_MAT_s015
                                 + 0.838194 w_k1_MSFT_s015
                                 + 0.998063 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.720589 w_k1_ADM_s015
                                 + 0.666702 w_k1_AXP_s015
                                 + 0.603479 w_k1_DIS_s015
                                 + 0.744643 w_k1_FDX_s015
                                 + 0.685408 w_k1_IBM_s015
                                 + 0.678901 w_k1_INTC_s015
                                 + 0.60581 w_k1_KO_s015 + 0.95263 w_k1_MAT_s015
                                 + 0.722526 w_k1_MSFT_s015
                                 + 0.782861 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.83733 w_k1_ADM_s015
                                 + 0.63826 w_k1_AXP_s015
                                 + 0.75944 w_k1_DIS_s015
                                 + 0.821329 w_k1_FDX_s015
                                 + 0.870528 w_k1_IBM_s015
                                 + 0.720124 w_k1_INTC_s015
                                 + 0.793478 w_k1_KO_s015
                                 + 0.590191 w_k1_MAT_s015
                                 + 0.68973 w_k1_MSFT_s015
                                 + 0.866576 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.02123 w_k1_ADM_s015
                                 + 1.28606 w_k1_AXP_s015
                                 + 1.61985 w_k1_DIS_s015
                                 + 1.40801 w_k1_FDX_s015
                                 + 1.25246 w_k1_IBM_s015
                                 + 1.14661 w_k1_INTC_s015
                                 + 1.36718 w_k1_KO_s015 + 1.5159 w_k1_MAT_s015
                                 + 1.31793 w_k1_MSFT_s015 + 1.2365 w_k1_PG_s015
                                 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.20414 w_k1_ADM_s015
                                 + 0.934166 w_k1_AXP_s015
                                 + 0.994673 w_k1_DIS_s015
                                 + 1.00756 w_k1_FDX_s015
                                 + 1.14943 w_k1_IBM_s015
                                 + 0.695435 w_k1_INTC_s015
                                 + 1.02767 w_k1_KO_s015 + 1.3159 w_k1_MAT_s015
                                 + 0.956346 w_k1_MSFT_s015
                                 + 1.03239 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.03896 w_k1_ADM_s015
                                 + 0.768344 w_k1_AXP_s015
                                 + 1.09405 w_k1_DIS_s015
                                 + 1.06029 w_k1_FDX_s015
                                 + 0.883846 w_k1_IBM_s015
                                 + 1.00834 w_k1_INTC_s015
                                 + 0.996472 w_k1_KO_s015
                                 + 0.758823 w_k1_MAT_s015
                                 + 1.16343 w_k1_MSFT_s015
                                 + 1.00766 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.04407 w_k1_ADM_s016
                                 + 0.93784 w_k1_AXP_s016
                                 + 1.22118 w_k1_DIS_s016
                                 + 1.05761 w_k1_FDX_s016
                                 + 1.04494 w_k1_IBM_s016
                                 + 1.0042 w_k1_INTC_s016 + 1.12367 w_k1_KO_s016
                                 + 0.672782 w_k1_MAT_s016
                                 + 1.07298 w_k1_MSFT_s016
                                 + 1.05223 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.16031 w_k1_ADM_s016
                                 + 1.59252 w_k1_AXP_s016
                                 + 1.78112 w_k1_DIS_s016
                                 + 1.51675 w_k1_FDX_s016
                                 + 1.15342 w_k1_IBM_s016
                                 + 1.47866 w_k1_INTC_s016
                                 + 1.23807 w_k1_KO_s016 + 1.5071 w_k1_MAT_s016
                                 + 1.45585 w_k1_MSFT_s016
                                 + 1.17065 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.77365 w_k1_ADM_s016
                                 + 1.00511 w_k1_AXP_s016
                                 + 0.901775 w_k1_DIS_s016
                                 + 0.918967 w_k1_FDX_s016
                                 + 0.992741 w_k1_IBM_s016
                                 + 0.960165 w_k1_INTC_s016
                                 + 0.869293 w_k1_KO_s016
                                 + 1.19598 w_k1_MAT_s016
                                 + 1.00018 w_k1_MSFT_s016
                                 + 0.956464 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.02499 w_k1_ADM_s016
                                 + 1.02218 w_k1_AXP_s016
                                 + 0.979231 w_k1_DIS_s016
                                 + 1.10877 w_k1_FDX_s016
                                 + 0.707756 w_k1_IBM_s016
                                 + 1.14047 w_k1_INTC_s016
                                 + 0.939637 w_k1_KO_s016
                                 + 0.912165 w_k1_MAT_s016
                                 + 1.12697 w_k1_MSFT_s016
                                 + 1.00974 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.783222 w_k1_ADM_s016
                                 + 0.739463 w_k1_AXP_s016
                                 + 0.744235 w_k1_DIS_s016
                                 + 0.743732 w_k1_FDX_s016
                                 + 1.13327 w_k1_IBM_s016
                                 + 0.798042 w_k1_INTC_s016
                                 + 0.949457 w_k1_KO_s016
                                 + 1.10801 w_k1_MAT_s016
                                 + 0.825606 w_k1_MSFT_s016
                                 + 0.890276 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.852788 w_k1_ADM_s016
                                 + 0.667982 w_k1_AXP_s016
                                 + 0.870977 w_k1_DIS_s016
                                 + 0.839843 w_k1_FDX_s016
                                 + 0.685588 w_k1_IBM_s016
                                 + 0.757741 w_k1_INTC_s016
                                 + 0.737915 w_k1_KO_s016
                                 + 0.522365 w_k1_MAT_s016
                                 + 0.868473 w_k1_MSFT_s016
                                 + 0.764572 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.653218 w_k1_ADM_s016
                                 + 0.620875 w_k1_AXP_s016
                                 + 0.536288 w_k1_DIS_s016
                                 + 0.668678 w_k1_FDX_s016
                                 + 0.782914 w_k1_IBM_s016
                                 + 0.811521 w_k1_INTC_s016
                                 + 0.778525 w_k1_KO_s016
                                 + 0.81606 w_k1_MAT_s016
                                 + 0.713542 w_k1_MSFT_s016
                                 + 0.798324 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.976958 w_k1_ADM_s016
                                 + 1.03671 w_k1_AXP_s016
                                 + 1.15091 w_k1_DIS_s016
                                 + 1.02483 w_k1_FDX_s016
                                 + 1.42073 w_k1_IBM_s016
                                 + 1.22768 w_k1_INTC_s016
                                 + 1.25618 w_k1_KO_s016 + 1.57004 w_k1_MAT_s016
                                 + 1.17019 w_k1_MSFT_s016
                                 + 1.14264 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.3402 w_k1_ADM_s017
                                 + 1.46947 w_k1_AXP_s017
                                 + 1.69015 w_k1_DIS_s017
                                 + 1.44566 w_k1_FDX_s017
                                 + 1.24656 w_k1_IBM_s017
                                 + 1.24726 w_k1_INTC_s017
                                 + 1.50608 w_k1_KO_s017 + 1.50074 w_k1_MAT_s017
                                 + 1.37123 w_k1_MSFT_s017
                                 + 1.11221 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.12612 w_k1_ADM_s017
                                 + 1.37835 w_k1_AXP_s017
                                 + 1.04208 w_k1_DIS_s017
                                 + 1.17805 w_k1_FDX_s017
                                 + 0.976413 w_k1_IBM_s017
                                 + 1.29026 w_k1_INTC_s017
                                 + 1.07916 w_k1_KO_s017 + 1.34965 w_k1_MAT_s017
                                 + 1.22936 w_k1_MSFT_s017
                                 + 1.06603 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.16275 w_k1_ADM_s017
                                 + 0.72106 w_k1_AXP_s017
                                 + 0.731082 w_k1_DIS_s017
                                 + 0.744425 w_k1_FDX_s017
                                 + 0.972657 w_k1_IBM_s017
                                 + 0.823991 w_k1_INTC_s017
                                 + 0.841454 w_k1_KO_s017
                                 + 0.715767 w_k1_MAT_s017
                                 + 0.828846 w_k1_MSFT_s017
                                 + 0.820607 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.35066 w_k1_ADM_s017
                                 + 1.07767 w_k1_AXP_s017 + 1.2879 w_k1_DIS_s017
                                 + 1.09157 w_k1_FDX_s017
                                 + 1.39787 w_k1_IBM_s017
                                 + 1.39833 w_k1_INTC_s017
                                 + 1.50673 w_k1_KO_s017 + 1.07739 w_k1_MAT_s017
                                 + 1.1903 w_k1_MSFT_s017 + 1.0614 w_k1_PG_s017
                                 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.47605 w_k1_ADM_s017
                                 + 1.02376 w_k1_AXP_s017
                                 + 1.30208 w_k1_DIS_s017
                                 + 1.16821 w_k1_FDX_s017
                                 + 1.01029 w_k1_IBM_s017
                                 + 1.14978 w_k1_INTC_s017
                                 + 1.30713 w_k1_KO_s017
                                 + 0.552849 w_k1_MAT_s017
                                 + 1.31249 w_k1_MSFT_s017
                                 + 1.19541 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.14234 w_k1_ADM_s017
                                 + 1.00091 w_k1_AXP_s017
                                 + 0.962127 w_k1_DIS_s017
                                 + 1.04638 w_k1_FDX_s017
                                 + 0.773473 w_k1_IBM_s017
                                 + 1.27066 w_k1_INTC_s017
                                 + 0.893165 w_k1_KO_s017
                                 + 0.636914 w_k1_MAT_s017
                                 + 1.12689 w_k1_MSFT_s017
                                 + 0.923392 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.12858 w_k1_ADM_s017
                                 + 1.07803 w_k1_AXP_s017
                                 + 0.992041 w_k1_DIS_s017
                                 + 0.940693 w_k1_FDX_s017
                                 + 1.36622 w_k1_IBM_s017
                                 + 0.949831 w_k1_INTC_s017
                                 + 1.03867 w_k1_KO_s017
                                 + 0.857513 w_k1_MAT_s017
                                 + 0.940323 w_k1_MSFT_s017
                                 + 0.989673 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.21295 w_k1_ADM_s017
                                 + 0.924076 w_k1_AXP_s017
                                 + 1.09035 w_k1_DIS_s017
                                 + 0.899707 w_k1_FDX_s017
                                 + 1.01135 w_k1_IBM_s017
                                 + 0.833713 w_k1_INTC_s017
                                 + 1.22808 w_k1_KO_s017
                                 + 0.958435 w_k1_MAT_s017
                                 + 1.07056 w_k1_MSFT_s017
                                 + 0.997178 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.04889 w_k1_ADM_s100
                                 + 0.823642 w_k1_AXP_s100
                                 + 0.954812 w_k1_DIS_s100
                                 + 0.861847 w_k1_FDX_s100
                                 + 0.962592 w_k1_IBM_s100
                                 + 0.736152 w_k1_INTC_s100
                                 + 0.938097 w_k1_KO_s100
                                 + 1.11198 w_k1_MAT_s100
                                 + 0.983398 w_k1_MSFT_s100
                                 + 1.08475 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.7673 w_k1_ADM_s100
                                 + 1.25001 w_k1_AXP_s100
                                 + 1.50094 w_k1_DIS_s100
                                 + 1.41125 w_k1_FDX_s100
                                 + 0.859208 w_k1_IBM_s100
                                 + 1.04467 w_k1_INTC_s100
                                 + 1.23753 w_k1_KO_s100 + 1.14253 w_k1_MAT_s100
                                 + 1.41738 w_k1_MSFT_s100
                                 + 1.28305 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.16892 w_k1_ADM_s100
                                 + 1.43162 w_k1_AXP_s100
                                 + 1.28429 w_k1_DIS_s100 + 1.147 w_k1_FDX_s100
                                 + 1.01796 w_k1_IBM_s100
                                 + 0.804678 w_k1_INTC_s100
                                 + 1.02064 w_k1_KO_s100 + 1.47123 w_k1_MAT_s100
                                 + 1.13338 w_k1_MSFT_s100
                                 + 1.14604 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.3627 w_k1_ADM_s100
                                 + 1.40052 w_k1_AXP_s100
                                 + 1.01318 w_k1_DIS_s100
                                 + 1.02272 w_k1_FDX_s100
                                 + 1.44493 w_k1_IBM_s100
                                 + 1.03138 w_k1_INTC_s100
                                 + 1.03657 w_k1_KO_s100 + 1.28968 w_k1_MAT_s100
                                 + 1.00907 w_k1_MSFT_s100
                                 + 1.04933 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.18295 w_k1_ADM_s100
                                 + 1.60663 w_k1_AXP_s100
                                 + 1.62743 w_k1_DIS_s100 + 1.1993 w_k1_FDX_s100
                                 + 1.92079 w_k1_IBM_s100
                                 + 1.16031 w_k1_INTC_s100 + 1.4479 w_k1_KO_s100
                                 + 2.57607 w_k1_MAT_s100
                                 + 1.20599 w_k1_MSFT_s100
                                 + 1.36916 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.910063 w_k1_ADM_s100
                                 + 0.760556 w_k1_AXP_s100
                                 + 0.615762 w_k1_DIS_s100
                                 + 0.678473 w_k1_FDX_s100
                                 + 0.793732 w_k1_IBM_s100
                                 + 0.641211 w_k1_INTC_s100
                                 + 0.889876 w_k1_KO_s100
                                 + 1.00484 w_k1_MAT_s100
                                 + 0.762961 w_k1_MSFT_s100
                                 + 0.908242 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.12675 w_k1_ADM_s100
                                 + 1.12875 w_k1_AXP_s100
                                 + 0.890622 w_k1_DIS_s100
                                 + 1.09145 w_k1_FDX_s100
                                 + 0.792033 w_k1_IBM_s100
                                 + 0.767355 w_k1_INTC_s100
                                 + 0.814353 w_k1_KO_s100
                                 + 0.990083 w_k1_MAT_s100
                                 + 1.00006 w_k1_MSFT_s100
                                 + 1.01753 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.0819 w_k1_ADM_s100
                                 + 0.689421 w_k1_AXP_s100
                                 + 0.684284 w_k1_DIS_s100
                                 + 0.835023 w_k1_FDX_s100
                                 + 0.695444 w_k1_IBM_s100
                                 + 0.820599 w_k1_INTC_s100
                                 + 0.740699 w_k1_KO_s100
                                 + 0.768264 w_k1_MAT_s100
                                 + 0.850004 w_k1_MSFT_s100
                                 + 0.850932 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.971264 w_k1_ADM_s101
                                 + 0.957883 w_k1_AXP_s101
                                 + 1.03372 w_k1_DIS_s101
                                 + 0.977962 w_k1_FDX_s101
                                 + 1.19769 w_k1_IBM_s101
                                 + 1.1742 w_k1_INTC_s101 + 1.15559 w_k1_KO_s101
                                 + 1.35475 w_k1_MAT_s101
                                 + 1.05938 w_k1_MSFT_s101
                                 + 1.02532 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.959048 w_k1_ADM_s101
                                 + 0.955036 w_k1_AXP_s101
                                 + 1.16097 w_k1_DIS_s101
                                 + 1.19564 w_k1_FDX_s101
                                 + 0.80821 w_k1_IBM_s101
                                 + 1.08302 w_k1_INTC_s101
                                 + 0.89305 w_k1_KO_s101
                                 + 0.611866 w_k1_MAT_s101
                                 + 1.15631 w_k1_MSFT_s101
                                 + 0.975855 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.927342 w_k1_ADM_s101
                                 + 0.882354 w_k1_AXP_s101
                                 + 0.953709 w_k1_DIS_s101
                                 + 1.04196 w_k1_FDX_s101
                                 + 0.710639 w_k1_IBM_s101
                                 + 1.3618 w_k1_INTC_s101
                                 + 0.861528 w_k1_KO_s101
                                 + 0.549231 w_k1_MAT_s101
                                 + 1.16435 w_k1_MSFT_s101
                                 + 0.964498 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.983276 w_k1_ADM_s101
                                 + 0.837083 w_k1_AXP_s101
                                 + 0.672906 w_k1_DIS_s101
                                 + 0.829967 w_k1_FDX_s101
                                 + 0.942045 w_k1_IBM_s101
                                 + 0.892899 w_k1_INTC_s101
                                 + 0.840884 w_k1_KO_s101
                                 + 0.886517 w_k1_MAT_s101
                                 + 0.887658 w_k1_MSFT_s101
                                 + 0.932753 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.06791 w_k1_ADM_s101
                                 + 1.0001 w_k1_AXP_s101
                                 + 0.849641 w_k1_DIS_s101
                                 + 0.987199 w_k1_FDX_s101
                                 + 0.898566 w_k1_IBM_s101
                                 + 0.963549 w_k1_INTC_s101
                                 + 0.805242 w_k1_KO_s101 + 0.8468 w_k1_MAT_s101
                                 + 0.951844 w_k1_MSFT_s101
                                 + 1.01042 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.32636 w_k1_ADM_s101
                                 + 1.46167 w_k1_AXP_s101
                                 + 1.55119 w_k1_DIS_s101 + 1.2506 w_k1_FDX_s101
                                 + 1.36792 w_k1_IBM_s101
                                 + 1.6066 w_k1_INTC_s101 + 1.40864 w_k1_KO_s101
                                 + 1.25623 w_k1_MAT_s101
                                 + 1.52616 w_k1_MSFT_s101
                                 + 1.16013 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.04173 w_k1_ADM_s101
                                 + 0.664745 w_k1_AXP_s101
                                 + 0.647903 w_k1_DIS_s101
                                 + 0.838464 w_k1_FDX_s101
                                 + 0.74675 w_k1_IBM_s101
                                 + 1.18411 w_k1_INTC_s101
                                 + 0.75065 w_k1_KO_s101
                                 + 0.498767 w_k1_MAT_s101
                                 + 0.924801 w_k1_MSFT_s101
                                 + 0.837629 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.28608 w_k1_ADM_s101
                                 + 1.1648 w_k1_AXP_s101 + 1.10037 w_k1_DIS_s101
                                 + 1.1456 w_k1_FDX_s101 + 1.14232 w_k1_IBM_s101
                                 + 1.20864 w_k1_INTC_s101
                                 + 1.01943 w_k1_KO_s101 + 1.12767 w_k1_MAT_s101
                                 + 1.04167 w_k1_MSFT_s101
                                 + 1.11864 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.03923 w_k1_ADM_s102
                                 + 1.04552 w_k1_AXP_s102
                                 + 1.46811 w_k1_DIS_s102
                                 + 1.28893 w_k1_FDX_s102
                                 + 1.28189 w_k1_IBM_s102
                                 + 1.18453 w_k1_INTC_s102
                                 + 1.44799 w_k1_KO_s102 + 1.08865 w_k1_MAT_s102
                                 + 1.35749 w_k1_MSFT_s102
                                 + 1.10878 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.972052 w_k1_ADM_s102
                                 + 1.183 w_k1_AXP_s102 + 1.51908 w_k1_DIS_s102
                                 + 1.39838 w_k1_FDX_s102
                                 + 0.920316 w_k1_IBM_s102
                                 + 1.35583 w_k1_INTC_s102
                                 + 1.31802 w_k1_KO_s102 + 1.00145 w_k1_MAT_s102
                                 + 1.5387 w_k1_MSFT_s102 + 1.22216 w_k1_PG_s102
                                 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.757892 w_k1_ADM_s102
                                 + 0.699885 w_k1_AXP_s102
                                 + 0.658494 w_k1_DIS_s102
                                 + 0.814151 w_k1_FDX_s102
                                 + 0.686085 w_k1_IBM_s102
                                 + 0.745682 w_k1_INTC_s102
                                 + 0.730609 w_k1_KO_s102
                                 + 0.79026 w_k1_MAT_s102
                                 + 0.827546 w_k1_MSFT_s102
                                 + 0.852071 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.22387 w_k1_ADM_s102
                                 + 1.02852 w_k1_AXP_s102
                                 + 0.978595 w_k1_DIS_s102
                                 + 1.11575 w_k1_FDX_s102
                                 + 0.977799 w_k1_IBM_s102
                                 + 1.02775 w_k1_INTC_s102
                                 + 1.08323 w_k1_KO_s102
                                 + 0.990503 w_k1_MAT_s102
                                 + 1.07497 w_k1_MSFT_s102
                                 + 1.07193 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.61605 w_k1_ADM_s102
                                 + 1.71175 w_k1_AXP_s102
                                 + 1.44779 w_k1_DIS_s102 + 1.3274 w_k1_FDX_s102
                                 + 1.48137 w_k1_IBM_s102
                                 + 1.31452 w_k1_INTC_s102
                                 + 1.39084 w_k1_KO_s102 + 2.06565 w_k1_MAT_s102
                                 + 1.34447 w_k1_MSFT_s102
                                 + 1.31145 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.10189 w_k1_ADM_s102
                                 + 1.05945 w_k1_AXP_s102
                                 + 1.03901 w_k1_DIS_s102
                                 + 1.16773 w_k1_FDX_s102
                                 + 0.746539 w_k1_IBM_s102
                                 + 1.09349 w_k1_INTC_s102
                                 + 0.861748 w_k1_KO_s102
                                 + 0.698901 w_k1_MAT_s102
                                 + 1.05111 w_k1_MSFT_s102
                                 + 1.09056 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.934074 w_k1_ADM_s102
                                 + 0.850541 w_k1_AXP_s102
                                 + 0.788793 w_k1_DIS_s102
                                 + 0.865855 w_k1_FDX_s102
                                 + 1.31181 w_k1_IBM_s102
                                 + 0.877176 w_k1_INTC_s102
                                 + 0.956977 w_k1_KO_s102 + 1.1003 w_k1_MAT_s102
                                 + 0.917416 w_k1_MSFT_s102
                                 + 0.959971 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.932612 w_k1_ADM_s102
                                 + 0.887342 w_k1_AXP_s102
                                 + 1.00775 w_k1_DIS_s102
                                 + 0.994297 w_k1_FDX_s102
                                 + 0.963788 w_k1_IBM_s102
                                 + 1.02709 w_k1_INTC_s102
                                 + 0.972656 w_k1_KO_s102
                                 + 0.667153 w_k1_MAT_s102
                                 + 1.05821 w_k1_MSFT_s102
                                 + 0.966726 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.923542 w_k1_ADM_s103
                                 + 1.05971 w_k1_AXP_s103
                                 + 0.904887 w_k1_DIS_s103
                                 + 0.903769 w_k1_FDX_s103
                                 + 1.49704 w_k1_IBM_s103
                                 + 0.874519 w_k1_INTC_s103
                                 + 1.04543 w_k1_KO_s103 + 1.42887 w_k1_MAT_s103
                                 + 0.889483 w_k1_MSFT_s103
                                 + 1.04432 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.922919 w_k1_ADM_s103
                                 + 1.35671 w_k1_AXP_s103
                                 + 1.46864 w_k1_DIS_s103
                                 + 1.03926 w_k1_FDX_s103
                                 + 1.95367 w_k1_IBM_s103
                                 + 1.19184 w_k1_INTC_s103
                                 + 1.70941 w_k1_KO_s103 + 1.7472 w_k1_MAT_s103
                                 + 1.21776 w_k1_MSFT_s103
                                 + 1.35761 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.3255 w_k1_ADM_s103
                                 + 1.07187 w_k1_AXP_s103
                                 + 1.49537 w_k1_DIS_s103
                                 + 1.33469 w_k1_FDX_s103
                                 + 0.727911 w_k1_IBM_s103
                                 + 1.03922 w_k1_INTC_s103
                                 + 1.05185 w_k1_KO_s103
                                 + 0.760521 w_k1_MAT_s103
                                 + 1.13972 w_k1_MSFT_s103
                                 + 1.12411 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.04299 w_k1_ADM_s103
                                 + 1.19731 w_k1_AXP_s103
                                 + 1.17595 w_k1_DIS_s103
                                 + 1.02068 w_k1_FDX_s103
                                 + 1.02075 w_k1_IBM_s103
                                 + 0.965687 w_k1_INTC_s103
                                 + 1.0004 w_k1_KO_s103 + 1.2548 w_k1_MAT_s103
                                 + 0.954996 w_k1_MSFT_s103
                                 + 0.998327 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.852439 w_k1_ADM_s103
                                 + 0.724627 w_k1_AXP_s103
                                 + 0.785486 w_k1_DIS_s103
                                 + 0.816456 w_k1_FDX_s103
                                 + 0.8116 w_k1_IBM_s103
                                 + 0.719623 w_k1_INTC_s103
                                 + 0.701203 w_k1_KO_s103
                                 + 0.764377 w_k1_MAT_s103
                                 + 0.751165 w_k1_MSFT_s103
                                 + 0.868932 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.54313 w_k1_ADM_s103
                                 + 1.70864 w_k1_AXP_s103 + 1.9259 w_k1_DIS_s103
                                 + 1.33575 w_k1_FDX_s103
                                 + 1.43907 w_k1_IBM_s103
                                 + 1.14666 w_k1_INTC_s103
                                 + 1.43678 w_k1_KO_s103 + 2.20699 w_k1_MAT_s103
                                 + 1.28872 w_k1_MSFT_s103
                                 + 1.37672 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.8893 w_k1_ADM_s103
                                 + 1.08883 w_k1_AXP_s103
                                 + 0.677139 w_k1_DIS_s103
                                 + 0.803859 w_k1_FDX_s103
                                 + 1.02987 w_k1_IBM_s103
                                 + 0.774781 w_k1_INTC_s103
                                 + 0.752689 w_k1_KO_s103
                                 + 1.15805 w_k1_MAT_s103
                                 + 0.701639 w_k1_MSFT_s103
                                 + 0.988434 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.02851 w_k1_ADM_s103
                                 + 1.29244 w_k1_AXP_s103
                                 + 1.29789 w_k1_DIS_s103 + 1.2287 w_k1_FDX_s103
                                 + 0.904637 w_k1_IBM_s103
                                 + 1.05211 w_k1_INTC_s103
                                 + 0.943592 w_k1_KO_s103
                                 + 1.17356 w_k1_MAT_s103
                                 + 1.13539 w_k1_MSFT_s103
                                 + 1.04924 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.08314 w_k1_ADM_s104
                                 + 1.34345 w_k1_AXP_s104
                                 + 1.28974 w_k1_DIS_s104
                                 + 1.14398 w_k1_FDX_s104
                                 + 1.26976 w_k1_IBM_s104
                                 + 1.14955 w_k1_INTC_s104
                                 + 1.33703 w_k1_KO_s104 + 1.87372 w_k1_MAT_s104
                                 + 1.09579 w_k1_MSFT_s104
                                 + 1.14785 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.18362 w_k1_ADM_s104
                                 + 1.18383 w_k1_AXP_s104
                                 + 1.04429 w_k1_DIS_s104
                                 + 1.16219 w_k1_FDX_s104
                                 + 0.798599 w_k1_IBM_s104
                                 + 1.10492 w_k1_INTC_s104
                                 + 0.785917 w_k1_KO_s104
                                 + 0.724415 w_k1_MAT_s104
                                 + 1.16371 w_k1_MSFT_s104
                                 + 1.02229 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.11528 w_k1_ADM_s104
                                 + 0.919695 w_k1_AXP_s104
                                 + 0.902044 w_k1_DIS_s104
                                 + 0.976224 w_k1_FDX_s104
                                 + 0.715067 w_k1_IBM_s104
                                 + 0.948533 w_k1_INTC_s104
                                 + 0.827327 w_k1_KO_s104
                                 + 0.929698 w_k1_MAT_s104
                                 + 1.00553 w_k1_MSFT_s104
                                 + 0.981105 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.9586 w_k1_ADM_s104
                                 + 1.20363 w_k1_AXP_s104
                                 + 1.16248 w_k1_DIS_s104
                                 + 0.907793 w_k1_FDX_s104
                                 + 1.62234 w_k1_IBM_s104
                                 + 0.96639 w_k1_INTC_s104
                                 + 1.18121 w_k1_KO_s104 + 2.07529 w_k1_MAT_s104
                                 + 1.02638 w_k1_MSFT_s104
                                 + 1.11205 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.61546 w_k1_ADM_s104
                                 + 1.56593 w_k1_AXP_s104
                                 + 1.32991 w_k1_DIS_s104
                                 + 1.41171 w_k1_FDX_s104
                                 + 0.996495 w_k1_IBM_s104
                                 + 1.29939 w_k1_INTC_s104
                                 + 1.03812 w_k1_KO_s104 + 1.34579 w_k1_MAT_s104
                                 + 1.32717 w_k1_MSFT_s104
                                 + 1.22156 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.01248 w_k1_ADM_s104
                                 + 1.12 w_k1_AXP_s104 + 0.812562 w_k1_DIS_s104
                                 + 0.863905 w_k1_FDX_s104
                                 + 1.09078 w_k1_IBM_s104
                                 + 1.01413 w_k1_INTC_s104
                                 + 0.88784 w_k1_KO_s104 + 1.47132 w_k1_MAT_s104
                                 + 0.915989 w_k1_MSFT_s104
                                 + 0.936935 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.868114 w_k1_ADM_s104
                                 + 0.61862 w_k1_AXP_s104
                                 + 0.606447 w_k1_DIS_s104
                                 + 0.714487 w_k1_FDX_s104
                                 + 0.696658 w_k1_IBM_s104
                                 + 0.913155 w_k1_INTC_s104
                                 + 0.702783 w_k1_KO_s104
                                 + 0.647321 w_k1_MAT_s104
                                 + 0.728673 w_k1_MSFT_s104
                                 + 0.737559 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.784217 w_k1_ADM_s104
                                 + 0.638423 w_k1_AXP_s104
                                 + 0.974953 w_k1_DIS_s104
                                 + 0.887961 w_k1_FDX_s104
                                 + 0.814977 w_k1_IBM_s104
                                 + 0.943141 w_k1_INTC_s104
                                 + 0.829632 w_k1_KO_s104
                                 + 0.627458 w_k1_MAT_s104
                                 + 0.940927 w_k1_MSFT_s104
                                 + 0.90134 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.04492 w_k1_ADM_s105
                                 + 0.999328 w_k1_AXP_s105
                                 + 1.05131 w_k1_DIS_s105
                                 + 1.00376 w_k1_FDX_s105
                                 + 1.48494 w_k1_IBM_s105
                                 + 0.911269 w_k1_INTC_s105
                                 + 1.11158 w_k1_KO_s105 + 1.70335 w_k1_MAT_s105
                                 + 1.10684 w_k1_MSFT_s105
                                 + 1.01633 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.802687 w_k1_ADM_s105
                                 + 0.767558 w_k1_AXP_s105
                                 + 0.906687 w_k1_DIS_s105
                                 + 0.874757 w_k1_FDX_s105
                                 + 0.977892 w_k1_IBM_s105
                                 + 0.811249 w_k1_INTC_s105
                                 + 0.979919 w_k1_KO_s105
                                 + 1.05384 w_k1_MAT_s105
                                 + 1.00896 w_k1_MSFT_s105
                                 + 0.898438 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.800654 w_k1_ADM_s105
                                 + 0.622547 w_k1_AXP_s105
                                 + 0.620053 w_k1_DIS_s105
                                 + 0.662856 w_k1_FDX_s105
                                 + 1.03799 w_k1_IBM_s105
                                 + 0.749862 w_k1_INTC_s105
                                 + 0.835393 w_k1_KO_s105
                                 + 0.717659 w_k1_MAT_s105
                                 + 0.790217 w_k1_MSFT_s105
                                 + 0.840373 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.00243 w_k1_ADM_s105
                                 + 1.10376 w_k1_AXP_s105
                                 + 1.15494 w_k1_DIS_s105
                                 + 1.06341 w_k1_FDX_s105 + 1.0894 w_k1_IBM_s105
                                 + 0.957605 w_k1_INTC_s105
                                 + 1.14237 w_k1_KO_s105 + 1.26781 w_k1_MAT_s105
                                 + 1.08685 w_k1_MSFT_s105
                                 + 1.16359 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.07538 w_k1_ADM_s105
                                 + 1.14066 w_k1_AXP_s105
                                 + 1.47523 w_k1_DIS_s105
                                 + 1.31565 w_k1_FDX_s105
                                 + 0.900715 w_k1_IBM_s105
                                 + 1.00771 w_k1_INTC_s105 + 1.1233 w_k1_KO_s105
                                 + 1.636 w_k1_MAT_s105 + 1.34308 w_k1_MSFT_s105
                                 + 1.15579 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.993515 w_k1_ADM_s105
                                 + 0.791546 w_k1_AXP_s105
                                 + 0.810879 w_k1_DIS_s105
                                 + 0.968798 w_k1_FDX_s105
                                 + 0.591094 w_k1_IBM_s105
                                 + 0.839033 w_k1_INTC_s105
                                 + 0.86119 w_k1_KO_s105
                                 + 0.580422 w_k1_MAT_s105
                                 + 0.932814 w_k1_MSFT_s105
                                 + 0.897413 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.909828 w_k1_ADM_s105
                                 + 0.775241 w_k1_AXP_s105
                                 + 0.617354 w_k1_DIS_s105
                                 + 0.775365 w_k1_FDX_s105
                                 + 0.75234 w_k1_IBM_s105
                                 + 0.877551 w_k1_INTC_s105
                                 + 0.756111 w_k1_KO_s105
                                 + 0.834825 w_k1_MAT_s105
                                 + 0.808005 w_k1_MSFT_s105
                                 + 0.857177 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.937661 w_k1_ADM_s105
                                 + 0.934597 w_k1_AXP_s105
                                 + 0.921626 w_k1_DIS_s105
                                 + 1.10765 w_k1_FDX_s105
                                 + 0.777205 w_k1_IBM_s105
                                 + 0.934603 w_k1_INTC_s105
                                 + 0.964702 w_k1_KO_s105
                                 + 1.01211 w_k1_MAT_s105
                                 + 1.07631 w_k1_MSFT_s105
                                 + 1.04205 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.39486 w_k1_ADM_s106
                                 + 1.61265 w_k1_AXP_s106
                                 + 1.35826 w_k1_DIS_s106
                                 + 1.40778 w_k1_FDX_s106
                                 + 1.02195 w_k1_IBM_s106
                                 + 1.38525 w_k1_INTC_s106
                                 + 1.13212 w_k1_KO_s106 + 1.0872 w_k1_MAT_s106
                                 + 1.15624 w_k1_MSFT_s106
                                 + 1.23229 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.02102 w_k1_ADM_s106
                                 + 0.844232 w_k1_AXP_s106
                                 + 0.975907 w_k1_DIS_s106
                                 + 0.966274 w_k1_FDX_s106
                                 + 0.822195 w_k1_IBM_s106
                                 + 0.950676 w_k1_INTC_s106
                                 + 0.852885 w_k1_KO_s106
                                 + 0.596853 w_k1_MAT_s106
                                 + 0.91137 w_k1_MSFT_s106
                                 + 0.935956 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.828264 w_k1_ADM_s106
                                 + 0.822881 w_k1_AXP_s106
                                 + 0.751999 w_k1_DIS_s106
                                 + 0.862509 w_k1_FDX_s106
                                 + 0.8406 w_k1_IBM_s106
                                 + 0.955996 w_k1_INTC_s106
                                 + 0.718399 w_k1_KO_s106
                                 + 0.961166 w_k1_MAT_s106
                                 + 0.932475 w_k1_MSFT_s106
                                 + 0.902276 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.909262 w_k1_ADM_s106
                                 + 1.01603 w_k1_AXP_s106 + 1.0171 w_k1_DIS_s106
                                 + 0.967653 w_k1_FDX_s106
                                 + 1.2547 w_k1_IBM_s106
                                 + 0.921523 w_k1_INTC_s106
                                 + 1.01175 w_k1_KO_s106 + 1.27496 w_k1_MAT_s106
                                 + 0.896071 w_k1_MSFT_s106
                                 + 1.06066 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.568351 w_k1_ADM_s106
                                 + 0.616875 w_k1_AXP_s106
                                 + 0.620036 w_k1_DIS_s106
                                 + 0.716461 w_k1_FDX_s106
                                 + 0.757216 w_k1_IBM_s106
                                 + 0.886416 w_k1_INTC_s106
                                 + 0.750938 w_k1_KO_s106
                                 + 0.794011 w_k1_MAT_s106
                                 + 0.809148 w_k1_MSFT_s106
                                 + 0.809718 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.662709 w_k1_ADM_s106
                                 + 0.695506 w_k1_AXP_s106
                                 + 0.666414 w_k1_DIS_s106
                                 + 0.731067 w_k1_FDX_s106
                                 + 1.04815 w_k1_IBM_s106
                                 + 0.89246 w_k1_INTC_s106
                                 + 0.844273 w_k1_KO_s106
                                 + 0.808398 w_k1_MAT_s106
                                 + 0.776179 w_k1_MSFT_s106
                                 + 0.886183 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.01251 w_k1_ADM_s106
                                 + 1.06402 w_k1_AXP_s106
                                 + 1.26749 w_k1_DIS_s106
                                 + 1.17655 w_k1_FDX_s106
                                 + 0.88648 w_k1_IBM_s106
                                 + 1.30529 w_k1_INTC_s106 + 1.0404 w_k1_KO_s106
                                 + 0.782712 w_k1_MAT_s106
                                 + 1.12412 w_k1_MSFT_s106
                                 + 1.02626 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.18253 w_k1_ADM_s106
                                 + 1.51095 w_k1_AXP_s106
                                 + 1.81384 w_k1_DIS_s106
                                 + 1.30388 w_k1_FDX_s106
                                 + 1.34484 w_k1_IBM_s106
                                 + 1.38673 w_k1_INTC_s106
                                 + 1.36276 w_k1_KO_s106 + 1.55884 w_k1_MAT_s106
                                 + 1.20503 w_k1_MSFT_s106
                                 + 1.17302 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.0631 w_k1_ADM_s107
                                 + 1.16601 w_k1_AXP_s107
                                 + 1.42474 w_k1_DIS_s107
                                 + 1.22236 w_k1_FDX_s107
                                 + 1.06351 w_k1_IBM_s107
                                 + 1.04255 w_k1_INTC_s107
                                 + 1.10032 w_k1_KO_s107 + 1.45464 w_k1_MAT_s107
                                 + 1.33736 w_k1_MSFT_s107
                                 + 1.10098 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.793356 w_k1_ADM_s107
                                 + 0.718906 w_k1_AXP_s107
                                 + 1.01658 w_k1_DIS_s107
                                 + 0.838121 w_k1_FDX_s107
                                 + 0.919437 w_k1_IBM_s107
                                 + 1.0081 w_k1_INTC_s107
                                 + 0.969957 w_k1_KO_s107
                                 + 0.867595 w_k1_MAT_s107
                                 + 1.0764 w_k1_MSFT_s107
                                 + 0.886262 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.00846 w_k1_ADM_s107
                                 + 0.90692 w_k1_AXP_s107
                                 + 0.886985 w_k1_DIS_s107
                                 + 0.896002 w_k1_FDX_s107
                                 + 1.25677 w_k1_IBM_s107
                                 + 0.911651 w_k1_INTC_s107
                                 + 1.06165 w_k1_KO_s107 + 1.13151 w_k1_MAT_s107
                                 + 0.955386 w_k1_MSFT_s107
                                 + 0.966193 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.917157 w_k1_ADM_s107
                                 + 0.6593 w_k1_AXP_s107
                                 + 0.605481 w_k1_DIS_s107
                                 + 0.719842 w_k1_FDX_s107
                                 + 0.816111 w_k1_IBM_s107
                                 + 0.811568 w_k1_INTC_s107
                                 + 0.92139 w_k1_KO_s107 + 0.7444 w_k1_MAT_s107
                                 + 0.857234 w_k1_MSFT_s107
                                 + 0.87479 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.93431 w_k1_ADM_s107
                                 + 0.959725 w_k1_AXP_s107
                                 + 0.993701 w_k1_DIS_s107
                                 + 1.00931 w_k1_FDX_s107
                                 + 0.884374 w_k1_IBM_s107
                                 + 0.849751 w_k1_INTC_s107
                                 + 0.850609 w_k1_KO_s107
                                 + 0.725269 w_k1_MAT_s107
                                 + 1.12187 w_k1_MSFT_s107
                                 + 0.92134 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.37963 w_k1_ADM_s107
                                 + 1.43355 w_k1_AXP_s107
                                 + 1.53854 w_k1_DIS_s107
                                 + 1.17556 w_k1_FDX_s107
                                 + 1.69861 w_k1_IBM_s107
                                 + 0.984934 w_k1_INTC_s107
                                 + 1.6353 w_k1_KO_s107 + 1.80099 w_k1_MAT_s107
                                 + 1.30055 w_k1_MSFT_s107
                                 + 1.35102 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.863094 w_k1_ADM_s107
                                 + 1.03194 w_k1_AXP_s107
                                 + 0.628473 w_k1_DIS_s107
                                 + 0.819502 w_k1_FDX_s107
                                 + 0.836524 w_k1_IBM_s107
                                 + 0.975756 w_k1_INTC_s107
                                 + 0.802758 w_k1_KO_s107
                                 + 0.895379 w_k1_MAT_s107
                                 + 0.842846 w_k1_MSFT_s107
                                 + 0.911035 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.992227 w_k1_ADM_s107
                                 + 0.648374 w_k1_AXP_s107
                                 + 0.888107 w_k1_DIS_s107
                                 + 0.909126 w_k1_FDX_s107
                                 + 0.653336 w_k1_IBM_s107
                                 + 0.873439 w_k1_INTC_s107
                                 + 0.84263 w_k1_KO_s107
                                 + 0.713327 w_k1_MAT_s107
                                 + 1.06371 w_k1_MSFT_s107
                                 + 0.879524 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.09416 w_k1_ADM_s110
                                 + 1.21604 w_k1_AXP_s110
                                 + 1.04972 w_k1_DIS_s110
                                 + 1.10372 w_k1_FDX_s110
                                 + 0.768487 w_k1_IBM_s110
                                 + 1.22324 w_k1_INTC_s110 + 0.8926 w_k1_KO_s110
                                 + 0.975273 w_k1_MAT_s110
                                 + 0.994283 w_k1_MSFT_s110
                                 + 1.06916 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.04011 w_k1_ADM_s110
                                 + 1.065 w_k1_AXP_s110 + 1.19085 w_k1_DIS_s110
                                 + 0.876078 w_k1_FDX_s110
                                 + 1.30901 w_k1_IBM_s110
                                 + 1.07495 w_k1_INTC_s110
                                 + 1.19017 w_k1_KO_s110 + 1.27624 w_k1_MAT_s110
                                 + 0.973914 w_k1_MSFT_s110
                                 + 1.04462 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.10646 w_k1_ADM_s110
                                 + 1.20071 w_k1_AXP_s110
                                 + 1.26733 w_k1_DIS_s110
                                 + 1.18086 w_k1_FDX_s110
                                 + 1.14094 w_k1_IBM_s110
                                 + 1.13584 w_k1_INTC_s110 + 1.0356 w_k1_KO_s110
                                 + 1.13161 w_k1_MAT_s110
                                 + 1.0209 w_k1_MSFT_s110 + 1.06183 w_k1_PG_s110
                                 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.3267 w_k1_ADM_s110
                                 + 1.39765 w_k1_AXP_s110
                                 + 1.85512 w_k1_DIS_s110
                                 + 1.45225 w_k1_FDX_s110
                                 + 0.903469 w_k1_IBM_s110
                                 + 1.21049 w_k1_INTC_s110
                                 + 1.31869 w_k1_KO_s110 + 1.55058 w_k1_MAT_s110
                                 + 1.18226 w_k1_MSFT_s110
                                 + 1.30979 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.638806 w_k1_ADM_s110
                                 + 0.844296 w_k1_AXP_s110
                                 + 0.694618 w_k1_DIS_s110
                                 + 0.648057 w_k1_FDX_s110
                                 + 1.30548 w_k1_IBM_s110
                                 + 0.82309 w_k1_INTC_s110
                                 + 0.978039 w_k1_KO_s110
                                 + 1.48158 w_k1_MAT_s110
                                 + 0.73958 w_k1_MSFT_s110
                                 + 0.836787 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.820417 w_k1_ADM_s110
                                 + 0.792081 w_k1_AXP_s110
                                 + 0.867465 w_k1_DIS_s110
                                 + 0.877104 w_k1_FDX_s110
                                 + 0.629718 w_k1_IBM_s110
                                 + 0.921666 w_k1_INTC_s110
                                 + 0.786466 w_k1_KO_s110
                                 + 0.773823 w_k1_MAT_s110
                                 + 0.964158 w_k1_MSFT_s110
                                 + 0.863111 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.834775 w_k1_ADM_s110
                                 + 1.32782 w_k1_AXP_s110
                                 + 1.44461 w_k1_DIS_s110
                                 + 1.20772 w_k1_FDX_s110
                                 + 1.72825 w_k1_IBM_s110
                                 + 1.57959 w_k1_INTC_s110
                                 + 1.59699 w_k1_KO_s110 + 2.01134 w_k1_MAT_s110
                                 + 1.21613 w_k1_MSFT_s110
                                 + 1.26878 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.730428 w_k1_ADM_s110
                                 + 0.838005 w_k1_AXP_s110
                                 + 0.925012 w_k1_DIS_s110
                                 + 0.960821 w_k1_FDX_s110
                                 + 0.974531 w_k1_IBM_s110
                                 + 1.04161 w_k1_INTC_s110
                                 + 0.862116 w_k1_KO_s110
                                 + 0.844376 w_k1_MAT_s110
                                 + 0.962801 w_k1_MSFT_s110
                                 + 0.82166 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.795279 w_k1_ADM_s111
                                 + 0.97133 w_k1_AXP_s111
                                 + 0.762073 w_k1_DIS_s111
                                 + 0.766617 w_k1_FDX_s111
                                 + 1.3621 w_k1_IBM_s111
                                 + 0.752307 w_k1_INTC_s111
                                 + 0.944186 w_k1_KO_s111
                                 + 1.32691 w_k1_MAT_s111
                                 + 0.742056 w_k1_MSFT_s111
                                 + 0.826149 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.883939 w_k1_ADM_s111
                                 + 0.822855 w_k1_AXP_s111
                                 + 1.14337 w_k1_DIS_s111
                                 + 0.993941 w_k1_FDX_s111
                                 + 1.03348 w_k1_IBM_s111
                                 + 0.943161 w_k1_INTC_s111
                                 + 1.12815 w_k1_KO_s111
                                 + 0.881672 w_k1_MAT_s111
                                 + 1.12016 w_k1_MSFT_s111
                                 + 0.90298 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.21555 w_k1_ADM_s111
                                 + 1.30037 w_k1_AXP_s111
                                 + 1.65783 w_k1_DIS_s111
                                 + 1.39444 w_k1_FDX_s111
                                 + 0.916237 w_k1_IBM_s111
                                 + 1.24695 w_k1_INTC_s111 + 1.4614 w_k1_KO_s111
                                 + 1.24696 w_k1_MAT_s111
                                 + 1.64381 w_k1_MSFT_s111
                                 + 1.16942 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.825952 w_k1_ADM_s111
                                 + 1.10088 w_k1_AXP_s111
                                 + 0.904392 w_k1_DIS_s111
                                 + 1.04303 w_k1_FDX_s111
                                 + 0.744531 w_k1_IBM_s111
                                 + 0.989827 w_k1_INTC_s111
                                 + 0.840757 w_k1_KO_s111
                                 + 1.10199 w_k1_MAT_s111
                                 + 1.01645 w_k1_MSFT_s111
                                 + 0.846081 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.06915 w_k1_ADM_s111
                                 + 1.20829 w_k1_AXP_s111
                                 + 1.26981 w_k1_DIS_s111
                                 + 1.31272 w_k1_FDX_s111
                                 + 0.684002 w_k1_IBM_s111
                                 + 1.30078 w_k1_INTC_s111
                                 + 1.05263 w_k1_KO_s111
                                 + 0.846364 w_k1_MAT_s111
                                 + 1.31394 w_k1_MSFT_s111
                                 + 1.05467 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.17963 w_k1_ADM_s111
                                 + 1.18831 w_k1_AXP_s111
                                 + 1.46448 w_k1_DIS_s111
                                 + 1.20111 w_k1_FDX_s111
                                 + 1.33433 w_k1_IBM_s111
                                 + 1.02716 w_k1_INTC_s111
                                 + 1.33861 w_k1_KO_s111 + 1.43197 w_k1_MAT_s111
                                 + 1.24345 w_k1_MSFT_s111
                                 + 1.12559 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.22499 w_k1_ADM_s111
                                 + 1.42663 w_k1_AXP_s111 + 1.1804 w_k1_DIS_s111
                                 + 1.1905 w_k1_FDX_s111 + 1.01691 w_k1_IBM_s111
                                 + 1.04268 w_k1_INTC_s111
                                 + 1.11626 w_k1_KO_s111 + 1.23467 w_k1_MAT_s111
                                 + 1.12263 w_k1_MSFT_s111
                                 + 1.09192 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.784479 w_k1_ADM_s111
                                 + 0.776725 w_k1_AXP_s111
                                 + 0.599733 w_k1_DIS_s111
                                 + 0.752354 w_k1_FDX_s111
                                 + 0.821174 w_k1_IBM_s111
                                 + 0.747844 w_k1_INTC_s111
                                 + 0.748344 w_k1_KO_s111
                                 + 0.926395 w_k1_MAT_s111
                                 + 0.710039 w_k1_MSFT_s111
                                 + 0.799772 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.795629 w_k1_ADM_s112
                                 + 0.887167 w_k1_AXP_s112
                                 + 1.01476 w_k1_DIS_s112
                                 + 0.841921 w_k1_FDX_s112
                                 + 1.48166 w_k1_IBM_s112
                                 + 0.934683 w_k1_INTC_s112
                                 + 1.16983 w_k1_KO_s112 + 1.42378 w_k1_MAT_s112
                                 + 1.0054 w_k1_MSFT_s112 + 1.04256 w_k1_PG_s112
                                 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.14562 w_k1_ADM_s112
                                 + 0.796288 w_k1_AXP_s112
                                 + 1.39164 w_k1_DIS_s112
                                 + 1.13256 w_k1_FDX_s112
                                 + 0.904613 w_k1_IBM_s112
                                 + 1.03538 w_k1_INTC_s112
                                 + 1.21439 w_k1_KO_s112 + 0.7005 w_k1_MAT_s112
                                 + 1.23702 w_k1_MSFT_s112
                                 + 1.10226 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.04163 w_k1_ADM_s112
                                 + 1.55142 w_k1_AXP_s112
                                 + 1.28301 w_k1_DIS_s112
                                 + 1.16517 w_k1_FDX_s112
                                 + 1.16895 w_k1_IBM_s112
                                 + 1.00927 w_k1_INTC_s112 + 1.03 w_k1_KO_s112
                                 + 1.61789 w_k1_MAT_s112
                                 + 1.05518 w_k1_MSFT_s112
                                 + 1.03748 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.79558 w_k1_ADM_s112
                                 + 0.820614 w_k1_AXP_s112
                                 + 0.803436 w_k1_DIS_s112
                                 + 0.838415 w_k1_FDX_s112
                                 + 0.873741 w_k1_IBM_s112
                                 + 0.740968 w_k1_INTC_s112
                                 + 0.829669 w_k1_KO_s112
                                 + 0.869685 w_k1_MAT_s112
                                 + 0.837415 w_k1_MSFT_s112
                                 + 0.917002 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.18504 w_k1_ADM_s112
                                 + 1.56976 w_k1_AXP_s112
                                 + 1.61989 w_k1_DIS_s112
                                 + 1.31465 w_k1_FDX_s112
                                 + 1.16378 w_k1_IBM_s112
                                 + 1.36114 w_k1_INTC_s112
                                 + 1.38182 w_k1_KO_s112 + 1.52095 w_k1_MAT_s112
                                 + 1.27728 w_k1_MSFT_s112
                                 + 1.34336 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.670147 w_k1_ADM_s112
                                 + 0.687657 w_k1_AXP_s112
                                 + 0.750946 w_k1_DIS_s112
                                 + 0.785559 w_k1_FDX_s112
                                 + 0.498735 w_k1_IBM_s112
                                 + 0.870413 w_k1_INTC_s112
                                 + 0.643469 w_k1_KO_s112
                                 + 0.468732 w_k1_MAT_s112
                                 + 0.847946 w_k1_MSFT_s112
                                 + 0.76591 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.995631 w_k1_ADM_s112
                                 + 1.61198 w_k1_AXP_s112
                                 + 1.60397 w_k1_DIS_s112
                                 + 1.19254 w_k1_FDX_s112
                                 + 1.73253 w_k1_IBM_s112
                                 + 1.16714 w_k1_INTC_s112
                                 + 1.46009 w_k1_KO_s112 + 1.559 w_k1_MAT_s112
                                 + 1.07711 w_k1_MSFT_s112
                                 + 1.17107 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.05586 w_k1_ADM_s112
                                 + 1.07057 w_k1_AXP_s112
                                 + 0.953007 w_k1_DIS_s112
                                 + 1.17472 w_k1_FDX_s112
                                 + 0.672162 w_k1_IBM_s112
                                 + 1.10304 w_k1_INTC_s112
                                 + 0.771533 w_k1_KO_s112
                                 + 0.748573 w_k1_MAT_s112
                                 + 0.933189 w_k1_MSFT_s112
                                 + 0.872614 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.31041 w_k1_ADM_s113
                                 + 0.993196 w_k1_AXP_s113
                                 + 1.08645 w_k1_DIS_s113
                                 + 1.04133 w_k1_FDX_s113
                                 + 0.894447 w_k1_IBM_s113
                                 + 0.941362 w_k1_INTC_s113
                                 + 0.952378 w_k1_KO_s113
                                 + 0.805219 w_k1_MAT_s113
                                 + 0.900023 w_k1_MSFT_s113
                                 + 1.00807 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.35019 w_k1_ADM_s113
                                 + 1.62732 w_k1_AXP_s113
                                 + 1.85639 w_k1_DIS_s113
                                 + 1.42482 w_k1_FDX_s113
                                 + 1.52242 w_k1_IBM_s113
                                 + 1.31958 w_k1_INTC_s113
                                 + 1.37997 w_k1_KO_s113 + 1.71472 w_k1_MAT_s113
                                 + 1.2943 w_k1_MSFT_s113 + 1.34555 w_k1_PG_s113
                                 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.945055 w_k1_ADM_s113
                                 + 0.667962 w_k1_AXP_s113
                                 + 0.735999 w_k1_DIS_s113
                                 + 0.879161 w_k1_FDX_s113
                                 + 0.626389 w_k1_IBM_s113
                                 + 0.760192 w_k1_INTC_s113
                                 + 0.812838 w_k1_KO_s113
                                 + 0.489436 w_k1_MAT_s113
                                 + 0.879632 w_k1_MSFT_s113
                                 + 0.85023 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.880644 w_k1_ADM_s113
                                 + 0.943043 w_k1_AXP_s113
                                 + 0.807536 w_k1_DIS_s113
                                 + 0.923441 w_k1_FDX_s113
                                 + 1.0301 w_k1_IBM_s113
                                 + 0.851146 w_k1_INTC_s113
                                 + 0.911897 w_k1_KO_s113
                                 + 0.99998 w_k1_MAT_s113
                                 + 0.835809 w_k1_MSFT_s113
                                 + 0.995824 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.40458 w_k1_ADM_s113
                                 + 1.34085 w_k1_AXP_s113
                                 + 1.16992 w_k1_DIS_s113
                                 + 1.07888 w_k1_FDX_s113
                                 + 1.45875 w_k1_IBM_s113
                                 + 0.892913 w_k1_INTC_s113
                                 + 1.16554 w_k1_KO_s113 + 1.43728 w_k1_MAT_s113
                                 + 0.915784 w_k1_MSFT_s113
                                 + 1.08988 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.2001 w_k1_ADM_s113
                                 + 0.954642 w_k1_AXP_s113
                                 + 1.16796 w_k1_DIS_s113
                                 + 0.93528 w_k1_FDX_s113
                                 + 1.55665 w_k1_IBM_s113
                                 + 0.83967 w_k1_INTC_s113
                                 + 1.10173 w_k1_KO_s113 + 1.00127 w_k1_MAT_s113
                                 + 0.87965 w_k1_MSFT_s113
                                 + 1.01255 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.817846 w_k1_ADM_s113
                                 + 0.71119 w_k1_AXP_s113
                                 + 0.594991 w_k1_DIS_s113
                                 + 0.674701 w_k1_FDX_s113
                                 + 0.878596 w_k1_IBM_s113
                                 + 0.633554 w_k1_INTC_s113
                                 + 0.703874 w_k1_KO_s113
                                 + 0.944162 w_k1_MAT_s113
                                 + 0.717635 w_k1_MSFT_s113
                                 + 0.842567 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.49981 w_k1_ADM_s113
                                 + 1.3466 w_k1_AXP_s113 + 1.27598 w_k1_DIS_s113
                                 + 1.34558 w_k1_FDX_s113
                                 + 0.913139 w_k1_IBM_s113
                                 + 1.07489 w_k1_INTC_s113
                                 + 1.01682 w_k1_KO_s113 + 1.11006 w_k1_MAT_s113
                                 + 1.11369 w_k1_MSFT_s113
                                 + 1.25314 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.26706 w_k1_ADM_s114
                                 + 1.39627 w_k1_AXP_s114
                                 + 1.45088 w_k1_DIS_s114
                                 + 1.27102 w_k1_FDX_s114
                                 + 0.980254 w_k1_IBM_s114
                                 + 1.09693 w_k1_INTC_s114
                                 + 1.30911 w_k1_KO_s114 + 1.35324 w_k1_MAT_s114
                                 + 1.27645 w_k1_MSFT_s114
                                 + 1.18499 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.705495 w_k1_ADM_s114
                                 + 0.948221 w_k1_AXP_s114
                                 + 0.956141 w_k1_DIS_s114
                                 + 0.831371 w_k1_FDX_s114
                                 + 1.45269 w_k1_IBM_s114
                                 + 0.831018 w_k1_INTC_s114
                                 + 1.3699 w_k1_KO_s114 + 1.55673 w_k1_MAT_s114
                                 + 0.942967 w_k1_MSFT_s114
                                 + 0.987038 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.765675 w_k1_ADM_s114
                                 + 0.627111 w_k1_AXP_s114
                                 + 0.67019 w_k1_DIS_s114
                                 + 0.813474 w_k1_FDX_s114
                                 + 0.667016 w_k1_IBM_s114
                                 + 0.82313 w_k1_INTC_s114
                                 + 0.892722 w_k1_KO_s114
                                 + 0.702295 w_k1_MAT_s114
                                 + 0.87668 w_k1_MSFT_s114
                                 + 0.883199 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.961598 w_k1_ADM_s114
                                 + 0.902676 w_k1_AXP_s114
                                 + 0.991417 w_k1_DIS_s114
                                 + 0.989594 w_k1_FDX_s114
                                 + 0.787835 w_k1_IBM_s114
                                 + 1.06416 w_k1_INTC_s114
                                 + 1.01715 w_k1_KO_s114
                                 + 0.669806 w_k1_MAT_s114
                                 + 1.03907 w_k1_MSFT_s114
                                 + 0.932933 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.807589 w_k1_ADM_s114
                                 + 0.921035 w_k1_AXP_s114
                                 + 1.03791 w_k1_DIS_s114
                                 + 0.966093 w_k1_FDX_s114
                                 + 1.13036 w_k1_IBM_s114
                                 + 0.946121 w_k1_INTC_s114
                                 + 1.1673 w_k1_KO_s114 + 0.972908 w_k1_MAT_s114
                                 + 1.01422 w_k1_MSFT_s114
                                 + 0.95235 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.08451 w_k1_ADM_s114
                                 + 1.21347 w_k1_AXP_s114
                                 + 1.26261 w_k1_DIS_s114
                                 + 1.10345 w_k1_FDX_s114
                                 + 1.48948 w_k1_IBM_s114
                                 + 1.07625 w_k1_INTC_s114
                                 + 1.50386 w_k1_KO_s114 + 1.98057 w_k1_MAT_s114
                                 + 1.04149 w_k1_MSFT_s114 + 1.1398 w_k1_PG_s114
                                 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.10584 w_k1_ADM_s114
                                 + 0.910103 w_k1_AXP_s114
                                 + 1.25829 w_k1_DIS_s114
                                 + 1.06353 w_k1_FDX_s114
                                 + 0.95819 w_k1_IBM_s114
                                 + 0.928426 w_k1_INTC_s114
                                 + 1.31855 w_k1_KO_s114 + 1.10493 w_k1_MAT_s114
                                 + 1.19359 w_k1_MSFT_s114
                                 + 1.06716 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.782891 w_k1_ADM_s114
                                 + 0.737478 w_k1_AXP_s114
                                 + 0.765032 w_k1_DIS_s114
                                 + 0.781071 w_k1_FDX_s114
                                 + 0.974591 w_k1_IBM_s114
                                 + 0.659762 w_k1_INTC_s114
                                 + 0.957679 w_k1_KO_s114
                                 + 0.822126 w_k1_MAT_s114
                                 + 0.769378 w_k1_MSFT_s114
                                 + 0.867129 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.894515 w_k1_ADM_s115
                                 + 0.714301 w_k1_AXP_s115
                                 + 0.806073 w_k1_DIS_s115
                                 + 0.809359 w_k1_FDX_s115
                                 + 0.906118 w_k1_IBM_s115
                                 + 1.0321 w_k1_INTC_s115
                                 + 0.850263 w_k1_KO_s115
                                 + 0.75238 w_k1_MAT_s115
                                 + 0.858232 w_k1_MSFT_s115
                                 + 0.911969 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.01789 w_k1_ADM_s115
                                 + 0.802145 w_k1_AXP_s115
                                 + 0.896977 w_k1_DIS_s115
                                 + 1.02848 w_k1_FDX_s115
                                 + 0.734119 w_k1_IBM_s115
                                 + 1.18689 w_k1_INTC_s115
                                 + 0.86535 w_k1_KO_s115
                                 + 0.755515 w_k1_MAT_s115
                                 + 1.11305 w_k1_MSFT_s115
                                 + 0.905116 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.887529 w_k1_ADM_s115
                                 + 0.651807 w_k1_AXP_s115
                                 + 0.656027 w_k1_DIS_s115
                                 + 0.764834 w_k1_FDX_s115
                                 + 0.700124 w_k1_IBM_s115
                                 + 0.796021 w_k1_INTC_s115
                                 + 0.65024 w_k1_KO_s115
                                 + 0.802226 w_k1_MAT_s115
                                 + 0.831578 w_k1_MSFT_s115
                                 + 0.803222 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.21873 w_k1_ADM_s115
                                 + 1.23572 w_k1_AXP_s115
                                 + 1.15037 w_k1_DIS_s115
                                 + 1.16474 w_k1_FDX_s115
                                 + 0.906012 w_k1_IBM_s115
                                 + 1.19981 w_k1_INTC_s115
                                 + 1.05592 w_k1_KO_s115
                                 + 0.986179 w_k1_MAT_s115
                                 + 1.09108 w_k1_MSFT_s115
                                 + 1.09238 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.01571 w_k1_ADM_s115
                                 + 1.02212 w_k1_AXP_s115
                                 + 1.05943 w_k1_DIS_s115
                                 + 0.933236 w_k1_FDX_s115
                                 + 1.2908 w_k1_IBM_s115 + 1.072 w_k1_INTC_s115
                                 + 1.20356 w_k1_KO_s115 + 1.35988 w_k1_MAT_s115
                                 + 1.0951 w_k1_MSFT_s115
                                 + 0.928923 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.1657 w_k1_ADM_s115
                                 + 1.11061 w_k1_AXP_s115
                                 + 1.37378 w_k1_DIS_s115
                                 + 1.30723 w_k1_FDX_s115
                                 + 0.968859 w_k1_IBM_s115
                                 + 1.1254 w_k1_INTC_s115 + 1.14034 w_k1_KO_s115
                                 + 1.224 w_k1_MAT_s115 + 1.1876 w_k1_MSFT_s115
                                 + 1.10211 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.938743 w_k1_ADM_s115
                                 + 1.07508 w_k1_AXP_s115
                                 + 0.870021 w_k1_DIS_s115
                                 + 0.952403 w_k1_FDX_s115
                                 + 0.847613 w_k1_IBM_s115
                                 + 0.998889 w_k1_INTC_s115
                                 + 0.837088 w_k1_KO_s115
                                 + 1.16383 w_k1_MAT_s115
                                 + 0.970349 w_k1_MSFT_s115
                                 + 0.862357 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.33444 w_k1_ADM_s115
                                 + 1.51211 w_k1_AXP_s115
                                 + 1.65066 w_k1_DIS_s115
                                 + 1.30286 w_k1_FDX_s115
                                 + 1.27052 w_k1_IBM_s115 + 1.202 w_k1_INTC_s115
                                 + 1.4304 w_k1_KO_s115 + 1.58577 w_k1_MAT_s115
                                 + 1.25938 w_k1_MSFT_s115
                                 + 1.15402 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.44353 w_k1_ADM_s116
                                 + 1.54891 w_k1_AXP_s116
                                 + 1.53764 w_k1_DIS_s116
                                 + 1.54038 w_k1_FDX_s116
                                 + 1.19682 w_k1_IBM_s116
                                 + 1.10383 w_k1_INTC_s116
                                 + 1.25872 w_k1_KO_s116 + 1.32395 w_k1_MAT_s116
                                 + 1.2723 w_k1_MSFT_s116 + 1.25301 w_k1_PG_s116
                                 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.887335 w_k1_ADM_s116
                                 + 0.559939 w_k1_AXP_s116
                                 + 0.57324 w_k1_DIS_s116
                                 + 0.720113 w_k1_FDX_s116
                                 + 0.683954 w_k1_IBM_s116
                                 + 0.693595 w_k1_INTC_s116
                                 + 0.636758 w_k1_KO_s116
                                 + 0.669051 w_k1_MAT_s116
                                 + 0.810287 w_k1_MSFT_s116
                                 + 0.797811 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.02241 w_k1_ADM_s116
                                 + 0.990675 w_k1_AXP_s116
                                 + 1.10754 w_k1_DIS_s116
                                 + 1.02711 w_k1_FDX_s116
                                 + 1.26701 w_k1_IBM_s116
                                 + 1.08753 w_k1_INTC_s116
                                 + 1.21503 w_k1_KO_s116 + 1.43713 w_k1_MAT_s116
                                 + 1.16455 w_k1_MSFT_s116
                                 + 1.06581 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.993442 w_k1_ADM_s116
                                 + 0.944492 w_k1_AXP_s116
                                 + 1.22127 w_k1_DIS_s116
                                 + 1.12905 w_k1_FDX_s116
                                 + 0.842533 w_k1_IBM_s116
                                 + 1.17877 w_k1_INTC_s116
                                 + 0.950557 w_k1_KO_s116
                                 + 0.760882 w_k1_MAT_s116
                                 + 1.14073 w_k1_MSFT_s116
                                 + 1.03298 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.768185 w_k1_ADM_s116
                                 + 0.79419 w_k1_AXP_s116
                                 + 0.654784 w_k1_DIS_s116
                                 + 0.761191 w_k1_FDX_s116
                                 + 1.04433 w_k1_IBM_s116
                                 + 0.894992 w_k1_INTC_s116
                                 + 0.868676 w_k1_KO_s116
                                 + 1.26069 w_k1_MAT_s116
                                 + 0.881266 w_k1_MSFT_s116
                                 + 0.894597 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.44691 w_k1_ADM_s116
                                 + 1.08445 w_k1_AXP_s116
                                 + 1.51892 w_k1_DIS_s116
                                 + 1.35004 w_k1_FDX_s116
                                 + 0.988741 w_k1_IBM_s116
                                 + 0.965192 w_k1_INTC_s116
                                 + 1.31281 w_k1_KO_s116 + 1.08883 w_k1_MAT_s116
                                 + 1.36396 w_k1_MSFT_s116
                                 + 1.24998 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.21124 w_k1_ADM_s116
                                 + 1.21194 w_k1_AXP_s116
                                 + 1.28163 w_k1_DIS_s116
                                 + 1.30894 w_k1_FDX_s116
                                 + 1.09844 w_k1_IBM_s116
                                 + 0.968975 w_k1_INTC_s116
                                 + 1.14687 w_k1_KO_s116 + 1.77927 w_k1_MAT_s116
                                 + 1.16734 w_k1_MSFT_s116
                                 + 1.14403 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.926969 w_k1_ADM_s116
                                 + 0.849625 w_k1_AXP_s116
                                 + 0.920953 w_k1_DIS_s116
                                 + 0.975435 w_k1_FDX_s116
                                 + 0.903162 w_k1_IBM_s116
                                 + 0.91449 w_k1_INTC_s116
                                 + 0.885058 w_k1_KO_s116
                                 + 0.906669 w_k1_MAT_s116
                                 + 1.03692 w_k1_MSFT_s116
                                 + 0.910513 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.891128 w_k1_ADM_s117
                                 + 0.825541 w_k1_AXP_s117
                                 + 0.823392 w_k1_DIS_s117
                                 + 0.869717 w_k1_FDX_s117
                                 + 0.658592 w_k1_IBM_s117
                                 + 0.97402 w_k1_INTC_s117
                                 + 0.775998 w_k1_KO_s117
                                 + 0.545784 w_k1_MAT_s117
                                 + 0.98384 w_k1_MSFT_s117
                                 + 0.817377 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.870652 w_k1_ADM_s117
                                 + 0.914409 w_k1_AXP_s117
                                 + 0.810625 w_k1_DIS_s117
                                 + 0.822692 w_k1_FDX_s117
                                 + 1.30598 w_k1_IBM_s117
                                 + 0.891695 w_k1_INTC_s117
                                 + 1.06842 w_k1_KO_s117 + 1.10784 w_k1_MAT_s117
                                 + 0.786294 w_k1_MSFT_s117
                                 + 0.886178 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.04002 w_k1_ADM_s117
                                 + 1.09705 w_k1_AXP_s117
                                 + 0.857375 w_k1_DIS_s117
                                 + 0.951299 w_k1_FDX_s117
                                 + 0.994363 w_k1_IBM_s117
                                 + 1.11513 w_k1_INTC_s117
                                 + 0.925127 w_k1_KO_s117
                                 + 1.03511 w_k1_MAT_s117
                                 + 1.0149 w_k1_MSFT_s117 + 1.03458 w_k1_PG_s117
                                 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.849276 w_k1_ADM_s117
                                 + 0.638022 w_k1_AXP_s117
                                 + 0.631793 w_k1_DIS_s117
                                 + 0.713341 w_k1_FDX_s117
                                 + 0.883828 w_k1_IBM_s117
                                 + 0.878424 w_k1_INTC_s117
                                 + 0.80214 w_k1_KO_s117
                                 + 0.707078 w_k1_MAT_s117
                                 + 0.850339 w_k1_MSFT_s117
                                 + 0.75775 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.12662 w_k1_ADM_s117
                                 + 0.928608 w_k1_AXP_s117
                                 + 1.30083 w_k1_DIS_s117 + 1.0618 w_k1_FDX_s117
                                 + 0.966713 w_k1_IBM_s117
                                 + 1.10463 w_k1_INTC_s117
                                 + 1.06081 w_k1_KO_s117 + 1.07685 w_k1_MAT_s117
                                 + 1.16881 w_k1_MSFT_s117
                                 + 1.04766 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.43527 w_k1_ADM_s117
                                 + 1.06362 w_k1_AXP_s117
                                 + 1.17177 w_k1_DIS_s117
                                 + 1.17717 w_k1_FDX_s117
                                 + 0.766065 w_k1_IBM_s117
                                 + 1.25703 w_k1_INTC_s117
                                 + 1.01614 w_k1_KO_s117
                                 + 0.709371 w_k1_MAT_s117
                                 + 1.17187 w_k1_MSFT_s117
                                 + 0.97374 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.3249 w_k1_ADM_s117
                                 + 1.53967 w_k1_AXP_s117
                                 + 1.49596 w_k1_DIS_s117
                                 + 1.23232 w_k1_FDX_s117
                                 + 1.49116 w_k1_IBM_s117
                                 + 1.09991 w_k1_INTC_s117
                                 + 1.58732 w_k1_KO_s117 + 1.96663 w_k1_MAT_s117
                                 + 1.25894 w_k1_MSFT_s117
                                 + 1.21142 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.10884 w_k1_ADM_s117
                                 + 1.13039 w_k1_AXP_s117
                                 + 1.24827 w_k1_DIS_s117
                                 + 1.06268 w_k1_FDX_s117
                                 + 1.32051 w_k1_IBM_s117
                                 + 1.2474 w_k1_INTC_s117 + 1.17048 w_k1_KO_s117
                                 + 1.30001 w_k1_MAT_s117
                                 + 1.13985 w_k1_MSFT_s117
                                 + 1.12958 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0142218829202046 w_k1_ADM_s000
                                 + 0.0235360762371336 w_k1_ADM_s001
                                 + 0.0244980251641993 w_k1_ADM_s002
                                 + 0.0236135064127789 w_k1_ADM_s003
                                 + 0.029258611419522 w_k1_ADM_s004
                                 + 0.0148734387282081 w_k1_ADM_s005
                                 + 0.0133918574174119 w_k1_ADM_s006
                                 + 0.0454959614386408 w_k1_ADM_s007
                                 + 0.0100904317582902 w_k1_ADM_s010
                                 + 0.0195531261109933 w_k1_ADM_s011
                                 + 0.0373310745629254 w_k1_ADM_s012
                                 + 0.0219101212672028 w_k1_ADM_s013
                                 + 0.0116670969345215 w_k1_ADM_s014
                                 + 0.0122578950397424 w_k1_ADM_s015
                                 + 0.0201702054043234 w_k1_ADM_s016
                                 + 0.0163389494970665 w_k1_ADM_s017
                                 + 0.0152839991692511 w_k1_ADM_s100
                                 + 0.0181021787356945 w_k1_ADM_s101
                                 + 0.033550949571225 w_k1_ADM_s102
                                 + 0.0573726613811921 w_k1_ADM_s103
                                 + 0.0663202182532765 w_k1_ADM_s104
                                 + 0.0583573075710443 w_k1_ADM_s105
                                 + 0.0373780464768758 w_k1_ADM_s106
                                 + 0.0229642177967068 w_k1_ADM_s107
                                 + 0.0182948220322901 w_k1_ADM_s110
                                 + 0.0561306692384895 w_k1_ADM_s111
                                 + 0.018323397678157 w_k1_ADM_s112
                                 + 0.0330482641282225 w_k1_ADM_s113
                                 + 0.0703575741600682 w_k1_ADM_s114
                                 + 0.0931235073355764 w_k1_ADM_s115
                                 + 0.0231515311078936 w_k1_ADM_s116
                                 + 0.0714690807674752 w_k1_ADM_s117
                                 + 0.0130424175316774 w_k1_AXP_s000
                                 + 0.0233190525240725 w_k1_AXP_s001
                                 + 0.0218455081491817 w_k1_AXP_s002
                                 + 0.0192492073992641 w_k1_AXP_s003
                                 + 0.0287272675185744 w_k1_AXP_s004
                                 + 0.0160176812461263 w_k1_AXP_s005
                                 + 0.0140608532550881 w_k1_AXP_s006
                                 + 0.0442145224325562 w_k1_AXP_s007
                                 + 0.0112340481329113 w_k1_AXP_s010
                                 + 0.0225794057185814 w_k1_AXP_s011
                                 + 0.0381247917867915 w_k1_AXP_s012
                                 + 0.020377952121916 w_k1_AXP_s013
                                 + 0.011593261925891 w_k1_AXP_s014
                                 + 0.0113640231504949 w_k1_AXP_s015
                                 + 0.0207269311560806 w_k1_AXP_s016
                                 + 0.013449064030804 w_k1_AXP_s017
                                 + 0.0135350336595555 w_k1_AXP_s100
                                 + 0.0167884886935359 w_k1_AXP_s101
                                 + 0.0322621915578769 w_k1_AXP_s102
                                 + 0.0628333063165833 w_k1_AXP_s103
                                 + 0.0683635246426451 w_k1_AXP_s104
                                 + 0.0571322297557529 w_k1_AXP_s105
                                 + 0.038584367971634 w_k1_AXP_s106
                                 + 0.0223620649072915 w_k1_AXP_s107
                                 + 0.0204150700837682 w_k1_AXP_s110
                                 + 0.0610781286403732 w_k1_AXP_s111
                                 + 0.0203085160297036 w_k1_AXP_s112
                                 + 0.0284313295953722 w_k1_AXP_s113
                                 + 0.0714149675237847 w_k1_AXP_s114
                                 + 0.0903173019269872 w_k1_AXP_s115
                                 + 0.0210228913090669 w_k1_AXP_s116
                                 + 0.0687875449907969 w_k1_AXP_s117
                                 + 0.013810660577028 w_k1_DIS_s000
                                 + 0.023149325741618 w_k1_DIS_s001
                                 + 0.0216143446550792 w_k1_DIS_s002
                                 + 0.0198238179039379 w_k1_DIS_s003
                                 + 0.0280618916277498 w_k1_DIS_s004
                                 + 0.016242294119644 w_k1_DIS_s005
                                 + 0.0153003137982004 w_k1_DIS_s006
                                 + 0.0451547579276295 w_k1_DIS_s007
                                 + 0.0112229689126149 w_k1_DIS_s010
                                 + 0.0225751456047813 w_k1_DIS_s011
                                 + 0.0384155913184492 w_k1_DIS_s012
                                 + 0.0218632616040883 w_k1_DIS_s013
                                 + 0.0124734965817467 w_k1_DIS_s014
                                 + 0.012519170576444 w_k1_DIS_s015
                                 + 0.0224148335012618 w_k1_DIS_s016
                                 + 0.0141527551563254 w_k1_DIS_s017
                                 + 0.0131494065807066 w_k1_DIS_s100
                                 + 0.0165108790254067 w_k1_DIS_s101
                                 + 0.032800110467641 w_k1_DIS_s102
                                 + 0.0620242651155284 w_k1_DIS_s103
                                 + 0.0620298592303278 w_k1_DIS_s104
                                 + 0.0611047817773215 w_k1_DIS_s105
                                 + 0.0410828467826217 w_k1_DIS_s106
                                 + 0.0243378071400812 w_k1_DIS_s107
                                 + 0.0213299926333118 w_k1_DIS_s110
                                 + 0.0606509208085732 w_k1_DIS_s111
                                 + 0.0211186863265481 w_k1_DIS_s112
                                 + 0.0292968715924644 w_k1_DIS_s113
                                 + 0.0783103776573719 w_k1_DIS_s114
                                 + 0.0945680140584787 w_k1_DIS_s115
                                 + 0.0233541699381389 w_k1_DIS_s116
                                 + 0.0685117629916681 w_k1_DIS_s117
                                 + 0.0131987736094902 w_k1_FDX_s000
                                 + 0.0226779521405804 w_k1_FDX_s001
                                 + 0.0214761575307622 w_k1_FDX_s002
                                 + 0.0195244237309739 w_k1_FDX_s003
                                 + 0.0277547569356026 w_k1_FDX_s004
                                 + 0.0155017273654222 w_k1_FDX_s005
                                 + 0.0140053895630343 w_k1_FDX_s006
                                 + 0.0432116132226096 w_k1_FDX_s007
                                 + 0.0109004455478086 w_k1_FDX_s010
                                 + 0.0223606045240642 w_k1_FDX_s011
                                 + 0.0370206269625384 w_k1_FDX_s012
                                 + 0.0228214421762519 w_k1_FDX_s013
                                 + 0.0119326848930931 w_k1_FDX_s014
                                 + 0.0126641932640132 w_k1_FDX_s015
                                 + 0.0214807736315504 w_k1_FDX_s016
                                 + 0.0133022469261524 w_k1_FDX_s017
                                 + 0.0127376613832237 w_k1_FDX_s100
                                 + 0.0169303224938332 w_k1_FDX_s101
                                 + 0.0338970901187358 w_k1_FDX_s102
                                 + 0.0576534092217601 w_k1_FDX_s103
                                 + 0.0621748995591656 w_k1_FDX_s104
                                 + 0.0606786855581329 w_k1_FDX_s105
                                 + 0.0397811550078456 w_k1_FDX_s106
                                 + 0.0226373203822921 w_k1_FDX_s107
                                 + 0.0199803672019089 w_k1_FDX_s110
                                 + 0.0596057882443043 w_k1_FDX_s111
                                 + 0.0198977345940516 w_k1_FDX_s112
                                 + 0.0286895371258852 w_k1_FDX_s113
                                 + 0.0730799739144616 w_k1_FDX_s114
                                 + 0.0891691010728692 w_k1_FDX_s115
                                 + 0.0235865911132119 w_k1_FDX_s116
                                 + 0.0654841294478842 w_k1_FDX_s117
                                 + 0.0133501241582733 w_k1_IBM_s000
                                 + 0.0228138606431443 w_k1_IBM_s001
                                 + 0.0218452039667522 w_k1_IBM_s002
                                 + 0.02125607358649 w_k1_IBM_s003
                                 + 0.0274649688839891 w_k1_IBM_s004
                                 + 0.0164603375304371 w_k1_IBM_s005
                                 + 0.0145832000054958 w_k1_IBM_s006
                                 + 0.0439722252506015 w_k1_IBM_s007
                                 + 0.0109431773865089 w_k1_IBM_s010
                                 + 0.0217987043141464 w_k1_IBM_s011
                                 + 0.0387206517805931 w_k1_IBM_s012
                                 + 0.0222176609022619 w_k1_IBM_s013
                                 + 0.0116274834764796 w_k1_IBM_s014
                                 + 0.0122666932681889 w_k1_IBM_s015
                                 + 0.0227954752727711 w_k1_IBM_s016
                                 + 0.0143470335662036 w_k1_IBM_s017
                                 + 0.0133014013544185 w_k1_IBM_s100
                                 + 0.0167231071859769 w_k1_IBM_s101
                                 + 0.0333310125754012 w_k1_IBM_s102
                                 + 0.0592172456079329 w_k1_IBM_s103
                                 + 0.0608170168270329 w_k1_IBM_s104
                                 + 0.0632839430141436 w_k1_IBM_s105
                                 + 0.0385043578757242 w_k1_IBM_s106
                                 + 0.0237673573193166 w_k1_IBM_s107
                                 + 0.0189931792382661 w_k1_IBM_s110
                                 + 0.0579743725454158 w_k1_IBM_s111
                                 + 0.0197680309339777 w_k1_IBM_s112
                                 + 0.0283632753845457 w_k1_IBM_s113
                                 + 0.07636584374874 w_k1_IBM_s114
                                 + 0.0912166932781151 w_k1_IBM_s115
                                 + 0.0215238497568026 w_k1_IBM_s116
                                 + 0.0682925010732407 w_k1_IBM_s117
                                 + 0.0143241937009947 w_k1_INTC_s000
                                 + 0.0221140135224379 w_k1_INTC_s001
                                 + 0.0194726671840639 w_k1_INTC_s002
                                 + 0.0169559777675809 w_k1_INTC_s003
                                 + 0.0326160723898596 w_k1_INTC_s004
                                 + 0.0170557697732362 w_k1_INTC_s005
                                 + 0.01630547454 w_k1_INTC_s006
                                 + 0.042239957827405 w_k1_INTC_s007
                                 + 0.0112321137560598 w_k1_INTC_s010
                                 + 0.0211779291749575 w_k1_INTC_s011
                                 + 0.0347105111242849 w_k1_INTC_s012
                                 + 0.0232595059496017 w_k1_INTC_s013
                                 + 0.0130062564143277 w_k1_INTC_s014
                                 + 0.0108441337880657 w_k1_INTC_s015
                                 + 0.0228661514963309 w_k1_INTC_s016
                                 + 0.01395495363954 w_k1_INTC_s017
                                 + 0.0111265096816658 w_k1_INTC_s100
                                 + 0.0196208622415752 w_k1_INTC_s101
                                 + 0.0324546120229842 w_k1_INTC_s102
                                 + 0.0521183533530444 w_k1_INTC_s103
                                 + 0.0639235673602021 w_k1_INTC_s104
                                 + 0.0546407783440402 w_k1_INTC_s105
                                 + 0.0417805402519186 w_k1_INTC_s106
                                 + 0.0213213433961 w_k1_INTC_s107
                                 + 0.0214211342658779 w_k1_INTC_s110
                                 + 0.0553661152079605 w_k1_INTC_s111
                                 + 0.0190189585665065 w_k1_INTC_s112
                                 + 0.0252693657011302 w_k1_INTC_s113
                                 + 0.0687535257670088 w_k1_INTC_s114
                                 + 0.0954915355878051 w_k1_INTC_s115
                                 + 0.0213188615925853 w_k1_INTC_s116
                                 + 0.0711424254731253 w_k1_INTC_s117
                                 + 0.0139363195110419 w_k1_KO_s000
                                 + 0.0245968187876775 w_k1_KO_s001
                                 + 0.0215300304320171 w_k1_KO_s002
                                 + 0.0185184906739157 w_k1_KO_s003
                                 + 0.0300092496461112 w_k1_KO_s004
                                 + 0.0166822214292205 w_k1_KO_s005
                                 + 0.0158853318909214 w_k1_KO_s006
                                 + 0.0451691603060519 w_k1_KO_s007
                                 + 0.010778905313067 w_k1_KO_s010
                                 + 0.0201911877147886 w_k1_KO_s011
                                 + 0.0394084163956502 w_k1_KO_s012
                                 + 0.0226678657458869 w_k1_KO_s013
                                 + 0.0117839428771555 w_k1_KO_s014
                                 + 0.0122024638749199 w_k1_KO_s015
                                 + 0.0225190459737072 w_k1_KO_s016
                                 + 0.0147822937938577 w_k1_KO_s017
                                 + 0.0126808997297198 w_k1_KO_s100
                                 + 0.016230050783293 w_k1_KO_s101
                                 + 0.0333391975757758 w_k1_KO_s102
                                 + 0.0542094811876202 w_k1_KO_s103
                                 + 0.0584733402645968 w_k1_KO_s104
                                 + 0.062241534637869 w_k1_KO_s105
                                 + 0.0371391010428048 w_k1_KO_s106
                                 + 0.0236739641070639 w_k1_KO_s107
                                 + 0.0194060417724519 w_k1_KO_s110
                                 + 0.0600130972562829 w_k1_KO_s111
                                 + 0.0196045128169878 w_k1_KO_s112
                                 + 0.027146758475306 w_k1_KO_s113
                                 + 0.0868186914484319 w_k1_KO_s114
                                 + 0.0934370495003919 w_k1_KO_s115
                                 + 0.0220053849793038 w_k1_KO_s116
                                 + 0.0688729720294358 w_k1_KO_s117
                                 + 0.0127481907145562 w_k1_MAT_s000
                                 + 0.0225181677744185 w_k1_MAT_s001
                                 + 0.0211861232369331 w_k1_MAT_s002
                                 + 0.0182293020401978 w_k1_MAT_s003
                                 + 0.0303408113678541 w_k1_MAT_s004
                                 + 0.018726521665006 w_k1_MAT_s005
                                 + 0.0141101299352953 w_k1_MAT_s006
                                 + 0.0520030209335728 w_k1_MAT_s007
                                 + 0.0131699003766111 w_k1_MAT_s010
                                 + 0.0248670011677434 w_k1_MAT_s011
                                 + 0.0433645506355776 w_k1_MAT_s012
                                 + 0.0235122912142041 w_k1_MAT_s013
                                 + 0.0113173865706988 w_k1_MAT_s014
                                 + 0.013177580929883 w_k1_MAT_s015
                                 + 0.02364474199988 w_k1_MAT_s016
                                 + 0.011874533100521 w_k1_MAT_s017
                                 + 0.0156920091587927 w_k1_MAT_s100
                                 + 0.0156283225765715 w_k1_MAT_s101
                                 + 0.0334251770796694 w_k1_MAT_s102
                                 + 0.0660693646173778 w_k1_MAT_s103
                                 + 0.0770525722934188 w_k1_MAT_s104
                                 + 0.0740536834739802 w_k1_MAT_s105
                                 + 0.0367945506404839 w_k1_MAT_s106
                                 + 0.0252069964053803 w_k1_MAT_s107
                                 + 0.021834072081819 w_k1_MAT_s110
                                 + 0.0652615781556497 w_k1_MAT_s111
                                 + 0.0203431825181222 w_k1_MAT_s112
                                 + 0.0265437851205587 w_k1_MAT_s113
                                 + 0.0815478867738353 w_k1_MAT_s114
                                 + 0.100269365516148 w_k1_MAT_s115
                                 + 0.0238854458421595 w_k1_MAT_s116
                                 + 0.0693780495959613 w_k1_MAT_s117
                                 + 0.0145747316014852 w_k1_MSFT_s000
                                 + 0.0212437290486531 w_k1_MSFT_s001
                                 + 0.0194462930500949 w_k1_MSFT_s002
                                 + 0.0181166696774085 w_k1_MSFT_s003
                                 + 0.0318732676693069 w_k1_MSFT_s004
                                 + 0.0195088376418084 w_k1_MSFT_s005
                                 + 0.0136730541924361 w_k1_MSFT_s006
                                 + 0.0440544114730133 w_k1_MSFT_s007
                                 + 0.0119163852460914 w_k1_MSFT_s010
                                 + 0.0225500458180324 w_k1_MSFT_s011
                                 + 0.0396356705809334 w_k1_MSFT_s012
                                 + 0.022319149532874 w_k1_MSFT_s013
                                 + 0.0122622199520952 w_k1_MSFT_s014
                                 + 0.0119370951513356 w_k1_MSFT_s015
                                 + 0.0227430541721195 w_k1_MSFT_s016
                                 + 0.0142733016626321 w_k1_MSFT_s017
                                 + 0.0131349941884986 w_k1_MSFT_s100
                                 + 0.0178567854996708 w_k1_MSFT_s101
                                 + 0.034365635978478 w_k1_MSFT_s102
                                 + 0.0535768577159108 w_k1_MSFT_s103
                                 + 0.0626098388673642 w_k1_MSFT_s104
                                 + 0.0639631031507804 w_k1_MSFT_s105
                                 + 0.0382729762885293 w_k1_MSFT_s106
                                 + 0.0252729359353126 w_k1_MSFT_s107
                                 + 0.0193379863185657 w_k1_MSFT_s110
                                 + 0.0604291083584233 w_k1_MSFT_s111
                                 + 0.0195319817624603 w_k1_MSFT_s112
                                 + 0.0259280475960892 w_k1_MSFT_s113
                                 + 0.0754445783916535 w_k1_MSFT_s114
                                 + 0.0936330083063429 w_k1_MSFT_s115
                                 + 0.0241070133057595 w_k1_MSFT_s116
                                 + 0.0691029666801547 w_k1_MSFT_s117
                                 + 0.0136488216885162 w_k1_PG_s000
                                 + 0.0230869556140652 w_k1_PG_s001
                                 + 0.022050262827283 w_k1_PG_s002
                                 + 0.0188528926448608 w_k1_PG_s003
                                 + 0.0268131621451804 w_k1_PG_s004
                                 + 0.0155120114119071 w_k1_PG_s005
                                 + 0.015600623781445 w_k1_PG_s006
                                 + 0.0458154320388816 w_k1_PG_s007
                                 + 0.0108166583081672 w_k1_PG_s010
                                 + 0.0224825777726091 w_k1_PG_s011
                                 + 0.0371985199307153 w_k1_PG_s012
                                 + 0.0211937965204155 w_k1_PG_s013
                                 + 0.0114330367929334 w_k1_PG_s014
                                 + 0.0123811764722772 w_k1_PG_s015
                                 + 0.0218432360377589 w_k1_PG_s016
                                 + 0.0131104320083326 w_k1_PG_s017
                                 + 0.0137266452612452 w_k1_PG_s100
                                 + 0.0167014970479479 w_k1_PG_s101
                                 + 0.0331029225676051 w_k1_PG_s102
                                 + 0.0583128446516918 w_k1_PG_s103
                                 + 0.0615157436646714 w_k1_PG_s104
                                 + 0.06180965791943 w_k1_PG_s105
                                 + 0.0391595125300498 w_k1_PG_s106
                                 + 0.0228650530786989 w_k1_PG_s107
                                 + 0.0194651841455355 w_k1_PG_s110
                                 + 0.0551923901897663 w_k1_PG_s111
                                 + 0.0196898983121846 w_k1_PG_s112
                                 + 0.0286677086984037 w_k1_PG_s113
                                 + 0.0739784591220679 w_k1_PG_s114
                                 + 0.0850675328272073 w_k1_PG_s115
                                 + 0.0223363805120067 w_k1_PG_s116
                                 + 0.0650563858045917 w_k1_PG_s117 >= 1.19
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
