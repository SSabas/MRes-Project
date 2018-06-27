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
 w_k1_ADM_s0_return_constraint:  1.14276 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.924731 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.15147 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.997746 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.07764 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.889878 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.747974 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.16424 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.854028 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.814567 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.17181 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.928366 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.718284 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.02979 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.09928 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.906759 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.809441 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.32529 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.57592 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.907486 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.810888 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.18223 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.24496 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.923059 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.22053 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.841425 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.8121 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.06028 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.941177 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.25467 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.13251 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.983036 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.39092 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.772733 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.937735 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.2997 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.866342 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.23472 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.24295 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.929008 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.987488 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.10477 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.06575 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.751009 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.681324 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.8089 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.94066 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.925983 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.33644 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.1619 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.696096 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.18745 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.05275 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.893817 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.771574 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.31695 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.56773 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.595672 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.864956 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.27298 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.24906 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.882266 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.33798 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.812009 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.640313 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.924069 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.939642 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.23212 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.24472 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.10644 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.43987 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.708402 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.897042 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.52689 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.859394 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.47751 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.31854 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.847347 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.07923 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.998896 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.20769 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.797916 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.598127 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.71721 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.907663 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.642375 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.37026 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.22669 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.856315 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.872142 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.21396 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.14844 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.864113 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.52869 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 2.19424 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.535152 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.810454 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.3454 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.39002 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.70969 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.28406 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.701334 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.676725 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.807242 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.09255 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.18506 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.21119 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.943106 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.30926 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.675669 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.931925 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.72303 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.688617 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.74255 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.1878 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.902216 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.12327 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.953521 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.12858 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.905932 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.71769 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.40248 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.983284 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.803142 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.27299 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.02333 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.785294 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.976898 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.13652 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.914327 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.924014 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.3341 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.63989 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.747467 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.84315 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.15415 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.2391 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.890757 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.05991 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.710082 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.707674 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.00425 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.997676 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.27981 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.13562 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.9028 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.1805 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.714597 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.03955 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.59057 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.849776 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.25071 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.20923 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.926358 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.750088 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.19575 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.21791 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.774552 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.887317 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.57311 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.734197 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.657128 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.913005 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.3201 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.03839 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.13207 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.868224 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.35227 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.797583 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.11863 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.54508 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.601096 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.12125 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.76671 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.06753 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.752663 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.66486 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.20788 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.806473 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.598013 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.807192 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.669554 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.813599 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.26464 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.1177 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.879264 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.746838 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.831793 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.614914 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.49289 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.10105 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.911839 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.06522 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.999779 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.04368 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.84306 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.644752 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.21658 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.932716 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.9126 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.04927 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.12791 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.984015 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.912836 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.21548 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.944232 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.01194 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.14305 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.43875 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.798549 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.784488 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.46452 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.31811 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.962575 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.14158 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.829071 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.899109 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.05118 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.139 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.16677 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.19867 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.13374 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.17085 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.925161 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.971326 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.40619 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.02862 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.49956 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.17183 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.943334 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.01352 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.06379 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.1493 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.793472 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.708933 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.38056 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.733601 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.632292 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.04965 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.23958 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.902299 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.964348 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.11354 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.22033 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.915453 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.30936 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.63666 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.640398 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.91209 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.55801 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.06748 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.757029 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.45067 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.890018 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.836486 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.727441 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.940194 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.902362 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.0537 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.1289 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.11845 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.735122 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.740552 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.11161 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.609598 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.53332 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.28411 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.997307 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.701226 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.14701 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.09216 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.728237 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.93835 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 2.3539 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.848818 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.861007 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.23884 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.69544 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.85055 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.35141 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.934202 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.2852 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.875 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.46278 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.67625 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.634774 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.29073 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.82873 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.96036 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.966233 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.91762 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.943577 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.72973 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.634663 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.821326 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.843003 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.854978 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.11674 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.16389 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.660895 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.584087 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.989564 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.459216 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.38779 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.10972 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.892727 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.17223 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.971647 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.15966 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.902611 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.709568 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.23343 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.873647 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.762511 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.21528 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.14236 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.911878 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.884575 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.26836 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.03516 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.03772 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.35425 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.69585 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.773599 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.83738 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.27336 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.15519 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.845912 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.06114 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.764951 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.776945 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.935205 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.07123 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.06921 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.17949 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.01067 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.17856 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.831293 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.07181 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.47418 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.813208 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.45262 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.0818 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.938703 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.0425 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.02498 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.10405 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.920435 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.819558 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.29098 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.886872 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.841663 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.16821 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.10344 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.95704 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.05836 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.09266 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.10089 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.911221 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.26477 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.39978 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.799534 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.910628 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.2313 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.16437 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.873565 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.15512 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.845705 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.822691 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.934345 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.94983 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.00457 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.01416 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.995577 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.11945 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.803921 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.93511 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.20236 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.842536 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.24706 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.567000695873866 z_k1_s0000
                                 - 0.167000369510314 z_k1_s0001
                                 - 0.209000601256936 z_k1_s0002
                                 - 0.288000853175164 z_k1_s0003
                                 - 0.178000491534569 z_k1_s0004
                                 - 0.154000277336313 z_k1_s0005
                                 - 0.535000601076626 z_k1_s0006
                                 - 0.764001190031674 z_k1_s0007
                                 - 0.245999776625855 z_k1_s0010
                                 - 0.329999649522436 z_k1_s0011
                                 - 0.330999970533436 z_k1_s0012
                                 - 0.241000255572963 z_k1_s0013
                                 - 0.263999302817528 z_k1_s0014
                                 - 0.0910000359714726 z_k1_s0015
                                 - 0.166000145772218 z_k1_s0016
                                 - 0.414999843430018 z_k1_s0017
                                 - 0.396999340426157 z_k1_s0020
                                 - 0.287000530867689 z_k1_s0021
                                 - 0.226000343603097 z_k1_s0022
                                 - 0.248999459128695 z_k1_s0023
                                 - 0.276999226725425 z_k1_s0024
                                 - 0.143000006026651 z_k1_s0025
                                 - 0.135000127913323 z_k1_s0026
                                 - 0.22800060443155 z_k1_s0027
                                 - 0.327999796311783 z_k1_s0030
                                 - 0.0980000554103103 z_k1_s0031
                                 - 0.252999851191717 z_k1_s0032
                                 - 0.149000181532071 z_k1_s0033
                                 - 0.10500005936819 z_k1_s0034
                                 - 0.0740001002119083 z_k1_s0035
                                 - 0.0790000057531818 z_k1_s0036
                                 - 0.275999961481961 z_k1_s0037
                                 - 0.265999670985196 z_k1_s0040
                                 - 0.119999996582814 z_k1_s0041
                                 - 0.232999832644943 z_k1_s0042
                                 - 0.318999106955864 z_k1_s0043
                                 - 0.460000790500893 z_k1_s0044
                                 - 0.231000877382033 z_k1_s0045
                                 - 0.224999993592776 z_k1_s0046
                                 - 0.154999995586135 z_k1_s0047
                                 - 0.221999748360041 z_k1_s0050
                                 - 0.443999496720082 z_k1_s0051
                                 - 0.161999912673462 z_k1_s0052
                                 - 0.117999879463434 z_k1_s0053
                                 - 0.235998920527577 z_k1_s0054
                                 - 0.543999572197419 z_k1_s0055
                                 - 0.171999920221196 z_k1_s0056
                                 - 0.19799973024548 z_k1_s0057
                                 - 0.0900001099532854 z_k1_s0060
                                 - 0.0940000873289912 z_k1_s0061
                                 - 0.0430000043887994 z_k1_s0062
                                 - 0.0549999984159074 z_k1_s0063
                                 - 0.0520000153841279 z_k1_s0064
                                 - 0.0749998852944367 z_k1_s0065
                                 - 0.094999772172966 z_k1_s0066
                                 - 0.115000278051399 z_k1_s0067
                                 - 0.22900105589479 z_k1_s0070
                                 - 0.119000498180931 z_k1_s0071
                                 - 0.13500045661154 z_k1_s0072
                                 - 0.106000519906011 z_k1_s0073
                                 - 0.2270017117936 z_k1_s0074
                                 - 0.474002648622533 z_k1_s0075
                                 - 0.324001712830197 z_k1_s0076
                                 - 0.314001136308624 z_k1_s0077
                                 - 0.338999025375803 z_k1_s0100
                                 - 0.504997123719769 z_k1_s0101
                                 - 0.260999458158719 z_k1_s0102
                                 - 0.375999132092092 z_k1_s0103
                                 - 0.722997251335929 z_k1_s0104
                                 - 0.280999317473431 z_k1_s0105
                                 - 0.784998244517921 z_k1_s0106
                                 - 0.592999904135904 z_k1_s0107
                                 - 0.155999970972782 z_k1_s0110
                                 - 0.178999994893118 z_k1_s0111
                                 - 0.441999297356456 z_k1_s0112
                                 - 0.124000033327062 z_k1_s0113
                                 - 0.256999472779705 z_k1_s0114
                                 - 0.101000009406726 z_k1_s0115
                                 - 0.331999035824475 z_k1_s0116
                                 - 0.101000009406726 z_k1_s0117
                                 - 0.439000015435755 z_k1_s0120
                                 - 0.402999789386555 z_k1_s0121
                                 - 0.498000048763571 z_k1_s0122
                                 - 0.440000007056565 z_k1_s0123
                                 - 1.51900046451101 z_k1_s0124
                                 - 0.600997602606744 z_k1_s0125
                                 - 0.8719979703464 z_k1_s0126
                                 - 0.504999990109237 z_k1_s0127
                                 - 0.138999921961056 z_k1_s0130
                                 - 0.277999450322438 z_k1_s0131
                                 - 0.300999447320302 z_k1_s0132
                                 - 0.360000038575198 z_k1_s0133
                                 - 0.128999931822847 z_k1_s0134
                                 - 0.308999360710934 z_k1_s0135
                                 - 0.0779998443580951 z_k1_s0136
                                 - 0.374000379008398 z_k1_s0137
                                 - 0.207000061779676 z_k1_s0140
                                 - 0.199000075121513 z_k1_s0141
                                 - 0.190000497130809 z_k1_s0142
                                 - 0.191000088463351 z_k1_s0143
                                 - 0.0699998832589194 z_k1_s0144
                                 - 0.342000243635891 z_k1_s0145
                                 - 0.312999071000868 z_k1_s0146
                                 - 0.115999969343243 z_k1_s0147
                                 - 0.614997913763583 z_k1_s0150
                                 - 0.417999320902343 z_k1_s0151
                                 - 0.342999358816891 z_k1_s0152
                                 - 1.07799738099692 z_k1_s0153
                                 - 0.478998636290237 z_k1_s0154
                                 - 0.245999588204748 z_k1_s0155
                                 - 0.478001065907826 z_k1_s0156
                                 - 0.569997344733271 z_k1_s0157
                                 - 0.213000421814194 z_k1_s0160
                                 - 0.371000704083799 z_k1_s0161
                                 - 0.811002545242368 z_k1_s0162
                                 - 0.345000974161765 z_k1_s0163
                                 - 0.583001624133012 z_k1_s0164
                                 - 0.31300097204148 z_k1_s0165
                                 - 0.664001856000565 z_k1_s0166
                                 - 0.324000927470211 z_k1_s0167
                                 - 0.19999996063048 z_k1_s0170
                                 - 0.162000034790678 z_k1_s0171
                                 - 0.253000550317463 z_k1_s0172
                                 - 0.153999936345475 z_k1_s0173
                                 - 0.12499997539405 z_k1_s0174
                                 - 0.161000001647531 z_k1_s0175
                                 - 0.345000598887473 z_k1_s0176
                                 - 0.26700001412168 z_k1_s0177
                                 - 0.879998168074487 z_k1_s1000
                                 - 2.15399757384326 z_k1_s1001
                                 - 1.97999290864879 z_k1_s1002
                                 - 1.06599329336845 z_k1_s1003
                                 - 0.866998071580122 z_k1_s1004
                                 - 0.472998661282025 z_k1_s1005
                                 - 0.844998048850903 z_k1_s1006
                                 - 0.871997827555464 z_k1_s1007
                                 - 0.843996706789677 z_k1_s1010
                                 - 0.682000576366674 z_k1_s1011
                                 - 0.775999405375552 z_k1_s1012
                                 - 0.392999427480134 z_k1_s1013
                                 - 0.725997806460072 z_k1_s1014
                                 - 0.51699791054073 z_k1_s1015
                                 - 0.19999943407488 z_k1_s1016
                                 - 0.212999762773066 z_k1_s1017
                                 - 1.19400322670129 z_k1_s1020
                                 - 0.870000475583468 z_k1_s1021
                                 - 0.370000845760631 z_k1_s1022
                                 - 0.690000377186889 z_k1_s1023
                                 - 1.08799930775067 z_k1_s1024
                                 - 0.385000853960346 z_k1_s1025
                                 - 0.752001698081034 z_k1_s1026
                                 - 1.08600445466534 z_k1_s1027
                                 - 0.498998817823211 z_k1_s1030
                                 - 0.322999624835104 z_k1_s1031
                                 - 0.86099674253777 z_k1_s1032
                                 - 0.381999472188962 z_k1_s1033
                                 - 0.752998562566568 z_k1_s1034
                                 - 0.738998615533434 z_k1_s1035
                                 - 1.08799941228604 z_k1_s1036
                                 - 0.294999730768835 z_k1_s1037
                                 - 0.492001028135944 z_k1_s1040
                                 - 0.467000585670652 z_k1_s1041
                                 - 0.500000591691859 z_k1_s1042
                                 - 0.59400114058347 z_k1_s1043
                                 - 0.316000307888343 z_k1_s1044
                                 - 1.4560031433163 z_k1_s1045
                                 - 0.76799938943772 z_k1_s1046
                                 - 0.567999978522376 z_k1_s1047
                                 - 0.493000218975649 z_k1_s1050
                                 - 0.198999004751807 z_k1_s1051
                                 - 0.139999963488104 z_k1_s1052
                                 - 0.0989998252310091 z_k1_s1053
                                 - 0.191999751326723 z_k1_s1054
                                 - 0.245999637943654 z_k1_s1055
                                 - 0.430000384355866 z_k1_s1056
                                 - 0.186999802280674 z_k1_s1057
                                 - 0.37999908090587 z_k1_s1060
                                 - 0.31499931761925 z_k1_s1061
                                 - 0.376999231261716 z_k1_s1062
                                 - 0.149999732599579 z_k1_s1063
                                 - 0.250999806313678 z_k1_s1064
                                 - 0.665999851964967 z_k1_s1065
                                 - 0.404000899055721 z_k1_s1066
                                 - 0.478000211274801 z_k1_s1067
                                 - 0.142999938881959 z_k1_s1070
                                 - 0.419000556808732 z_k1_s1071
                                 - 0.154999989765711 z_k1_s1072
                                 - 0.377998702740719 z_k1_s1073
                                 - 0.291998384107125 z_k1_s1074
                                 - 0.63999903113665 z_k1_s1075
                                 - 0.404000147954345 z_k1_s1076
                                 - 0.331000551927563 z_k1_s1077
                                 - 0.446001671486878 z_k1_s1100
                                 - 0.202000514627231 z_k1_s1101
                                 - 0.146000249118459 z_k1_s1102
                                 - 0.685001401487731 z_k1_s1103
                                 - 0.486001861136 z_k1_s1104
                                 - 0.175000312763933 z_k1_s1105
                                 - 0.638000218598196 z_k1_s1106
                                 - 0.176000243655022 z_k1_s1107
                                 - 0.822001462238478 z_k1_s1110
                                 - 0.37800109550948 z_k1_s1111
                                 - 0.859999936706991 z_k1_s1112
                                 - 0.78900504683842 z_k1_s1113
                                 - 0.286000777763103 z_k1_s1114
                                 - 0.248000693189741 z_k1_s1115
                                 - 1.11600124089817 z_k1_s1116
                                 - 0.868002157813283 z_k1_s1117
                                 - 0.342000981073801 z_k1_s1120
                                 - 0.0900001203244513 z_k1_s1121
                                 - 0.104000095388181 z_k1_s1122
                                 - 0.114000217891082 z_k1_s1123
                                 - 0.111000246620323 z_k1_s1124
                                 - 0.370000276400153 z_k1_s1125
                                 - 0.22400091062189 z_k1_s1126
                                 - 0.281999918655839 z_k1_s1127
                                 - 0.527000735460579 z_k1_s1130
                                 - 0.42099930679969 z_k1_s1131
                                 - 0.374999425191114 z_k1_s1132
                                 - 0.478999497818729 z_k1_s1133
                                 - 0.304999385160977 z_k1_s1134
                                 - 1.3909999930457 z_k1_s1135
                                 - 0.44399947780366 z_k1_s1136
                                 - 0.661999036835442 z_k1_s1137
                                 - 0.424001003375935 z_k1_s1140
                                 - 0.381001848068128 z_k1_s1141
                                 - 0.422000304126015 z_k1_s1142
                                 - 0.169000768977779 z_k1_s1143
                                 - 0.247000901812761 z_k1_s1144
                                 - 0.180000861736864 z_k1_s1145
                                 - 0.437001217982559 z_k1_s1146
                                 - 0.627001245451167 z_k1_s1147
                                 - 0.0790000353974138 z_k1_s1150
                                 - 0.376001202475264 z_k1_s1151
                                 - 0.24500100077808 z_k1_s1152
                                 - 0.334999985102718 z_k1_s1153
                                 - 0.12400022276074 z_k1_s1154
                                 - 0.170000282971949 z_k1_s1155
                                 - 0.182999498995697 z_k1_s1156
                                 - 0.226001157265017 z_k1_s1157
                                 - 0.258999300847458 z_k1_s1160
                                 - 0.545999251811983 z_k1_s1161
                                 - 0.251000161240307 z_k1_s1162
                                 - 0.384000198053006 z_k1_s1163
                                 - 0.111999671274634 z_k1_s1164
                                 - 0.47499803919276 z_k1_s1165
                                 - 0.141999740144284 z_k1_s1166
                                 - 0.272000038579434 z_k1_s1167
                                 - 0.351000431464525 z_k1_s1170
                                 - 0.357000083762692 z_k1_s1171
                                 - 0.121000142820528 z_k1_s1172
                                 - 0.245000804190866 z_k1_s1173
                                 - 0.0760001540815564 z_k1_s1174
                                 - 0.130000140568323 z_k1_s1175
                                 - 0.247000111289649 z_k1_s1176
                                 - 0.204000295150369 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.970478 w_k1_ADM_s000
                                 + 1.055 w_k1_AXP_s000 + 1.0707 w_k1_DIS_s000
                                 + 1.076 w_k1_FDX_s000 + 0.783085 w_k1_IBM_s000
                                 + 1.34914 w_k1_INTC_s000
                                 + 0.988262 w_k1_KO_s000
                                 + 0.866653 w_k1_MAT_s000
                                 + 1.11321 w_k1_MSFT_s000
                                 + 1.07685 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.2301 w_k1_ADM_s000
                                 + 1.53766 w_k1_AXP_s000
                                 + 1.26078 w_k1_DIS_s000
                                 + 1.03569 w_k1_FDX_s000
                                 + 1.92254 w_k1_IBM_s000
                                 + 1.37526 w_k1_INTC_s000
                                 + 1.34346 w_k1_KO_s000 + 2.03542 w_k1_MAT_s000
                                 + 1.05631 w_k1_MSFT_s000
                                 + 1.08399 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.599108 w_k1_ADM_s000
                                 + 0.63115 w_k1_AXP_s000
                                 + 0.495915 w_k1_DIS_s000
                                 + 0.651193 w_k1_FDX_s000
                                 + 0.627367 w_k1_IBM_s000
                                 + 0.911197 w_k1_INTC_s000
                                 + 0.574939 w_k1_KO_s000
                                 + 0.435179 w_k1_MAT_s000
                                 + 0.694123 w_k1_MSFT_s000
                                 + 0.69684 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.806196 w_k1_ADM_s000
                                 + 1.00828 w_k1_AXP_s000
                                 + 0.636419 w_k1_DIS_s000
                                 + 0.726586 w_k1_FDX_s000
                                 + 1.01584 w_k1_IBM_s000
                                 + 1.13121 w_k1_INTC_s000
                                 + 0.90118 w_k1_KO_s000
                                 + 0.860497 w_k1_MAT_s000
                                 + 0.738993 w_k1_MSFT_s000
                                 + 0.907647 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.722372 w_k1_ADM_s000
                                 + 0.803969 w_k1_AXP_s000
                                 + 0.827862 w_k1_DIS_s000
                                 + 0.772039 w_k1_FDX_s000
                                 + 1.09868 w_k1_IBM_s000
                                 + 1.10556 w_k1_INTC_s000
                                 + 1.02696 w_k1_KO_s000
                                 + 0.870743 w_k1_MAT_s000
                                 + 1.00063 w_k1_MSFT_s000
                                 + 1.01733 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.831766 w_k1_ADM_s000
                                 + 0.704443 w_k1_AXP_s000
                                 + 0.732847 w_k1_DIS_s000
                                 + 0.825672 w_k1_FDX_s000
                                 + 0.55736 w_k1_IBM_s000
                                 + 1.19425 w_k1_INTC_s000
                                 + 0.730315 w_k1_KO_s000
                                 + 0.379996 w_k1_MAT_s000
                                 + 0.843432 w_k1_MSFT_s000
                                 + 0.807252 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.793647 w_k1_ADM_s000
                                 + 0.913623 w_k1_AXP_s000
                                 + 0.941196 w_k1_DIS_s000
                                 + 0.912833 w_k1_FDX_s000
                                 + 0.918117 w_k1_IBM_s000
                                 + 1.27926 w_k1_INTC_s000
                                 + 0.941022 w_k1_KO_s000
                                 + 0.604033 w_k1_MAT_s000
                                 + 0.923305 w_k1_MSFT_s000
                                 + 0.886387 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.23039 w_k1_ADM_s000
                                 + 1.42793 w_k1_AXP_s000 + 1.4811 w_k1_DIS_s000
                                 + 1.20643 w_k1_FDX_s000
                                 + 1.16032 w_k1_IBM_s000
                                 + 1.26883 w_k1_INTC_s000
                                 + 1.22568 w_k1_KO_s000 + 1.27651 w_k1_MAT_s000
                                 + 1.14183 w_k1_MSFT_s000
                                 + 1.12059 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.06919 w_k1_ADM_s001
                                 + 0.924007 w_k1_AXP_s001
                                 + 0.979143 w_k1_DIS_s001
                                 + 0.93757 w_k1_FDX_s001
                                 + 1.51518 w_k1_IBM_s001
                                 + 1.04194 w_k1_INTC_s001
                                 + 1.18924 w_k1_KO_s001 + 1.06544 w_k1_MAT_s001
                                 + 0.93128 w_k1_MSFT_s001
                                 + 0.950323 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.24413 w_k1_ADM_s001
                                 + 1.23116 w_k1_AXP_s001
                                 + 1.20823 w_k1_DIS_s001 + 1.1006 w_k1_FDX_s001
                                 + 1.05285 w_k1_IBM_s001
                                 + 1.16816 w_k1_INTC_s001
                                 + 1.19179 w_k1_KO_s001
                                 + 0.970184 w_k1_MAT_s001
                                 + 1.23153 w_k1_MSFT_s001
                                 + 1.12759 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.0707 w_k1_ADM_s001
                                 + 0.822352 w_k1_AXP_s001
                                 + 0.789079 w_k1_DIS_s001
                                 + 0.95327 w_k1_FDX_s001
                                 + 0.658589 w_k1_IBM_s001
                                 + 0.982731 w_k1_INTC_s001
                                 + 0.680889 w_k1_KO_s001
                                 + 0.655471 w_k1_MAT_s001
                                 + 0.871875 w_k1_MSFT_s001
                                 + 0.814172 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.32823 w_k1_ADM_s001
                                 + 1.03148 w_k1_AXP_s001
                                 + 1.43185 w_k1_DIS_s001
                                 + 1.22016 w_k1_FDX_s001
                                 + 0.788094 w_k1_IBM_s001
                                 + 1.04332 w_k1_INTC_s001
                                 + 1.00481 w_k1_KO_s001
                                 + 0.670971 w_k1_MAT_s001
                                 + 1.25138 w_k1_MSFT_s001 + 1.0176 w_k1_PG_s001
                                 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.14072 w_k1_ADM_s001
                                 + 0.996878 w_k1_AXP_s001
                                 + 0.970784 w_k1_DIS_s001
                                 + 0.948037 w_k1_FDX_s001
                                 + 1.07259 w_k1_IBM_s001
                                 + 1.05471 w_k1_INTC_s001
                                 + 1.05214 w_k1_KO_s001
                                 + 0.915346 w_k1_MAT_s001
                                 + 1.04912 w_k1_MSFT_s001
                                 + 1.00622 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.28224 w_k1_ADM_s001
                                 + 1.11854 w_k1_AXP_s001
                                 + 1.84254 w_k1_DIS_s001 + 1.2253 w_k1_FDX_s001
                                 + 1.27268 w_k1_IBM_s001
                                 + 1.26813 w_k1_INTC_s001 + 1.3402 w_k1_KO_s001
                                 + 0.805552 w_k1_MAT_s001
                                 + 1.39412 w_k1_MSFT_s001
                                 + 1.10431 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.32489 w_k1_ADM_s001
                                 + 1.65855 w_k1_AXP_s001
                                 + 2.07005 w_k1_DIS_s001
                                 + 1.53824 w_k1_FDX_s001
                                 + 1.23284 w_k1_IBM_s001
                                 + 1.60641 w_k1_INTC_s001
                                 + 1.55093 w_k1_KO_s001 + 1.23932 w_k1_MAT_s001
                                 + 1.43507 w_k1_MSFT_s001
                                 + 1.51301 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.05716 w_k1_ADM_s001
                                 + 0.71008 w_k1_AXP_s001
                                 + 0.733184 w_k1_DIS_s001
                                 + 0.742466 w_k1_FDX_s001
                                 + 1.01133 w_k1_IBM_s001
                                 + 0.855425 w_k1_INTC_s001
                                 + 0.830095 w_k1_KO_s001
                                 + 0.83577 w_k1_MAT_s001
                                 + 0.748886 w_k1_MSFT_s001
                                 + 0.888838 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.07061 w_k1_ADM_s002
                                 + 0.697044 w_k1_AXP_s002
                                 + 0.86684 w_k1_DIS_s002
                                 + 0.884705 w_k1_FDX_s002
                                 + 0.794899 w_k1_IBM_s002
                                 + 0.818704 w_k1_INTC_s002
                                 + 1.02099 w_k1_KO_s002
                                 + 0.593428 w_k1_MAT_s002
                                 + 1.0221 w_k1_MSFT_s002
                                 + 0.945912 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.23705 w_k1_ADM_s002
                                 + 1.1932 w_k1_AXP_s002 + 1.17649 w_k1_DIS_s002
                                 + 1.11546 w_k1_FDX_s002
                                 + 1.45846 w_k1_IBM_s002
                                 + 1.13221 w_k1_INTC_s002
                                 + 1.31799 w_k1_KO_s002 + 1.21075 w_k1_MAT_s002
                                 + 1.01304 w_k1_MSFT_s002
                                 + 1.13979 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.04763 w_k1_ADM_s002
                                 + 1.08128 w_k1_AXP_s002
                                 + 1.34944 w_k1_DIS_s002
                                 + 1.15618 w_k1_FDX_s002
                                 + 1.04608 w_k1_IBM_s002
                                 + 1.39693 w_k1_INTC_s002
                                 + 1.46985 w_k1_KO_s002 + 1.09652 w_k1_MAT_s002
                                 + 1.38696 w_k1_MSFT_s002
                                 + 1.21674 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.45392 w_k1_ADM_s002
                                 + 1.09463 w_k1_AXP_s002
                                 + 1.01009 w_k1_DIS_s002 + 1.1084 w_k1_FDX_s002
                                 + 0.867364 w_k1_IBM_s002
                                 + 1.15979 w_k1_INTC_s002
                                 + 0.972434 w_k1_KO_s002
                                 + 0.973398 w_k1_MAT_s002
                                 + 1.14064 w_k1_MSFT_s002
                                 + 0.98015 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.981902 w_k1_ADM_s002
                                 + 0.912295 w_k1_AXP_s002
                                 + 0.821947 w_k1_DIS_s002
                                 + 0.909346 w_k1_FDX_s002
                                 + 0.901982 w_k1_IBM_s002
                                 + 0.948671 w_k1_INTC_s002
                                 + 1.04574 w_k1_KO_s002 + 0.96238 w_k1_MAT_s002
                                 + 0.929713 w_k1_MSFT_s002
                                 + 0.960531 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.64769 w_k1_ADM_s002
                                 + 1.63921 w_k1_AXP_s002
                                 + 1.70395 w_k1_DIS_s002
                                 + 1.45613 w_k1_FDX_s002
                                 + 1.25907 w_k1_IBM_s002
                                 + 1.23071 w_k1_INTC_s002
                                 + 1.54587 w_k1_KO_s002 + 1.20275 w_k1_MAT_s002
                                 + 1.39923 w_k1_MSFT_s002
                                 + 1.40171 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.882122 w_k1_ADM_s002
                                 + 0.760779 w_k1_AXP_s002
                                 + 1.24148 w_k1_DIS_s002
                                 + 1.04338 w_k1_FDX_s002
                                 + 1.05788 w_k1_IBM_s002
                                 + 1.09778 w_k1_INTC_s002
                                 + 1.34275 w_k1_KO_s002 + 1.0451 w_k1_MAT_s002
                                 + 1.17548 w_k1_MSFT_s002
                                 + 1.20883 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.954206 w_k1_ADM_s002
                                 + 0.753219 w_k1_AXP_s002
                                 + 0.646373 w_k1_DIS_s002
                                 + 0.74343 w_k1_FDX_s002
                                 + 0.971999 w_k1_IBM_s002
                                 + 0.894402 w_k1_INTC_s002
                                 + 0.915672 w_k1_KO_s002
                                 + 1.14628 w_k1_MAT_s002
                                 + 0.884919 w_k1_MSFT_s002
                                 + 0.858446 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.02257 w_k1_ADM_s003
                                 + 0.999113 w_k1_AXP_s003
                                 + 1.0111 w_k1_DIS_s003 + 1.10865 w_k1_FDX_s003
                                 + 0.822368 w_k1_IBM_s003
                                 + 1.07106 w_k1_INTC_s003
                                 + 1.03911 w_k1_KO_s003
                                 + 0.816768 w_k1_MAT_s003
                                 + 1.16731 w_k1_MSFT_s003
                                 + 1.03226 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.15771 w_k1_ADM_s003
                                 + 1.44801 w_k1_AXP_s003
                                 + 1.38109 w_k1_DIS_s003
                                 + 1.23774 w_k1_FDX_s003
                                 + 1.16023 w_k1_IBM_s003
                                 + 1.00187 w_k1_INTC_s003 + 1.4092 w_k1_KO_s003
                                 + 1.59641 w_k1_MAT_s003
                                 + 1.20598 w_k1_MSFT_s003
                                 + 1.21578 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.852211 w_k1_ADM_s003
                                 + 0.674843 w_k1_AXP_s003
                                 + 0.648054 w_k1_DIS_s003
                                 + 0.773855 w_k1_FDX_s003
                                 + 0.816963 w_k1_IBM_s003
                                 + 0.752592 w_k1_INTC_s003
                                 + 0.893195 w_k1_KO_s003
                                 + 0.606059 w_k1_MAT_s003
                                 + 0.715641 w_k1_MSFT_s003
                                 + 0.841612 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.19641 w_k1_ADM_s003
                                 + 1.24789 w_k1_AXP_s003
                                 + 1.54639 w_k1_DIS_s003
                                 + 1.24569 w_k1_FDX_s003
                                 + 1.48171 w_k1_IBM_s003
                                 + 1.26897 w_k1_INTC_s003
                                 + 1.52773 w_k1_KO_s003 + 1.3636 w_k1_MAT_s003
                                 + 1.25421 w_k1_MSFT_s003
                                 + 1.19624 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.852207 w_k1_ADM_s003
                                 + 1.05517 w_k1_AXP_s003
                                 + 1.08113 w_k1_DIS_s003
                                 + 0.937538 w_k1_FDX_s003
                                 + 1.60702 w_k1_IBM_s003
                                 + 0.906579 w_k1_INTC_s003
                                 + 1.23853 w_k1_KO_s003 + 1.74361 w_k1_MAT_s003
                                 + 0.968986 w_k1_MSFT_s003
                                 + 1.0842 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.930707 w_k1_ADM_s003
                                 + 0.549244 w_k1_AXP_s003
                                 + 0.920032 w_k1_DIS_s003
                                 + 0.921235 w_k1_FDX_s003
                                 + 0.944013 w_k1_IBM_s003
                                 + 0.915358 w_k1_INTC_s003
                                 + 1.0209 w_k1_KO_s003 + 0.751893 w_k1_MAT_s003
                                 + 1.07423 w_k1_MSFT_s003
                                 + 0.926257 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.3731 w_k1_ADM_s003
                                 + 1.12965 w_k1_AXP_s003
                                 + 1.47613 w_k1_DIS_s003
                                 + 1.40804 w_k1_FDX_s003
                                 + 1.05189 w_k1_IBM_s003
                                 + 1.16474 w_k1_INTC_s003
                                 + 1.18004 w_k1_KO_s003
                                 + 0.750996 w_k1_MAT_s003
                                 + 1.24247 w_k1_MSFT_s003
                                 + 1.19982 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.924079 w_k1_ADM_s003
                                 + 0.912507 w_k1_AXP_s003
                                 + 1.04286 w_k1_DIS_s003
                                 + 0.964289 w_k1_FDX_s003
                                 + 1.15674 w_k1_IBM_s003
                                 + 0.990325 w_k1_INTC_s003
                                 + 1.04657 w_k1_KO_s003 + 0.90224 w_k1_MAT_s003
                                 + 1.00956 w_k1_MSFT_s003
                                 + 1.09073 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.858641 w_k1_ADM_s004
                                 + 0.70369 w_k1_AXP_s004
                                 + 0.656859 w_k1_DIS_s004
                                 + 0.850684 w_k1_FDX_s004
                                 + 0.722529 w_k1_IBM_s004
                                 + 0.922786 w_k1_INTC_s004
                                 + 0.727377 w_k1_KO_s004
                                 + 0.644317 w_k1_MAT_s004
                                 + 0.783038 w_k1_MSFT_s004
                                 + 0.793446 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.900005 w_k1_ADM_s004
                                 + 0.91549 w_k1_AXP_s004
                                 + 1.55653 w_k1_DIS_s004 + 1.0585 w_k1_FDX_s004
                                 + 1.39806 w_k1_IBM_s004
                                 + 1.31639 w_k1_INTC_s004
                                 + 1.33409 w_k1_KO_s004
                                 + 0.955346 w_k1_MAT_s004
                                 + 1.17844 w_k1_MSFT_s004
                                 + 1.06758 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.10339 w_k1_ADM_s004
                                 + 1.60316 w_k1_AXP_s004
                                 + 1.71496 w_k1_DIS_s004
                                 + 1.43747 w_k1_FDX_s004
                                 + 1.26457 w_k1_IBM_s004
                                 + 1.30766 w_k1_INTC_s004
                                 + 1.25233 w_k1_KO_s004 + 1.55171 w_k1_MAT_s004
                                 + 1.25247 w_k1_MSFT_s004
                                 + 1.13318 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.98584 w_k1_ADM_s004
                                 + 0.770872 w_k1_AXP_s004
                                 + 0.769193 w_k1_DIS_s004
                                 + 0.831993 w_k1_FDX_s004
                                 + 0.977067 w_k1_IBM_s004
                                 + 1.00864 w_k1_INTC_s004
                                 + 0.89432 w_k1_KO_s004
                                 + 0.778839 w_k1_MAT_s004
                                 + 0.874304 w_k1_MSFT_s004
                                 + 0.9805 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.843232 w_k1_ADM_s004
                                 + 0.960042 w_k1_AXP_s004
                                 + 1.03558 w_k1_DIS_s004
                                 + 1.01021 w_k1_FDX_s004 + 1.0987 w_k1_IBM_s004
                                 + 1.15031 w_k1_INTC_s004
                                 + 1.00722 w_k1_KO_s004 + 1.15146 w_k1_MAT_s004
                                 + 0.901902 w_k1_MSFT_s004
                                 + 0.947311 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.92068 w_k1_ADM_s004
                                 + 1.17394 w_k1_AXP_s004
                                 + 0.989211 w_k1_DIS_s004
                                 + 1.03593 w_k1_FDX_s004
                                 + 1.38637 w_k1_IBM_s004
                                 + 1.13488 w_k1_INTC_s004
                                 + 1.03587 w_k1_KO_s004 + 1.45328 w_k1_MAT_s004
                                 + 0.864749 w_k1_MSFT_s004
                                 + 0.909615 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.0886 w_k1_ADM_s004
                                 + 1.34091 w_k1_AXP_s004
                                 + 1.07515 w_k1_DIS_s004
                                 + 1.28702 w_k1_FDX_s004 + 1.0079 w_k1_IBM_s004
                                 + 1.32056 w_k1_INTC_s004
                                 + 1.00051 w_k1_KO_s004
                                 + 0.993885 w_k1_MAT_s004
                                 + 1.06408 w_k1_MSFT_s004
                                 + 0.945346 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.639634 w_k1_ADM_s004
                                 + 0.788912 w_k1_AXP_s004
                                 + 0.843216 w_k1_DIS_s004
                                 + 0.881695 w_k1_FDX_s004
                                 + 0.697051 w_k1_IBM_s004
                                 + 0.942811 w_k1_INTC_s004
                                 + 0.828646 w_k1_KO_s004
                                 + 0.768419 w_k1_MAT_s004
                                 + 0.976749 w_k1_MSFT_s004
                                 + 0.934014 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.976923 w_k1_ADM_s005
                                 + 1.15055 w_k1_AXP_s005 + 1.0353 w_k1_DIS_s005
                                 + 0.965154 w_k1_FDX_s005
                                 + 1.51055 w_k1_IBM_s005
                                 + 0.981144 w_k1_INTC_s005
                                 + 1.12227 w_k1_KO_s005 + 1.22306 w_k1_MAT_s005
                                 + 0.87345 w_k1_MSFT_s005
                                 + 1.05434 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.961867 w_k1_ADM_s005
                                 + 0.769801 w_k1_AXP_s005
                                 + 0.798519 w_k1_DIS_s005
                                 + 0.830241 w_k1_FDX_s005
                                 + 0.734725 w_k1_IBM_s005
                                 + 1.03622 w_k1_INTC_s005
                                 + 0.800542 w_k1_KO_s005
                                 + 0.557187 w_k1_MAT_s005
                                 + 0.838404 w_k1_MSFT_s005
                                 + 0.858405 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.956237 w_k1_ADM_s005
                                 + 1.49977 w_k1_AXP_s005
                                 + 1.43953 w_k1_DIS_s005
                                 + 1.26208 w_k1_FDX_s005
                                 + 1.45036 w_k1_IBM_s005
                                 + 1.04314 w_k1_INTC_s005
                                 + 1.13746 w_k1_KO_s005 + 1.76077 w_k1_MAT_s005
                                 + 1.13122 w_k1_MSFT_s005
                                 + 1.12488 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.791575 w_k1_ADM_s005
                                 + 0.69114 w_k1_AXP_s005
                                 + 0.658201 w_k1_DIS_s005
                                 + 0.751665 w_k1_FDX_s005
                                 + 0.896612 w_k1_IBM_s005
                                 + 0.8094 w_k1_INTC_s005
                                 + 0.735918 w_k1_KO_s005
                                 + 0.480327 w_k1_MAT_s005
                                 + 0.68932 w_k1_MSFT_s005
                                 + 0.688551 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.10045 w_k1_ADM_s005
                                 + 1.13208 w_k1_AXP_s005
                                 + 1.60598 w_k1_DIS_s005
                                 + 1.27917 w_k1_FDX_s005
                                 + 0.988423 w_k1_IBM_s005
                                 + 1.21283 w_k1_INTC_s005
                                 + 1.09967 w_k1_KO_s005 + 0.98329 w_k1_MAT_s005
                                 + 1.1743 w_k1_MSFT_s005 + 1.0472 w_k1_PG_s005
                                 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.06944 w_k1_ADM_s005
                                 + 1.10703 w_k1_AXP_s005
                                 + 1.04463 w_k1_DIS_s005
                                 + 1.11003 w_k1_FDX_s005
                                 + 0.908391 w_k1_IBM_s005
                                 + 1.16584 w_k1_INTC_s005
                                 + 1.08502 w_k1_KO_s005
                                 + 0.761487 w_k1_MAT_s005
                                 + 1.00727 w_k1_MSFT_s005 + 1.0487 w_k1_PG_s005
                                 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.68663 w_k1_ADM_s005
                                 + 0.858159 w_k1_AXP_s005
                                 + 0.694698 w_k1_DIS_s005
                                 + 0.770367 w_k1_FDX_s005
                                 + 1.23038 w_k1_IBM_s005
                                 + 1.00807 w_k1_INTC_s005
                                 + 0.885568 w_k1_KO_s005
                                 + 0.976999 w_k1_MAT_s005
                                 + 0.806448 w_k1_MSFT_s005
                                 + 0.914501 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.905381 w_k1_ADM_s005
                                 + 0.861136 w_k1_AXP_s005
                                 + 1.1608 w_k1_DIS_s005 + 1.04004 w_k1_FDX_s005
                                 + 1.26665 w_k1_IBM_s005
                                 + 0.962596 w_k1_INTC_s005
                                 + 1.15668 w_k1_KO_s005 + 1.14498 w_k1_MAT_s005
                                 + 0.994664 w_k1_MSFT_s005
                                 + 1.06492 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.00535 w_k1_ADM_s006
                                 + 1.44603 w_k1_AXP_s006
                                 + 1.62068 w_k1_DIS_s006
                                 + 1.24899 w_k1_FDX_s006
                                 + 1.37033 w_k1_IBM_s006
                                 + 1.22952 w_k1_INTC_s006
                                 + 1.27473 w_k1_KO_s006 + 1.45143 w_k1_MAT_s006
                                 + 1.1004 w_k1_MSFT_s006 + 1.06199 w_k1_PG_s006
                                 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.03444 w_k1_ADM_s006
                                 + 1.33608 w_k1_AXP_s006 + 1.0767 w_k1_DIS_s006
                                 + 1.03813 w_k1_FDX_s006
                                 + 1.29423 w_k1_IBM_s006
                                 + 1.05058 w_k1_INTC_s006 + 1.0927 w_k1_KO_s006
                                 + 1.76486 w_k1_MAT_s006
                                 + 1.00113 w_k1_MSFT_s006
                                 + 0.925849 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.15408 w_k1_ADM_s006
                                 + 0.971261 w_k1_AXP_s006
                                 + 1.32776 w_k1_DIS_s006
                                 + 1.05279 w_k1_FDX_s006
                                 + 0.870508 w_k1_IBM_s006
                                 + 0.998556 w_k1_INTC_s006
                                 + 1.03573 w_k1_KO_s006
                                 + 0.580587 w_k1_MAT_s006
                                 + 0.895493 w_k1_MSFT_s006
                                 + 0.909539 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.876137 w_k1_ADM_s006
                                 + 0.992928 w_k1_AXP_s006
                                 + 0.990433 w_k1_DIS_s006
                                 + 0.84392 w_k1_FDX_s006
                                 + 1.60283 w_k1_IBM_s006
                                 + 0.893234 w_k1_INTC_s006
                                 + 1.08474 w_k1_KO_s006 + 1.60725 w_k1_MAT_s006
                                 + 0.85939 w_k1_MSFT_s006
                                 + 0.949707 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.865726 w_k1_ADM_s006
                                 + 0.713233 w_k1_AXP_s006
                                 + 0.692561 w_k1_DIS_s006
                                 + 0.796182 w_k1_FDX_s006
                                 + 0.651913 w_k1_IBM_s006
                                 + 0.879959 w_k1_INTC_s006
                                 + 0.733067 w_k1_KO_s006
                                 + 0.538335 w_k1_MAT_s006
                                 + 0.912869 w_k1_MSFT_s006
                                 + 0.787014 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.21024 w_k1_ADM_s006
                                 + 0.830673 w_k1_AXP_s006
                                 + 0.736658 w_k1_DIS_s006
                                 + 0.794827 w_k1_FDX_s006
                                 + 0.965706 w_k1_IBM_s006
                                 + 0.811254 w_k1_INTC_s006
                                 + 0.911664 w_k1_KO_s006
                                 + 1.18732 w_k1_MAT_s006
                                 + 0.864341 w_k1_MSFT_s006
                                 + 0.81588 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.944601 w_k1_ADM_s006
                                 + 1.00028 w_k1_AXP_s006
                                 + 0.881447 w_k1_DIS_s006
                                 + 0.899251 w_k1_FDX_s006
                                 + 0.932229 w_k1_IBM_s006
                                 + 1.10901 w_k1_INTC_s006
                                 + 0.99997 w_k1_KO_s006 + 0.96922 w_k1_MAT_s006
                                 + 0.903325 w_k1_MSFT_s006
                                 + 0.980475 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.10431 w_k1_ADM_s006
                                 + 1.03516 w_k1_AXP_s006
                                 + 1.10417 w_k1_DIS_s006
                                 + 1.08405 w_k1_FDX_s006
                                 + 0.951213 w_k1_IBM_s006
                                 + 1.02063 w_k1_INTC_s006
                                 + 1.04588 w_k1_KO_s006 + 1.09478 w_k1_MAT_s006
                                 + 0.991918 w_k1_MSFT_s006
                                 + 0.890049 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.988966 w_k1_ADM_s007
                                 + 1.13689 w_k1_AXP_s007
                                 + 1.42032 w_k1_DIS_s007
                                 + 1.06845 w_k1_FDX_s007
                                 + 1.57154 w_k1_IBM_s007
                                 + 1.16812 w_k1_INTC_s007 + 1.3318 w_k1_KO_s007
                                 + 1.24716 w_k1_MAT_s007
                                 + 1.11972 w_k1_MSFT_s007
                                 + 0.953047 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.940966 w_k1_ADM_s007
                                 + 1.61167 w_k1_AXP_s007
                                 + 1.65831 w_k1_DIS_s007
                                 + 1.44535 w_k1_FDX_s007
                                 + 1.16466 w_k1_IBM_s007
                                 + 1.42785 w_k1_INTC_s007
                                 + 1.20025 w_k1_KO_s007 + 1.28331 w_k1_MAT_s007
                                 + 1.36717 w_k1_MSFT_s007
                                 + 1.18784 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.15218 w_k1_ADM_s007
                                 + 0.885612 w_k1_AXP_s007
                                 + 0.926418 w_k1_DIS_s007
                                 + 0.990979 w_k1_FDX_s007
                                 + 0.81577 w_k1_IBM_s007
                                 + 0.834221 w_k1_INTC_s007
                                 + 0.821682 w_k1_KO_s007
                                 + 0.810092 w_k1_MAT_s007
                                 + 0.901134 w_k1_MSFT_s007
                                 + 0.913442 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.07791 w_k1_ADM_s007
                                 + 0.864529 w_k1_AXP_s007
                                 + 1.18103 w_k1_DIS_s007
                                 + 1.13956 w_k1_FDX_s007
                                 + 0.739302 w_k1_IBM_s007
                                 + 1.1857 w_k1_INTC_s007
                                 + 0.869778 w_k1_KO_s007 + 1.0021 w_k1_MAT_s007
                                 + 1.1266 w_k1_MSFT_s007
                                 + 0.999606 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.953989 w_k1_ADM_s007
                                 + 0.793826 w_k1_AXP_s007
                                 + 0.751794 w_k1_DIS_s007
                                 + 0.830399 w_k1_FDX_s007
                                 + 0.847188 w_k1_IBM_s007
                                 + 1.08278 w_k1_INTC_s007
                                 + 0.694827 w_k1_KO_s007
                                 + 0.727552 w_k1_MAT_s007
                                 + 0.873133 w_k1_MSFT_s007
                                 + 0.834327 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.873098 w_k1_ADM_s007
                                 + 0.787904 w_k1_AXP_s007
                                 + 0.85513 w_k1_DIS_s007
                                 + 0.826585 w_k1_FDX_s007
                                 + 0.920908 w_k1_IBM_s007
                                 + 0.989689 w_k1_INTC_s007
                                 + 0.899274 w_k1_KO_s007
                                 + 0.937407 w_k1_MAT_s007
                                 + 0.889088 w_k1_MSFT_s007
                                 + 0.863195 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.895518 w_k1_ADM_s007
                                 + 1.02668 w_k1_AXP_s007
                                 + 0.961482 w_k1_DIS_s007
                                 + 1.01924 w_k1_FDX_s007
                                 + 1.05712 w_k1_IBM_s007
                                 + 0.981948 w_k1_INTC_s007
                                 + 0.900417 w_k1_KO_s007
                                 + 1.14032 w_k1_MAT_s007
                                 + 0.932307 w_k1_MSFT_s007
                                 + 0.916799 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.22872 w_k1_ADM_s007
                                 + 1.20632 w_k1_AXP_s007
                                 + 1.32601 w_k1_DIS_s007
                                 + 1.11437 w_k1_FDX_s007
                                 + 1.04158 w_k1_IBM_s007
                                 + 1.22908 w_k1_INTC_s007
                                 + 1.06911 w_k1_KO_s007 + 1.04546 w_k1_MAT_s007
                                 + 1.2502 w_k1_MSFT_s007 + 1.02605 w_k1_PG_s007
                                 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.950948 w_k1_ADM_s010
                                 + 0.698356 w_k1_AXP_s010
                                 + 0.638909 w_k1_DIS_s010
                                 + 0.781706 w_k1_FDX_s010
                                 + 0.701667 w_k1_IBM_s010
                                 + 0.674044 w_k1_INTC_s010
                                 + 0.703841 w_k1_KO_s010
                                 + 0.464388 w_k1_MAT_s010
                                 + 0.637753 w_k1_MSFT_s010
                                 + 0.760562 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.902045 w_k1_ADM_s010
                                 + 1.01974 w_k1_AXP_s010
                                 + 0.825851 w_k1_DIS_s010
                                 + 0.977053 w_k1_FDX_s010
                                 + 1.00279 w_k1_IBM_s010
                                 + 0.823424 w_k1_INTC_s010
                                 + 0.840376 w_k1_KO_s010
                                 + 1.15094 w_k1_MAT_s010
                                 + 0.730289 w_k1_MSFT_s010
                                 + 0.913637 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.992446 w_k1_ADM_s010
                                 + 0.802102 w_k1_AXP_s010
                                 + 0.767398 w_k1_DIS_s010
                                 + 0.924462 w_k1_FDX_s010
                                 + 1.31233 w_k1_IBM_s010
                                 + 0.858428 w_k1_INTC_s010
                                 + 1.01228 w_k1_KO_s010 + 1.2422 w_k1_MAT_s010
                                 + 0.829836 w_k1_MSFT_s010
                                 + 1.01767 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.13982 w_k1_ADM_s010
                                 + 1.35749 w_k1_AXP_s010
                                 + 1.72603 w_k1_DIS_s010
                                 + 1.44897 w_k1_FDX_s010
                                 + 1.31674 w_k1_IBM_s010
                                 + 1.09854 w_k1_INTC_s010
                                 + 1.39016 w_k1_KO_s010 + 1.47146 w_k1_MAT_s010
                                 + 1.28372 w_k1_MSFT_s010
                                 + 1.24826 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.16753 w_k1_ADM_s010
                                 + 1.20584 w_k1_AXP_s010
                                 + 1.16246 w_k1_DIS_s010 + 1.0817 w_k1_FDX_s010
                                 + 1.32279 w_k1_IBM_s010
                                 + 0.889218 w_k1_INTC_s010
                                 + 1.11736 w_k1_KO_s010 + 1.42495 w_k1_MAT_s010
                                 + 0.85866 w_k1_MSFT_s010
                                 + 1.12929 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.876738 w_k1_ADM_s010
                                 + 1.15216 w_k1_AXP_s010
                                 + 0.958571 w_k1_DIS_s010
                                 + 1.18995 w_k1_FDX_s010
                                 + 0.651399 w_k1_IBM_s010
                                 + 0.965458 w_k1_INTC_s010
                                 + 0.795836 w_k1_KO_s010
                                 + 0.56543 w_k1_MAT_s010
                                 + 0.90583 w_k1_MSFT_s010
                                 + 0.908799 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.976901 w_k1_ADM_s010
                                 + 0.99812 w_k1_AXP_s010
                                 + 1.24319 w_k1_DIS_s010
                                 + 1.14737 w_k1_FDX_s010
                                 + 0.874753 w_k1_IBM_s010
                                 + 0.832824 w_k1_INTC_s010
                                 + 0.894037 w_k1_KO_s010
                                 + 0.807363 w_k1_MAT_s010
                                 + 1.11178 w_k1_MSFT_s010
                                 + 1.02847 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.897862 w_k1_ADM_s010
                                 + 0.723014 w_k1_AXP_s010
                                 + 0.75295 w_k1_DIS_s010
                                 + 0.919324 w_k1_FDX_s010
                                 + 0.822363 w_k1_IBM_s010
                                 + 0.771162 w_k1_INTC_s010
                                 + 0.737246 w_k1_KO_s010
                                 + 0.744154 w_k1_MAT_s010
                                 + 0.802022 w_k1_MSFT_s010
                                 + 0.878502 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.07344 w_k1_ADM_s011
                                 + 1.2504 w_k1_AXP_s011 + 1.55833 w_k1_DIS_s011
                                 + 1.39587 w_k1_FDX_s011
                                 + 1.24211 w_k1_IBM_s011
                                 + 1.39542 w_k1_INTC_s011
                                 + 1.37531 w_k1_KO_s011 + 1.34288 w_k1_MAT_s011
                                 + 1.29637 w_k1_MSFT_s011
                                 + 1.17404 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.984966 w_k1_ADM_s011
                                 + 1.02073 w_k1_AXP_s011
                                 + 0.962216 w_k1_DIS_s011
                                 + 0.984736 w_k1_FDX_s011
                                 + 1.50528 w_k1_IBM_s011
                                 + 0.925574 w_k1_INTC_s011
                                 + 1.18749 w_k1_KO_s011 + 1.81725 w_k1_MAT_s011
                                 + 1.01693 w_k1_MSFT_s011
                                 + 1.02367 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.13189 w_k1_ADM_s011
                                 + 0.769143 w_k1_AXP_s011
                                 + 0.928405 w_k1_DIS_s011
                                 + 0.9397 w_k1_FDX_s011
                                 + 0.825007 w_k1_IBM_s011
                                 + 1.03693 w_k1_INTC_s011
                                 + 0.859175 w_k1_KO_s011
                                 + 0.468489 w_k1_MAT_s011
                                 + 0.929736 w_k1_MSFT_s011
                                 + 0.905138 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.02773 w_k1_ADM_s011
                                 + 0.981633 w_k1_AXP_s011
                                 + 1.01179 w_k1_DIS_s011
                                 + 1.11821 w_k1_FDX_s011
                                 + 1.35137 w_k1_IBM_s011
                                 + 1.02398 w_k1_INTC_s011
                                 + 1.22353 w_k1_KO_s011 + 1.04513 w_k1_MAT_s011
                                 + 1.00461 w_k1_MSFT_s011
                                 + 1.02062 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.882088 w_k1_ADM_s011
                                 + 1.0611 w_k1_AXP_s011 + 1.28942 w_k1_DIS_s011
                                 + 1.20935 w_k1_FDX_s011
                                 + 0.825391 w_k1_IBM_s011
                                 + 1.13318 w_k1_INTC_s011 + 1.025 w_k1_KO_s011
                                 + 0.743533 w_k1_MAT_s011
                                 + 1.15204 w_k1_MSFT_s011
                                 + 0.999191 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.39389 w_k1_ADM_s011
                                 + 1.34421 w_k1_AXP_s011
                                 + 1.50636 w_k1_DIS_s011
                                 + 1.45798 w_k1_FDX_s011
                                 + 1.09693 w_k1_IBM_s011
                                 + 1.15668 w_k1_INTC_s011
                                 + 1.36024 w_k1_KO_s011 + 1.41049 w_k1_MAT_s011
                                 + 1.23996 w_k1_MSFT_s011
                                 + 1.24531 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.896609 w_k1_ADM_s011
                                 + 0.753755 w_k1_AXP_s011
                                 + 0.763832 w_k1_DIS_s011
                                 + 0.72832 w_k1_FDX_s011
                                 + 0.861976 w_k1_IBM_s011
                                 + 0.849319 w_k1_INTC_s011
                                 + 0.854713 w_k1_KO_s011
                                 + 0.833231 w_k1_MAT_s011
                                 + 0.894395 w_k1_MSFT_s011
                                 + 0.777096 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.790533 w_k1_ADM_s011
                                 + 0.882511 w_k1_AXP_s011
                                 + 0.693366 w_k1_DIS_s011
                                 + 0.782873 w_k1_FDX_s011
                                 + 1.12554 w_k1_IBM_s011
                                 + 0.734693 w_k1_INTC_s011
                                 + 0.858553 w_k1_KO_s011
                                 + 0.929674 w_k1_MAT_s011
                                 + 0.701389 w_k1_MSFT_s011
                                 + 0.896998 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.41067 w_k1_ADM_s012
                                 + 1.12213 w_k1_AXP_s012
                                 + 1.22985 w_k1_DIS_s012
                                 + 1.21664 w_k1_FDX_s012
                                 + 0.873911 w_k1_IBM_s012
                                 + 1.17351 w_k1_INTC_s012
                                 + 1.02685 w_k1_KO_s012
                                 + 0.738075 w_k1_MAT_s012
                                 + 1.14297 w_k1_MSFT_s012
                                 + 1.11192 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.205 w_k1_ADM_s012
                                 + 1.35798 w_k1_AXP_s012
                                 + 1.90416 w_k1_DIS_s012 + 1.3852 w_k1_FDX_s012
                                 + 1.29768 w_k1_IBM_s012
                                 + 1.22508 w_k1_INTC_s012
                                 + 1.36559 w_k1_KO_s012 + 1.14631 w_k1_MAT_s012
                                 + 1.41445 w_k1_MSFT_s012
                                 + 1.23117 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.5101 w_k1_ADM_s012
                                 + 1.66157 w_k1_AXP_s012 + 1.3143 w_k1_DIS_s012
                                 + 1.32324 w_k1_FDX_s012
                                 + 1.19183 w_k1_IBM_s012
                                 + 1.14731 w_k1_INTC_s012
                                 + 1.17844 w_k1_KO_s012 + 1.347 w_k1_MAT_s012
                                 + 1.26346 w_k1_MSFT_s012
                                 + 1.26142 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.1524 w_k1_ADM_s012
                                 + 0.934883 w_k1_AXP_s012
                                 + 1.09777 w_k1_DIS_s012
                                 + 1.06106 w_k1_FDX_s012
                                 + 0.766334 w_k1_IBM_s012
                                 + 1.18067 w_k1_INTC_s012
                                 + 0.954784 w_k1_KO_s012
                                 + 0.596065 w_k1_MAT_s012
                                 + 1.13804 w_k1_MSFT_s012
                                 + 1.17155 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.15789 w_k1_ADM_s012
                                 + 0.857849 w_k1_AXP_s012
                                 + 0.716725 w_k1_DIS_s012
                                 + 0.921698 w_k1_FDX_s012
                                 + 0.789437 w_k1_IBM_s012
                                 + 0.916406 w_k1_INTC_s012
                                 + 0.807322 w_k1_KO_s012
                                 + 0.598791 w_k1_MAT_s012
                                 + 0.855979 w_k1_MSFT_s012
                                 + 0.912463 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.990096 w_k1_ADM_s012
                                 + 0.825561 w_k1_AXP_s012
                                 + 1.11094 w_k1_DIS_s012
                                 + 0.90359 w_k1_FDX_s012
                                 + 1.07967 w_k1_IBM_s012
                                 + 0.910136 w_k1_INTC_s012
                                 + 1.15203 w_k1_KO_s012
                                 + 0.786877 w_k1_MAT_s012
                                 + 0.903038 w_k1_MSFT_s012
                                 + 0.964202 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.23832 w_k1_ADM_s012
                                 + 1.24345 w_k1_AXP_s012
                                 + 0.936639 w_k1_DIS_s012
                                 + 0.996823 w_k1_FDX_s012
                                 + 1.20631 w_k1_IBM_s012
                                 + 0.969501 w_k1_INTC_s012
                                 + 1.03946 w_k1_KO_s012 + 1.20385 w_k1_MAT_s012
                                 + 0.919406 w_k1_MSFT_s012
                                 + 1.02146 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.05264 w_k1_ADM_s012
                                 + 1.07168 w_k1_AXP_s012
                                 + 1.04461 w_k1_DIS_s012
                                 + 0.885498 w_k1_FDX_s012
                                 + 1.52795 w_k1_IBM_s012
                                 + 1.08926 w_k1_INTC_s012
                                 + 1.25152 w_k1_KO_s012 + 1.15707 w_k1_MAT_s012
                                 + 0.989268 w_k1_MSFT_s012
                                 + 1.00937 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.95569 w_k1_ADM_s013
                                 + 1.36963 w_k1_AXP_s013
                                 + 1.15768 w_k1_DIS_s013
                                 + 1.13419 w_k1_FDX_s013
                                 + 1.66421 w_k1_IBM_s013
                                 + 1.49569 w_k1_INTC_s013
                                 + 1.38567 w_k1_KO_s013 + 1.67815 w_k1_MAT_s013
                                 + 1.13214 w_k1_MSFT_s013
                                 + 1.07705 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.693816 w_k1_ADM_s013
                                 + 0.663302 w_k1_AXP_s013
                                 + 0.645437 w_k1_DIS_s013
                                 + 0.80855 w_k1_FDX_s013
                                 + 0.778067 w_k1_IBM_s013
                                 + 0.96699 w_k1_INTC_s013
                                 + 0.745709 w_k1_KO_s013
                                 + 0.417486 w_k1_MAT_s013
                                 + 0.79478 w_k1_MSFT_s013
                                 + 0.725672 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.14379 w_k1_ADM_s013
                                 + 1.09363 w_k1_AXP_s013
                                 + 1.37792 w_k1_DIS_s013
                                 + 1.32278 w_k1_FDX_s013
                                 + 0.837594 w_k1_IBM_s013
                                 + 1.38381 w_k1_INTC_s013
                                 + 0.981626 w_k1_KO_s013
                                 + 0.648074 w_k1_MAT_s013
                                 + 1.12497 w_k1_MSFT_s013
                                 + 1.04431 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.766533 w_k1_ADM_s013
                                 + 1.07479 w_k1_AXP_s013
                                 + 1.33127 w_k1_DIS_s013 + 1.2205 w_k1_FDX_s013
                                 + 1.23764 w_k1_IBM_s013
                                 + 1.23489 w_k1_INTC_s013
                                 + 1.22126 w_k1_KO_s013 + 1.21964 w_k1_MAT_s013
                                 + 1.24256 w_k1_MSFT_s013
                                 + 1.14862 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.1058 w_k1_ADM_s013
                                 + 0.910942 w_k1_AXP_s013
                                 + 1.06688 w_k1_DIS_s013
                                 + 0.934465 w_k1_FDX_s013
                                 + 1.45619 w_k1_IBM_s013
                                 + 1.20132 w_k1_INTC_s013
                                 + 1.12885 w_k1_KO_s013
                                 + 0.717195 w_k1_MAT_s013
                                 + 1.07413 w_k1_MSFT_s013
                                 + 0.958194 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.931364 w_k1_ADM_s013
                                 + 0.865759 w_k1_AXP_s013
                                 + 0.833437 w_k1_DIS_s013
                                 + 0.849726 w_k1_FDX_s013
                                 + 1.07268 w_k1_IBM_s013
                                 + 0.938783 w_k1_INTC_s013
                                 + 0.936393 w_k1_KO_s013
                                 + 0.849664 w_k1_MAT_s013
                                 + 0.901905 w_k1_MSFT_s013
                                 + 0.943718 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.02222 w_k1_ADM_s013
                                 + 1.47281 w_k1_AXP_s013
                                 + 1.00794 w_k1_DIS_s013
                                 + 1.35255 w_k1_FDX_s013
                                 + 0.772989 w_k1_IBM_s013
                                 + 1.60184 w_k1_INTC_s013
                                 + 0.798117 w_k1_KO_s013
                                 + 0.702324 w_k1_MAT_s013
                                 + 1.05908 w_k1_MSFT_s013
                                 + 0.983846 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.870237 w_k1_ADM_s013
                                 + 0.986435 w_k1_AXP_s013
                                 + 1.01494 w_k1_DIS_s013
                                 + 1.01173 w_k1_FDX_s013
                                 + 0.951331 w_k1_IBM_s013
                                 + 0.900789 w_k1_INTC_s013
                                 + 0.897896 w_k1_KO_s013
                                 + 0.900486 w_k1_MAT_s013
                                 + 0.902017 w_k1_MSFT_s013
                                 + 1.04486 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.990294 w_k1_ADM_s014
                                 + 1.31082 w_k1_AXP_s014 + 1.2946 w_k1_DIS_s014
                                 + 1.0418 w_k1_FDX_s014 + 1.75809 w_k1_IBM_s014
                                 + 1.08607 w_k1_INTC_s014
                                 + 1.48518 w_k1_KO_s014 + 2.17032 w_k1_MAT_s014
                                 + 1.06473 w_k1_MSFT_s014
                                 + 1.07221 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.990185 w_k1_ADM_s014
                                 + 0.879421 w_k1_AXP_s014
                                 + 0.703214 w_k1_DIS_s014
                                 + 0.732466 w_k1_FDX_s014
                                 + 1.10685 w_k1_IBM_s014
                                 + 0.787604 w_k1_INTC_s014
                                 + 0.974323 w_k1_KO_s014
                                 + 0.890783 w_k1_MAT_s014
                                 + 0.78434 w_k1_MSFT_s014
                                 + 0.90459 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.857883 w_k1_ADM_s014
                                 + 1.119 w_k1_AXP_s014 + 1.2353 w_k1_DIS_s014
                                 + 1.00883 w_k1_FDX_s014 + 1.1602 w_k1_IBM_s014
                                 + 0.982898 w_k1_INTC_s014
                                 + 1.00313 w_k1_KO_s014
                                 + 0.951705 w_k1_MAT_s014
                                 + 0.936136 w_k1_MSFT_s014
                                 + 1.13634 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.795077 w_k1_ADM_s014
                                 + 0.853496 w_k1_AXP_s014
                                 + 0.912905 w_k1_DIS_s014
                                 + 0.892027 w_k1_FDX_s014
                                 + 1.09271 w_k1_IBM_s014
                                 + 0.868367 w_k1_INTC_s014
                                 + 1.04821 w_k1_KO_s014 + 1.28783 w_k1_MAT_s014
                                 + 0.978256 w_k1_MSFT_s014
                                 + 0.973494 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.29109 w_k1_ADM_s014
                                 + 1.41988 w_k1_AXP_s014
                                 + 1.74339 w_k1_DIS_s014
                                 + 1.27153 w_k1_FDX_s014
                                 + 1.39849 w_k1_IBM_s014
                                 + 1.01288 w_k1_INTC_s014
                                 + 1.44373 w_k1_KO_s014 + 1.81773 w_k1_MAT_s014
                                 + 1.37906 w_k1_MSFT_s014
                                 + 1.30332 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.693757 w_k1_ADM_s014
                                 + 0.702282 w_k1_AXP_s014
                                 + 0.66161 w_k1_DIS_s014
                                 + 0.701175 w_k1_FDX_s014
                                 + 0.696747 w_k1_IBM_s014
                                 + 0.612978 w_k1_INTC_s014
                                 + 0.75177 w_k1_KO_s014
                                 + 0.684331 w_k1_MAT_s014
                                 + 0.642573 w_k1_MSFT_s014
                                 + 0.865101 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.871579 w_k1_ADM_s014
                                 + 1.07467 w_k1_AXP_s014
                                 + 1.00437 w_k1_DIS_s014
                                 + 0.977269 w_k1_FDX_s014
                                 + 0.711897 w_k1_IBM_s014
                                 + 1.13453 w_k1_INTC_s014
                                 + 0.966115 w_k1_KO_s014
                                 + 0.699819 w_k1_MAT_s014
                                 + 0.910128 w_k1_MSFT_s014
                                 + 0.943042 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.10186 w_k1_ADM_s014
                                 + 1.62327 w_k1_AXP_s014
                                 + 1.47186 w_k1_DIS_s014
                                 + 1.42608 w_k1_FDX_s014
                                 + 0.929573 w_k1_IBM_s014
                                 + 1.12015 w_k1_INTC_s014
                                 + 1.14396 w_k1_KO_s014 + 1.3254 w_k1_MAT_s014
                                 + 1.10191 w_k1_MSFT_s014 + 1.285 w_k1_PG_s014
                                 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.26758 w_k1_ADM_s015
                                 + 1.40996 w_k1_AXP_s015
                                 + 1.69127 w_k1_DIS_s015
                                 + 1.38205 w_k1_FDX_s015
                                 + 0.992878 w_k1_IBM_s015
                                 + 1.21235 w_k1_INTC_s015
                                 + 1.37455 w_k1_KO_s015 + 1.02652 w_k1_MAT_s015
                                 + 1.32254 w_k1_MSFT_s015
                                 + 1.23095 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.848135 w_k1_ADM_s015
                                 + 1.08753 w_k1_AXP_s015
                                 + 1.16106 w_k1_DIS_s015
                                 + 1.06694 w_k1_FDX_s015
                                 + 0.741699 w_k1_IBM_s015
                                 + 0.918134 w_k1_INTC_s015
                                 + 0.851069 w_k1_KO_s015
                                 + 0.871122 w_k1_MAT_s015
                                 + 1.05569 w_k1_MSFT_s015
                                 + 0.950744 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.09993 w_k1_ADM_s015
                                 + 1.55104 w_k1_AXP_s015
                                 + 1.66482 w_k1_DIS_s015
                                 + 1.19042 w_k1_FDX_s015
                                 + 1.74691 w_k1_IBM_s015
                                 + 1.03357 w_k1_INTC_s015
                                 + 1.90866 w_k1_KO_s015 + 1.95815 w_k1_MAT_s015
                                 + 1.341 w_k1_MSFT_s015 + 1.34027 w_k1_PG_s015
                                 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.744172 w_k1_ADM_s015
                                 + 0.931639 w_k1_AXP_s015
                                 + 1.08937 w_k1_DIS_s015
                                 + 0.914187 w_k1_FDX_s015
                                 + 1.06923 w_k1_IBM_s015
                                 + 0.941083 w_k1_INTC_s015
                                 + 1.23326 w_k1_KO_s015
                                 + 0.917648 w_k1_MAT_s015
                                 + 0.954656 w_k1_MSFT_s015
                                 + 0.981831 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.03238 w_k1_ADM_s015
                                 + 0.778788 w_k1_AXP_s015
                                 + 0.955389 w_k1_DIS_s015
                                 + 0.972092 w_k1_FDX_s015
                                 + 0.836894 w_k1_IBM_s015
                                 + 0.935724 w_k1_INTC_s015
                                 + 1.04102 w_k1_KO_s015
                                 + 0.742426 w_k1_MAT_s015
                                 + 1.07918 w_k1_MSFT_s015
                                 + 0.988446 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.684916 w_k1_ADM_s015
                                 + 0.686621 w_k1_AXP_s015
                                 + 0.799205 w_k1_DIS_s015
                                 + 0.757355 w_k1_FDX_s015
                                 + 0.961659 w_k1_IBM_s015
                                 + 0.80181 w_k1_INTC_s015
                                 + 1.02736 w_k1_KO_s015
                                 + 0.889996 w_k1_MAT_s015
                                 + 0.803709 w_k1_MSFT_s015
                                 + 0.873258 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.732355 w_k1_ADM_s015
                                 + 0.695179 w_k1_AXP_s015
                                 + 0.591949 w_k1_DIS_s015
                                 + 0.684261 w_k1_FDX_s015
                                 + 0.689304 w_k1_IBM_s015
                                 + 0.709424 w_k1_INTC_s015
                                 + 0.69681 w_k1_KO_s015
                                 + 0.594821 w_k1_MAT_s015
                                 + 0.757861 w_k1_MSFT_s015
                                 + 0.788438 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.738656 w_k1_ADM_s015
                                 + 0.898814 w_k1_AXP_s015
                                 + 1.13124 w_k1_DIS_s015
                                 + 0.909392 w_k1_FDX_s015
                                 + 1.44913 w_k1_IBM_s015
                                 + 0.974934 w_k1_INTC_s015
                                 + 1.07936 w_k1_KO_s015 + 1.01727 w_k1_MAT_s015
                                 + 0.995454 w_k1_MSFT_s015
                                 + 1.00566 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.779779 w_k1_ADM_s016
                                 + 0.769551 w_k1_AXP_s016
                                 + 0.722554 w_k1_DIS_s016
                                 + 0.697991 w_k1_FDX_s016
                                 + 1.20566 w_k1_IBM_s016
                                 + 0.910231 w_k1_INTC_s016
                                 + 1.12206 w_k1_KO_s016 + 1.0302 w_k1_MAT_s016
                                 + 0.860446 w_k1_MSFT_s016
                                 + 0.930879 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.246 w_k1_ADM_s016
                                 + 1.559 w_k1_AXP_s016 + 1.48402 w_k1_DIS_s016
                                 + 1.3231 w_k1_FDX_s016 + 1.22211 w_k1_IBM_s016
                                 + 1.46126 w_k1_INTC_s016
                                 + 1.39205 w_k1_KO_s016 + 1.43858 w_k1_MAT_s016
                                 + 1.38271 w_k1_MSFT_s016
                                 + 1.42769 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.00837 w_k1_ADM_s016
                                 + 0.881709 w_k1_AXP_s016
                                 + 0.814279 w_k1_DIS_s016
                                 + 0.868096 w_k1_FDX_s016
                                 + 1.04119 w_k1_IBM_s016
                                 + 1.13786 w_k1_INTC_s016
                                 + 1.05766 w_k1_KO_s016 + 0.85452 w_k1_MAT_s016
                                 + 0.956682 w_k1_MSFT_s016
                                 + 1.01985 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.04435 w_k1_ADM_s016
                                 + 1.21214 w_k1_AXP_s016
                                 + 0.986831 w_k1_DIS_s016
                                 + 1.04292 w_k1_FDX_s016
                                 + 1.61231 w_k1_IBM_s016
                                 + 1.12832 w_k1_INTC_s016
                                 + 1.16637 w_k1_KO_s016 + 1.52526 w_k1_MAT_s016
                                 + 0.92596 w_k1_MSFT_s016 + 1.1955 w_k1_PG_s016
                                 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.756461 w_k1_ADM_s016
                                 + 0.567897 w_k1_AXP_s016
                                 + 0.632081 w_k1_DIS_s016
                                 + 0.706901 w_k1_FDX_s016
                                 + 0.724389 w_k1_IBM_s016
                                 + 1.10133 w_k1_INTC_s016
                                 + 0.828096 w_k1_KO_s016 + 0.5735 w_k1_MAT_s016
                                 + 0.870715 w_k1_MSFT_s016
                                 + 0.839914 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.13124 w_k1_ADM_s016
                                 + 1.17153 w_k1_AXP_s016
                                 + 1.23765 w_k1_DIS_s016
                                 + 1.28997 w_k1_FDX_s016
                                 + 0.94965 w_k1_IBM_s016
                                 + 1.36428 w_k1_INTC_s016
                                 + 1.07499 w_k1_KO_s016 + 1.04239 w_k1_MAT_s016
                                 + 1.22962 w_k1_MSFT_s016
                                 + 1.21174 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.16569 w_k1_ADM_s016
                                 + 1.03316 w_k1_AXP_s016
                                 + 1.00214 w_k1_DIS_s016
                                 + 1.03419 w_k1_FDX_s016
                                 + 0.832372 w_k1_IBM_s016
                                 + 1.08009 w_k1_INTC_s016
                                 + 0.926326 w_k1_KO_s016
                                 + 0.687703 w_k1_MAT_s016
                                 + 1.06953 w_k1_MSFT_s016
                                 + 1.02719 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.2599 w_k1_ADM_s016
                                 + 1.29097 w_k1_AXP_s016
                                 + 1.44354 w_k1_DIS_s016
                                 + 1.27097 w_k1_FDX_s016
                                 + 1.37294 w_k1_IBM_s016
                                 + 1.1802 w_k1_INTC_s016 + 1.24523 w_k1_KO_s016
                                 + 1.0708 w_k1_MAT_s016
                                 + 1.28001 w_k1_MSFT_s016
                                 + 1.20589 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.29221 w_k1_ADM_s017
                                 + 1.35181 w_k1_AXP_s017
                                 + 0.926261 w_k1_DIS_s017
                                 + 1.20025 w_k1_FDX_s017
                                 + 0.718048 w_k1_IBM_s017
                                 + 0.953744 w_k1_INTC_s017
                                 + 0.818578 w_k1_KO_s017
                                 + 0.81815 w_k1_MAT_s017
                                 + 0.962229 w_k1_MSFT_s017
                                 + 1.10651 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.54108 w_k1_ADM_s017
                                 + 1.25678 w_k1_AXP_s017
                                 + 1.72068 w_k1_DIS_s017
                                 + 1.28473 w_k1_FDX_s017
                                 + 1.62169 w_k1_IBM_s017
                                 + 0.833404 w_k1_INTC_s017
                                 + 1.37842 w_k1_KO_s017 + 1.41738 w_k1_MAT_s017
                                 + 1.10834 w_k1_MSFT_s017
                                 + 1.35462 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.842023 w_k1_ADM_s017
                                 + 1.19834 w_k1_AXP_s017
                                 + 1.02956 w_k1_DIS_s017
                                 + 0.972961 w_k1_FDX_s017
                                 + 0.974508 w_k1_IBM_s017
                                 + 0.850188 w_k1_INTC_s017
                                 + 0.884087 w_k1_KO_s017
                                 + 1.04587 w_k1_MAT_s017
                                 + 0.848568 w_k1_MSFT_s017
                                 + 1.02853 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.870714 w_k1_ADM_s017
                                 + 0.851657 w_k1_AXP_s017
                                 + 0.703096 w_k1_DIS_s017
                                 + 0.752023 w_k1_FDX_s017
                                 + 1.11228 w_k1_IBM_s017
                                 + 0.701987 w_k1_INTC_s017
                                 + 0.807965 w_k1_KO_s017
                                 + 1.00757 w_k1_MAT_s017
                                 + 0.667442 w_k1_MSFT_s017
                                 + 0.802959 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.33411 w_k1_ADM_s017
                                 + 1.36729 w_k1_AXP_s017
                                 + 1.43919 w_k1_DIS_s017
                                 + 1.44389 w_k1_FDX_s017
                                 + 1.01382 w_k1_IBM_s017
                                 + 0.906368 w_k1_INTC_s017
                                 + 1.05066 w_k1_KO_s017 + 1.18283 w_k1_MAT_s017
                                 + 1.26847 w_k1_MSFT_s017
                                 + 1.12506 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.27476 w_k1_ADM_s017
                                 + 0.80737 w_k1_AXP_s017
                                 + 1.16381 w_k1_DIS_s017
                                 + 1.09365 w_k1_FDX_s017
                                 + 0.891627 w_k1_IBM_s017
                                 + 0.919685 w_k1_INTC_s017
                                 + 0.985021 w_k1_KO_s017
                                 + 0.729324 w_k1_MAT_s017
                                 + 1.05481 w_k1_MSFT_s017
                                 + 1.02447 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.02741 w_k1_ADM_s017
                                 + 0.891058 w_k1_AXP_s017
                                 + 0.855644 w_k1_DIS_s017
                                 + 0.89768 w_k1_FDX_s017
                                 + 0.724116 w_k1_IBM_s017
                                 + 0.719477 w_k1_INTC_s017
                                 + 0.773674 w_k1_KO_s017
                                 + 0.786276 w_k1_MAT_s017
                                 + 0.871207 w_k1_MSFT_s017
                                 + 0.935136 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.891188 w_k1_ADM_s017
                                 + 1.35267 w_k1_AXP_s017
                                 + 1.08092 w_k1_DIS_s017
                                 + 1.04745 w_k1_FDX_s017
                                 + 1.31641 w_k1_IBM_s017
                                 + 0.726221 w_k1_INTC_s017
                                 + 0.987043 w_k1_KO_s017
                                 + 1.74846 w_k1_MAT_s017
                                 + 0.889541 w_k1_MSFT_s017
                                 + 0.966907 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.99182 w_k1_ADM_s100
                                 + 1.33037 w_k1_AXP_s100
                                 + 0.80921 w_k1_DIS_s100
                                 + 1.04008 w_k1_FDX_s100
                                 + 1.06484 w_k1_IBM_s100
                                 + 0.946951 w_k1_INTC_s100
                                 + 0.915023 w_k1_KO_s100
                                 + 1.29889 w_k1_MAT_s100
                                 + 0.849937 w_k1_MSFT_s100
                                 + 0.952466 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.28697 w_k1_ADM_s100
                                 + 1.49423 w_k1_AXP_s100 + 1.5302 w_k1_DIS_s100
                                 + 1.38013 w_k1_FDX_s100
                                 + 1.11666 w_k1_IBM_s100
                                 + 1.12922 w_k1_INTC_s100
                                 + 1.44627 w_k1_KO_s100 + 1.9963 w_k1_MAT_s100
                                 + 1.24414 w_k1_MSFT_s100
                                 + 1.17074 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.89169 w_k1_ADM_s100
                                 + 1.01694 w_k1_AXP_s100
                                 + 0.907193 w_k1_DIS_s100
                                 + 0.954336 w_k1_FDX_s100
                                 + 0.901896 w_k1_IBM_s100
                                 + 0.84942 w_k1_INTC_s100
                                 + 0.974234 w_k1_KO_s100
                                 + 1.02738 w_k1_MAT_s100
                                 + 0.948204 w_k1_MSFT_s100
                                 + 0.92487 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.638782 w_k1_ADM_s100
                                 + 0.710722 w_k1_AXP_s100
                                 + 0.554324 w_k1_DIS_s100
                                 + 0.674857 w_k1_FDX_s100
                                 + 0.810538 w_k1_IBM_s100
                                 + 0.763561 w_k1_INTC_s100
                                 + 0.775128 w_k1_KO_s100
                                 + 0.922875 w_k1_MAT_s100
                                 + 0.72989 w_k1_MSFT_s100
                                 + 0.734349 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.891221 w_k1_ADM_s100
                                 + 0.795666 w_k1_AXP_s100
                                 + 0.83958 w_k1_DIS_s100
                                 + 0.800522 w_k1_FDX_s100
                                 + 1.19663 w_k1_IBM_s100
                                 + 0.895275 w_k1_INTC_s100
                                 + 0.970044 w_k1_KO_s100
                                 + 1.27078 w_k1_MAT_s100
                                 + 0.89067 w_k1_MSFT_s100
                                 + 0.863635 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.805745 w_k1_ADM_s100
                                 + 1.2815 w_k1_AXP_s100
                                 + 0.859915 w_k1_DIS_s100
                                 + 0.873839 w_k1_FDX_s100
                                 + 1.46824 w_k1_IBM_s100
                                 + 0.918113 w_k1_INTC_s100
                                 + 1.17724 w_k1_KO_s100 + 1.74091 w_k1_MAT_s100
                                 + 0.881415 w_k1_MSFT_s100
                                 + 0.968745 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.04432 w_k1_ADM_s100
                                 + 1.07169 w_k1_AXP_s100
                                 + 0.80718 w_k1_DIS_s100
                                 + 1.02191 w_k1_FDX_s100
                                 + 0.614797 w_k1_IBM_s100
                                 + 0.885122 w_k1_INTC_s100
                                 + 0.731052 w_k1_KO_s100
                                 + 0.714022 w_k1_MAT_s100
                                 + 1.00177 w_k1_MSFT_s100
                                 + 0.792859 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.01246 w_k1_ADM_s100
                                 + 1.18408 w_k1_AXP_s100
                                 + 1.33754 w_k1_DIS_s100 + 1.0335 w_k1_FDX_s100
                                 + 1.27235 w_k1_IBM_s100
                                 + 1.2574 w_k1_INTC_s100 + 1.35445 w_k1_KO_s100
                                 + 1.50926 w_k1_MAT_s100
                                 + 1.22472 w_k1_MSFT_s100
                                 + 1.13184 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.11597 w_k1_ADM_s101
                                 + 1.24546 w_k1_AXP_s101
                                 + 1.22171 w_k1_DIS_s101
                                 + 1.18059 w_k1_FDX_s101
                                 + 0.943244 w_k1_IBM_s101
                                 + 1.17212 w_k1_INTC_s101
                                 + 1.02812 w_k1_KO_s101
                                 + 0.772609 w_k1_MAT_s101
                                 + 1.11114 w_k1_MSFT_s101
                                 + 0.961367 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.985483 w_k1_ADM_s101
                                 + 1.03231 w_k1_AXP_s101
                                 + 0.807517 w_k1_DIS_s101
                                 + 0.941464 w_k1_FDX_s101
                                 + 0.805232 w_k1_IBM_s101
                                 + 1.00768 w_k1_INTC_s101
                                 + 0.832994 w_k1_KO_s101 + 0.5173 w_k1_MAT_s101
                                 + 0.913924 w_k1_MSFT_s101
                                 + 0.890714 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.18144 w_k1_ADM_s101
                                 + 1.14579 w_k1_AXP_s101
                                 + 1.04426 w_k1_DIS_s101
                                 + 1.02293 w_k1_FDX_s101
                                 + 1.25386 w_k1_IBM_s101
                                 + 0.891388 w_k1_INTC_s101
                                 + 1.09558 w_k1_KO_s101
                                 + 0.883855 w_k1_MAT_s101
                                 + 1.02158 w_k1_MSFT_s101
                                 + 1.11273 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.950144 w_k1_ADM_s101
                                 + 0.969121 w_k1_AXP_s101
                                 + 0.893546 w_k1_DIS_s101
                                 + 0.828784 w_k1_FDX_s101
                                 + 1.4467 w_k1_IBM_s101
                                 + 1.03754 w_k1_INTC_s101
                                 + 1.03214 w_k1_KO_s101
                                 + 0.961944 w_k1_MAT_s101
                                 + 0.916553 w_k1_MSFT_s101
                                 + 0.963678 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.891262 w_k1_ADM_s101
                                 + 1.47891 w_k1_AXP_s101
                                 + 1.60828 w_k1_DIS_s101
                                 + 1.22337 w_k1_FDX_s101
                                 + 1.43529 w_k1_IBM_s101
                                 + 1.21162 w_k1_INTC_s101
                                 + 1.32232 w_k1_KO_s101 + 1.40636 w_k1_MAT_s101
                                 + 1.15011 w_k1_MSFT_s101
                                 + 1.02886 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.721777 w_k1_ADM_s101
                                 + 0.811805 w_k1_AXP_s101
                                 + 0.744795 w_k1_DIS_s101
                                 + 0.790221 w_k1_FDX_s101
                                 + 0.863926 w_k1_IBM_s101
                                 + 0.885428 w_k1_INTC_s101
                                 + 0.868635 w_k1_KO_s101
                                 + 0.589109 w_k1_MAT_s101
                                 + 0.874694 w_k1_MSFT_s101
                                 + 0.818088 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.732984 w_k1_ADM_s101
                                 + 0.709205 w_k1_AXP_s101
                                 + 0.663109 w_k1_DIS_s101
                                 + 0.797304 w_k1_FDX_s101
                                 + 0.557433 w_k1_IBM_s101
                                 + 0.848158 w_k1_INTC_s101
                                 + 0.655224 w_k1_KO_s101
                                 + 0.581651 w_k1_MAT_s101
                                 + 0.802862 w_k1_MSFT_s101
                                 + 0.832877 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.12921 w_k1_ADM_s101
                                 + 0.736169 w_k1_AXP_s101
                                 + 0.825894 w_k1_DIS_s101
                                 + 0.88999 w_k1_FDX_s101
                                 + 0.83882 w_k1_IBM_s101
                                 + 0.837413 w_k1_INTC_s101
                                 + 0.908396 w_k1_KO_s101
                                 + 0.521399 w_k1_MAT_s101
                                 + 0.937034 w_k1_MSFT_s101
                                 + 0.918479 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.722417 w_k1_ADM_s102
                                 + 0.724225 w_k1_AXP_s102
                                 + 0.764545 w_k1_DIS_s102
                                 + 0.867543 w_k1_FDX_s102
                                 + 0.73731 w_k1_IBM_s102
                                 + 1.10733 w_k1_INTC_s102
                                 + 0.712551 w_k1_KO_s102
                                 + 0.610411 w_k1_MAT_s102
                                 + 0.885554 w_k1_MSFT_s102
                                 + 0.84834 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.08751 w_k1_ADM_s102
                                 + 1.12439 w_k1_AXP_s102
                                 + 1.11479 w_k1_DIS_s102
                                 + 1.12434 w_k1_FDX_s102 + 1.0164 w_k1_IBM_s102
                                 + 1.18631 w_k1_INTC_s102
                                 + 0.863936 w_k1_KO_s102
                                 + 0.859937 w_k1_MAT_s102
                                 + 1.15118 w_k1_MSFT_s102
                                 + 0.997017 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.831944 w_k1_ADM_s102
                                 + 1.11352 w_k1_AXP_s102
                                 + 0.735602 w_k1_DIS_s102
                                 + 0.926793 w_k1_FDX_s102
                                 + 0.848019 w_k1_IBM_s102
                                 + 0.893874 w_k1_INTC_s102
                                 + 0.803711 w_k1_KO_s102
                                 + 0.800762 w_k1_MAT_s102
                                 + 0.951433 w_k1_MSFT_s102
                                 + 0.901704 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.67888 w_k1_ADM_s102
                                 + 0.761047 w_k1_AXP_s102
                                 + 0.631997 w_k1_DIS_s102
                                 + 0.715038 w_k1_FDX_s102
                                 + 0.970315 w_k1_IBM_s102
                                 + 0.70267 w_k1_INTC_s102
                                 + 0.750852 w_k1_KO_s102
                                 + 0.888642 w_k1_MAT_s102
                                 + 0.704202 w_k1_MSFT_s102
                                 + 0.725943 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.04352 w_k1_ADM_s102
                                 + 1.46398 w_k1_AXP_s102
                                 + 1.57777 w_k1_DIS_s102
                                 + 1.24322 w_k1_FDX_s102
                                 + 1.52647 w_k1_IBM_s102
                                 + 1.33188 w_k1_INTC_s102
                                 + 1.26821 w_k1_KO_s102 + 1.26789 w_k1_MAT_s102
                                 + 1.26361 w_k1_MSFT_s102
                                 + 1.20655 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.717844 w_k1_ADM_s102
                                 + 0.737252 w_k1_AXP_s102
                                 + 0.630234 w_k1_DIS_s102
                                 + 0.752705 w_k1_FDX_s102
                                 + 0.558911 w_k1_IBM_s102
                                 + 0.852783 w_k1_INTC_s102
                                 + 0.576302 w_k1_KO_s102
                                 + 0.467671 w_k1_MAT_s102
                                 + 0.745323 w_k1_MSFT_s102
                                 + 0.832877 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.66772 w_k1_ADM_s102
                                 + 0.87442 w_k1_AXP_s102
                                 + 0.857594 w_k1_DIS_s102
                                 + 0.790626 w_k1_FDX_s102
                                 + 1.38155 w_k1_IBM_s102
                                 + 0.767137 w_k1_INTC_s102
                                 + 0.923649 w_k1_KO_s102
                                 + 1.22723 w_k1_MAT_s102
                                 + 0.734796 w_k1_MSFT_s102
                                 + 0.936251 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.985396 w_k1_ADM_s102
                                 + 1.12901 w_k1_AXP_s102
                                 + 1.45999 w_k1_DIS_s102
                                 + 1.21345 w_k1_FDX_s102
                                 + 0.84531 w_k1_IBM_s102
                                 + 1.04565 w_k1_INTC_s102
                                 + 0.97215 w_k1_KO_s102
                                 + 0.905937 w_k1_MAT_s102
                                 + 1.28711 w_k1_MSFT_s102
                                 + 1.04756 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.764722 w_k1_ADM_s103
                                 + 0.911425 w_k1_AXP_s103
                                 + 0.788665 w_k1_DIS_s103
                                 + 0.767431 w_k1_FDX_s103
                                 + 1.14056 w_k1_IBM_s103
                                 + 1.06818 w_k1_INTC_s103
                                 + 0.834861 w_k1_KO_s103
                                 + 1.22294 w_k1_MAT_s103
                                 + 0.938436 w_k1_MSFT_s103
                                 + 0.998178 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.892816 w_k1_ADM_s103
                                 + 0.941364 w_k1_AXP_s103
                                 + 1.09132 w_k1_DIS_s103
                                 + 0.957775 w_k1_FDX_s103
                                 + 1.03706 w_k1_IBM_s103
                                 + 1.09359 w_k1_INTC_s103
                                 + 1.07764 w_k1_KO_s103 + 1.21007 w_k1_MAT_s103
                                 + 1.16478 w_k1_MSFT_s103
                                 + 1.11079 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.869153 w_k1_ADM_s103
                                 + 1.1975 w_k1_AXP_s103 + 1.1844 w_k1_DIS_s103
                                 + 1.12872 w_k1_FDX_s103
                                 + 1.21011 w_k1_IBM_s103
                                 + 1.13232 w_k1_INTC_s103
                                 + 1.18376 w_k1_KO_s103 + 1.61119 w_k1_MAT_s103
                                 + 1.09947 w_k1_MSFT_s103
                                 + 1.18982 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.4907 w_k1_ADM_s103
                                 + 1.77748 w_k1_AXP_s103
                                 + 1.78391 w_k1_DIS_s103
                                 + 1.42894 w_k1_FDX_s103
                                 + 1.29076 w_k1_IBM_s103
                                 + 1.3102 w_k1_INTC_s103 + 1.45958 w_k1_KO_s103
                                 + 1.65028 w_k1_MAT_s103
                                 + 1.45892 w_k1_MSFT_s103
                                 + 1.38218 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.890608 w_k1_ADM_s103
                                 + 0.871136 w_k1_AXP_s103
                                 + 0.885753 w_k1_DIS_s103
                                 + 0.889735 w_k1_FDX_s103
                                 + 0.917501 w_k1_IBM_s103
                                 + 1.00084 w_k1_INTC_s103
                                 + 0.971193 w_k1_KO_s103
                                 + 1.13655 w_k1_MAT_s103
                                 + 0.915194 w_k1_MSFT_s103
                                 + 1.01287 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.716886 w_k1_ADM_s103
                                 + 0.710441 w_k1_AXP_s103
                                 + 0.701786 w_k1_DIS_s103
                                 + 0.75839 w_k1_FDX_s103
                                 + 0.698322 w_k1_IBM_s103
                                 + 0.864002 w_k1_INTC_s103
                                 + 0.77068 w_k1_KO_s103
                                 + 0.736151 w_k1_MAT_s103
                                 + 0.794109 w_k1_MSFT_s103
                                 + 0.769531 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.03418 w_k1_ADM_s103
                                 + 1.14378 w_k1_AXP_s103
                                 + 1.01796 w_k1_DIS_s103
                                 + 1.19734 w_k1_FDX_s103
                                 + 0.762734 w_k1_IBM_s103
                                 + 1.03961 w_k1_INTC_s103
                                 + 0.924766 w_k1_KO_s103
                                 + 1.06182 w_k1_MAT_s103
                                 + 1.00505 w_k1_MSFT_s103
                                 + 1.07104 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.864416 w_k1_ADM_s103
                                 + 1.27202 w_k1_AXP_s103
                                 + 1.21788 w_k1_DIS_s103
                                 + 1.05967 w_k1_FDX_s103
                                 + 1.65394 w_k1_IBM_s103
                                 + 1.19975 w_k1_INTC_s103 + 1.1875 w_k1_KO_s103
                                 + 2.08772 w_k1_MAT_s103
                                 + 0.983042 w_k1_MSFT_s103
                                 + 1.20927 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.0021 w_k1_ADM_s104
                                 + 0.944328 w_k1_AXP_s104
                                 + 0.850781 w_k1_DIS_s104
                                 + 1.04251 w_k1_FDX_s104
                                 + 0.677183 w_k1_IBM_s104
                                 + 0.828664 w_k1_INTC_s104
                                 + 0.781978 w_k1_KO_s104
                                 + 0.71134 w_k1_MAT_s104
                                 + 0.845233 w_k1_MSFT_s104
                                 + 0.950217 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.01833 w_k1_ADM_s104
                                 + 1.09266 w_k1_AXP_s104
                                 + 0.908859 w_k1_DIS_s104
                                 + 0.901295 w_k1_FDX_s104
                                 + 1.42583 w_k1_IBM_s104
                                 + 0.833528 w_k1_INTC_s104
                                 + 0.979628 w_k1_KO_s104
                                 + 1.31946 w_k1_MAT_s104
                                 + 0.952962 w_k1_MSFT_s104
                                 + 0.982441 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.814963 w_k1_ADM_s104
                                 + 1.10273 w_k1_AXP_s104
                                 + 0.78796 w_k1_DIS_s104
                                 + 0.84346 w_k1_FDX_s104
                                 + 1.17237 w_k1_IBM_s104
                                 + 0.937066 w_k1_INTC_s104
                                 + 0.887828 w_k1_KO_s104
                                 + 1.29456 w_k1_MAT_s104
                                 + 0.807866 w_k1_MSFT_s104
                                 + 0.883612 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.12021 w_k1_ADM_s104
                                 + 1.31725 w_k1_AXP_s104
                                 + 1.04939 w_k1_DIS_s104
                                 + 1.14057 w_k1_FDX_s104
                                 + 0.992026 w_k1_IBM_s104
                                 + 0.979211 w_k1_INTC_s104
                                 + 0.924556 w_k1_KO_s104
                                 + 1.06127 w_k1_MAT_s104
                                 + 1.05362 w_k1_MSFT_s104
                                 + 0.985025 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.884304 w_k1_ADM_s104
                                 + 0.82276 w_k1_AXP_s104
                                 + 0.720861 w_k1_DIS_s104
                                 + 0.835771 w_k1_FDX_s104
                                 + 0.70261 w_k1_IBM_s104
                                 + 0.773873 w_k1_INTC_s104
                                 + 0.748682 w_k1_KO_s104
                                 + 0.771476 w_k1_MAT_s104
                                 + 0.787277 w_k1_MSFT_s104
                                 + 0.850057 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.17528 w_k1_ADM_s104
                                 + 1.44705 w_k1_AXP_s104
                                 + 1.59927 w_k1_DIS_s104
                                 + 1.25487 w_k1_FDX_s104
                                 + 1.31576 w_k1_IBM_s104
                                 + 1.09002 w_k1_INTC_s104
                                 + 1.30407 w_k1_KO_s104 + 1.15357 w_k1_MAT_s104
                                 + 1.18145 w_k1_MSFT_s104
                                 + 1.17858 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.889718 w_k1_ADM_s104
                                 + 1.05194 w_k1_AXP_s104
                                 + 0.980819 w_k1_DIS_s104
                                 + 0.964051 w_k1_FDX_s104
                                 + 0.978756 w_k1_IBM_s104
                                 + 0.904789 w_k1_INTC_s104
                                 + 0.869752 w_k1_KO_s104
                                 + 0.951839 w_k1_MAT_s104
                                 + 0.960027 w_k1_MSFT_s104
                                 + 0.885255 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.696079 w_k1_ADM_s104
                                 + 0.842616 w_k1_AXP_s104
                                 + 0.624897 w_k1_DIS_s104
                                 + 0.728728 w_k1_FDX_s104
                                 + 0.756148 w_k1_IBM_s104
                                 + 0.669374 w_k1_INTC_s104
                                 + 0.603411 w_k1_KO_s104
                                 + 0.661404 w_k1_MAT_s104
                                 + 0.727456 w_k1_MSFT_s104
                                 + 0.790364 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.17624 w_k1_ADM_s105
                                 + 1.3199 w_k1_AXP_s105 + 1.39368 w_k1_DIS_s105
                                 + 1.23746 w_k1_FDX_s105
                                 + 1.23446 w_k1_IBM_s105
                                 + 0.88978 w_k1_INTC_s105 + 1.2336 w_k1_KO_s105
                                 + 1.67297 w_k1_MAT_s105
                                 + 1.21554 w_k1_MSFT_s105
                                 + 1.12002 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.863623 w_k1_ADM_s105
                                 + 0.800149 w_k1_AXP_s105
                                 + 0.836275 w_k1_DIS_s105
                                 + 0.87204 w_k1_FDX_s105
                                 + 0.735218 w_k1_IBM_s105
                                 + 0.736351 w_k1_INTC_s105
                                 + 0.839891 w_k1_KO_s105
                                 + 1.04237 w_k1_MAT_s105
                                 + 0.884209 w_k1_MSFT_s105
                                 + 0.880955 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.776923 w_k1_ADM_s105
                                 + 0.897663 w_k1_AXP_s105
                                 + 0.910068 w_k1_DIS_s105
                                 + 0.834262 w_k1_FDX_s105
                                 + 1.17216 w_k1_IBM_s105
                                 + 0.810694 w_k1_INTC_s105
                                 + 0.905898 w_k1_KO_s105
                                 + 0.845464 w_k1_MAT_s105
                                 + 0.824961 w_k1_MSFT_s105
                                 + 0.864427 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.933895 w_k1_ADM_s105
                                 + 0.817896 w_k1_AXP_s105
                                 + 1.00334 w_k1_DIS_s105
                                 + 0.999744 w_k1_FDX_s105
                                 + 0.700605 w_k1_IBM_s105
                                 + 1.01111 w_k1_INTC_s105
                                 + 0.944125 w_k1_KO_s105
                                 + 0.536166 w_k1_MAT_s105
                                 + 1.05797 w_k1_MSFT_s105
                                 + 0.852621 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.922389 w_k1_ADM_s105
                                 + 1.08617 w_k1_AXP_s105
                                 + 0.837618 w_k1_DIS_s105
                                 + 0.949441 w_k1_FDX_s105
                                 + 1.02863 w_k1_IBM_s105
                                 + 0.894691 w_k1_INTC_s105
                                 + 1.06266 w_k1_KO_s105 + 1.17669 w_k1_MAT_s105
                                 + 0.88004 w_k1_MSFT_s105
                                 + 0.942865 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.795983 w_k1_ADM_s105
                                 + 0.685301 w_k1_AXP_s105
                                 + 0.759003 w_k1_DIS_s105
                                 + 0.799633 w_k1_FDX_s105
                                 + 0.800763 w_k1_IBM_s105
                                 + 0.622997 w_k1_INTC_s105
                                 + 0.826623 w_k1_KO_s105
                                 + 0.988732 w_k1_MAT_s105
                                 + 0.729884 w_k1_MSFT_s105
                                 + 0.82668 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.0411 w_k1_ADM_s105
                                 + 1.02833 w_k1_AXP_s105
                                 + 1.27501 w_k1_DIS_s105
                                 + 1.13718 w_k1_FDX_s105
                                 + 1.16281 w_k1_IBM_s105
                                 + 0.943098 w_k1_INTC_s105
                                 + 1.16303 w_k1_KO_s105 + 1.41559 w_k1_MAT_s105
                                 + 1.05102 w_k1_MSFT_s105 + 1.0115 w_k1_PG_s105
                                 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.01946 w_k1_ADM_s105
                                 + 1.19674 w_k1_AXP_s105
                                 + 0.948653 w_k1_DIS_s105
                                 + 1.07931 w_k1_FDX_s105
                                 + 0.604502 w_k1_IBM_s105
                                 + 0.993055 w_k1_INTC_s105
                                 + 0.864165 w_k1_KO_s105
                                 + 0.782843 w_k1_MAT_s105
                                 + 0.983939 w_k1_MSFT_s105
                                 + 0.980693 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.889738 w_k1_ADM_s106
                                 + 0.730751 w_k1_AXP_s106
                                 + 0.774801 w_k1_DIS_s106
                                 + 0.846567 w_k1_FDX_s106
                                 + 0.683016 w_k1_IBM_s106
                                 + 0.824792 w_k1_INTC_s106
                                 + 0.753063 w_k1_KO_s106
                                 + 0.635296 w_k1_MAT_s106
                                 + 0.838383 w_k1_MSFT_s106
                                 + 0.96147 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.19311 w_k1_ADM_s106
                                 + 1.04531 w_k1_AXP_s106
                                 + 0.831668 w_k1_DIS_s106
                                 + 0.956458 w_k1_FDX_s106
                                 + 0.938058 w_k1_IBM_s106
                                 + 0.774636 w_k1_INTC_s106
                                 + 0.928506 w_k1_KO_s106
                                 + 0.854045 w_k1_MAT_s106
                                 + 0.849101 w_k1_MSFT_s106
                                 + 1.04843 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.977422 w_k1_ADM_s106
                                 + 0.970437 w_k1_AXP_s106
                                 + 1.02546 w_k1_DIS_s106
                                 + 0.960424 w_k1_FDX_s106
                                 + 0.911989 w_k1_IBM_s106
                                 + 0.876726 w_k1_INTC_s106
                                 + 0.942361 w_k1_KO_s106
                                 + 0.825475 w_k1_MAT_s106
                                 + 0.964892 w_k1_MSFT_s106
                                 + 0.996908 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.746969 w_k1_ADM_s106
                                 + 0.744631 w_k1_AXP_s106
                                 + 1.02218 w_k1_DIS_s106
                                 + 0.807713 w_k1_FDX_s106
                                 + 1.29277 w_k1_IBM_s106
                                 + 0.967438 w_k1_INTC_s106
                                 + 1.03968 w_k1_KO_s106
                                 + 0.984478 w_k1_MAT_s106
                                 + 0.781445 w_k1_MSFT_s106
                                 + 0.940393 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.839125 w_k1_ADM_s106
                                 + 0.739942 w_k1_AXP_s106
                                 + 0.731436 w_k1_DIS_s106
                                 + 0.762419 w_k1_FDX_s106
                                 + 0.974544 w_k1_IBM_s106
                                 + 0.803486 w_k1_INTC_s106
                                 + 0.787379 w_k1_KO_s106
                                 + 0.865601 w_k1_MAT_s106
                                 + 0.808874 w_k1_MSFT_s106
                                 + 0.791396 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.47229 w_k1_ADM_s106
                                 + 1.25952 w_k1_AXP_s106
                                 + 1.32146 w_k1_DIS_s106
                                 + 1.18469 w_k1_FDX_s106
                                 + 1.40551 w_k1_IBM_s106
                                 + 1.05397 w_k1_INTC_s106
                                 + 1.17691 w_k1_KO_s106 + 1.37957 w_k1_MAT_s106
                                 + 1.10694 w_k1_MSFT_s106
                                 + 1.20365 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.00283 w_k1_ADM_s106
                                 + 0.844606 w_k1_AXP_s106
                                 + 1.00955 w_k1_DIS_s106
                                 + 0.975913 w_k1_FDX_s106
                                 + 1.20513 w_k1_IBM_s106
                                 + 1.12358 w_k1_INTC_s106
                                 + 1.16408 w_k1_KO_s106
                                 + 0.977655 w_k1_MAT_s106
                                 + 0.960858 w_k1_MSFT_s106
                                 + 1.05934 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.30633 w_k1_ADM_s106
                                 + 1.033 w_k1_AXP_s106 + 1.22496 w_k1_DIS_s106
                                 + 1.22493 w_k1_FDX_s106
                                 + 0.906034 w_k1_IBM_s106
                                 + 1.27786 w_k1_INTC_s106
                                 + 0.936116 w_k1_KO_s106
                                 + 0.652718 w_k1_MAT_s106
                                 + 1.20701 w_k1_MSFT_s106
                                 + 1.14803 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.35344 w_k1_ADM_s107
                                 + 1.39674 w_k1_AXP_s107
                                 + 1.43496 w_k1_DIS_s107
                                 + 1.14349 w_k1_FDX_s107
                                 + 1.55794 w_k1_IBM_s107
                                 + 1.35861 w_k1_INTC_s107
                                 + 1.06353 w_k1_KO_s107 + 1.20835 w_k1_MAT_s107
                                 + 1.14304 w_k1_MSFT_s107
                                 + 1.16336 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.0951 w_k1_ADM_s107
                                 + 1.10946 w_k1_AXP_s107
                                 + 1.22365 w_k1_DIS_s107
                                 + 1.18556 w_k1_FDX_s107
                                 + 0.738196 w_k1_IBM_s107
                                 + 0.861486 w_k1_INTC_s107
                                 + 0.830706 w_k1_KO_s107
                                 + 1.02059 w_k1_MAT_s107
                                 + 1.21657 w_k1_MSFT_s107
                                 + 0.986493 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.27338 w_k1_ADM_s107
                                 + 0.953225 w_k1_AXP_s107
                                 + 1.12318 w_k1_DIS_s107
                                 + 1.12687 w_k1_FDX_s107
                                 + 1.11721 w_k1_IBM_s107
                                 + 1.01638 w_k1_INTC_s107
                                 + 1.02805 w_k1_KO_s107
                                 + 0.983878 w_k1_MAT_s107
                                 + 1.16332 w_k1_MSFT_s107
                                 + 1.02611 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.909205 w_k1_ADM_s107
                                 + 0.858482 w_k1_AXP_s107
                                 + 0.579182 w_k1_DIS_s107
                                 + 0.727568 w_k1_FDX_s107
                                 + 0.844291 w_k1_IBM_s107
                                 + 0.974198 w_k1_INTC_s107
                                 + 0.735891 w_k1_KO_s107
                                 + 0.721593 w_k1_MAT_s107
                                 + 0.845546 w_k1_MSFT_s107
                                 + 0.792645 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.994806 w_k1_ADM_s107
                                 + 0.827887 w_k1_AXP_s107
                                 + 1.27775 w_k1_DIS_s107
                                 + 0.993689 w_k1_FDX_s107
                                 + 1.21656 w_k1_IBM_s107
                                 + 1.05808 w_k1_INTC_s107
                                 + 1.19598 w_k1_KO_s107 + 1.14491 w_k1_MAT_s107
                                 + 1.20894 w_k1_MSFT_s107
                                 + 1.02037 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.829078 w_k1_ADM_s107
                                 + 0.904621 w_k1_AXP_s107
                                 + 0.901405 w_k1_DIS_s107
                                 + 0.863616 w_k1_FDX_s107
                                 + 0.903773 w_k1_IBM_s107
                                 + 0.89644 w_k1_INTC_s107
                                 + 1.03685 w_k1_KO_s107 + 1.04565 w_k1_MAT_s107
                                 + 0.911689 w_k1_MSFT_s107
                                 + 1.03386 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.02164 w_k1_ADM_s107
                                 + 1.04398 w_k1_AXP_s107
                                 + 0.891241 w_k1_DIS_s107
                                 + 0.986361 w_k1_FDX_s107
                                 + 0.961518 w_k1_IBM_s107
                                 + 1.03315 w_k1_INTC_s107
                                 + 0.966703 w_k1_KO_s107
                                 + 0.785056 w_k1_MAT_s107
                                 + 0.918546 w_k1_MSFT_s107
                                 + 0.980807 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.01864 w_k1_ADM_s107
                                 + 1.26919 w_k1_AXP_s107
                                 + 1.94682 w_k1_DIS_s107
                                 + 1.37041 w_k1_FDX_s107
                                 + 1.16921 w_k1_IBM_s107
                                 + 1.38826 w_k1_INTC_s107
                                 + 1.37008 w_k1_KO_s107
                                 + 0.859956 w_k1_MAT_s107
                                 + 1.59371 w_k1_MSFT_s107
                                 + 1.16276 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.894311 w_k1_ADM_s110
                                 + 0.703914 w_k1_AXP_s110
                                 + 0.711576 w_k1_DIS_s110
                                 + 0.774537 w_k1_FDX_s110
                                 + 0.762511 w_k1_IBM_s110
                                 + 0.810376 w_k1_INTC_s110
                                 + 0.70625 w_k1_KO_s110
                                 + 0.886176 w_k1_MAT_s110
                                 + 0.812882 w_k1_MSFT_s110
                                 + 0.830002 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.3604 w_k1_ADM_s110
                                 + 1.16065 w_k1_AXP_s110
                                 + 1.68175 w_k1_DIS_s110
                                 + 1.47132 w_k1_FDX_s110
                                 + 0.942661 w_k1_IBM_s110
                                 + 1.00119 w_k1_INTC_s110
                                 + 1.08243 w_k1_KO_s110 + 1.1593 w_k1_MAT_s110
                                 + 1.29471 w_k1_MSFT_s110
                                 + 1.14886 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.848177 w_k1_ADM_s110
                                 + 0.869522 w_k1_AXP_s110
                                 + 0.980049 w_k1_DIS_s110
                                 + 0.935853 w_k1_FDX_s110
                                 + 1.54689 w_k1_IBM_s110
                                 + 0.994791 w_k1_INTC_s110
                                 + 1.22853 w_k1_KO_s110 + 1.45775 w_k1_MAT_s110
                                 + 0.950431 w_k1_MSFT_s110
                                 + 1.07249 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.12243 w_k1_ADM_s110
                                 + 0.932099 w_k1_AXP_s110
                                 + 0.720229 w_k1_DIS_s110
                                 + 0.91498 w_k1_FDX_s110
                                 + 0.922602 w_k1_IBM_s110
                                 + 0.798322 w_k1_INTC_s110
                                 + 0.802066 w_k1_KO_s110
                                 + 1.15098 w_k1_MAT_s110
                                 + 0.885463 w_k1_MSFT_s110
                                 + 0.90319 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.20719 w_k1_ADM_s110
                                 + 1.02452 w_k1_AXP_s110
                                 + 1.19834 w_k1_DIS_s110
                                 + 1.14754 w_k1_FDX_s110
                                 + 0.864558 w_k1_IBM_s110
                                 + 1.10148 w_k1_INTC_s110
                                 + 0.91971 w_k1_KO_s110
                                 + 0.752022 w_k1_MAT_s110
                                 + 1.21257 w_k1_MSFT_s110
                                 + 1.02903 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.3616 w_k1_ADM_s110
                                 + 1.33201 w_k1_AXP_s110
                                 + 1.37703 w_k1_DIS_s110
                                 + 1.28196 w_k1_FDX_s110 + 1.4171 w_k1_IBM_s110
                                 + 1.28197 w_k1_INTC_s110
                                 + 1.19613 w_k1_KO_s110 + 1.14601 w_k1_MAT_s110
                                 + 1.16146 w_k1_MSFT_s110
                                 + 1.23156 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.00339 w_k1_ADM_s110
                                 + 1.01815 w_k1_AXP_s110
                                 + 1.04323 w_k1_DIS_s110
                                 + 1.04095 w_k1_FDX_s110
                                 + 1.21337 w_k1_IBM_s110
                                 + 0.897607 w_k1_INTC_s110
                                 + 1.15215 w_k1_KO_s110 + 1.43549 w_k1_MAT_s110
                                 + 1.03148 w_k1_MSFT_s110
                                 + 1.06536 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.988899 w_k1_ADM_s110
                                 + 0.981134 w_k1_AXP_s110
                                 + 1.43907 w_k1_DIS_s110
                                 + 1.06376 w_k1_FDX_s110
                                 + 1.20582 w_k1_IBM_s110
                                 + 1.20372 w_k1_INTC_s110
                                 + 1.20226 w_k1_KO_s110
                                 + 0.948591 w_k1_MAT_s110
                                 + 1.21543 w_k1_MSFT_s110
                                 + 1.11283 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.04888 w_k1_ADM_s111
                                 + 0.891864 w_k1_AXP_s111
                                 + 1.37989 w_k1_DIS_s111
                                 + 1.16127 w_k1_FDX_s111
                                 + 1.00182 w_k1_IBM_s111
                                 + 0.977118 w_k1_INTC_s111
                                 + 1.11112 w_k1_KO_s111
                                 + 0.953985 w_k1_MAT_s111
                                 + 1.12052 w_k1_MSFT_s111
                                 + 1.04698 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.17758 w_k1_ADM_s111
                                 + 1.33351 w_k1_AXP_s111
                                 + 1.56411 w_k1_DIS_s111
                                 + 1.43547 w_k1_FDX_s111
                                 + 0.933728 w_k1_IBM_s111
                                 + 1.22081 w_k1_INTC_s111
                                 + 0.949032 w_k1_KO_s111
                                 + 0.790326 w_k1_MAT_s111
                                 + 1.19805 w_k1_MSFT_s111
                                 + 1.06183 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.13456 w_k1_ADM_s111
                                 + 0.930999 w_k1_AXP_s111
                                 + 1.33394 w_k1_DIS_s111
                                 + 0.979653 w_k1_FDX_s111
                                 + 1.53141 w_k1_IBM_s111
                                 + 0.872263 w_k1_INTC_s111
                                 + 1.1605 w_k1_KO_s111 + 1.74364 w_k1_MAT_s111
                                 + 1.11378 w_k1_MSFT_s111
                                 + 1.15717 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.840938 w_k1_ADM_s111
                                 + 0.775592 w_k1_AXP_s111
                                 + 1.07937 w_k1_DIS_s111
                                 + 0.992201 w_k1_FDX_s111
                                 + 0.762792 w_k1_IBM_s111
                                 + 0.973272 w_k1_INTC_s111
                                 + 0.936123 w_k1_KO_s111
                                 + 0.978559 w_k1_MAT_s111
                                 + 0.944704 w_k1_MSFT_s111
                                 + 1.04817 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.14312 w_k1_ADM_s111
                                 + 1.65034 w_k1_AXP_s111
                                 + 1.49508 w_k1_DIS_s111
                                 + 1.34756 w_k1_FDX_s111
                                 + 1.36688 w_k1_IBM_s111
                                 + 1.3155 w_k1_INTC_s111 + 1.40462 w_k1_KO_s111
                                 + 2.11692 w_k1_MAT_s111
                                 + 1.22449 w_k1_MSFT_s111
                                 + 1.30229 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.861639 w_k1_ADM_s111
                                 + 0.768496 w_k1_AXP_s111
                                 + 0.691076 w_k1_DIS_s111
                                 + 0.670218 w_k1_FDX_s111
                                 + 1.29641 w_k1_IBM_s111
                                 + 0.742203 w_k1_INTC_s111
                                 + 0.684498 w_k1_KO_s111
                                 + 0.921427 w_k1_MAT_s111
                                 + 0.657818 w_k1_MSFT_s111
                                 + 0.877681 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.767294 w_k1_ADM_s111
                                 + 0.80331 w_k1_AXP_s111
                                 + 0.652383 w_k1_DIS_s111
                                 + 0.785991 w_k1_FDX_s111
                                 + 0.796904 w_k1_IBM_s111
                                 + 0.817769 w_k1_INTC_s111
                                 + 0.818901 w_k1_KO_s111
                                 + 0.902867 w_k1_MAT_s111
                                 + 0.831409 w_k1_MSFT_s111
                                 + 0.883174 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.924942 w_k1_ADM_s111
                                 + 1.08893 w_k1_AXP_s111
                                 + 1.00892 w_k1_DIS_s111
                                 + 0.990111 w_k1_FDX_s111
                                 + 1.11495 w_k1_IBM_s111
                                 + 0.879612 w_k1_INTC_s111
                                 + 0.975189 w_k1_KO_s111
                                 + 1.58331 w_k1_MAT_s111
                                 + 0.879153 w_k1_MSFT_s111
                                 + 1.03681 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.893298 w_k1_ADM_s112
                                 + 0.874314 w_k1_AXP_s112
                                 + 0.764692 w_k1_DIS_s112
                                 + 0.910473 w_k1_FDX_s112
                                 + 0.832281 w_k1_IBM_s112
                                 + 0.978575 w_k1_INTC_s112
                                 + 0.830622 w_k1_KO_s112
                                 + 0.732358 w_k1_MAT_s112
                                 + 0.762678 w_k1_MSFT_s112
                                 + 0.864934 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.864874 w_k1_ADM_s112
                                 + 1.09658 w_k1_AXP_s112
                                 + 1.46664 w_k1_DIS_s112 + 1.256 w_k1_FDX_s112
                                 + 1.14733 w_k1_IBM_s112
                                 + 1.4113 w_k1_INTC_s112 + 1.21251 w_k1_KO_s112
                                 + 1.24547 w_k1_MAT_s112
                                 + 1.19535 w_k1_MSFT_s112
                                 + 1.13061 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.12338 w_k1_ADM_s112
                                 + 1.11309 w_k1_AXP_s112
                                 + 1.24035 w_k1_DIS_s112 + 1.21 w_k1_FDX_s112
                                 + 1.35805 w_k1_IBM_s112
                                 + 1.11436 w_k1_INTC_s112
                                 + 1.53262 w_k1_KO_s112 + 1.99966 w_k1_MAT_s112
                                 + 1.15166 w_k1_MSFT_s112
                                 + 1.25918 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.736765 w_k1_ADM_s112
                                 + 0.760133 w_k1_AXP_s112
                                 + 0.60298 w_k1_DIS_s112
                                 + 0.706628 w_k1_FDX_s112
                                 + 0.78303 w_k1_IBM_s112
                                 + 0.777944 w_k1_INTC_s112
                                 + 0.639227 w_k1_KO_s112
                                 + 0.685726 w_k1_MAT_s112
                                 + 0.70193 w_k1_MSFT_s112
                                 + 0.781362 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.01538 w_k1_ADM_s112
                                 + 1.58809 w_k1_AXP_s112
                                 + 1.34253 w_k1_DIS_s112
                                 + 1.26357 w_k1_FDX_s112
                                 + 1.66855 w_k1_IBM_s112
                                 + 1.20359 w_k1_INTC_s112
                                 + 1.31586 w_k1_KO_s112 + 2.30935 w_k1_MAT_s112
                                 + 1.20269 w_k1_MSFT_s112
                                 + 1.21304 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.939047 w_k1_ADM_s112
                                 + 1.06662 w_k1_AXP_s112
                                 + 0.947811 w_k1_DIS_s112
                                 + 0.971776 w_k1_FDX_s112
                                 + 1.29766 w_k1_IBM_s112
                                 + 0.946354 w_k1_INTC_s112
                                 + 1.06028 w_k1_KO_s112 + 1.59255 w_k1_MAT_s112
                                 + 1.01582 w_k1_MSFT_s112
                                 + 1.01232 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.0835 w_k1_ADM_s112
                                 + 1.4496 w_k1_AXP_s112 + 1.13773 w_k1_DIS_s112
                                 + 1.29816 w_k1_FDX_s112
                                 + 0.948266 w_k1_IBM_s112
                                 + 1.33612 w_k1_INTC_s112
                                 + 1.03081 w_k1_KO_s112
                                 + 0.831045 w_k1_MAT_s112
                                 + 1.13436 w_k1_MSFT_s112
                                 + 0.949287 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.971945 w_k1_ADM_s112
                                 + 0.976899 w_k1_AXP_s112
                                 + 1.02098 w_k1_DIS_s112
                                 + 1.08846 w_k1_FDX_s112
                                 + 0.94445 w_k1_IBM_s112
                                 + 1.13223 w_k1_INTC_s112
                                 + 1.07985 w_k1_KO_s112 + 1.13111 w_k1_MAT_s112
                                 + 1.04454 w_k1_MSFT_s112
                                 + 0.989923 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.88296 w_k1_ADM_s113
                                 + 0.843333 w_k1_AXP_s113
                                 + 0.933917 w_k1_DIS_s113
                                 + 1.02136 w_k1_FDX_s113
                                 + 0.666004 w_k1_IBM_s113
                                 + 0.884134 w_k1_INTC_s113
                                 + 0.979246 w_k1_KO_s113
                                 + 0.883718 w_k1_MAT_s113
                                 + 1.05623 w_k1_MSFT_s113
                                 + 0.901933 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.26797 w_k1_ADM_s113
                                 + 1.45107 w_k1_AXP_s113 + 1.7293 w_k1_DIS_s113
                                 + 1.44164 w_k1_FDX_s113
                                 + 1.19298 w_k1_IBM_s113
                                 + 1.18527 w_k1_INTC_s113
                                 + 1.68622 w_k1_KO_s113 + 1.33518 w_k1_MAT_s113
                                 + 1.18336 w_k1_MSFT_s113
                                 + 1.29562 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.719364 w_k1_ADM_s113
                                 + 0.678036 w_k1_AXP_s113
                                 + 0.487755 w_k1_DIS_s113
                                 + 0.6998 w_k1_FDX_s113 + 0.69828 w_k1_IBM_s113
                                 + 0.662753 w_k1_INTC_s113
                                 + 0.692944 w_k1_KO_s113
                                 + 0.726496 w_k1_MAT_s113
                                 + 0.693231 w_k1_MSFT_s113
                                 + 0.754581 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.766899 w_k1_ADM_s113
                                 + 0.732158 w_k1_AXP_s113
                                 + 0.6409 w_k1_DIS_s113 + 0.74954 w_k1_FDX_s113
                                 + 1.04601 w_k1_IBM_s113
                                 + 0.759444 w_k1_INTC_s113
                                 + 0.964653 w_k1_KO_s113
                                 + 0.974734 w_k1_MAT_s113
                                 + 0.779132 w_k1_MSFT_s113
                                 + 0.888738 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.27898 w_k1_ADM_s113
                                 + 1.39112 w_k1_AXP_s113
                                 + 1.28257 w_k1_DIS_s113
                                 + 1.16863 w_k1_FDX_s113
                                 + 1.79024 w_k1_IBM_s113
                                 + 1.06373 w_k1_INTC_s113
                                 + 1.53433 w_k1_KO_s113 + 1.65407 w_k1_MAT_s113
                                 + 1.05551 w_k1_MSFT_s113
                                 + 1.19442 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.05487 w_k1_ADM_s113
                                 + 1.03362 w_k1_AXP_s113
                                 + 0.871118 w_k1_DIS_s113
                                 + 0.967901 w_k1_FDX_s113
                                 + 0.956576 w_k1_IBM_s113
                                 + 0.884061 w_k1_INTC_s113
                                 + 1.04193 w_k1_KO_s113 + 1.156 w_k1_MAT_s113
                                 + 0.921477 w_k1_MSFT_s113
                                 + 0.99318 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.02344 w_k1_ADM_s113
                                 + 1.13323 w_k1_AXP_s113
                                 + 1.43525 w_k1_DIS_s113
                                 + 1.18871 w_k1_FDX_s113
                                 + 1.01478 w_k1_IBM_s113
                                 + 1.17034 w_k1_INTC_s113
                                 + 1.29767 w_k1_KO_s113 + 1.3295 w_k1_MAT_s113
                                 + 1.23774 w_k1_MSFT_s113 + 1.1775 w_k1_PG_s113
                                 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.96645 w_k1_ADM_s113
                                 + 1.03059 w_k1_AXP_s113
                                 + 1.00701 w_k1_DIS_s113
                                 + 0.914865 w_k1_FDX_s113
                                 + 1.41583 w_k1_IBM_s113
                                 + 0.902503 w_k1_INTC_s113
                                 + 1.25831 w_k1_KO_s113 + 1.31338 w_k1_MAT_s113
                                 + 0.859483 w_k1_MSFT_s113
                                 + 1.04806 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.05494 w_k1_ADM_s114
                                 + 0.818255 w_k1_AXP_s114
                                 + 0.777321 w_k1_DIS_s114
                                 + 0.780113 w_k1_FDX_s114
                                 + 0.86822 w_k1_IBM_s114
                                 + 1.07104 w_k1_INTC_s114
                                 + 1.06076 w_k1_KO_s114
                                 + 0.897576 w_k1_MAT_s114
                                 + 0.800419 w_k1_MSFT_s114
                                 + 0.901913 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.28894 w_k1_ADM_s114
                                 + 1.21261 w_k1_AXP_s114
                                 + 1.22397 w_k1_DIS_s114
                                 + 1.08902 w_k1_FDX_s114
                                 + 1.15925 w_k1_IBM_s114
                                 + 1.12451 w_k1_INTC_s114
                                 + 1.12027 w_k1_KO_s114 + 1.23477 w_k1_MAT_s114
                                 + 1.0878 w_k1_MSFT_s114 + 0.93957 w_k1_PG_s114
                                 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.16601 w_k1_ADM_s114
                                 + 0.908028 w_k1_AXP_s114
                                 + 1.01949 w_k1_DIS_s114
                                 + 0.88304 w_k1_FDX_s114
                                 + 1.45782 w_k1_IBM_s114
                                 + 0.871924 w_k1_INTC_s114
                                 + 1.3613 w_k1_KO_s114 + 1.61464 w_k1_MAT_s114
                                 + 0.807276 w_k1_MSFT_s114
                                 + 1.05625 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.885298 w_k1_ADM_s114
                                 + 0.72745 w_k1_AXP_s114
                                 + 0.656664 w_k1_DIS_s114
                                 + 0.805709 w_k1_FDX_s114
                                 + 0.613829 w_k1_IBM_s114
                                 + 1.09266 w_k1_INTC_s114
                                 + 0.814166 w_k1_KO_s114
                                 + 0.662989 w_k1_MAT_s114
                                 + 0.885303 w_k1_MSFT_s114
                                 + 0.753448 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.18879 w_k1_ADM_s114
                                 + 0.972456 w_k1_AXP_s114
                                 + 0.887122 w_k1_DIS_s114
                                 + 1.01157 w_k1_FDX_s114
                                 + 0.799164 w_k1_IBM_s114
                                 + 1.1244 w_k1_INTC_s114 + 0.87561 w_k1_KO_s114
                                 + 0.837614 w_k1_MAT_s114
                                 + 0.913223 w_k1_MSFT_s114
                                 + 0.925435 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.00157 w_k1_ADM_s114
                                 + 1.0592 w_k1_AXP_s114 + 1.10841 w_k1_DIS_s114
                                 + 1.14141 w_k1_FDX_s114
                                 + 0.669562 w_k1_IBM_s114
                                 + 1.1762 w_k1_INTC_s114 + 1.02334 w_k1_KO_s114
                                 + 1.24449 w_k1_MAT_s114
                                 + 1.15989 w_k1_MSFT_s114
                                 + 0.998617 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.881402 w_k1_ADM_s114
                                 + 0.934672 w_k1_AXP_s114
                                 + 0.959342 w_k1_DIS_s114
                                 + 0.870398 w_k1_FDX_s114
                                 + 1.05423 w_k1_IBM_s114
                                 + 1.11776 w_k1_INTC_s114
                                 + 1.17066 w_k1_KO_s114 + 1.05164 w_k1_MAT_s114
                                 + 0.882645 w_k1_MSFT_s114
                                 + 1.00564 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.3186 w_k1_ADM_s114
                                 + 1.38831 w_k1_AXP_s114 + 1.4319 w_k1_DIS_s114
                                 + 1.2857 w_k1_FDX_s114 + 1.25562 w_k1_IBM_s114
                                 + 1.3741 w_k1_INTC_s114 + 1.51318 w_k1_KO_s114
                                 + 1.68154 w_k1_MAT_s114
                                 + 1.16659 w_k1_MSFT_s114
                                 + 1.17035 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.28219 w_k1_ADM_s115
                                 + 1.59802 w_k1_AXP_s115
                                 + 1.12649 w_k1_DIS_s115 + 1.0993 w_k1_FDX_s115
                                 + 1.91597 w_k1_IBM_s115
                                 + 0.961362 w_k1_INTC_s115
                                 + 1.3033 w_k1_KO_s115 + 2.63503 w_k1_MAT_s115
                                 + 0.933768 w_k1_MSFT_s115
                                 + 1.2168 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.05771 w_k1_ADM_s115
                                 + 0.981232 w_k1_AXP_s115
                                 + 0.994873 w_k1_DIS_s115
                                 + 0.964791 w_k1_FDX_s115
                                 + 0.950237 w_k1_IBM_s115
                                 + 1.10891 w_k1_INTC_s115
                                 + 0.974682 w_k1_KO_s115
                                 + 0.94259 w_k1_MAT_s115
                                 + 1.00677 w_k1_MSFT_s115
                                 + 0.997763 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.04375 w_k1_ADM_s115
                                 + 0.894951 w_k1_AXP_s115
                                 + 1.1586 w_k1_DIS_s115
                                 + 0.970542 w_k1_FDX_s115
                                 + 1.44973 w_k1_IBM_s115
                                 + 1.00245 w_k1_INTC_s115
                                 + 1.41939 w_k1_KO_s115
                                 + 0.977209 w_k1_MAT_s115
                                 + 1.02465 w_k1_MSFT_s115
                                 + 1.04926 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.8412 w_k1_ADM_s115
                                 + 0.829139 w_k1_AXP_s115
                                 + 0.71277 w_k1_DIS_s115
                                 + 0.733251 w_k1_FDX_s115
                                 + 0.964334 w_k1_IBM_s115
                                 + 0.811674 w_k1_INTC_s115
                                 + 0.889328 w_k1_KO_s115
                                 + 0.874662 w_k1_MAT_s115
                                 + 0.813415 w_k1_MSFT_s115
                                 + 0.811401 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.18217 w_k1_ADM_s115
                                 + 1.45868 w_k1_AXP_s115
                                 + 0.988455 w_k1_DIS_s115
                                 + 1.05648 w_k1_FDX_s115
                                 + 0.983474 w_k1_IBM_s115
                                 + 1.11969 w_k1_INTC_s115
                                 + 1.18672 w_k1_KO_s115 + 1.76833 w_k1_MAT_s115
                                 + 1.00432 w_k1_MSFT_s115
                                 + 1.07443 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.49113 w_k1_ADM_s115
                                 + 1.64224 w_k1_AXP_s115
                                 + 1.65332 w_k1_DIS_s115
                                 + 1.36765 w_k1_FDX_s115
                                 + 1.11507 w_k1_IBM_s115
                                 + 1.48053 w_k1_INTC_s115 + 1.3281 w_k1_KO_s115
                                 + 1.09593 w_k1_MAT_s115
                                 + 1.36109 w_k1_MSFT_s115 + 1.161 w_k1_PG_s115
                                 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.21997 w_k1_ADM_s115
                                 + 1.00471 w_k1_AXP_s115
                                 + 0.748489 w_k1_DIS_s115
                                 + 0.941029 w_k1_FDX_s115
                                 + 0.749347 w_k1_IBM_s115
                                 + 1.02213 w_k1_INTC_s115
                                 + 0.938397 w_k1_KO_s115
                                 + 0.893216 w_k1_MAT_s115
                                 + 0.868531 w_k1_MSFT_s115
                                 + 0.859094 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.20783 w_k1_ADM_s115
                                 + 1.32717 w_k1_AXP_s115
                                 + 1.17024 w_k1_DIS_s115
                                 + 1.15134 w_k1_FDX_s115
                                 + 1.10964 w_k1_IBM_s115
                                 + 1.16003 w_k1_INTC_s115
                                 + 0.950475 w_k1_KO_s115
                                 + 0.882965 w_k1_MAT_s115
                                 + 1.04137 w_k1_MSFT_s115
                                 + 1.02386 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.16142 w_k1_ADM_s116
                                 + 0.932837 w_k1_AXP_s116
                                 + 0.907432 w_k1_DIS_s116
                                 + 1.05634 w_k1_FDX_s116
                                 + 0.678618 w_k1_IBM_s116
                                 + 0.839525 w_k1_INTC_s116
                                 + 0.937889 w_k1_KO_s116
                                 + 1.23833 w_k1_MAT_s116
                                 + 1.01591 w_k1_MSFT_s116
                                 + 0.920724 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.10344 w_k1_ADM_s116
                                 + 1.334 w_k1_AXP_s116 + 1.43597 w_k1_DIS_s116
                                 + 1.22652 w_k1_FDX_s116
                                 + 1.16632 w_k1_IBM_s116
                                 + 1.1924 w_k1_INTC_s116 + 1.51642 w_k1_KO_s116
                                 + 1.63699 w_k1_MAT_s116
                                 + 1.33657 w_k1_MSFT_s116
                                 + 1.21674 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.983571 w_k1_ADM_s116
                                 + 1.05352 w_k1_AXP_s116
                                 + 1.03333 w_k1_DIS_s116
                                 + 0.945781 w_k1_FDX_s116
                                 + 1.13468 w_k1_IBM_s116
                                 + 0.855032 w_k1_INTC_s116
                                 + 1.26801 w_k1_KO_s116 + 2.08808 w_k1_MAT_s116
                                 + 1.10931 w_k1_MSFT_s116 + 1.0685 w_k1_PG_s116
                                 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.2564 w_k1_ADM_s116
                                 + 1.25931 w_k1_AXP_s116
                                 + 1.13833 w_k1_DIS_s116
                                 + 0.97647 w_k1_FDX_s116
                                 + 1.64517 w_k1_IBM_s116
                                 + 0.944056 w_k1_INTC_s116
                                 + 1.49174 w_k1_KO_s116 + 2.03751 w_k1_MAT_s116
                                 + 1.12077 w_k1_MSFT_s116
                                 + 1.11364 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.996068 w_k1_ADM_s116
                                 + 1.10347 w_k1_AXP_s116
                                 + 0.844573 w_k1_DIS_s116
                                 + 0.82087 w_k1_FDX_s116
                                 + 1.25589 w_k1_IBM_s116
                                 + 0.635422 w_k1_INTC_s116
                                 + 1.06704 w_k1_KO_s116 + 1.59904 w_k1_MAT_s116
                                 + 0.829646 w_k1_MSFT_s116
                                 + 0.851751 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.709197 w_k1_ADM_s116
                                 + 0.586555 w_k1_AXP_s116
                                 + 0.604198 w_k1_DIS_s116
                                 + 0.734642 w_k1_FDX_s116
                                 + 0.762848 w_k1_IBM_s116
                                 + 0.628935 w_k1_INTC_s116
                                 + 0.743687 w_k1_KO_s116
                                 + 0.747779 w_k1_MAT_s116
                                 + 0.7576 w_k1_MSFT_s116
                                 + 0.732441 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.39381 w_k1_ADM_s116
                                 + 0.886075 w_k1_AXP_s116
                                 + 0.90583 w_k1_DIS_s116
                                 + 0.97209 w_k1_FDX_s116
                                 + 0.840452 w_k1_IBM_s116
                                 + 0.84597 w_k1_INTC_s116
                                 + 1.05666 w_k1_KO_s116 + 1.57688 w_k1_MAT_s116
                                 + 1.09803 w_k1_MSFT_s116
                                 + 0.936794 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.995009 w_k1_ADM_s116
                                 + 0.767056 w_k1_AXP_s116
                                 + 0.993773 w_k1_DIS_s116
                                 + 0.926078 w_k1_FDX_s116
                                 + 0.931655 w_k1_IBM_s116
                                 + 0.979747 w_k1_INTC_s116
                                 + 0.959155 w_k1_KO_s116
                                 + 0.895788 w_k1_MAT_s116
                                 + 1.09612 w_k1_MSFT_s116
                                 + 0.899108 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.05107 w_k1_ADM_s117
                                 + 0.81534 w_k1_AXP_s117
                                 + 1.02496 w_k1_DIS_s117
                                 + 1.04637 w_k1_FDX_s117
                                 + 0.826195 w_k1_IBM_s117
                                 + 1.02939 w_k1_INTC_s117
                                 + 0.913449 w_k1_KO_s117
                                 + 0.625696 w_k1_MAT_s117
                                 + 0.971825 w_k1_MSFT_s117
                                 + 0.920248 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.151 w_k1_ADM_s117
                                 + 1.05625 w_k1_AXP_s117
                                 + 0.933223 w_k1_DIS_s117
                                 + 0.906328 w_k1_FDX_s117
                                 + 1.33536 w_k1_IBM_s117
                                 + 0.963958 w_k1_INTC_s117
                                 + 1.00108 w_k1_KO_s117 + 1.23165 w_k1_MAT_s117
                                 + 0.900056 w_k1_MSFT_s117
                                 + 0.904876 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.658322 w_k1_ADM_s117
                                 + 0.692988 w_k1_AXP_s117
                                 + 0.736104 w_k1_DIS_s117
                                 + 0.788085 w_k1_FDX_s117
                                 + 0.970487 w_k1_IBM_s117
                                 + 0.788936 w_k1_INTC_s117
                                 + 0.819756 w_k1_KO_s117
                                 + 0.693284 w_k1_MAT_s117
                                 + 0.747744 w_k1_MSFT_s117
                                 + 0.821089 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.987205 w_k1_ADM_s117
                                 + 0.859064 w_k1_AXP_s117
                                 + 0.782321 w_k1_DIS_s117
                                 + 0.972251 w_k1_FDX_s117
                                 + 0.757324 w_k1_IBM_s117
                                 + 0.869554 w_k1_INTC_s117
                                 + 0.723848 w_k1_KO_s117
                                 + 0.808302 w_k1_MAT_s117
                                 + 0.779879 w_k1_MSFT_s117
                                 + 0.862003 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.62001 w_k1_ADM_s117
                                 + 1.78069 w_k1_AXP_s117
                                 + 1.85837 w_k1_DIS_s117
                                 + 1.50488 w_k1_FDX_s117
                                 + 1.48435 w_k1_IBM_s117
                                 + 1.33937 w_k1_INTC_s117
                                 + 1.37291 w_k1_KO_s117 + 1.43302 w_k1_MAT_s117
                                 + 1.3136 w_k1_MSFT_s117 + 1.23444 w_k1_PG_s117
                                 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.854739 w_k1_ADM_s117
                                 + 0.467051 w_k1_AXP_s117
                                 + 0.638253 w_k1_DIS_s117
                                 + 0.740132 w_k1_FDX_s117
                                 + 0.681957 w_k1_IBM_s117
                                 + 0.809284 w_k1_INTC_s117
                                 + 0.695577 w_k1_KO_s117
                                 + 0.562899 w_k1_MAT_s117
                                 + 0.68516 w_k1_MSFT_s117
                                 + 0.794543 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.23347 w_k1_ADM_s117
                                 + 0.996181 w_k1_AXP_s117
                                 + 1.39029 w_k1_DIS_s117
                                 + 1.20969 w_k1_FDX_s117
                                 + 0.801677 w_k1_IBM_s117
                                 + 1.04344 w_k1_INTC_s117
                                 + 1.04338 w_k1_KO_s117
                                 + 0.752582 w_k1_MAT_s117
                                 + 1.0357 w_k1_MSFT_s117 + 1.07028 w_k1_PG_s117
                                 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.37503 w_k1_ADM_s117
                                 + 1.38282 w_k1_AXP_s117
                                 + 1.24613 w_k1_DIS_s117
                                 + 1.28133 w_k1_FDX_s117 + 1.1144 w_k1_IBM_s117
                                 + 1.42286 w_k1_INTC_s117
                                 + 1.33463 w_k1_KO_s117 + 1.04355 w_k1_MAT_s117
                                 + 1.21475 w_k1_MSFT_s117
                                 + 1.04374 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0273438373590462 w_k1_ADM_s000
                                 + 0.0242457474148748 w_k1_ADM_s001
                                 + 0.022231074063864 w_k1_ADM_s002
                                 + 0.0136460783240224 w_k1_ADM_s003
                                 + 0.0185261411992797 w_k1_ADM_s004
                                 + 0.0203110660273644 w_k1_ADM_s005
                                 + 0.00638050502901756 w_k1_ADM_s006
                                 + 0.0191463027014065 w_k1_ADM_s007
                                 + 0.038552811850922 w_k1_ADM_s010
                                 + 0.0171649487475206 w_k1_ADM_s011
                                 + 0.0632926126618654 w_k1_ADM_s012
                                 + 0.0178159434338927 w_k1_ADM_s013
                                 + 0.0144515613820476 w_k1_ADM_s014
                                 + 0.0374766575669904 w_k1_ADM_s015
                                 + 0.0378377884036866 w_k1_ADM_s016
                                 + 0.0181962349885782 w_k1_ADM_s017
                                 + 0.0901052172285541 w_k1_ADM_s100
                                 + 0.043115070695507 w_k1_ADM_s101
                                 + 0.0556894049600213 w_k1_ADM_s102
                                 + 0.0456835314674503 w_k1_ADM_s103
                                 + 0.0511081038496631 w_k1_ADM_s104
                                 + 0.0196419353548763 w_k1_ADM_s105
                                 + 0.0341519632308732 w_k1_ADM_s106
                                 + 0.0276444551914323 w_k1_ADM_s107
                                 + 0.0320555239419 w_k1_ADM_s110
                                 + 0.0514630626437297 w_k1_ADM_s111
                                 + 0.015611190317113 w_k1_ADM_s112
                                 + 0.0458784845381624 w_k1_ADM_s113
                                 + 0.0326591001287877 w_k1_ADM_s114
                                 + 0.0193282181114506 w_k1_ADM_s115
                                 + 0.0254960832800072 w_k1_ADM_s116
                                 + 0.0192076714770936 w_k1_ADM_s117
                                 + 0.0310859212003647 w_k1_AXP_s000
                                 + 0.0208933822331624 w_k1_AXP_s001
                                 + 0.0189765867016406 w_k1_AXP_s002
                                 + 0.0131881621287239 w_k1_AXP_s003
                                 + 0.0205327094346178 w_k1_AXP_s004
                                 + 0.0210923159237524 w_k1_AXP_s005
                                 + 0.00665568327990562 w_k1_AXP_s006
                                 + 0.0192843746420085 w_k1_AXP_s007
                                 + 0.0387931587919731 w_k1_AXP_s010
                                 + 0.0158730313453378 w_k1_AXP_s011
                                 + 0.0570340868584666 w_k1_AXP_s012
                                 + 0.0195971896960694 w_k1_AXP_s013
                                 + 0.0168621506504905 w_k1_AXP_s014
                                 + 0.0424458415710273 w_k1_AXP_s015
                                 + 0.0367763304625551 w_k1_AXP_s016
                                 + 0.0187777239419187 w_k1_AXP_s017
                                 + 0.103945296502755 w_k1_AXP_s100
                                 + 0.0481722783599393 w_k1_AXP_s101
                                 + 0.065403999707699 w_k1_AXP_s102
                                 + 0.0526955683391849 w_k1_AXP_s103
                                 + 0.0596209377122819 w_k1_AXP_s104
                                 + 0.0205968479836789 w_k1_AXP_s105
                                 + 0.0294406633276991 w_k1_AXP_s106
                                 + 0.0279942193469317 w_k1_AXP_s107
                                 + 0.0286711880682413 w_k1_AXP_s110
                                 + 0.0515406144717312 w_k1_AXP_s111
                                 + 0.0177125060655544 w_k1_AXP_s112
                                 + 0.0470776227303957 w_k1_AXP_s113
                                 + 0.0302485461676252 w_k1_AXP_s114
                                 + 0.0193607478393628 w_k1_AXP_s115
                                 + 0.0245463854548291 w_k1_AXP_s116
                                 + 0.016817909641175 w_k1_AXP_s117
                                 + 0.0299989519154729 w_k1_DIS_s000
                                 + 0.0231770312994443 w_k1_DIS_s001
                                 + 0.0202459086732707 w_k1_DIS_s002
                                 + 0.0144740221956746 w_k1_DIS_s003
                                 + 0.0208394859256649 w_k1_DIS_s004
                                 + 0.0219186365349236 w_k1_DIS_s005
                                 + 0.00660618340838784 w_k1_DIS_s006
                                 + 0.020767340143294 w_k1_DIS_s007
                                 + 0.0401513257904108 w_k1_DIS_s010
                                 + 0.0175829653965001 w_k1_DIS_s011
                                 + 0.055454737509603 w_k1_DIS_s012
                                 + 0.0208772502592832 w_k1_DIS_s013
                                 + 0.0165040493815619 w_k1_DIS_s014
                                 + 0.0485281019085697 w_k1_DIS_s015
                                 + 0.0359433779097192 w_k1_DIS_s016
                                 + 0.0178383384343043 w_k1_DIS_s017
                                 + 0.0937834216230509 w_k1_DIS_s100
                                 + 0.0460455634332299 w_k1_DIS_s101
                                 + 0.067807096416738 w_k1_DIS_s102
                                 + 0.0509965984555715 w_k1_DIS_s103
                                 + 0.0552488645584075 w_k1_DIS_s104
                                 + 0.0215343203369525 w_k1_DIS_s105
                                 + 0.0315339558388235 w_k1_DIS_s106
                                 + 0.030653914091141 w_k1_DIS_s107
                                 + 0.0303575751304856 w_k1_DIS_s110
                                 + 0.059271099286692 w_k1_DIS_s111
                                 + 0.0163373988645201 w_k1_DIS_s112
                                 + 0.0461690576386322 w_k1_DIS_s113
                                 + 0.0307279292193764 w_k1_DIS_s114
                                 + 0.0179078450738191 w_k1_DIS_s115
                                 + 0.0249606642386091 w_k1_DIS_s116
                                 + 0.0179548211324863 w_k1_DIS_s117
                                 + 0.0280306767574304 w_k1_FDX_s000
                                 + 0.0212868562769165 w_k1_FDX_s001
                                 + 0.0197912665181487 w_k1_FDX_s002
                                 + 0.0141032072167811 w_k1_FDX_s003
                                 + 0.0208387730426131 w_k1_FDX_s004
                                 + 0.0212021304675198 w_k1_FDX_s005
                                 + 0.00612787222898454 w_k1_FDX_s006
                                 + 0.0193170454400978 w_k1_FDX_s007
                                 + 0.0410678581464794 w_k1_FDX_s010
                                 + 0.0172695825970227 w_k1_FDX_s011
                                 + 0.0547769950696602 w_k1_FDX_s012
                                 + 0.020869616780656 w_k1_FDX_s013
                                 + 0.0152358738860876 w_k1_FDX_s014
                                 + 0.0418711082377904 w_k1_FDX_s015
                                 + 0.0361776572373293 w_k1_FDX_s016
                                 + 0.0175608058868572 w_k1_FDX_s017
                                 + 0.0936913413300351 w_k1_FDX_s100
                                 + 0.0440373232195331 w_k1_FDX_s101
                                 + 0.0640509492679031 w_k1_FDX_s102
                                 + 0.050557126135742 w_k1_FDX_s103
                                 + 0.0527855829020069 w_k1_FDX_s104
                                 + 0.0206919473196119 w_k1_FDX_s105
                                 + 0.0306637105711047 w_k1_FDX_s106
                                 + 0.0280491934074782 w_k1_FDX_s107
                                 + 0.0303944787070039 w_k1_FDX_s110
                                 + 0.0541073113757777 w_k1_FDX_s111
                                 + 0.0172836729557048 w_k1_FDX_s112
                                 + 0.0460284887701256 w_k1_FDX_s113
                                 + 0.0289630988970548 w_k1_FDX_s114
                                 + 0.0172894680071818 w_k1_FDX_s115
                                 + 0.0238644663947276 w_k1_FDX_s116
                                 + 0.0179516980654246 w_k1_FDX_s117
                                 + 0.0284783567951655 w_k1_IBM_s000
                                 + 0.0215142821691595 w_k1_IBM_s001
                                 + 0.0198086640819077 w_k1_IBM_s002
                                 + 0.0145185948069263 w_k1_IBM_s003
                                 + 0.0212676372066862 w_k1_IBM_s004
                                 + 0.0219217024795328 w_k1_IBM_s005
                                 + 0.0067485370633394 w_k1_IBM_s006
                                 + 0.0198536711002192 w_k1_IBM_s007
                                 + 0.0389563796899891 w_k1_IBM_s010
                                 + 0.0171820641925618 w_k1_IBM_s011
                                 + 0.0550887893051077 w_k1_IBM_s012
                                 + 0.0208069005848332 w_k1_IBM_s013
                                 + 0.0168016957484039 w_k1_IBM_s014
                                 + 0.0446539082815059 w_k1_IBM_s015
                                 + 0.0382795774261114 w_k1_IBM_s016
                                 + 0.0169574688341451 w_k1_IBM_s017
                                 + 0.0935305609462447 w_k1_IBM_s100
                                 + 0.0466563047800757 w_k1_IBM_s101
                                 + 0.0658082235282035 w_k1_IBM_s102
                                 + 0.0496378041321126 w_k1_IBM_s103
                                 + 0.0549344017681674 w_k1_IBM_s104
                                 + 0.0199589344404748 w_k1_IBM_s105
                                 + 0.0319340524244379 w_k1_IBM_s106
                                 + 0.0273350826923437 w_k1_IBM_s107
                                 + 0.0304285385534277 w_k1_IBM_s110
                                 + 0.0566487113306564 w_k1_IBM_s111
                                 + 0.0176249496759616 w_k1_IBM_s112
                                 + 0.04880580409228 w_k1_IBM_s113
                                 + 0.0309463166343558 w_k1_IBM_s114
                                 + 0.0188631238928333 w_k1_IBM_s115
                                 + 0.0260488663660856 w_k1_IBM_s116
                                 + 0.0169651005967213 w_k1_IBM_s117
                                 + 0.0354535996649884 w_k1_INTC_s000
                                 + 0.022340418320576 w_k1_INTC_s001
                                 + 0.0204536086960968 w_k1_INTC_s002
                                 + 0.0135724476282577 w_k1_INTC_s003
                                 + 0.0226443086570969 w_k1_INTC_s004
                                 + 0.0222681731088701 w_k1_INTC_s005
                                 + 0.00630807597510547 w_k1_INTC_s006
                                 + 0.0209471407843981 w_k1_INTC_s007
                                 + 0.033066859190523 w_k1_INTC_s010
                                 + 0.0173288754876566 w_k1_INTC_s011
                                 + 0.0543422302157607 w_k1_INTC_s012
                                 + 0.0224470221761454 w_k1_INTC_s013
                                 + 0.0149974340349178 w_k1_INTC_s014
                                 + 0.0403864163507501 w_k1_INTC_s015
                                 + 0.0421675273518469 w_k1_INTC_s016
                                 + 0.0135245050198929 w_k1_INTC_s017
                                 + 0.0881629311693653 w_k1_INTC_s100
                                 + 0.044613707232487 w_k1_INTC_s101
                                 + 0.0665969814418535 w_k1_INTC_s102
                                 + 0.0523881761609201 w_k1_INTC_s103
                                 + 0.0475384607921585 w_k1_INTC_s104
                                 + 0.0171506277177083 w_k1_INTC_s105
                                 + 0.0300143385716756 w_k1_INTC_s106
                                 + 0.0284061469126422 w_k1_INTC_s107
                                 + 0.0279995632489753 w_k1_INTC_s110
                                 + 0.0501915980136823 w_k1_INTC_s111
                                 + 0.0176860100467399 w_k1_INTC_s112
                                 + 0.042484947117044 w_k1_INTC_s113
                                 + 0.0327464471659269 w_k1_INTC_s114
                                 + 0.0185016088534687 w_k1_INTC_s115
                                 + 0.0220214544561164 w_k1_INTC_s116
                                 + 0.0176894523506626 w_k1_INTC_s117
                                 + 0.0289954317523205 w_k1_KO_s000
                                 + 0.0215504313818764 w_k1_KO_s001
                                 + 0.022586921132323 w_k1_KO_s002
                                 + 0.0152020866815971 w_k1_KO_s003
                                 + 0.0198681654486639 w_k1_KO_s004
                                 + 0.0210680186075283 w_k1_KO_s005
                                 + 0.00643404432105285 w_k1_KO_s006
                                 + 0.0186236190091277 w_k1_KO_s007
                                 + 0.0362037422526885 w_k1_KO_s010
                                 + 0.0172986799647452 w_k1_KO_s011
                                 + 0.0546520215145982 w_k1_KO_s012
                                 + 0.0196807010205195 w_k1_KO_s013
                                 + 0.0168538498038701 w_k1_KO_s014
                                 + 0.0488489740672791 w_k1_KO_s015
                                 + 0.0385340660770302 w_k1_KO_s016
                                 + 0.0155550005687977 w_k1_KO_s017
                                 + 0.0987241248823154 w_k1_KO_s100
                                 + 0.0442524860107239 w_k1_KO_s101
                                 + 0.0576990804827685 w_k1_KO_s102
                                 + 0.0499873969210221 w_k1_KO_s103
                                 + 0.049813428751903 w_k1_KO_s104
                                 + 0.0206467788576457 w_k1_KO_s105
                                 + 0.029890699734518 w_k1_KO_s106
                                 + 0.02794517198452 w_k1_KO_s107
                                 + 0.0286539601827703 w_k1_KO_s110
                                 + 0.0534055217780309 w_k1_KO_s111
                                 + 0.0169924889668981 w_k1_KO_s112
                                 + 0.052743418573576 w_k1_KO_s113
                                 + 0.0344947607067998 w_k1_KO_s114
                                 + 0.0187458473839048 w_k1_KO_s115
                                 + 0.0284566958903771 w_k1_KO_s116
                                 + 0.0167928694151406 w_k1_KO_s117
                                 + 0.0268201063754335 w_k1_MAT_s000
                                 + 0.0182845113276373 w_k1_MAT_s001
                                 + 0.0191427882921767 w_k1_MAT_s002
                                 + 0.0132792369487379 w_k1_MAT_s003
                                 + 0.0210413726807364 w_k1_MAT_s004
                                 + 0.0190188633614141 w_k1_MAT_s005
                                 + 0.00744907820395186 w_k1_MAT_s006
                                 + 0.0196113160848139 w_k1_MAT_s007
                                 + 0.0388031184452445 w_k1_MAT_s010
                                 + 0.0157552100901645 w_k1_MAT_s011
                                 + 0.0473560238770376 w_k1_MAT_s012
                                 + 0.0173009096733027 w_k1_MAT_s013
                                 + 0.0178739363780639 w_k1_MAT_s014
                                 + 0.0409502835995551 w_k1_MAT_s015
                                 + 0.034365780844432 w_k1_MAT_s016
                                 + 0.0181639640020454 w_k1_MAT_s017
                                 + 0.123056603991666 w_k1_MAT_s100
                                 + 0.0362176287680983 w_k1_MAT_s101
                                 + 0.058526720732815 w_k1_MAT_s102
                                 + 0.0618970466084923 w_k1_MAT_s103
                                 + 0.0527392238877233 w_k1_MAT_s104
                                 + 0.0242789807006723 w_k1_MAT_s105
                                 + 0.0281234230548466 w_k1_MAT_s106
                                 + 0.0263101955990092 w_k1_MAT_s107
                                 + 0.0327950133104599 w_k1_MAT_s110
                                 + 0.0657041192807151 w_k1_MAT_s111
                                 + 0.0199940747298753 w_k1_MAT_s112
                                 + 0.05339402495016 w_k1_MAT_s113
                                 + 0.0358924331937212 w_k1_MAT_s114
                                 + 0.0186360229300506 w_k1_MAT_s115
                                 + 0.03522891013562 w_k1_MAT_s116
                                 + 0.015220998578627 w_k1_MAT_s117
                                 + 0.0283982734635741 w_k1_MSFT_s000
                                 + 0.0217851396014097 w_k1_MSFT_s001
                                 + 0.0211205992100585 w_k1_MSFT_s002
                                 + 0.0142702850668193 w_k1_MSFT_s003
                                 + 0.0192587563648325 w_k1_MSFT_s004
                                 + 0.0199149402918965 w_k1_MSFT_s005
                                 + 0.00591096272890358 w_k1_MSFT_s006
                                 + 0.0197445069857745 w_k1_MSFT_s007
                                 + 0.0350794659110863 w_k1_MSFT_s010
                                 + 0.0170886669691345 w_k1_MSFT_s011
                                 + 0.0534598431719604 w_k1_MSFT_s012
                                 + 0.0200146631118584 w_k1_MSFT_s013
                                 + 0.0147018108889312 w_k1_MSFT_s014
                                 + 0.0438801818584097 w_k1_MSFT_s015
                                 + 0.0380898523013248 w_k1_MSFT_s016
                                 + 0.0155592860401619 w_k1_MSFT_s017
                                 + 0.0918687674755846 w_k1_MSFT_s100
                                 + 0.0436139774523721 w_k1_MSFT_s101
                                 + 0.0650894397133132 w_k1_MSFT_s102
                                 + 0.0500792486880683 w_k1_MSFT_s103
                                 + 0.0501014410716975 w_k1_MSFT_s104
                                 + 0.0197990561546539 w_k1_MSFT_s105
                                 + 0.0297241871267074 w_k1_MSFT_s106
                                 + 0.0300822824943404 w_k1_MSFT_s107
                                 + 0.0303395295739583 w_k1_MSFT_s110
                                 + 0.0528146483331431 w_k1_MSFT_s111
                                 + 0.016262216262587 w_k1_MSFT_s112
                                 + 0.0441023002322535 w_k1_MSFT_s113
                                 + 0.0279563830179486 w_k1_MSFT_s114
                                 + 0.0172606187587524 w_k1_MSFT_s115
                                 + 0.0260854234128071 w_k1_MSFT_s116
                                 + 0.0163651124717381 w_k1_MSFT_s117
                                 + 0.0283439646251628 w_k1_PG_s000
                                 + 0.021067774582018 w_k1_PG_s001
                                 + 0.0204711654731022 w_k1_PG_s002
                                 + 0.0142710666510361 w_k1_PG_s003
                                 + 0.0191933602777362 w_k1_PG_s004
                                 + 0.0206445239215561 w_k1_PG_s005
                                 + 0.00571569598732317 w_k1_PG_s006
                                 + 0.0180665184956975 w_k1_PG_s007
                                 + 0.0385431404007573 w_k1_PG_s010
                                 + 0.0162417421581941 w_k1_PG_s011
                                 + 0.0549392107572517 w_k1_PG_s012
                                 + 0.0196201935044936 w_k1_PG_s013
                                 + 0.0163513185036266 w_k1_PG_s014
                                 + 0.043109481540721 w_k1_PG_s015
                                 + 0.0390921379413307 w_k1_PG_s016
                                 + 0.0171098354117169 w_k1_PG_s017
                                 + 0.0883789790199535 w_k1_PG_s100
                                 + 0.0419317139157186 w_k1_PG_s101
                                 + 0.0618995923731629 w_k1_PG_s102
                                 + 0.0526269478256268 w_k1_PG_s103
                                 + 0.0506665725256229 w_k1_PG_s104
                                 + 0.0193563596401859 w_k1_PG_s105
                                 + 0.0318950855855784 w_k1_PG_s106
                                 + 0.0277910351081432 w_k1_PG_s107
                                 + 0.0296871495794357 w_k1_PG_s110
                                 + 0.055598675873271 w_k1_PG_s111
                                 + 0.016185993678496 w_k1_PG_s112
                                 + 0.0469188180754121 w_k1_PG_s113
                                 + 0.0289507304854498 w_k1_PG_s114
                                 + 0.0171938360818446 w_k1_PG_s115
                                 + 0.0241952344085623 w_k1_PG_s116
                                 + 0.0163098244639587 w_k1_PG_s117 >= 1.14
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
