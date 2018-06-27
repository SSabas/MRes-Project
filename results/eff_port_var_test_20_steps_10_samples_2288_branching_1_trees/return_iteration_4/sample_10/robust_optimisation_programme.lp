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
 w_k1_ADM_s0_return_constraint:  0.975014 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  1.05836 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.29997 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.941393 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.0017 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.763166 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.09596 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.997543 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.26874 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.760773 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.901835 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.20569 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.40101 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.98048 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.26072 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.804569 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.2305 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.686329 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.00929 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.56697 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.2197 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.690189 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.15863 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.05278 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.01332 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.25867 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.703223 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.52635 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.57192 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.955884 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.871006 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.16936 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.39385 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.01632 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.995812 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.0716 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.52985 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.907863 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  0.976608 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.17872 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.49605 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.965981 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.07806 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.698018 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.06528 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.737757 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.33085 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.662738 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.877527 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.07433 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.57176 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.11626 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.14824 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.770765 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.16713 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.569297 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.927391 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.7686 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.34989 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.666011 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.790694 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.17873 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.17672 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.30288 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.710925 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.34174 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.80144 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.799457 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.76187 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.934921 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.57128 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.935496 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.01056 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.10388 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.07746 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.81186 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  0.986419 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.15933 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.64688 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.91882 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.20732 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.719353 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.67149 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.904334 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.18585 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.604697 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.826697 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.10676 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.84919 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.24802 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.17707 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.811197 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.31612 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.56222 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.872804 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.67023 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.6742 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.622045 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.905118 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.07861 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.957322 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.21036 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.597614 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.6938 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.62037 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.802423 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.657819 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.902106 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.58905 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.764887 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.958526 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.1696 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.17176 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.861176 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  0.998954 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  1.05432 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.40829 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.934096 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.10559 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.806219 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.25342 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.97958 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.36291 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.777347 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.863758 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.16844 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.46013 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.02181 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.1797 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.759033 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.15775 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.727286 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.01036 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.48947 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.20903 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.713343 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.995203 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.945199 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.01455 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.29288 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.718062 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.29653 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.48625 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.785525 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.775776 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.04072 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.47163 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.849224 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.842815 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.10666 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.22719 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.853571 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.929394 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.35199 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.0651 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.971139 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.10143 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.718333 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.10533 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.72577 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.989969 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.645072 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.10625 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.03505 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.7329 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.58504 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.01001 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.36454 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.27074 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.585478 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.757077 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.30797 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.87768 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.866951 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.850745 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.62762 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.957009 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.657524 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.720441 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.27597 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.18131 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.98319 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.757183 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.67927 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.08938 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.21036 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.72026 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.10091 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.888159 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.989041 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 0.95609 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.06425 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.25606 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.00086 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.07855 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.846905 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.43734 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.05114 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.32991 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.8469 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.980364 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.20794 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.30884 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.10243 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.17154 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.799671 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.11656 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.69975 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.08093 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.23063 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.14571 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.844203 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.01472 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.06205 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.93713 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.34744 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.776768 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.12881 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.17457 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.952767 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.695543 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.968589 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.29099 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.756435 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.813683 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.1687 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.02665 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.876498 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.957537 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.16803 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.29547 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.936209 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.15462 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.740017 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.43877 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.901526 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.02063 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.631446 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.937605 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.973617 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.62935 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.28102 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.00304 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.09528 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.35408 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.617932 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.854285 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.36765 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.75476 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.812519 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.979378 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.29727 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.873477 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.912114 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.71695 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.25217 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.18071 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.932178 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.789645 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.922134 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.3426 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.985252 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.20098 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.17739 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.11358 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.988726 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.958153 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.39656 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.17345 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.984818 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.06801 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.686972 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.951594 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.664326 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.845869 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.668755 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.999036 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.01686 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 2.21094 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.91259 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.03541 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.21902 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.03798 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.590153 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.706056 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.79992 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.79216 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.669183 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.759727 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.43273 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.957625 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.717466 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.769831 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.19706 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.53172 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.933195 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.595445 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.711415 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.26476 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.14069 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.63931 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.03683 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.02653 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.835672 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 0.947392 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.05283 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.3903 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.01833 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.13731 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.814675 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.38437 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.969448 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.15351 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.838653 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.892101 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.07963 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.37843 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.08494 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.04124 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.833927 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.12958 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.713178 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.942422 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.17823 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.19202 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.751127 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.021 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.858879 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.868184 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.16118 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.744442 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.3045 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.20622 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.836689 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.715669 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.0738 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.3951 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.839256 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.794798 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.19028 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.27895 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.851346 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   0.979851 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.05698 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.16636 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.997032 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.06435 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.883365 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.14488 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.999538 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.17174 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.845592 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.980269 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.05492 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.38615 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.18779 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.00228 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.901169 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.08963 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.752588 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.932842 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.14311 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.2474 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.811161 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.09078 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.12811 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.06018 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.02291 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.877703 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.32329 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.28562 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.920221 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.835421 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.00346 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.25813 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.884377 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.04208 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.07879 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.02865 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.895174 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.0989999287652199 z_k1_s0000
                                 - 0.135000044679829 z_k1_s0001
                                 - 0.122999746041656 z_k1_s0002
                                 - 0.0829997905809589 z_k1_s0003
                                 - 0.0910002496730451 z_k1_s0004
                                 - 0.0960000491056544 z_k1_s0005
                                 - 0.03699999780114 z_k1_s0006
                                 - 0.116000026836003 z_k1_s0007
                                 - 0.173999631484089 z_k1_s0010
                                 - 0.311998364650042 z_k1_s0011
                                 - 0.545998610133481 z_k1_s0012
                                 - 0.175999485124531 z_k1_s0013
                                 - 0.180999560824409 z_k1_s0014
                                 - 0.773999000296412 z_k1_s0015
                                 - 0.19599949352486 z_k1_s0016
                                 - 0.584997669716782 z_k1_s0017
                                 - 0.154000050414047 z_k1_s0020
                                 - 0.144999709539693 z_k1_s0021
                                 - 0.18199989358048 z_k1_s0022
                                 - 0.120999713540084 z_k1_s0023
                                 - 0.073999911582239 z_k1_s0024
                                 - 0.103999678332083 z_k1_s0025
                                 - 0.0959999839984364 z_k1_s0026
                                 - 0.0369999101411861 z_k1_s0027
                                 - 0.353999384885778 z_k1_s0030
                                 - 0.162999775908327 z_k1_s0031
                                 - 0.295999458785892 z_k1_s0032
                                 - 0.212999613760113 z_k1_s0033
                                 - 0.475999065372011 z_k1_s0034
                                 - 0.154999687660227 z_k1_s0035
                                 - 0.425999703319455 z_k1_s0036
                                 - 0.295999458785892 z_k1_s0037
                                 - 0.517998529332733 z_k1_s0040
                                 - 0.13199978530996 z_k1_s0041
                                 - 0.287999612639754 z_k1_s0042
                                 - 0.186999584405994 z_k1_s0043
                                 - 0.191999667459981 z_k1_s0044
                                 - 0.189999589658473 z_k1_s0045
                                 - 0.597999412397401 z_k1_s0046
                                 - 0.123999697003493 z_k1_s0047
                                 - 0.274001159103459 z_k1_s0050
                                 - 0.0790001565586283 z_k1_s0051
                                 - 0.262000442206924 z_k1_s0052
                                 - 0.29800107189312 z_k1_s0053
                                 - 0.188999883594858 z_k1_s0054
                                 - 0.123000351573802 z_k1_s0055
                                 - 0.129000253721047 z_k1_s0056
                                 - 0.167000090287953 z_k1_s0057
                                 - 0.0559999281697939 z_k1_s0060
                                 - 0.0899997896301212 z_k1_s0061
                                 - 0.116000319522674 z_k1_s0062
                                 - 0.227999467062974 z_k1_s0063
                                 - 0.0549999009739333 z_k1_s0064
                                 - 0.152999642371206 z_k1_s0065
                                 - 0.0970000686010563 z_k1_s0066
                                 - 0.0910000826257149 z_k1_s0067
                                 - 0.233998958116278 z_k1_s0070
                                 - 0.114999686485726 z_k1_s0071
                                 - 0.189999650332532 z_k1_s0072
                                 - 0.382999473422515 z_k1_s0073
                                 - 0.207999073881136 z_k1_s0074
                                 - 0.0949998251662658 z_k1_s0075
                                 - 0.297999417616368 z_k1_s0076
                                 - 0.0899998598364008 z_k1_s0077
                                 - 0.685998707217101 z_k1_s0100
                                 - 1.42699497337975 z_k1_s0101
                                 - 1.02299637669064 z_k1_s0102
                                 - 0.763998017024381 z_k1_s0103
                                 - 0.760998115954703 z_k1_s0104
                                 - 0.975993535275832 z_k1_s0105
                                 - 0.6379984081066 z_k1_s0106
                                 - 3.13499071759615 z_k1_s0107
                                 - 1.13500139865103 z_k1_s0110
                                 - 0.357000231916155 z_k1_s0111
                                 - 0.392000109639259 z_k1_s0112
                                 - 0.975999483485232 z_k1_s0113
                                 - 0.700998818803154 z_k1_s0114
                                 - 0.425000303181147 z_k1_s0115
                                 - 0.430999924624985 z_k1_s0116
                                 - 1.27199825432055 z_k1_s0117
                                 - 0.695996933519675 z_k1_s0120
                                 - 1.39699633916592 z_k1_s0121
                                 - 0.541998566774648 z_k1_s0122
                                 - 0.399998666757072 z_k1_s0123
                                 - 1.13499497252714 z_k1_s0124
                                 - 0.537998455667472 z_k1_s0125
                                 - 1.11199853349755 z_k1_s0126
                                 - 0.401999033409673 z_k1_s0127
                                 - 0.94800695242286 z_k1_s0130
                                 - 0.352001425202664 z_k1_s0131
                                 - 1.08100566229838 z_k1_s0132
                                 - 0.316001467737293 z_k1_s0133
                                 - 0.587002351340128 z_k1_s0134
                                 - 1.2980072470059 z_k1_s0135
                                 - 1.18700822789134 z_k1_s0136
                                 - 0.604002402065747 z_k1_s0137
                                 - 1.54999287536697 z_k1_s0140
                                 - 3.84699375539639 z_k1_s0141
                                 - 1.78000457835881 z_k1_s0142
                                 - 1.04199925878654 z_k1_s0143
                                 - 1.48799714447087 z_k1_s0144
                                 - 1.34099882078152 z_k1_s0145
                                 - 1.0179992130435 z_k1_s0146
                                 - 2.16299220580875 z_k1_s0147
                                 - 0.790998551087641 z_k1_s0150
                                 - 0.401000888894534 z_k1_s0151
                                 - 0.784998753069342 z_k1_s0152
                                 - 0.180000055451047 z_k1_s0153
                                 - 0.464998734423063 z_k1_s0154
                                 - 0.213000023652058 z_k1_s0155
                                 - 0.559999133380685 z_k1_s0156
                                 - 0.202000034251721 z_k1_s0157
                                 - 1.0770004004295 z_k1_s0160
                                 - 0.447001249412615 z_k1_s0161
                                 - 0.493001197144034 z_k1_s0162
                                 - 0.474001431742547 z_k1_s0163
                                 - 0.365001262708547 z_k1_s0164
                                 - 0.372001015119151 z_k1_s0165
                                 - 1.25300467364101 z_k1_s0166
                                 - 1.64300662684917 z_k1_s0167
                                 - 1.17800214971074 z_k1_s0170
                                 - 0.877002586544987 z_k1_s0171
                                 - 0.436000784505816 z_k1_s0172
                                 - 0.842999605127248 z_k1_s0173
                                 - 0.314000896149447 z_k1_s0174
                                 - 1.31800438858416 z_k1_s0175
                                 - 0.791000836207591 z_k1_s0176
                                 - 0.806003888807659 z_k1_s0177
                                 - 0.328000961307949 z_k1_s1000
                                 - 0.154000015735835 z_k1_s1001
                                 - 0.209999503275884 z_k1_s1002
                                 - 0.265000730324997 z_k1_s1003
                                 - 0.293001044095301 z_k1_s1004
                                 - 0.317999356389249 z_k1_s1005
                                 - 0.234999715570765 z_k1_s1006
                                 - 0.0969999877037292 z_k1_s1007
                                 - 0.900997676547081 z_k1_s1010
                                 - 0.472999838861065 z_k1_s1011
                                 - 0.650000274556994 z_k1_s1012
                                 - 0.350999840581083 z_k1_s1013
                                 - 0.816997274217004 z_k1_s1014
                                 - 0.247999832575863 z_k1_s1015
                                 - 0.198999918382999 z_k1_s1016
                                 - 0.206999975014416 z_k1_s1017
                                 - 0.35500018359818 z_k1_s1020
                                 - 0.301999633478684 z_k1_s1021
                                 - 0.35500018359818 z_k1_s1022
                                 - 0.380997856023019 z_k1_s1023
                                 - 0.5989995681682 z_k1_s1024
                                 - 0.351000297687038 z_k1_s1025
                                 - 0.154999861250243 z_k1_s1026
                                 - 0.67399822200091 z_k1_s1027
                                 - 0.190000097249269 z_k1_s1030
                                 - 0.0930000234114939 z_k1_s1031
                                 - 0.100000035057516 z_k1_s1032
                                 - 0.195999406888993 z_k1_s1033
                                 - 0.346999395481868 z_k1_s1034
                                 - 0.107999982710139 z_k1_s1035
                                 - 0.122999942008785 z_k1_s1036
                                 - 0.374999748465837 z_k1_s1037
                                 - 0.967999115178028 z_k1_s1040
                                 - 0.832000079491091 z_k1_s1041
                                 - 0.263999978684716 z_k1_s1042
                                 - 0.40699958397262 z_k1_s1043
                                 - 1.02000002764539 z_k1_s1044
                                 - 0.292999780176777 z_k1_s1045
                                 - 0.434999442378702 z_k1_s1046
                                 - 0.62100047236016 z_k1_s1047
                                 - 0.285000777733286 z_k1_s1050
                                 - 0.362000562524358 z_k1_s1051
                                 - 0.124999981777735 z_k1_s1052
                                 - 0.219000663626634 z_k1_s1053
                                 - 0.424999436444267 z_k1_s1054
                                 - 0.21399929331544 z_k1_s1055
                                 - 0.232999484497667 z_k1_s1056
                                 - 0.226999094124312 z_k1_s1057
                                 - 0.0959999116077642 z_k1_s1060
                                 - 0.11300000657994 z_k1_s1061
                                 - 0.14399995741162 z_k1_s1062
                                 - 0.33199910181036 z_k1_s1063
                                 - 0.159999972679572 z_k1_s1064
                                 - 0.358999093825039 z_k1_s1065
                                 - 0.190999743511305 z_k1_s1066
                                 - 0.40499988022018 z_k1_s1067
                                 - 0.512002114135851 z_k1_s1070
                                 - 0.284000779863543 z_k1_s1071
                                 - 0.288000740262917 z_k1_s1072
                                 - 0.399999304944663 z_k1_s1073
                                 - 0.546999727000833 z_k1_s1074
                                 - 0.527001557507596 z_k1_s1075
                                 - 0.345999349802024 z_k1_s1076
                                 - 0.361002058181033 z_k1_s1077
                                 - 0.139000214520964 z_k1_s1100
                                 - 0.17300049008741 z_k1_s1101
                                 - 0.0640000809247469 z_k1_s1102
                                 - 0.0690000484844636 z_k1_s1103
                                 - 0.173999898838914 z_k1_s1104
                                 - 0.106000233706687 z_k1_s1105
                                 - 0.0710000177883369 z_k1_s1106
                                 - 0.0899999476753004 z_k1_s1107
                                 - 0.0989998812722135 z_k1_s1110
                                 - 0.0369998783544948 z_k1_s1111
                                 - 0.1019998962198 z_k1_s1112
                                 - 0.0529998356753274 z_k1_s1113
                                 - 0.107999926114974 z_k1_s1114
                                 - 0.0899998364294529 z_k1_s1115
                                 - 0.0599998093532148 z_k1_s1116
                                 - 0.062999701901171 z_k1_s1117
                                 - 0.104000334184521 z_k1_s1120
                                 - 0.224000794551436 z_k1_s1121
                                 - 0.165000268503729 z_k1_s1122
                                 - 0.170000247185597 z_k1_s1123
                                 - 0.0800001449109304 z_k1_s1124
                                 - 0.126000629185572 z_k1_s1125
                                 - 0.0460000954738109 z_k1_s1126
                                 - 0.0570001457741287 z_k1_s1127
                                 - 0.18299906145855 z_k1_s1130
                                 - 0.151000096043004 z_k1_s1131
                                 - 0.263999064660538 z_k1_s1132
                                 - 0.10199978785486 z_k1_s1133
                                 - 0.107999761069885 z_k1_s1134
                                 - 0.0919997919632461 z_k1_s1135
                                 - 0.170999601427366 z_k1_s1136
                                 - 0.144999150030246 z_k1_s1137
                                 - 0.0179999883992429 z_k1_s1140
                                 - 0.0409998671761644 z_k1_s1141
                                 - 0.0209999864657834 z_k1_s1142
                                 - 0.0390000812649151 z_k1_s1143
                                 - 0.0189999877547564 z_k1_s1144
                                 - 0.0420000261315112 z_k1_s1145
                                 - 0.0249999838878373 z_k1_s1146
                                 - 0.0609998542864343 z_k1_s1147
                                 - 0.0870001512849261 z_k1_s1150
                                 - 0.244000759135151 z_k1_s1151
                                 - 0.102999863962222 z_k1_s1152
                                 - 0.0830001279154839 z_k1_s1153
                                 - 0.0800000865883726 z_k1_s1154
                                 - 0.0770001404613796 z_k1_s1155
                                 - 0.112000083143674 z_k1_s1156
                                 - 0.166000446231204 z_k1_s1157
                                 - 0.0539998190255028 z_k1_s1160
                                 - 0.073999982664343 z_k1_s1161
                                 - 0.126000096725645 z_k1_s1162
                                 - 0.0399999639780504 z_k1_s1163
                                 - 0.0579997479533765 z_k1_s1164
                                 - 0.0469999693516973 z_k1_s1165
                                 - 0.0629997888630866 z_k1_s1166
                                 - 0.0570001549710118 z_k1_s1167
                                 - 0.0729998584587244 z_k1_s1170
                                 - 0.0629998823684819 z_k1_s1171
                                 - 0.156999921616394 z_k1_s1172
                                 - 0.031999923488776 z_k1_s1173
                                 - 0.0379999751428297 z_k1_s1174
                                 - 0.080999707331102 z_k1_s1175
                                 - 0.110999965601371 z_k1_s1176
                                 - 0.104999913947317 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.958541 w_k1_ADM_s000
                                 + 1.08136 w_k1_AXP_s000
                                 + 0.892612 w_k1_DIS_s000
                                 + 0.934695 w_k1_FDX_s000
                                 + 1.14208 w_k1_IBM_s000
                                 + 0.750119 w_k1_INTC_s000
                                 + 0.940906 w_k1_KO_s000
                                 + 1.10855 w_k1_MAT_s000
                                 + 0.982052 w_k1_MSFT_s000
                                 + 1.03987 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.884854 w_k1_ADM_s000
                                 + 1.00426 w_k1_AXP_s000
                                 + 0.85034 w_k1_DIS_s000
                                 + 0.99927 w_k1_FDX_s000
                                 + 0.715152 w_k1_IBM_s000
                                 + 0.994805 w_k1_INTC_s000
                                 + 0.635351 w_k1_KO_s000
                                 + 0.931317 w_k1_MAT_s000
                                 + 0.949865 w_k1_MSFT_s000
                                 + 0.984976 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.05973 w_k1_ADM_s000
                                 + 1.30827 w_k1_AXP_s000 + 1.3127 w_k1_DIS_s000
                                 + 1.3198 w_k1_FDX_s000 + 1.03935 w_k1_IBM_s000
                                 + 1.33038 w_k1_INTC_s000
                                 + 1.02563 w_k1_KO_s000
                                 + 0.976367 w_k1_MAT_s000
                                 + 1.12452 w_k1_MSFT_s000
                                 + 1.23202 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.723788 w_k1_ADM_s000
                                 + 0.739167 w_k1_AXP_s000
                                 + 0.676017 w_k1_DIS_s000
                                 + 0.709925 w_k1_FDX_s000
                                 + 0.683561 w_k1_IBM_s000
                                 + 0.688096 w_k1_INTC_s000
                                 + 0.68404 w_k1_KO_s000
                                 + 0.919796 w_k1_MAT_s000
                                 + 0.7352 w_k1_MSFT_s000
                                 + 0.863713 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.54124 w_k1_ADM_s000
                                 + 1.2178 w_k1_AXP_s000 + 1.04576 w_k1_DIS_s000
                                 + 1.17702 w_k1_FDX_s000
                                 + 0.941338 w_k1_IBM_s000
                                 + 0.909803 w_k1_INTC_s000
                                 + 0.965312 w_k1_KO_s000
                                 + 1.54206 w_k1_MAT_s000
                                 + 1.04711 w_k1_MSFT_s000
                                 + 1.12058 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.04455 w_k1_ADM_s000
                                 + 1.10197 w_k1_AXP_s000
                                 + 1.09652 w_k1_DIS_s000
                                 + 0.981189 w_k1_FDX_s000
                                 + 1.47139 w_k1_IBM_s000
                                 + 1.11376 w_k1_INTC_s000
                                 + 1.12046 w_k1_KO_s000 + 1.43656 w_k1_MAT_s000
                                 + 0.93089 w_k1_MSFT_s000
                                 + 1.08694 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.28732 w_k1_ADM_s000
                                 + 1.37849 w_k1_AXP_s000
                                 + 1.55338 w_k1_DIS_s000
                                 + 1.15639 w_k1_FDX_s000
                                 + 1.77146 w_k1_IBM_s000
                                 + 1.17847 w_k1_INTC_s000
                                 + 1.49816 w_k1_KO_s000 + 2.01117 w_k1_MAT_s000
                                 + 1.39353 w_k1_MSFT_s000
                                 + 1.22797 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.07764 w_k1_ADM_s000
                                 + 0.823111 w_k1_AXP_s000
                                 + 0.817878 w_k1_DIS_s000
                                 + 0.840638 w_k1_FDX_s000
                                 + 1.19623 w_k1_IBM_s000
                                 + 0.847058 w_k1_INTC_s000
                                 + 0.764254 w_k1_KO_s000
                                 + 0.885482 w_k1_MAT_s000
                                 + 0.827424 w_k1_MSFT_s000
                                 + 0.844453 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.65684 w_k1_ADM_s001
                                 + 0.881091 w_k1_AXP_s001
                                 + 0.767579 w_k1_DIS_s001
                                 + 0.806926 w_k1_FDX_s001
                                 + 0.824424 w_k1_IBM_s001
                                 + 0.763548 w_k1_INTC_s001
                                 + 0.695487 w_k1_KO_s001
                                 + 1.36461 w_k1_MAT_s001
                                 + 0.89189 w_k1_MSFT_s001
                                 + 0.788337 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.692027 w_k1_ADM_s001
                                 + 0.762848 w_k1_AXP_s001
                                 + 0.636054 w_k1_DIS_s001
                                 + 0.757743 w_k1_FDX_s001
                                 + 0.841193 w_k1_IBM_s001
                                 + 0.780528 w_k1_INTC_s001
                                 + 0.808797 w_k1_KO_s001
                                 + 1.26004 w_k1_MAT_s001
                                 + 0.776972 w_k1_MSFT_s001
                                 + 0.862613 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.952178 w_k1_ADM_s001
                                 + 1.57393 w_k1_AXP_s001
                                 + 1.37472 w_k1_DIS_s001
                                 + 1.17458 w_k1_FDX_s001 + 1.307 w_k1_IBM_s001
                                 + 1.17926 w_k1_INTC_s001
                                 + 1.34961 w_k1_KO_s001 + 1.49156 w_k1_MAT_s001
                                 + 1.1686 w_k1_MSFT_s001 + 1.05277 w_k1_PG_s001
                                 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.89726 w_k1_ADM_s001
                                 + 1.00872 w_k1_AXP_s001
                                 + 0.87594 w_k1_DIS_s001
                                 + 0.790075 w_k1_FDX_s001
                                 + 1.45938 w_k1_IBM_s001
                                 + 0.840136 w_k1_INTC_s001
                                 + 1.10303 w_k1_KO_s001 + 1.33892 w_k1_MAT_s001
                                 + 0.979546 w_k1_MSFT_s001
                                 + 1.01311 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.766308 w_k1_ADM_s001
                                 + 0.879997 w_k1_AXP_s001
                                 + 0.784508 w_k1_DIS_s001
                                 + 0.888272 w_k1_FDX_s001
                                 + 0.617742 w_k1_IBM_s001
                                 + 0.943213 w_k1_INTC_s001
                                 + 0.620448 w_k1_KO_s001
                                 + 0.646431 w_k1_MAT_s001
                                 + 0.929174 w_k1_MSFT_s001
                                 + 0.820766 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.06094 w_k1_ADM_s001
                                 + 1.22522 w_k1_AXP_s001
                                 + 1.03207 w_k1_DIS_s001 + 1.0853 w_k1_FDX_s001
                                 + 0.843334 w_k1_IBM_s001
                                 + 1.03221 w_k1_INTC_s001
                                 + 0.80324 w_k1_KO_s001 + 1.05286 w_k1_MAT_s001
                                 + 1.12006 w_k1_MSFT_s001
                                 + 0.899275 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.16765 w_k1_ADM_s001
                                 + 1.64895 w_k1_AXP_s001
                                 + 1.80364 w_k1_DIS_s001
                                 + 1.40973 w_k1_FDX_s001
                                 + 1.15772 w_k1_IBM_s001
                                 + 1.14248 w_k1_INTC_s001
                                 + 1.18239 w_k1_KO_s001 + 1.01318 w_k1_MAT_s001
                                 + 1.21667 w_k1_MSFT_s001
                                 + 1.17834 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.963064 w_k1_ADM_s001
                                 + 1.09426 w_k1_AXP_s001
                                 + 0.957794 w_k1_DIS_s001
                                 + 0.965849 w_k1_FDX_s001
                                 + 1.1425 w_k1_IBM_s001
                                 + 0.88411 w_k1_INTC_s001
                                 + 0.962343 w_k1_KO_s001
                                 + 1.40946 w_k1_MAT_s001
                                 + 0.976258 w_k1_MSFT_s001
                                 + 0.981266 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.12229 w_k1_ADM_s002
                                 + 0.801223 w_k1_AXP_s002
                                 + 1.05838 w_k1_DIS_s002
                                 + 0.955532 w_k1_FDX_s002
                                 + 0.945251 w_k1_IBM_s002
                                 + 0.964802 w_k1_INTC_s002
                                 + 1.15139 w_k1_KO_s002 + 0.83225 w_k1_MAT_s002
                                 + 1.08626 w_k1_MSFT_s002
                                 + 0.985114 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.929723 w_k1_ADM_s002
                                 + 1.19965 w_k1_AXP_s002
                                 + 0.935479 w_k1_DIS_s002
                                 + 0.94545 w_k1_FDX_s002
                                 + 1.28496 w_k1_IBM_s002
                                 + 1.08593 w_k1_INTC_s002 + 1.1257 w_k1_KO_s002
                                 + 1.64126 w_k1_MAT_s002
                                 + 0.977615 w_k1_MSFT_s002
                                 + 1.13282 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.972638 w_k1_ADM_s002
                                 + 1.5908 w_k1_AXP_s002 + 1.61308 w_k1_DIS_s002
                                 + 1.30259 w_k1_FDX_s002 + 1.0457 w_k1_IBM_s002
                                 + 1.15698 w_k1_INTC_s002
                                 + 1.35252 w_k1_KO_s002 + 1.59153 w_k1_MAT_s002
                                 + 1.29129 w_k1_MSFT_s002
                                 + 1.05062 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.15305 w_k1_ADM_s002
                                 + 0.911137 w_k1_AXP_s002
                                 + 1.27709 w_k1_DIS_s002 + 1.0705 w_k1_FDX_s002
                                 + 0.786939 w_k1_IBM_s002
                                 + 1.01029 w_k1_INTC_s002
                                 + 0.998884 w_k1_KO_s002
                                 + 0.665212 w_k1_MAT_s002
                                 + 1.19152 w_k1_MSFT_s002
                                 + 0.908977 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.851638 w_k1_ADM_s002
                                 + 0.910708 w_k1_AXP_s002
                                 + 0.885751 w_k1_DIS_s002
                                 + 0.891829 w_k1_FDX_s002
                                 + 0.657947 w_k1_IBM_s002
                                 + 0.765336 w_k1_INTC_s002
                                 + 0.859447 w_k1_KO_s002
                                 + 0.830529 w_k1_MAT_s002
                                 + 0.8147 w_k1_MSFT_s002
                                 + 0.895708 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.667498 w_k1_ADM_s002
                                 + 0.784596 w_k1_AXP_s002
                                 + 0.648346 w_k1_DIS_s002
                                 + 0.754505 w_k1_FDX_s002
                                 + 0.723824 w_k1_IBM_s002
                                 + 0.833238 w_k1_INTC_s002
                                 + 0.765932 w_k1_KO_s002
                                 + 1.15475 w_k1_MAT_s002
                                 + 0.873782 w_k1_MSFT_s002
                                 + 0.824902 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.22988 w_k1_ADM_s002
                                 + 1.02527 w_k1_AXP_s002
                                 + 0.952696 w_k1_DIS_s002
                                 + 1.04031 w_k1_FDX_s002
                                 + 0.71419 w_k1_IBM_s002 + 1.012 w_k1_INTC_s002
                                 + 0.846505 w_k1_KO_s002
                                 + 0.573165 w_k1_MAT_s002
                                 + 1.01477 w_k1_MSFT_s002
                                 + 0.888668 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.00078 w_k1_ADM_s002
                                 + 0.757714 w_k1_AXP_s002
                                 + 0.770959 w_k1_DIS_s002
                                 + 0.842659 w_k1_FDX_s002
                                 + 0.497659 w_k1_IBM_s002
                                 + 0.708124 w_k1_INTC_s002
                                 + 0.665081 w_k1_KO_s002
                                 + 0.637824 w_k1_MAT_s002
                                 + 0.964993 w_k1_MSFT_s002
                                 + 0.830394 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.814504 w_k1_ADM_s003
                                 + 0.860117 w_k1_AXP_s003
                                 + 0.991127 w_k1_DIS_s003
                                 + 0.948488 w_k1_FDX_s003
                                 + 0.80767 w_k1_IBM_s003
                                 + 1.01928 w_k1_INTC_s003
                                 + 0.950258 w_k1_KO_s003
                                 + 0.660721 w_k1_MAT_s003
                                 + 0.937357 w_k1_MSFT_s003
                                 + 0.917221 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.21051 w_k1_ADM_s003
                                 + 1.4818 w_k1_AXP_s003 + 1.27968 w_k1_DIS_s003
                                 + 1.21317 w_k1_FDX_s003
                                 + 1.45117 w_k1_IBM_s003
                                 + 1.24938 w_k1_INTC_s003
                                 + 1.39062 w_k1_KO_s003 + 1.75649 w_k1_MAT_s003
                                 + 1.1189 w_k1_MSFT_s003 + 1.16797 w_k1_PG_s003
                                 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.814697 w_k1_ADM_s003
                                 + 0.977243 w_k1_AXP_s003
                                 + 0.869183 w_k1_DIS_s003
                                 + 0.782649 w_k1_FDX_s003
                                 + 1.47449 w_k1_IBM_s003
                                 + 1.01385 w_k1_INTC_s003
                                 + 1.13783 w_k1_KO_s003 + 1.62821 w_k1_MAT_s003
                                 + 0.776408 w_k1_MSFT_s003
                                 + 1.07119 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.50644 w_k1_ADM_s003
                                 + 1.38133 w_k1_AXP_s003
                                 + 1.83394 w_k1_DIS_s003
                                 + 1.34967 w_k1_FDX_s003
                                 + 1.17039 w_k1_IBM_s003
                                 + 1.38944 w_k1_INTC_s003
                                 + 1.40165 w_k1_KO_s003 + 0.91943 w_k1_MAT_s003
                                 + 1.27012 w_k1_MSFT_s003 + 1.2869 w_k1_PG_s003
                                 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.823579 w_k1_ADM_s003
                                 + 0.89779 w_k1_AXP_s003
                                 + 0.731985 w_k1_DIS_s003
                                 + 0.830864 w_k1_FDX_s003
                                 + 0.92409 w_k1_IBM_s003
                                 + 0.955721 w_k1_INTC_s003
                                 + 0.842149 w_k1_KO_s003
                                 + 0.967624 w_k1_MAT_s003
                                 + 0.798565 w_k1_MSFT_s003
                                 + 0.930098 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.36967 w_k1_ADM_s003
                                 + 1.21578 w_k1_AXP_s003
                                 + 1.10338 w_k1_DIS_s003
                                 + 1.31509 w_k1_FDX_s003
                                 + 0.721082 w_k1_IBM_s003
                                 + 1.20314 w_k1_INTC_s003
                                 + 0.957359 w_k1_KO_s003
                                 + 0.659754 w_k1_MAT_s003
                                 + 1.08847 w_k1_MSFT_s003
                                 + 1.03563 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.04337 w_k1_ADM_s003
                                 + 0.962937 w_k1_AXP_s003
                                 + 0.986261 w_k1_DIS_s003
                                 + 0.944341 w_k1_FDX_s003
                                 + 1.27685 w_k1_IBM_s003
                                 + 0.906029 w_k1_INTC_s003
                                 + 1.09824 w_k1_KO_s003 + 1.39499 w_k1_MAT_s003
                                 + 0.938402 w_k1_MSFT_s003
                                 + 1.06609 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.2622 w_k1_ADM_s003
                                 + 1.17281 w_k1_AXP_s003
                                 + 0.984349 w_k1_DIS_s003
                                 + 1.05909 w_k1_FDX_s003
                                 + 0.929291 w_k1_IBM_s003
                                 + 1.10822 w_k1_INTC_s003
                                 + 1.05912 w_k1_KO_s003 + 1.08586 w_k1_MAT_s003
                                 + 0.984883 w_k1_MSFT_s003
                                 + 1.06726 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.841568 w_k1_ADM_s004
                                 + 0.737876 w_k1_AXP_s004
                                 + 0.900406 w_k1_DIS_s004
                                 + 0.853157 w_k1_FDX_s004
                                 + 1.07633 w_k1_IBM_s004
                                 + 0.967404 w_k1_INTC_s004
                                 + 0.991282 w_k1_KO_s004
                                 + 0.987411 w_k1_MAT_s004
                                 + 1.02578 w_k1_MSFT_s004
                                 + 0.901044 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.13567 w_k1_ADM_s004
                                 + 1.20844 w_k1_AXP_s004
                                 + 0.779718 w_k1_DIS_s004 + 1.08 w_k1_FDX_s004
                                 + 0.829756 w_k1_IBM_s004
                                 + 0.843181 w_k1_INTC_s004
                                 + 0.668697 w_k1_KO_s004
                                 + 0.896192 w_k1_MAT_s004
                                 + 0.96861 w_k1_MSFT_s004
                                 + 0.967348 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.837464 w_k1_ADM_s004
                                 + 0.708437 w_k1_AXP_s004
                                 + 0.641926 w_k1_DIS_s004
                                 + 0.731743 w_k1_FDX_s004
                                 + 0.785497 w_k1_IBM_s004
                                 + 0.873756 w_k1_INTC_s004
                                 + 0.740278 w_k1_KO_s004
                                 + 0.593164 w_k1_MAT_s004
                                 + 0.726833 w_k1_MSFT_s004
                                 + 0.768279 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.16222 w_k1_ADM_s004
                                 + 0.928895 w_k1_AXP_s004
                                 + 0.958154 w_k1_DIS_s004
                                 + 1.05902 w_k1_FDX_s004
                                 + 0.771223 w_k1_IBM_s004
                                 + 1.22112 w_k1_INTC_s004
                                 + 0.984154 w_k1_KO_s004
                                 + 0.627801 w_k1_MAT_s004
                                 + 1.09148 w_k1_MSFT_s004
                                 + 0.987155 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.09539 w_k1_ADM_s004
                                 + 0.865339 w_k1_AXP_s004
                                 + 1.56665 w_k1_DIS_s004
                                 + 1.03681 w_k1_FDX_s004
                                 + 1.17626 w_k1_IBM_s004
                                 + 1.07457 w_k1_INTC_s004
                                 + 1.31031 w_k1_KO_s004 + 1.03691 w_k1_MAT_s004
                                 + 1.236 w_k1_MSFT_s004 + 0.995544 w_k1_PG_s004
                                 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.38055 w_k1_ADM_s004
                                 + 1.25969 w_k1_AXP_s004
                                 + 1.30799 w_k1_DIS_s004
                                 + 1.27644 w_k1_FDX_s004
                                 + 0.935775 w_k1_IBM_s004
                                 + 1.2881 w_k1_INTC_s004 + 1.18745 w_k1_KO_s004
                                 + 0.793125 w_k1_MAT_s004
                                 + 1.13629 w_k1_MSFT_s004
                                 + 1.12676 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.98841 w_k1_ADM_s004
                                 + 1.20478 w_k1_AXP_s004
                                 + 1.13388 w_k1_DIS_s004
                                 + 1.07651 w_k1_FDX_s004
                                 + 1.23308 w_k1_IBM_s004
                                 + 0.951879 w_k1_INTC_s004
                                 + 1.1183 w_k1_KO_s004 + 1.57976 w_k1_MAT_s004
                                 + 0.996934 w_k1_MSFT_s004
                                 + 1.08879 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.13835 w_k1_ADM_s004
                                 + 1.69677 w_k1_AXP_s004
                                 + 1.63071 w_k1_DIS_s004
                                 + 1.26699 w_k1_FDX_s004
                                 + 1.74322 w_k1_IBM_s004
                                 + 1.27868 w_k1_INTC_s004
                                 + 1.61999 w_k1_KO_s004 + 2.43428 w_k1_MAT_s004
                                 + 1.30151 w_k1_MSFT_s004
                                 + 1.31611 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.908636 w_k1_ADM_s005
                                 + 1.3911 w_k1_AXP_s005 + 1.01294 w_k1_DIS_s005
                                 + 1.09936 w_k1_FDX_s005
                                 + 0.992682 w_k1_IBM_s005
                                 + 1.07748 w_k1_INTC_s005
                                 + 1.06053 w_k1_KO_s005 + 1.37613 w_k1_MAT_s005
                                 + 1.12587 w_k1_MSFT_s005
                                 + 1.08873 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.874704 w_k1_ADM_s005
                                 + 0.682789 w_k1_AXP_s005
                                 + 0.606875 w_k1_DIS_s005
                                 + 0.680267 w_k1_FDX_s005
                                 + 0.720709 w_k1_IBM_s005
                                 + 0.642254 w_k1_INTC_s005
                                 + 0.735832 w_k1_KO_s005
                                 + 0.712499 w_k1_MAT_s005
                                 + 0.688224 w_k1_MSFT_s005
                                 + 0.870315 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.932586 w_k1_ADM_s005
                                 + 0.938166 w_k1_AXP_s005
                                 + 0.793184 w_k1_DIS_s005
                                 + 0.927988 w_k1_FDX_s005
                                 + 0.81209 w_k1_IBM_s005
                                 + 0.960988 w_k1_INTC_s005
                                 + 0.853988 w_k1_KO_s005
                                 + 0.85483 w_k1_MAT_s005
                                 + 0.948632 w_k1_MSFT_s005
                                 + 0.985041 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.06319 w_k1_ADM_s005
                                 + 1.10082 w_k1_AXP_s005 + 1.2041 w_k1_DIS_s005
                                 + 1.05646 w_k1_FDX_s005
                                 + 1.22156 w_k1_IBM_s005
                                 + 1.0962 w_k1_INTC_s005 + 1.31995 w_k1_KO_s005
                                 + 1.1376 w_k1_MAT_s005
                                 + 1.17007 w_k1_MSFT_s005
                                 + 1.14884 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.906454 w_k1_ADM_s005
                                 + 0.665158 w_k1_AXP_s005
                                 + 0.999908 w_k1_DIS_s005
                                 + 0.903894 w_k1_FDX_s005
                                 + 1.00601 w_k1_IBM_s005
                                 + 1.03436 w_k1_INTC_s005
                                 + 1.10311 w_k1_KO_s005 + 0.54536 w_k1_MAT_s005
                                 + 1.1023 w_k1_MSFT_s005
                                 + 0.920027 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.953829 w_k1_ADM_s005
                                 + 1.86738 w_k1_AXP_s005
                                 + 1.56148 w_k1_DIS_s005
                                 + 1.30058 w_k1_FDX_s005
                                 + 1.47519 w_k1_IBM_s005
                                 + 1.04855 w_k1_INTC_s005
                                 + 1.47739 w_k1_KO_s005 + 1.6317 w_k1_MAT_s005
                                 + 1.29635 w_k1_MSFT_s005
                                 + 1.27688 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.969213 w_k1_ADM_s005
                                 + 1.03921 w_k1_AXP_s005
                                 + 0.97259 w_k1_DIS_s005
                                 + 0.98825 w_k1_FDX_s005
                                 + 1.24473 w_k1_IBM_s005
                                 + 0.9554 w_k1_INTC_s005 + 1.09494 w_k1_KO_s005
                                 + 1.21653 w_k1_MAT_s005
                                 + 0.987527 w_k1_MSFT_s005
                                 + 1.07371 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.96111 w_k1_ADM_s005
                                 + 0.970849 w_k1_AXP_s005
                                 + 0.803573 w_k1_DIS_s005
                                 + 0.777602 w_k1_FDX_s005
                                 + 0.986765 w_k1_IBM_s005
                                 + 0.791581 w_k1_INTC_s005
                                 + 1.02735 w_k1_KO_s005 + 1.05237 w_k1_MAT_s005
                                 + 0.887402 w_k1_MSFT_s005
                                 + 0.9593 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.665064 w_k1_ADM_s006
                                 + 0.741768 w_k1_AXP_s006
                                 + 0.630365 w_k1_DIS_s006
                                 + 0.693658 w_k1_FDX_s006
                                 + 1.12269 w_k1_IBM_s006
                                 + 0.817114 w_k1_INTC_s006
                                 + 0.733958 w_k1_KO_s006
                                 + 0.824023 w_k1_MAT_s006
                                 + 0.679572 w_k1_MSFT_s006
                                 + 0.767042 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.02107 w_k1_ADM_s006
                                 + 0.794572 w_k1_AXP_s006
                                 + 0.74375 w_k1_DIS_s006
                                 + 0.880669 w_k1_FDX_s006
                                 + 0.616198 w_k1_IBM_s006
                                 + 0.956137 w_k1_INTC_s006
                                 + 0.815498 w_k1_KO_s006
                                 + 0.846675 w_k1_MAT_s006
                                 + 0.844973 w_k1_MSFT_s006
                                 + 1.01854 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.03632 w_k1_ADM_s006
                                 + 1.16214 w_k1_AXP_s006
                                 + 1.51796 w_k1_DIS_s006
                                 + 1.20357 w_k1_FDX_s006
                                 + 0.985666 w_k1_IBM_s006
                                 + 1.16111 w_k1_INTC_s006
                                 + 1.24001 w_k1_KO_s006 + 1.19498 w_k1_MAT_s006
                                 + 1.2666 w_k1_MSFT_s006 + 1.0986 w_k1_PG_s006
                                 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.06611 w_k1_ADM_s006
                                 + 1.07811 w_k1_AXP_s006
                                 + 0.916494 w_k1_DIS_s006
                                 + 1.04256 w_k1_FDX_s006
                                 + 0.913838 w_k1_IBM_s006
                                 + 1.07483 w_k1_INTC_s006
                                 + 0.844848 w_k1_KO_s006
                                 + 0.685823 w_k1_MAT_s006
                                 + 0.953828 w_k1_MSFT_s006
                                 + 1.06563 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.9494 w_k1_ADM_s006
                                 + 1.14455 w_k1_AXP_s006
                                 + 0.978843 w_k1_DIS_s006
                                 + 0.901045 w_k1_FDX_s006
                                 + 1.88397 w_k1_IBM_s006
                                 + 1.30953 w_k1_INTC_s006
                                 + 1.42764 w_k1_KO_s006 + 1.59157 w_k1_MAT_s006
                                 + 1.26169 w_k1_MSFT_s006
                                 + 1.21681 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.09916 w_k1_ADM_s006
                                 + 1.1029 w_k1_AXP_s006
                                 + 0.943916 w_k1_DIS_s006
                                 + 0.939742 w_k1_FDX_s006
                                 + 1.32882 w_k1_IBM_s006
                                 + 1.11792 w_k1_INTC_s006
                                 + 1.10116 w_k1_KO_s006 + 1.24855 w_k1_MAT_s006
                                 + 0.88811 w_k1_MSFT_s006
                                 + 0.961648 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.0708 w_k1_ADM_s006
                                 + 1.702 w_k1_AXP_s006 + 1.60852 w_k1_DIS_s006
                                 + 1.34111 w_k1_FDX_s006
                                 + 1.34345 w_k1_IBM_s006
                                 + 1.36995 w_k1_INTC_s006
                                 + 1.30336 w_k1_KO_s006 + 1.03534 w_k1_MAT_s006
                                 + 1.50338 w_k1_MSFT_s006
                                 + 1.17552 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.852194 w_k1_ADM_s006
                                 + 0.844999 w_k1_AXP_s006
                                 + 0.639112 w_k1_DIS_s006
                                 + 0.742092 w_k1_FDX_s006
                                 + 0.808733 w_k1_IBM_s006
                                 + 0.848551 w_k1_INTC_s006
                                 + 0.732146 w_k1_KO_s006
                                 + 0.597627 w_k1_MAT_s006
                                 + 0.790869 w_k1_MSFT_s006
                                 + 0.796599 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.28443 w_k1_ADM_s007
                                 + 0.736027 w_k1_AXP_s007
                                 + 1.11055 w_k1_DIS_s007
                                 + 1.04144 w_k1_FDX_s007
                                 + 0.803328 w_k1_IBM_s007
                                 + 1.00213 w_k1_INTC_s007
                                 + 0.937255 w_k1_KO_s007
                                 + 0.511701 w_k1_MAT_s007
                                 + 0.955073 w_k1_MSFT_s007
                                 + 1.01027 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.45019 w_k1_ADM_s007
                                 + 1.74952 w_k1_AXP_s007
                                 + 1.29474 w_k1_DIS_s007
                                 + 1.29515 w_k1_FDX_s007
                                 + 1.63541 w_k1_IBM_s007
                                 + 1.05909 w_k1_INTC_s007
                                 + 1.19472 w_k1_KO_s007 + 1.32359 w_k1_MAT_s007
                                 + 1.02822 w_k1_MSFT_s007
                                 + 0.979522 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.3033 w_k1_ADM_s007
                                 + 1.15433 w_k1_AXP_s007
                                 + 1.36307 w_k1_DIS_s007
                                 + 1.23027 w_k1_FDX_s007
                                 + 1.34308 w_k1_IBM_s007
                                 + 1.31614 w_k1_INTC_s007
                                 + 1.42795 w_k1_KO_s007
                                 + 0.862494 w_k1_MAT_s007
                                 + 1.16559 w_k1_MSFT_s007
                                 + 1.13518 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.1803 w_k1_ADM_s007
                                 + 1.19731 w_k1_AXP_s007
                                 + 0.959362 w_k1_DIS_s007
                                 + 1.01377 w_k1_FDX_s007
                                 + 1.09634 w_k1_IBM_s007
                                 + 1.03623 w_k1_INTC_s007
                                 + 1.00683 w_k1_KO_s007 + 1.15433 w_k1_MAT_s007
                                 + 0.963853 w_k1_MSFT_s007
                                 + 0.950036 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.966492 w_k1_ADM_s007
                                 + 0.774139 w_k1_AXP_s007
                                 + 0.68381 w_k1_DIS_s007
                                 + 0.766768 w_k1_FDX_s007
                                 + 0.812876 w_k1_IBM_s007
                                 + 0.928203 w_k1_INTC_s007
                                 + 0.764955 w_k1_KO_s007
                                 + 0.461166 w_k1_MAT_s007
                                 + 0.862488 w_k1_MSFT_s007
                                 + 0.848964 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.00566 w_k1_ADM_s007
                                 + 0.904269 w_k1_AXP_s007
                                 + 0.738869 w_k1_DIS_s007
                                 + 0.940701 w_k1_FDX_s007
                                 + 0.590228 w_k1_IBM_s007
                                 + 0.897363 w_k1_INTC_s007
                                 + 0.771791 w_k1_KO_s007
                                 + 0.732328 w_k1_MAT_s007
                                 + 0.844865 w_k1_MSFT_s007
                                 + 0.915126 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.00294 w_k1_ADM_s007
                                 + 1.05949 w_k1_AXP_s007
                                 + 0.77249 w_k1_DIS_s007
                                 + 0.874632 w_k1_FDX_s007
                                 + 0.96429 w_k1_IBM_s007
                                 + 0.941027 w_k1_INTC_s007
                                 + 0.82088 w_k1_KO_s007
                                 + 0.823374 w_k1_MAT_s007
                                 + 0.829093 w_k1_MSFT_s007
                                 + 0.868196 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.69959 w_k1_ADM_s007
                                 + 1.48341 w_k1_AXP_s007
                                 + 1.30595 w_k1_DIS_s007
                                 + 1.40666 w_k1_FDX_s007
                                 + 0.854463 w_k1_IBM_s007
                                 + 1.20439 w_k1_INTC_s007
                                 + 0.973206 w_k1_KO_s007
                                 + 0.509775 w_k1_MAT_s007
                                 + 1.1459 w_k1_MSFT_s007 + 1.06508 w_k1_PG_s007
                                 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.41441 w_k1_ADM_s010
                                 + 1.71635 w_k1_AXP_s010
                                 + 1.44077 w_k1_DIS_s010
                                 + 1.42945 w_k1_FDX_s010
                                 + 1.27259 w_k1_IBM_s010
                                 + 1.20033 w_k1_INTC_s010
                                 + 1.24193 w_k1_KO_s010 + 1.27971 w_k1_MAT_s010
                                 + 1.41951 w_k1_MSFT_s010
                                 + 1.34225 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.89522 w_k1_ADM_s010
                                 + 1.11906 w_k1_AXP_s010
                                 + 0.996782 w_k1_DIS_s010
                                 + 1.14695 w_k1_FDX_s010
                                 + 0.791836 w_k1_IBM_s010
                                 + 1.0212 w_k1_INTC_s010
                                 + 0.944333 w_k1_KO_s010
                                 + 1.15958 w_k1_MAT_s010
                                 + 0.982059 w_k1_MSFT_s010
                                 + 0.97568 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.92335 w_k1_ADM_s010
                                 + 1.29091 w_k1_AXP_s010
                                 + 0.963312 w_k1_DIS_s010
                                 + 1.06499 w_k1_FDX_s010
                                 + 1.27498 w_k1_IBM_s010
                                 + 0.960388 w_k1_INTC_s010
                                 + 1.1377 w_k1_KO_s010 + 1.34314 w_k1_MAT_s010
                                 + 0.952662 w_k1_MSFT_s010
                                 + 1.08686 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.842592 w_k1_ADM_s010
                                 + 1.216 w_k1_AXP_s010 + 1.05737 w_k1_DIS_s010
                                 + 0.89255 w_k1_FDX_s010
                                 + 1.61806 w_k1_IBM_s010
                                 + 1.03464 w_k1_INTC_s010
                                 + 1.32834 w_k1_KO_s010 + 1.66034 w_k1_MAT_s010
                                 + 0.958162 w_k1_MSFT_s010
                                 + 1.02389 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.32121 w_k1_ADM_s010
                                 + 0.854146 w_k1_AXP_s010
                                 + 1.31085 w_k1_DIS_s010
                                 + 1.23336 w_k1_FDX_s010
                                 + 0.672117 w_k1_IBM_s010
                                 + 1.01374 w_k1_INTC_s010
                                 + 1.17334 w_k1_KO_s010
                                 + 0.743908 w_k1_MAT_s010
                                 + 1.3364 w_k1_MSFT_s010 + 1.09626 w_k1_PG_s010
                                 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.832347 w_k1_ADM_s010
                                 + 1.11865 w_k1_AXP_s010 + 1.2928 w_k1_DIS_s010
                                 + 1.08176 w_k1_FDX_s010 + 1.1036 w_k1_IBM_s010
                                 + 0.924941 w_k1_INTC_s010
                                 + 1.28063 w_k1_KO_s010
                                 + 0.981156 w_k1_MAT_s010
                                 + 1.1456 w_k1_MSFT_s010 + 1.05186 w_k1_PG_s010
                                 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.869983 w_k1_ADM_s010
                                 + 0.793507 w_k1_AXP_s010
                                 + 0.625506 w_k1_DIS_s010
                                 + 0.815665 w_k1_FDX_s010
                                 + 0.577592 w_k1_IBM_s010
                                 + 0.804926 w_k1_INTC_s010
                                 + 0.76776 w_k1_KO_s010
                                 + 0.610229 w_k1_MAT_s010
                                 + 0.894495 w_k1_MSFT_s010
                                 + 0.875964 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.781282 w_k1_ADM_s010
                                 + 0.901015 w_k1_AXP_s010
                                 + 0.786085 w_k1_DIS_s010
                                 + 0.829697 w_k1_FDX_s010
                                 + 0.937334 w_k1_IBM_s010
                                 + 0.83938 w_k1_INTC_s010
                                 + 0.939995 w_k1_KO_s010
                                 + 1.04954 w_k1_MAT_s010
                                 + 0.903183 w_k1_MSFT_s010
                                 + 0.897294 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.0591 w_k1_ADM_s011
                                 + 1.49572 w_k1_AXP_s011
                                 + 1.28109 w_k1_DIS_s011
                                 + 1.23616 w_k1_FDX_s011 + 1.2773 w_k1_IBM_s011
                                 + 1.3378 w_k1_INTC_s011 + 1.16139 w_k1_KO_s011
                                 + 1.27313 w_k1_MAT_s011
                                 + 1.14262 w_k1_MSFT_s011
                                 + 1.17661 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.32638 w_k1_ADM_s011
                                 + 1.29534 w_k1_AXP_s011
                                 + 1.21378 w_k1_DIS_s011
                                 + 1.29138 w_k1_FDX_s011
                                 + 0.980739 w_k1_IBM_s011
                                 + 1.40592 w_k1_INTC_s011
                                 + 1.05814 w_k1_KO_s011
                                 + 0.913317 w_k1_MAT_s011
                                 + 1.25155 w_k1_MSFT_s011
                                 + 1.12569 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.911947 w_k1_ADM_s011
                                 + 0.783519 w_k1_AXP_s011
                                 + 0.75426 w_k1_DIS_s011
                                 + 0.917195 w_k1_FDX_s011
                                 + 0.561007 w_k1_IBM_s011
                                 + 0.84757 w_k1_INTC_s011
                                 + 0.661275 w_k1_KO_s011
                                 + 0.690675 w_k1_MAT_s011
                                 + 0.962166 w_k1_MSFT_s011
                                 + 0.862042 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.813314 w_k1_ADM_s011
                                 + 0.82899 w_k1_AXP_s011
                                 + 0.659272 w_k1_DIS_s011
                                 + 0.811954 w_k1_FDX_s011
                                 + 0.766236 w_k1_IBM_s011
                                 + 0.906555 w_k1_INTC_s011
                                 + 0.756201 w_k1_KO_s011
                                 + 0.778586 w_k1_MAT_s011
                                 + 0.76151 w_k1_MSFT_s011
                                 + 0.862365 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.899116 w_k1_ADM_s011
                                 + 0.801958 w_k1_AXP_s011
                                 + 0.684557 w_k1_DIS_s011
                                 + 0.788767 w_k1_FDX_s011
                                 + 1.09364 w_k1_IBM_s011
                                 + 0.936502 w_k1_INTC_s011
                                 + 0.846132 w_k1_KO_s011
                                 + 0.71155 w_k1_MAT_s011
                                 + 0.860379 w_k1_MSFT_s011
                                 + 0.923553 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.981904 w_k1_ADM_s011
                                 + 1.22952 w_k1_AXP_s011
                                 + 1.09316 w_k1_DIS_s011
                                 + 0.859467 w_k1_FDX_s011
                                 + 1.55121 w_k1_IBM_s011
                                 + 1.01647 w_k1_INTC_s011
                                 + 1.08865 w_k1_KO_s011 + 1.49727 w_k1_MAT_s011
                                 + 0.894244 w_k1_MSFT_s011
                                 + 0.949664 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.26437 w_k1_ADM_s011
                                 + 1.18637 w_k1_AXP_s011
                                 + 1.16623 w_k1_DIS_s011
                                 + 1.32558 w_k1_FDX_s011
                                 + 0.860215 w_k1_IBM_s011
                                 + 1.11638 w_k1_INTC_s011
                                 + 0.91652 w_k1_KO_s011 + 0.96546 w_k1_MAT_s011
                                 + 0.996319 w_k1_MSFT_s011
                                 + 1.12346 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.96936 w_k1_ADM_s011
                                 + 1.12965 w_k1_AXP_s011
                                 + 1.03354 w_k1_DIS_s011
                                 + 1.05437 w_k1_FDX_s011
                                 + 0.850593 w_k1_IBM_s011
                                 + 1.10801 w_k1_INTC_s011
                                 + 0.902128 w_k1_KO_s011
                                 + 0.775562 w_k1_MAT_s011
                                 + 1.05064 w_k1_MSFT_s011 + 1.0177 w_k1_PG_s011
                                 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.898636 w_k1_ADM_s012
                                 + 1.15755 w_k1_AXP_s012
                                 + 1.39179 w_k1_DIS_s012
                                 + 1.15564 w_k1_FDX_s012
                                 + 1.17886 w_k1_IBM_s012
                                 + 1.29414 w_k1_INTC_s012
                                 + 1.14432 w_k1_KO_s012 + 1.31467 w_k1_MAT_s012
                                 + 1.33513 w_k1_MSFT_s012
                                 + 1.13475 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.10645 w_k1_ADM_s012
                                 + 0.852951 w_k1_AXP_s012
                                 + 0.857599 w_k1_DIS_s012
                                 + 0.999178 w_k1_FDX_s012
                                 + 0.813991 w_k1_IBM_s012
                                 + 0.8766 w_k1_INTC_s012
                                 + 0.756712 w_k1_KO_s012
                                 + 0.96499 w_k1_MAT_s012
                                 + 0.886674 w_k1_MSFT_s012
                                 + 0.947426 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.743387 w_k1_ADM_s012
                                 + 0.623588 w_k1_AXP_s012
                                 + 0.599799 w_k1_DIS_s012
                                 + 0.673503 w_k1_FDX_s012
                                 + 0.797804 w_k1_IBM_s012
                                 + 0.815361 w_k1_INTC_s012
                                 + 0.683631 w_k1_KO_s012
                                 + 0.763678 w_k1_MAT_s012
                                 + 0.711779 w_k1_MSFT_s012
                                 + 0.752856 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.909866 w_k1_ADM_s012
                                 + 1.19861 w_k1_AXP_s012
                                 + 1.28923 w_k1_DIS_s012
                                 + 1.10577 w_k1_FDX_s012
                                 + 1.61154 w_k1_IBM_s012
                                 + 0.987589 w_k1_INTC_s012
                                 + 1.37305 w_k1_KO_s012 + 1.89667 w_k1_MAT_s012
                                 + 1.21969 w_k1_MSFT_s012
                                 + 1.23988 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.708841 w_k1_ADM_s012
                                 + 0.799635 w_k1_AXP_s012
                                 + 0.82485 w_k1_DIS_s012
                                 + 0.802859 w_k1_FDX_s012
                                 + 1.16169 w_k1_IBM_s012
                                 + 0.981135 w_k1_INTC_s012
                                 + 0.942512 w_k1_KO_s012
                                 + 1.22723 w_k1_MAT_s012
                                 + 0.868176 w_k1_MSFT_s012
                                 + 0.898461 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.995687 w_k1_ADM_s012
                                 + 0.894528 w_k1_AXP_s012
                                 + 1.33736 w_k1_DIS_s012
                                 + 1.26939 w_k1_FDX_s012
                                 + 0.665528 w_k1_IBM_s012
                                 + 0.974209 w_k1_INTC_s012
                                 + 0.89782 w_k1_KO_s012
                                 + 0.666601 w_k1_MAT_s012
                                 + 1.16883 w_k1_MSFT_s012
                                 + 1.00501 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.940658 w_k1_ADM_s012
                                 + 0.88205 w_k1_AXP_s012
                                 + 1.08507 w_k1_DIS_s012
                                 + 0.985031 w_k1_FDX_s012
                                 + 0.99698 w_k1_IBM_s012
                                 + 1.0486 w_k1_INTC_s012
                                 + 0.950534 w_k1_KO_s012 + 1.0905 w_k1_MAT_s012
                                 + 1.05043 w_k1_MSFT_s012
                                 + 1.08104 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.34858 w_k1_ADM_s012
                                 + 1.57215 w_k1_AXP_s012
                                 + 1.84866 w_k1_DIS_s012
                                 + 1.47476 w_k1_FDX_s012
                                 + 1.11188 w_k1_IBM_s012
                                 + 1.31969 w_k1_INTC_s012
                                 + 1.24042 w_k1_KO_s012 + 1.57419 w_k1_MAT_s012
                                 + 1.34707 w_k1_MSFT_s012 + 1.3348 w_k1_PG_s012
                                 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.11884 w_k1_ADM_s013
                                 + 0.871866 w_k1_AXP_s013
                                 + 0.73504 w_k1_DIS_s013
                                 + 0.935482 w_k1_FDX_s013
                                 + 0.80291 w_k1_IBM_s013
                                 + 0.943473 w_k1_INTC_s013
                                 + 0.847619 w_k1_KO_s013
                                 + 0.672687 w_k1_MAT_s013
                                 + 0.942349 w_k1_MSFT_s013
                                 + 0.899226 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.70373 w_k1_ADM_s013
                                 + 1.6386 w_k1_AXP_s013 + 2.09504 w_k1_DIS_s013
                                 + 1.44212 w_k1_FDX_s013
                                 + 1.47683 w_k1_IBM_s013
                                 + 1.43976 w_k1_INTC_s013
                                 + 1.68755 w_k1_KO_s013 + 1.10393 w_k1_MAT_s013
                                 + 1.68707 w_k1_MSFT_s013
                                 + 1.36673 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.23939 w_k1_ADM_s013
                                 + 1.24561 w_k1_AXP_s013 + 1.186 w_k1_DIS_s013
                                 + 1.02619 w_k1_FDX_s013
                                 + 1.11874 w_k1_IBM_s013
                                 + 1.21855 w_k1_INTC_s013
                                 + 1.18797 w_k1_KO_s013 + 1.20152 w_k1_MAT_s013
                                 + 1.03411 w_k1_MSFT_s013
                                 + 1.04442 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.35378 w_k1_ADM_s013
                                 + 0.903175 w_k1_AXP_s013
                                 + 1.15648 w_k1_DIS_s013
                                 + 1.08406 w_k1_FDX_s013
                                 + 0.757251 w_k1_IBM_s013
                                 + 1.09783 w_k1_INTC_s013
                                 + 1.01886 w_k1_KO_s013
                                 + 0.561259 w_k1_MAT_s013
                                 + 1.17937 w_k1_MSFT_s013
                                 + 1.04344 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.04317 w_k1_ADM_s013
                                 + 0.824606 w_k1_AXP_s013
                                 + 0.911527 w_k1_DIS_s013
                                 + 0.835101 w_k1_FDX_s013
                                 + 1.0441 w_k1_IBM_s013
                                 + 0.933604 w_k1_INTC_s013
                                 + 0.97254 w_k1_KO_s013
                                 + 0.633496 w_k1_MAT_s013
                                 + 0.905082 w_k1_MSFT_s013
                                 + 0.803234 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.930703 w_k1_ADM_s013
                                 + 0.831723 w_k1_AXP_s013
                                 + 0.638478 w_k1_DIS_s013
                                 + 0.73378 w_k1_FDX_s013
                                 + 0.844615 w_k1_IBM_s013
                                 + 0.868024 w_k1_INTC_s013
                                 + 0.87195 w_k1_KO_s013
                                 + 0.695901 w_k1_MAT_s013
                                 + 0.744777 w_k1_MSFT_s013
                                 + 0.817211 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.31756 w_k1_ADM_s013
                                 + 1.33424 w_k1_AXP_s013
                                 + 1.25669 w_k1_DIS_s013
                                 + 1.30245 w_k1_FDX_s013
                                 + 0.903083 w_k1_IBM_s013
                                 + 1.31291 w_k1_INTC_s013
                                 + 0.972569 w_k1_KO_s013
                                 + 0.815031 w_k1_MAT_s013
                                 + 1.14033 w_k1_MSFT_s013
                                 + 1.05084 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.01775 w_k1_ADM_s013
                                 + 1.32775 w_k1_AXP_s013
                                 + 1.27378 w_k1_DIS_s013
                                 + 1.00031 w_k1_FDX_s013 + 1.5907 w_k1_IBM_s013
                                 + 1.2281 w_k1_INTC_s013 + 1.28754 w_k1_KO_s013
                                 + 0.961903 w_k1_MAT_s013
                                 + 1.11522 w_k1_MSFT_s013
                                 + 1.00638 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.07462 w_k1_ADM_s014
                                 + 1.04888 w_k1_AXP_s014
                                 + 1.19981 w_k1_DIS_s014
                                 + 1.12331 w_k1_FDX_s014
                                 + 0.939603 w_k1_IBM_s014
                                 + 0.993706 w_k1_INTC_s014
                                 + 1.00393 w_k1_KO_s014
                                 + 0.896337 w_k1_MAT_s014
                                 + 1.14946 w_k1_MSFT_s014
                                 + 0.957905 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.835114 w_k1_ADM_s014
                                 + 0.727215 w_k1_AXP_s014
                                 + 0.692764 w_k1_DIS_s014
                                 + 0.788558 w_k1_FDX_s014
                                 + 0.809135 w_k1_IBM_s014
                                 + 0.651807 w_k1_INTC_s014
                                 + 0.779618 w_k1_KO_s014
                                 + 0.705481 w_k1_MAT_s014
                                 + 0.821422 w_k1_MSFT_s014
                                 + 0.879745 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.23984 w_k1_ADM_s014
                                 + 1.33712 w_k1_AXP_s014
                                 + 1.58886 w_k1_DIS_s014
                                 + 1.27628 w_k1_FDX_s014
                                 + 1.14822 w_k1_IBM_s014
                                 + 1.05727 w_k1_INTC_s014
                                 + 1.17749 w_k1_KO_s014 + 1.02152 w_k1_MAT_s014
                                 + 1.2161 w_k1_MSFT_s014 + 1.15609 w_k1_PG_s014
                                 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.1964 w_k1_ADM_s014
                                 + 0.774399 w_k1_AXP_s014
                                 + 0.860369 w_k1_DIS_s014
                                 + 0.937251 w_k1_FDX_s014
                                 + 0.838182 w_k1_IBM_s014
                                 + 0.82323 w_k1_INTC_s014
                                 + 0.990221 w_k1_KO_s014
                                 + 0.94335 w_k1_MAT_s014
                                 + 0.90967 w_k1_MSFT_s014
                                 + 0.941267 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.08824 w_k1_ADM_s014
                                 + 1.18487 w_k1_AXP_s014
                                 + 0.964683 w_k1_DIS_s014
                                 + 0.990476 w_k1_FDX_s014
                                 + 1.17971 w_k1_IBM_s014
                                 + 0.858017 w_k1_INTC_s014
                                 + 1.03002 w_k1_KO_s014 + 1.44092 w_k1_MAT_s014
                                 + 0.99303 w_k1_MSFT_s014
                                 + 0.985025 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.16124 w_k1_ADM_s014
                                 + 0.864386 w_k1_AXP_s014
                                 + 1.1578 w_k1_DIS_s014 + 1.18272 w_k1_FDX_s014
                                 + 0.620549 w_k1_IBM_s014
                                 + 0.868573 w_k1_INTC_s014
                                 + 0.846694 w_k1_KO_s014
                                 + 0.529051 w_k1_MAT_s014
                                 + 1.04123 w_k1_MSFT_s014
                                 + 1.00808 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.18523 w_k1_ADM_s014
                                 + 1.5247 w_k1_AXP_s014 + 1.53876 w_k1_DIS_s014
                                 + 1.22316 w_k1_FDX_s014
                                 + 1.80961 w_k1_IBM_s014
                                 + 0.999145 w_k1_INTC_s014
                                 + 1.61648 w_k1_KO_s014 + 2.13322 w_k1_MAT_s014
                                 + 1.13578 w_k1_MSFT_s014 + 1.2853 w_k1_PG_s014
                                 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.84898 w_k1_ADM_s014
                                 + 0.865651 w_k1_AXP_s014
                                 + 0.92549 w_k1_DIS_s014
                                 + 0.873169 w_k1_FDX_s014
                                 + 1.43417 w_k1_IBM_s014
                                 + 0.845362 w_k1_INTC_s014
                                 + 1.00924 w_k1_KO_s014 + 1.30596 w_k1_MAT_s014
                                 + 1.01433 w_k1_MSFT_s014 + 1.0392 w_k1_PG_s014
                                 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.919992 w_k1_ADM_s015
                                 + 0.945528 w_k1_AXP_s015
                                 + 0.920409 w_k1_DIS_s015
                                 + 0.953907 w_k1_FDX_s015
                                 + 0.766429 w_k1_IBM_s015
                                 + 0.980044 w_k1_INTC_s015
                                 + 0.990095 w_k1_KO_s015
                                 + 0.653597 w_k1_MAT_s015
                                 + 1.05823 w_k1_MSFT_s015
                                 + 1.05601 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.794303 w_k1_ADM_s015
                                 + 1.10761 w_k1_AXP_s015
                                 + 0.800009 w_k1_DIS_s015
                                 + 0.839532 w_k1_FDX_s015
                                 + 1.34442 w_k1_IBM_s015
                                 + 0.999795 w_k1_INTC_s015
                                 + 1.02626 w_k1_KO_s015 + 1.3058 w_k1_MAT_s015
                                 + 1.01104 w_k1_MSFT_s015
                                 + 1.07182 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.05518 w_k1_ADM_s015
                                 + 1.09043 w_k1_AXP_s015
                                 + 1.37242 w_k1_DIS_s015 + 1.2321 w_k1_FDX_s015
                                 + 1.05277 w_k1_IBM_s015
                                 + 1.19195 w_k1_INTC_s015
                                 + 1.20004 w_k1_KO_s015 + 1.00906 w_k1_MAT_s015
                                 + 1.29376 w_k1_MSFT_s015
                                 + 1.06957 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.27245 w_k1_ADM_s015
                                 + 1.45031 w_k1_AXP_s015
                                 + 1.30587 w_k1_DIS_s015
                                 + 1.36006 w_k1_FDX_s015 + 1.2059 w_k1_IBM_s015
                                 + 1.45848 w_k1_INTC_s015
                                 + 1.25588 w_k1_KO_s015 + 1.08237 w_k1_MAT_s015
                                 + 1.36781 w_k1_MSFT_s015
                                 + 1.36393 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.931058 w_k1_ADM_s015
                                 + 0.986268 w_k1_AXP_s015
                                 + 1.08654 w_k1_DIS_s015
                                 + 0.946272 w_k1_FDX_s015
                                 + 1.08539 w_k1_IBM_s015
                                 + 1.15614 w_k1_INTC_s015
                                 + 1.26858 w_k1_KO_s015 + 0.89288 w_k1_MAT_s015
                                 + 1.18033 w_k1_MSFT_s015
                                 + 1.12892 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.563233 w_k1_ADM_s015
                                 + 0.66528 w_k1_AXP_s015
                                 + 0.573993 w_k1_DIS_s015
                                 + 0.662366 w_k1_FDX_s015
                                 + 0.824766 w_k1_IBM_s015
                                 + 0.804915 w_k1_INTC_s015
                                 + 0.706955 w_k1_KO_s015
                                 + 0.719307 w_k1_MAT_s015
                                 + 0.727481 w_k1_MSFT_s015
                                 + 0.854988 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.816735 w_k1_ADM_s015
                                 + 0.68722 w_k1_AXP_s015
                                 + 0.646935 w_k1_DIS_s015
                                 + 0.838318 w_k1_FDX_s015
                                 + 0.73916 w_k1_IBM_s015
                                 + 0.935387 w_k1_INTC_s015
                                 + 0.81676 w_k1_KO_s015
                                 + 0.625302 w_k1_MAT_s015
                                 + 0.9105 w_k1_MSFT_s015
                                 + 0.911978 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.13167 w_k1_ADM_s015
                                 + 1.87559 w_k1_AXP_s015 + 1.8457 w_k1_DIS_s015
                                 + 1.51242 w_k1_FDX_s015 + 1.5218 w_k1_IBM_s015
                                 + 1.37776 w_k1_INTC_s015 + 1.4939 w_k1_KO_s015
                                 + 1.84929 w_k1_MAT_s015
                                 + 1.6168 w_k1_MSFT_s015 + 1.2351 w_k1_PG_s015
                                 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.789873 w_k1_ADM_s016
                                 + 0.783543 w_k1_AXP_s016
                                 + 0.679864 w_k1_DIS_s016
                                 + 0.78798 w_k1_FDX_s016
                                 + 1.07165 w_k1_IBM_s016
                                 + 1.0863 w_k1_INTC_s016
                                 + 0.817419 w_k1_KO_s016
                                 + 1.20794 w_k1_MAT_s016
                                 + 0.876045 w_k1_MSFT_s016
                                 + 0.855403 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.31582 w_k1_ADM_s016
                                 + 1.57448 w_k1_AXP_s016
                                 + 1.77528 w_k1_DIS_s016
                                 + 1.42851 w_k1_FDX_s016
                                 + 1.62615 w_k1_IBM_s016
                                 + 1.38483 w_k1_INTC_s016
                                 + 1.41305 w_k1_KO_s016 + 2.11892 w_k1_MAT_s016
                                 + 1.30686 w_k1_MSFT_s016
                                 + 1.19561 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.937946 w_k1_ADM_s016
                                 + 1.11225 w_k1_AXP_s016
                                 + 1.01454 w_k1_DIS_s016
                                 + 1.00992 w_k1_FDX_s016
                                 + 1.24011 w_k1_IBM_s016
                                 + 1.01866 w_k1_INTC_s016
                                 + 0.908246 w_k1_KO_s016
                                 + 1.40767 w_k1_MAT_s016
                                 + 0.972514 w_k1_MSFT_s016
                                 + 1.14688 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.19846 w_k1_ADM_s016
                                 + 1.28439 w_k1_AXP_s016 + 1.456 w_k1_DIS_s016
                                 + 1.26287 w_k1_FDX_s016
                                 + 1.21585 w_k1_IBM_s016
                                 + 1.08688 w_k1_INTC_s016
                                 + 1.29858 w_k1_KO_s016 + 1.28203 w_k1_MAT_s016
                                 + 1.23309 w_k1_MSFT_s016
                                 + 1.18849 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.973077 w_k1_ADM_s016
                                 + 1.21227 w_k1_AXP_s016
                                 + 1.47076 w_k1_DIS_s016
                                 + 1.44301 w_k1_FDX_s016
                                 + 0.882719 w_k1_IBM_s016
                                 + 1.4598 w_k1_INTC_s016 + 1.02826 w_k1_KO_s016
                                 + 1.18034 w_k1_MAT_s016
                                 + 1.35064 w_k1_MSFT_s016 + 1.1559 w_k1_PG_s016
                                 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.766273 w_k1_ADM_s016
                                 + 0.873948 w_k1_AXP_s016
                                 + 0.851089 w_k1_DIS_s016
                                 + 0.850829 w_k1_FDX_s016
                                 + 1.64689 w_k1_IBM_s016
                                 + 1.05431 w_k1_INTC_s016
                                 + 1.18683 w_k1_KO_s016 + 1.75515 w_k1_MAT_s016
                                 + 1.00573 w_k1_MSFT_s016
                                 + 0.979707 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.888996 w_k1_ADM_s016
                                 + 0.696267 w_k1_AXP_s016
                                 + 0.624936 w_k1_DIS_s016
                                 + 0.795348 w_k1_FDX_s016
                                 + 0.671641 w_k1_IBM_s016
                                 + 0.687873 w_k1_INTC_s016
                                 + 0.528587 w_k1_KO_s016
                                 + 0.640632 w_k1_MAT_s016
                                 + 0.779176 w_k1_MSFT_s016
                                 + 0.803229 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.05589 w_k1_ADM_s016
                                 + 1.11082 w_k1_AXP_s016
                                 + 0.992581 w_k1_DIS_s016
                                 + 1.08272 w_k1_FDX_s016
                                 + 0.908273 w_k1_IBM_s016
                                 + 1.0631 w_k1_INTC_s016
                                 + 0.841386 w_k1_KO_s016
                                 + 0.810693 w_k1_MAT_s016
                                 + 0.955713 w_k1_MSFT_s016
                                 + 0.942801 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.982138 w_k1_ADM_s017
                                 + 0.680351 w_k1_AXP_s017
                                 + 0.595096 w_k1_DIS_s017
                                 + 0.758987 w_k1_FDX_s017
                                 + 0.815704 w_k1_IBM_s017
                                 + 0.863149 w_k1_INTC_s017
                                 + 0.774862 w_k1_KO_s017
                                 + 0.812027 w_k1_MAT_s017
                                 + 0.772806 w_k1_MSFT_s017
                                 + 0.842587 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.32044 w_k1_ADM_s017
                                 + 0.957359 w_k1_AXP_s017
                                 + 1.23911 w_k1_DIS_s017
                                 + 1.10526 w_k1_FDX_s017
                                 + 1.07202 w_k1_IBM_s017
                                 + 0.836209 w_k1_INTC_s017
                                 + 1.07738 w_k1_KO_s017 + 1.17646 w_k1_MAT_s017
                                 + 1.02131 w_k1_MSFT_s017
                                 + 1.13963 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.4254 w_k1_ADM_s017
                                 + 1.38405 w_k1_AXP_s017
                                 + 1.77826 w_k1_DIS_s017
                                 + 1.23648 w_k1_FDX_s017
                                 + 1.51142 w_k1_IBM_s017
                                 + 1.19207 w_k1_INTC_s017
                                 + 1.61912 w_k1_KO_s017 + 1.91004 w_k1_MAT_s017
                                 + 1.34487 w_k1_MSFT_s017
                                 + 1.36335 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.14862 w_k1_ADM_s017
                                 + 1.1954 w_k1_AXP_s017 + 1.12493 w_k1_DIS_s017
                                 + 1.09615 w_k1_FDX_s017
                                 + 1.27993 w_k1_IBM_s017
                                 + 0.984404 w_k1_INTC_s017
                                 + 1.09496 w_k1_KO_s017 + 1.37315 w_k1_MAT_s017
                                 + 1.01425 w_k1_MSFT_s017
                                 + 0.988337 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.31711 w_k1_ADM_s017
                                 + 1.73272 w_k1_AXP_s017
                                 + 1.30347 w_k1_DIS_s017
                                 + 1.16146 w_k1_FDX_s017
                                 + 1.51453 w_k1_IBM_s017
                                 + 0.958033 w_k1_INTC_s017
                                 + 1.35212 w_k1_KO_s017 + 2.43117 w_k1_MAT_s017
                                 + 1.14387 w_k1_MSFT_s017
                                 + 1.16903 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.20813 w_k1_ADM_s017
                                 + 0.842338 w_k1_AXP_s017
                                 + 0.903271 w_k1_DIS_s017
                                 + 1.00525 w_k1_FDX_s017
                                 + 0.795017 w_k1_IBM_s017
                                 + 1.01971 w_k1_INTC_s017
                                 + 0.90892 w_k1_KO_s017
                                 + 0.738229 w_k1_MAT_s017
                                 + 0.992093 w_k1_MSFT_s017
                                 + 0.845542 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.54903 w_k1_ADM_s017
                                 + 1.29748 w_k1_AXP_s017
                                 + 1.49041 w_k1_DIS_s017
                                 + 1.32156 w_k1_FDX_s017
                                 + 0.960027 w_k1_IBM_s017
                                 + 1.18149 w_k1_INTC_s017
                                 + 1.11215 w_k1_KO_s017
                                 + 0.749054 w_k1_MAT_s017
                                 + 1.28146 w_k1_MSFT_s017
                                 + 1.15938 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.07165 w_k1_ADM_s017
                                 + 0.86332 w_k1_AXP_s017
                                 + 0.754003 w_k1_DIS_s017
                                 + 0.834494 w_k1_FDX_s017
                                 + 1.24813 w_k1_IBM_s017
                                 + 0.848236 w_k1_INTC_s017
                                 + 1.00127 w_k1_KO_s017 + 1.30317 w_k1_MAT_s017
                                 + 0.943372 w_k1_MSFT_s017
                                 + 0.865779 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.883364 w_k1_ADM_s100
                                 + 0.832282 w_k1_AXP_s100
                                 + 0.5739 w_k1_DIS_s100
                                 + 0.731268 w_k1_FDX_s100
                                 + 0.843321 w_k1_IBM_s100
                                 + 0.850637 w_k1_INTC_s100
                                 + 0.690091 w_k1_KO_s100
                                 + 0.807565 w_k1_MAT_s100
                                 + 0.735528 w_k1_MSFT_s100
                                 + 0.771466 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.0314 w_k1_ADM_s100
                                 + 1.08204 w_k1_AXP_s100
                                 + 0.899563 w_k1_DIS_s100
                                 + 0.951781 w_k1_FDX_s100
                                 + 0.875722 w_k1_IBM_s100
                                 + 0.823554 w_k1_INTC_s100
                                 + 0.789669 w_k1_KO_s100
                                 + 1.08944 w_k1_MAT_s100
                                 + 0.974221 w_k1_MSFT_s100
                                 + 0.833527 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.821112 w_k1_ADM_s100
                                 + 1.18073 w_k1_AXP_s100 + 1.1082 w_k1_DIS_s100
                                 + 1.02852 w_k1_FDX_s100
                                 + 1.61872 w_k1_IBM_s100
                                 + 1.08367 w_k1_INTC_s100
                                 + 1.18238 w_k1_KO_s100 + 1.55263 w_k1_MAT_s100
                                 + 0.97603 w_k1_MSFT_s100
                                 + 0.993696 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.965991 w_k1_ADM_s100
                                 + 0.913518 w_k1_AXP_s100
                                 + 0.975298 w_k1_DIS_s100
                                 + 0.914062 w_k1_FDX_s100
                                 + 1.13321 w_k1_IBM_s100
                                 + 0.828947 w_k1_INTC_s100
                                 + 0.942037 w_k1_KO_s100
                                 + 0.832648 w_k1_MAT_s100
                                 + 0.836526 w_k1_MSFT_s100
                                 + 0.80904 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.21766 w_k1_ADM_s100
                                 + 1.25406 w_k1_AXP_s100
                                 + 1.19631 w_k1_DIS_s100
                                 + 1.12542 w_k1_FDX_s100
                                 + 0.974763 w_k1_IBM_s100
                                 + 1.02034 w_k1_INTC_s100
                                 + 0.996089 w_k1_KO_s100
                                 + 0.866802 w_k1_MAT_s100
                                 + 1.03133 w_k1_MSFT_s100
                                 + 0.97426 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.930106 w_k1_ADM_s100
                                 + 1.13022 w_k1_AXP_s100
                                 + 0.914877 w_k1_DIS_s100
                                 + 1.04697 w_k1_FDX_s100
                                 + 0.665734 w_k1_IBM_s100
                                 + 0.983974 w_k1_INTC_s100
                                 + 0.730751 w_k1_KO_s100
                                 + 0.660128 w_k1_MAT_s100
                                 + 0.90077 w_k1_MSFT_s100
                                 + 0.799453 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.04094 w_k1_ADM_s100
                                 + 1.32419 w_k1_AXP_s100
                                 + 1.57386 w_k1_DIS_s100
                                 + 1.26408 w_k1_FDX_s100
                                 + 1.29973 w_k1_IBM_s100
                                 + 1.14224 w_k1_INTC_s100
                                 + 1.34683 w_k1_KO_s100 + 1.73328 w_k1_MAT_s100
                                 + 1.31197 w_k1_MSFT_s100 + 1.2129 w_k1_PG_s100
                                 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.791891 w_k1_ADM_s100
                                 + 0.925981 w_k1_AXP_s100
                                 + 0.673945 w_k1_DIS_s100
                                 + 0.784728 w_k1_FDX_s100
                                 + 1.20144 w_k1_IBM_s100
                                 + 0.733089 w_k1_INTC_s100
                                 + 0.787405 w_k1_KO_s100
                                 + 1.06257 w_k1_MAT_s100
                                 + 0.725146 w_k1_MSFT_s100
                                 + 0.78698 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.768059 w_k1_ADM_s101
                                 + 0.69209 w_k1_AXP_s101
                                 + 0.785776 w_k1_DIS_s101
                                 + 0.884969 w_k1_FDX_s101
                                 + 0.744087 w_k1_IBM_s101
                                 + 0.821165 w_k1_INTC_s101
                                 + 0.760667 w_k1_KO_s101
                                 + 0.814921 w_k1_MAT_s101
                                 + 0.910934 w_k1_MSFT_s101
                                 + 0.830806 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.25193 w_k1_ADM_s101
                                 + 1.55822 w_k1_AXP_s101
                                 + 1.75381 w_k1_DIS_s101
                                 + 1.20922 w_k1_FDX_s101
                                 + 1.68644 w_k1_IBM_s101
                                 + 0.970706 w_k1_INTC_s101
                                 + 1.57306 w_k1_KO_s101 + 2.31214 w_k1_MAT_s101
                                 + 1.44027 w_k1_MSFT_s101
                                 + 1.29563 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.07839 w_k1_ADM_s101
                                 + 1.12045 w_k1_AXP_s101
                                 + 1.00503 w_k1_DIS_s101
                                 + 1.16427 w_k1_FDX_s101
                                 + 0.830809 w_k1_IBM_s101
                                 + 0.925251 w_k1_INTC_s101
                                 + 1.02821 w_k1_KO_s101 + 1.19753 w_k1_MAT_s101
                                 + 1.21462 w_k1_MSFT_s101
                                 + 0.995604 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.627331 w_k1_ADM_s101
                                 + 0.658138 w_k1_AXP_s101
                                 + 0.632259 w_k1_DIS_s101
                                 + 0.748719 w_k1_FDX_s101
                                 + 1.11402 w_k1_IBM_s101
                                 + 0.793259 w_k1_INTC_s101
                                 + 0.807038 w_k1_KO_s101
                                 + 1.23916 w_k1_MAT_s101
                                 + 0.899309 w_k1_MSFT_s101
                                 + 0.885318 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.09403 w_k1_ADM_s101
                                 + 1.24173 w_k1_AXP_s101
                                 + 1.08317 w_k1_DIS_s101
                                 + 1.11234 w_k1_FDX_s101
                                 + 1.15064 w_k1_IBM_s101
                                 + 0.970907 w_k1_INTC_s101
                                 + 1.04083 w_k1_KO_s101 + 1.27654 w_k1_MAT_s101
                                 + 1.14963 w_k1_MSFT_s101 + 1.0373 w_k1_PG_s101
                                 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.09394 w_k1_ADM_s101
                                 + 1.11556 w_k1_AXP_s101
                                 + 1.29686 w_k1_DIS_s101
                                 + 1.21286 w_k1_FDX_s101
                                 + 0.87196 w_k1_IBM_s101
                                 + 1.05354 w_k1_INTC_s101
                                 + 0.974211 w_k1_KO_s101
                                 + 0.790022 w_k1_MAT_s101
                                 + 1.27575 w_k1_MSFT_s101
                                 + 1.08461 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.539481 w_k1_ADM_s101
                                 + 0.5236 w_k1_AXP_s101
                                 + 0.425401 w_k1_DIS_s101
                                 + 0.588699 w_k1_FDX_s101
                                 + 0.690261 w_k1_IBM_s101
                                 + 0.691559 w_k1_INTC_s101
                                 + 0.528013 w_k1_KO_s101
                                 + 0.827676 w_k1_MAT_s101
                                 + 0.757989 w_k1_MSFT_s101
                                 + 0.700376 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.83806 w_k1_ADM_s101
                                 + 1.708 w_k1_AXP_s101 + 2.04129 w_k1_DIS_s101
                                 + 1.56882 w_k1_FDX_s101
                                 + 1.12282 w_k1_IBM_s101
                                 + 1.53537 w_k1_INTC_s101
                                 + 1.50679 w_k1_KO_s101 + 1.34936 w_k1_MAT_s101
                                 + 1.55635 w_k1_MSFT_s101
                                 + 1.21174 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.19227 w_k1_ADM_s102
                                 + 1.05917 w_k1_AXP_s102 + 1.0865 w_k1_DIS_s102
                                 + 1.20686 w_k1_FDX_s102
                                 + 0.746175 w_k1_IBM_s102
                                 + 1.06354 w_k1_INTC_s102
                                 + 0.998235 w_k1_KO_s102
                                 + 1.22956 w_k1_MAT_s102
                                 + 1.22881 w_k1_MSFT_s102
                                 + 1.08536 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.26165 w_k1_ADM_s102
                                 + 1.43446 w_k1_AXP_s102
                                 + 1.46218 w_k1_DIS_s102
                                 + 1.18559 w_k1_FDX_s102
                                 + 1.47968 w_k1_IBM_s102
                                 + 1.28187 w_k1_INTC_s102
                                 + 1.57899 w_k1_KO_s102 + 1.58989 w_k1_MAT_s102
                                 + 1.17096 w_k1_MSFT_s102
                                 + 1.24109 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.778256 w_k1_ADM_s102
                                 + 0.661996 w_k1_AXP_s102
                                 + 0.653532 w_k1_DIS_s102
                                 + 0.795334 w_k1_FDX_s102
                                 + 0.711646 w_k1_IBM_s102
                                 + 0.842821 w_k1_INTC_s102
                                 + 0.761331 w_k1_KO_s102
                                 + 0.764529 w_k1_MAT_s102
                                 + 0.931094 w_k1_MSFT_s102
                                 + 0.784203 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.0607 w_k1_ADM_s102
                                 + 0.873697 w_k1_AXP_s102
                                 + 0.683502 w_k1_DIS_s102
                                 + 0.785289 w_k1_FDX_s102
                                 + 1.00111 w_k1_IBM_s102
                                 + 0.704524 w_k1_INTC_s102
                                 + 0.925768 w_k1_KO_s102
                                 + 1.44218 w_k1_MAT_s102
                                 + 0.829938 w_k1_MSFT_s102
                                 + 0.891985 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.50387 w_k1_ADM_s102
                                 + 1.2263 w_k1_AXP_s102 + 1.43529 w_k1_DIS_s102
                                 + 1.24216 w_k1_FDX_s102
                                 + 1.07027 w_k1_IBM_s102
                                 + 1.30189 w_k1_INTC_s102
                                 + 1.34413 w_k1_KO_s102 + 1.4985 w_k1_MAT_s102
                                 + 1.42256 w_k1_MSFT_s102
                                 + 1.09853 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.05857 w_k1_ADM_s102
                                 + 0.904655 w_k1_AXP_s102
                                 + 0.960364 w_k1_DIS_s102
                                 + 0.953505 w_k1_FDX_s102
                                 + 1.23499 w_k1_IBM_s102
                                 + 1.09565 w_k1_INTC_s102
                                 + 1.10206 w_k1_KO_s102 + 1.18986 w_k1_MAT_s102
                                 + 1.01833 w_k1_MSFT_s102
                                 + 1.00263 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.979469 w_k1_ADM_s102
                                 + 1.07917 w_k1_AXP_s102
                                 + 0.797481 w_k1_DIS_s102
                                 + 0.827896 w_k1_FDX_s102
                                 + 1.47373 w_k1_IBM_s102
                                 + 1.0955 w_k1_INTC_s102 + 1.07446 w_k1_KO_s102
                                 + 2.1149 w_k1_MAT_s102
                                 + 0.906773 w_k1_MSFT_s102
                                 + 0.889392 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.981425 w_k1_ADM_s102
                                 + 0.965787 w_k1_AXP_s102
                                 + 0.931982 w_k1_DIS_s102
                                 + 0.984824 w_k1_FDX_s102
                                 + 0.874115 w_k1_IBM_s102
                                 + 1.08796 w_k1_INTC_s102
                                 + 0.95709 w_k1_KO_s102
                                 + 0.999017 w_k1_MAT_s102
                                 + 1.01881 w_k1_MSFT_s102
                                 + 0.923817 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.11684 w_k1_ADM_s103
                                 + 0.834428 w_k1_AXP_s103
                                 + 0.754203 w_k1_DIS_s103
                                 + 0.889361 w_k1_FDX_s103
                                 + 0.814131 w_k1_IBM_s103
                                 + 0.74501 w_k1_INTC_s103
                                 + 0.869835 w_k1_KO_s103
                                 + 0.950844 w_k1_MAT_s103
                                 + 0.880877 w_k1_MSFT_s103
                                 + 1.00371 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.4366 w_k1_ADM_s103
                                 + 1.40656 w_k1_AXP_s103
                                 + 2.05616 w_k1_DIS_s103
                                 + 1.66567 w_k1_FDX_s103
                                 + 1.39886 w_k1_IBM_s103
                                 + 1.41649 w_k1_INTC_s103
                                 + 1.40485 w_k1_KO_s103 + 1.45536 w_k1_MAT_s103
                                 + 1.64671 w_k1_MSFT_s103
                                 + 1.46431 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.870368 w_k1_ADM_s103
                                 + 0.538249 w_k1_AXP_s103
                                 + 0.679025 w_k1_DIS_s103
                                 + 0.746462 w_k1_FDX_s103
                                 + 0.715554 w_k1_IBM_s103
                                 + 0.652073 w_k1_INTC_s103
                                 + 0.826413 w_k1_KO_s103
                                 + 0.640098 w_k1_MAT_s103
                                 + 0.82251 w_k1_MSFT_s103
                                 + 0.882522 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.26929 w_k1_ADM_s103
                                 + 1.09203 w_k1_AXP_s103
                                 + 1.11017 w_k1_DIS_s103
                                 + 1.22039 w_k1_FDX_s103
                                 + 0.857568 w_k1_IBM_s103
                                 + 0.841678 w_k1_INTC_s103
                                 + 0.929876 w_k1_KO_s103
                                 + 0.739045 w_k1_MAT_s103
                                 + 1.01081 w_k1_MSFT_s103
                                 + 1.02532 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.54558 w_k1_ADM_s103
                                 + 1.18885 w_k1_AXP_s103
                                 + 1.32796 w_k1_DIS_s103
                                 + 1.18609 w_k1_FDX_s103 + 1.4154 w_k1_IBM_s103
                                 + 0.860649 w_k1_INTC_s103
                                 + 1.37497 w_k1_KO_s103 + 1.78808 w_k1_MAT_s103
                                 + 1.2084 w_k1_MSFT_s103 + 1.34316 w_k1_PG_s103
                                 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.700831 w_k1_ADM_s103
                                 + 0.698135 w_k1_AXP_s103
                                 + 0.512655 w_k1_DIS_s103
                                 + 0.709465 w_k1_FDX_s103
                                 + 0.740566 w_k1_IBM_s103
                                 + 0.675163 w_k1_INTC_s103
                                 + 0.655375 w_k1_KO_s103
                                 + 0.764574 w_k1_MAT_s103
                                 + 0.750501 w_k1_MSFT_s103
                                 + 0.818809 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.706961 w_k1_ADM_s103
                                 + 0.960032 w_k1_AXP_s103
                                 + 0.869825 w_k1_DIS_s103
                                 + 0.874302 w_k1_FDX_s103
                                 + 1.23525 w_k1_IBM_s103
                                 + 0.926055 w_k1_INTC_s103
                                 + 0.991231 w_k1_KO_s103
                                 + 1.58439 w_k1_MAT_s103
                                 + 0.907987 w_k1_MSFT_s103
                                 + 1.10029 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.916929 w_k1_ADM_s103
                                 + 0.724744 w_k1_AXP_s103
                                 + 1.13689 w_k1_DIS_s103
                                 + 0.953032 w_k1_FDX_s103
                                 + 0.965178 w_k1_IBM_s103
                                 + 0.920991 w_k1_INTC_s103
                                 + 1.20296 w_k1_KO_s103
                                 + 0.845104 w_k1_MAT_s103
                                 + 1.16392 w_k1_MSFT_s103 + 1.1152 w_k1_PG_s103
                                 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.29805 w_k1_ADM_s104
                                 + 1.14759 w_k1_AXP_s104
                                 + 0.990904 w_k1_DIS_s104
                                 + 1.01862 w_k1_FDX_s104
                                 + 1.13984 w_k1_IBM_s104
                                 + 1.01142 w_k1_INTC_s104
                                 + 1.15823 w_k1_KO_s104 + 1.11993 w_k1_MAT_s104
                                 + 1.03622 w_k1_MSFT_s104
                                 + 0.96898 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.978584 w_k1_ADM_s104
                                 + 0.892036 w_k1_AXP_s104
                                 + 0.961976 w_k1_DIS_s104
                                 + 1.09903 w_k1_FDX_s104 + 0.7129 w_k1_IBM_s104
                                 + 1.06475 w_k1_INTC_s104
                                 + 0.799704 w_k1_KO_s104
                                 + 0.786999 w_k1_MAT_s104
                                 + 1.0929 w_k1_MSFT_s104
                                 + 0.922038 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.71097 w_k1_ADM_s104
                                 + 0.644584 w_k1_AXP_s104
                                 + 0.64617 w_k1_DIS_s104
                                 + 0.649547 w_k1_FDX_s104
                                 + 1.24576 w_k1_IBM_s104
                                 + 0.857161 w_k1_INTC_s104
                                 + 0.842685 w_k1_KO_s104
                                 + 0.777947 w_k1_MAT_s104
                                 + 0.745322 w_k1_MSFT_s104
                                 + 0.771559 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.69593 w_k1_ADM_s104
                                 + 1.07562 w_k1_AXP_s104
                                 + 1.31376 w_k1_DIS_s104
                                 + 1.34229 w_k1_FDX_s104
                                 + 0.67681 w_k1_IBM_s104
                                 + 1.25709 w_k1_INTC_s104
                                 + 1.03806 w_k1_KO_s104
                                 + 0.644536 w_k1_MAT_s104
                                 + 1.24841 w_k1_MSFT_s104
                                 + 1.10705 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.02513 w_k1_ADM_s104
                                 + 0.686245 w_k1_AXP_s104
                                 + 0.604527 w_k1_DIS_s104
                                 + 0.78079 w_k1_FDX_s104
                                 + 0.858331 w_k1_IBM_s104
                                 + 0.940367 w_k1_INTC_s104
                                 + 0.74643 w_k1_KO_s104
                                 + 0.701497 w_k1_MAT_s104
                                 + 0.816707 w_k1_MSFT_s104
                                 + 0.752753 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.18812 w_k1_ADM_s104
                                 + 0.776457 w_k1_AXP_s104
                                 + 0.863396 w_k1_DIS_s104
                                 + 0.800887 w_k1_FDX_s104
                                 + 1.62341 w_k1_IBM_s104
                                 + 0.973572 w_k1_INTC_s104
                                 + 1.29074 w_k1_KO_s104 + 1.35613 w_k1_MAT_s104
                                 + 0.748358 w_k1_MSFT_s104
                                 + 0.914242 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.227 w_k1_ADM_s104
                                 + 1.12136 w_k1_AXP_s104 + 1.2244 w_k1_DIS_s104
                                 + 1.18839 w_k1_FDX_s104
                                 + 1.65872 w_k1_IBM_s104
                                 + 1.05058 w_k1_INTC_s104
                                 + 1.36556 w_k1_KO_s104 + 1.3791 w_k1_MAT_s104
                                 + 1.11368 w_k1_MSFT_s104
                                 + 1.12981 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.47003 w_k1_ADM_s104
                                 + 1.22371 w_k1_AXP_s104
                                 + 1.53746 w_k1_DIS_s104
                                 + 1.38341 w_k1_FDX_s104
                                 + 1.16524 w_k1_IBM_s104
                                 + 1.44176 w_k1_INTC_s104
                                 + 1.11222 w_k1_KO_s104 + 1.09193 w_k1_MAT_s104
                                 + 1.329 w_k1_MSFT_s104 + 0.967028 w_k1_PG_s104
                                 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.07341 w_k1_ADM_s105
                                 + 1.32529 w_k1_AXP_s105
                                 + 1.77555 w_k1_DIS_s105
                                 + 1.28735 w_k1_FDX_s105
                                 + 1.36635 w_k1_IBM_s105
                                 + 1.33122 w_k1_INTC_s105
                                 + 1.71711 w_k1_KO_s105 + 1.80971 w_k1_MAT_s105
                                 + 1.35864 w_k1_MSFT_s105
                                 + 1.17832 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.882623 w_k1_ADM_s105
                                 + 1.01857 w_k1_AXP_s105
                                 + 1.07059 w_k1_DIS_s105
                                 + 0.979169 w_k1_FDX_s105
                                 + 1.36272 w_k1_IBM_s105
                                 + 1.00603 w_k1_INTC_s105
                                 + 1.25339 w_k1_KO_s105 + 1.32286 w_k1_MAT_s105
                                 + 1.04855 w_k1_MSFT_s105
                                 + 0.961214 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.819657 w_k1_ADM_s105
                                 + 0.668858 w_k1_AXP_s105
                                 + 0.677076 w_k1_DIS_s105
                                 + 0.762368 w_k1_FDX_s105
                                 + 0.586301 w_k1_IBM_s105
                                 + 0.902903 w_k1_INTC_s105
                                 + 0.795259 w_k1_KO_s105
                                 + 0.631264 w_k1_MAT_s105
                                 + 0.831054 w_k1_MSFT_s105
                                 + 0.809253 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.840822 w_k1_ADM_s105
                                 + 0.696193 w_k1_AXP_s105
                                 + 0.657281 w_k1_DIS_s105
                                 + 0.722157 w_k1_FDX_s105
                                 + 1.02464 w_k1_IBM_s105
                                 + 0.860827 w_k1_INTC_s105
                                 + 0.937752 w_k1_KO_s105
                                 + 0.919373 w_k1_MAT_s105
                                 + 0.770519 w_k1_MSFT_s105
                                 + 0.784596 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.688029 w_k1_ADM_s105
                                 + 0.867976 w_k1_AXP_s105
                                 + 0.66936 w_k1_DIS_s105
                                 + 0.806657 w_k1_FDX_s105
                                 + 0.841149 w_k1_IBM_s105
                                 + 0.869593 w_k1_INTC_s105
                                 + 0.919539 w_k1_KO_s105
                                 + 1.30846 w_k1_MAT_s105
                                 + 0.816491 w_k1_MSFT_s105
                                 + 0.820329 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.00428 w_k1_ADM_s105
                                 + 1.2844 w_k1_AXP_s105 + 1.33484 w_k1_DIS_s105
                                 + 1.32668 w_k1_FDX_s105
                                 + 1.09271 w_k1_IBM_s105
                                 + 1.44635 w_k1_INTC_s105
                                 + 1.18894 w_k1_KO_s105 + 1.45654 w_k1_MAT_s105
                                 + 1.09141 w_k1_MSFT_s105
                                 + 1.05988 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.81778 w_k1_ADM_s105
                                 + 0.803403 w_k1_AXP_s105
                                 + 0.912431 w_k1_DIS_s105
                                 + 0.934608 w_k1_FDX_s105
                                 + 0.845853 w_k1_IBM_s105
                                 + 0.881858 w_k1_INTC_s105
                                 + 0.889313 w_k1_KO_s105
                                 + 0.982406 w_k1_MAT_s105
                                 + 0.971285 w_k1_MSFT_s105
                                 + 0.884726 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.888085 w_k1_ADM_s105
                                 + 0.832416 w_k1_AXP_s105
                                 + 1.20937 w_k1_DIS_s105 + 1.1248 w_k1_FDX_s105
                                 + 0.737438 w_k1_IBM_s105
                                 + 1.23483 w_k1_INTC_s105
                                 + 1.03233 w_k1_KO_s105
                                 + 0.690625 w_k1_MAT_s105
                                 + 1.15302 w_k1_MSFT_s105
                                 + 0.982146 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.640519 w_k1_ADM_s106
                                 + 0.515544 w_k1_AXP_s106
                                 + 0.534408 w_k1_DIS_s106
                                 + 0.724823 w_k1_FDX_s106
                                 + 0.597841 w_k1_IBM_s106
                                 + 0.871004 w_k1_INTC_s106
                                 + 0.710326 w_k1_KO_s106
                                 + 0.478304 w_k1_MAT_s106
                                 + 0.78516 w_k1_MSFT_s106
                                 + 0.831988 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.12789 w_k1_ADM_s106
                                 + 1.29921 w_k1_AXP_s106
                                 + 1.43313 w_k1_DIS_s106
                                 + 1.34647 w_k1_FDX_s106
                                 + 0.895746 w_k1_IBM_s106
                                 + 1.0498 w_k1_INTC_s106 + 1.08533 w_k1_KO_s106
                                 + 0.954121 w_k1_MAT_s106
                                 + 1.12728 w_k1_MSFT_s106 + 1.0564 w_k1_PG_s106
                                 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.08387 w_k1_ADM_s106
                                 + 1.50522 w_k1_AXP_s106
                                 + 1.21447 w_k1_DIS_s106
                                 + 1.17127 w_k1_FDX_s106
                                 + 1.83473 w_k1_IBM_s106
                                 + 1.67776 w_k1_INTC_s106
                                 + 1.35857 w_k1_KO_s106 + 1.58447 w_k1_MAT_s106
                                 + 1.21524 w_k1_MSFT_s106
                                 + 1.15831 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.17896 w_k1_ADM_s106
                                 + 1.13809 w_k1_AXP_s106
                                 + 1.23275 w_k1_DIS_s106
                                 + 1.05703 w_k1_FDX_s106
                                 + 1.24521 w_k1_IBM_s106 + 1.271 w_k1_INTC_s106
                                 + 1.35002 w_k1_KO_s106 + 1.26122 w_k1_MAT_s106
                                 + 1.26112 w_k1_MSFT_s106
                                 + 1.09939 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.46749 w_k1_ADM_s106
                                 + 1.14919 w_k1_AXP_s106
                                 + 1.60849 w_k1_DIS_s106
                                 + 1.37011 w_k1_FDX_s106
                                 + 0.963736 w_k1_IBM_s106
                                 + 1.36359 w_k1_INTC_s106
                                 + 1.22148 w_k1_KO_s106
                                 + 0.817093 w_k1_MAT_s106
                                 + 1.27576 w_k1_MSFT_s106
                                 + 1.06404 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.721047 w_k1_ADM_s106
                                 + 0.708046 w_k1_AXP_s106
                                 + 0.613844 w_k1_DIS_s106
                                 + 0.80994 w_k1_FDX_s106
                                 + 0.751554 w_k1_IBM_s106
                                 + 0.859473 w_k1_INTC_s106
                                 + 0.799975 w_k1_KO_s106
                                 + 0.747599 w_k1_MAT_s106
                                 + 0.870663 w_k1_MSFT_s106
                                 + 0.855473 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.15551 w_k1_ADM_s106
                                 + 0.947249 w_k1_AXP_s106
                                 + 0.970397 w_k1_DIS_s106
                                 + 1.15809 w_k1_FDX_s106
                                 + 0.808799 w_k1_IBM_s106
                                 + 1.38127 w_k1_INTC_s106
                                 + 0.871709 w_k1_KO_s106
                                 + 0.640358 w_k1_MAT_s106
                                 + 1.2289 w_k1_MSFT_s106 + 1.01466 w_k1_PG_s106
                                 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.939971 w_k1_ADM_s106
                                 + 0.969428 w_k1_AXP_s106
                                 + 0.886005 w_k1_DIS_s106
                                 + 0.868252 w_k1_FDX_s106
                                 + 0.946342 w_k1_IBM_s106
                                 + 1.15858 w_k1_INTC_s106
                                 + 1.03833 w_k1_KO_s106
                                 + 0.797849 w_k1_MAT_s106
                                 + 1.01492 w_k1_MSFT_s106
                                 + 0.875958 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.26209 w_k1_ADM_s107
                                 + 1.10549 w_k1_AXP_s107
                                 + 1.49431 w_k1_DIS_s107
                                 + 1.26109 w_k1_FDX_s107
                                 + 0.889963 w_k1_IBM_s107
                                 + 0.924563 w_k1_INTC_s107
                                 + 1.10679 w_k1_KO_s107 + 1.13278 w_k1_MAT_s107
                                 + 1.36262 w_k1_MSFT_s107
                                 + 1.09746 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.23294 w_k1_ADM_s107
                                 + 1.34625 w_k1_AXP_s107 + 1.2322 w_k1_DIS_s107
                                 + 1.14637 w_k1_FDX_s107
                                 + 1.27626 w_k1_IBM_s107
                                 + 0.906764 w_k1_INTC_s107
                                 + 1.07804 w_k1_KO_s107 + 1.2946 w_k1_MAT_s107
                                 + 0.975062 w_k1_MSFT_s107
                                 + 1.07782 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.0402 w_k1_ADM_s107
                                 + 0.949533 w_k1_AXP_s107
                                 + 1.06315 w_k1_DIS_s107
                                 + 1.13013 w_k1_FDX_s107
                                 + 0.833234 w_k1_IBM_s107
                                 + 1.02621 w_k1_INTC_s107
                                 + 0.98883 w_k1_KO_s107
                                 + 0.917471 w_k1_MAT_s107
                                 + 0.973033 w_k1_MSFT_s107
                                 + 1.02307 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.901155 w_k1_ADM_s107
                                 + 1.20223 w_k1_AXP_s107
                                 + 1.49202 w_k1_DIS_s107
                                 + 1.21475 w_k1_FDX_s107
                                 + 1.61226 w_k1_IBM_s107
                                 + 1.24732 w_k1_INTC_s107
                                 + 1.47098 w_k1_KO_s107 + 1.0774 w_k1_MAT_s107
                                 + 1.31779 w_k1_MSFT_s107
                                 + 1.11914 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.848144 w_k1_ADM_s107
                                 + 0.860548 w_k1_AXP_s107
                                 + 0.926112 w_k1_DIS_s107
                                 + 0.931637 w_k1_FDX_s107
                                 + 1.23916 w_k1_IBM_s107
                                 + 0.81295 w_k1_INTC_s107
                                 + 1.13665 w_k1_KO_s107 + 1.04829 w_k1_MAT_s107
                                 + 1.00148 w_k1_MSFT_s107
                                 + 1.07942 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.811818 w_k1_ADM_s107
                                 + 0.790516 w_k1_AXP_s107
                                 + 0.674023 w_k1_DIS_s107
                                 + 0.751704 w_k1_FDX_s107
                                 + 0.956155 w_k1_IBM_s107
                                 + 0.816373 w_k1_INTC_s107
                                 + 0.895085 w_k1_KO_s107
                                 + 0.890772 w_k1_MAT_s107
                                 + 0.787416 w_k1_MSFT_s107
                                 + 0.892162 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.896732 w_k1_ADM_s107
                                 + 1.07572 w_k1_AXP_s107
                                 + 0.838034 w_k1_DIS_s107
                                 + 0.925125 w_k1_FDX_s107
                                 + 1.09386 w_k1_IBM_s107
                                 + 0.854739 w_k1_INTC_s107
                                 + 0.879313 w_k1_KO_s107
                                 + 1.24393 w_k1_MAT_s107
                                 + 0.812889 w_k1_MSFT_s107
                                 + 0.900686 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.845758 w_k1_ADM_s107
                                 + 0.951114 w_k1_AXP_s107
                                 + 0.758717 w_k1_DIS_s107
                                 + 0.971074 w_k1_FDX_s107
                                 + 0.665254 w_k1_IBM_s107
                                 + 0.909931 w_k1_INTC_s107
                                 + 0.691501 w_k1_KO_s107
                                 + 0.575785 w_k1_MAT_s107
                                 + 0.966937 w_k1_MSFT_s107
                                 + 0.854241 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.751708 w_k1_ADM_s110
                                 + 0.717368 w_k1_AXP_s110
                                 + 0.619871 w_k1_DIS_s110
                                 + 0.815903 w_k1_FDX_s110
                                 + 1.0569 w_k1_IBM_s110
                                 + 1.26348 w_k1_INTC_s110
                                 + 0.856178 w_k1_KO_s110
                                 + 1.15918 w_k1_MAT_s110
                                 + 1.03628 w_k1_MSFT_s110
                                 + 0.90684 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.20411 w_k1_ADM_s110
                                 + 1.08529 w_k1_AXP_s110
                                 + 1.19403 w_k1_DIS_s110
                                 + 1.11488 w_k1_FDX_s110
                                 + 1.38496 w_k1_IBM_s110
                                 + 1.0237 w_k1_INTC_s110 + 1.19333 w_k1_KO_s110
                                 + 1.61032 w_k1_MAT_s110
                                 + 1.33346 w_k1_MSFT_s110
                                 + 1.09119 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.769399 w_k1_ADM_s110
                                 + 0.638975 w_k1_AXP_s110
                                 + 0.471055 w_k1_DIS_s110
                                 + 0.677021 w_k1_FDX_s110
                                 + 0.758614 w_k1_IBM_s110
                                 + 0.872801 w_k1_INTC_s110
                                 + 0.765166 w_k1_KO_s110
                                 + 0.88981 w_k1_MAT_s110
                                 + 0.812097 w_k1_MSFT_s110
                                 + 0.771134 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.18393 w_k1_ADM_s110
                                 + 1.34624 w_k1_AXP_s110
                                 + 0.839803 w_k1_DIS_s110
                                 + 1.15974 w_k1_FDX_s110
                                 + 0.986279 w_k1_IBM_s110
                                 + 1.3042 w_k1_INTC_s110 + 1.04189 w_k1_KO_s110
                                 + 1.74995 w_k1_MAT_s110
                                 + 1.18125 w_k1_MSFT_s110
                                 + 0.966053 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.792491 w_k1_ADM_s110
                                 + 0.871257 w_k1_AXP_s110
                                 + 0.932788 w_k1_DIS_s110
                                 + 0.897364 w_k1_FDX_s110
                                 + 0.845108 w_k1_IBM_s110
                                 + 1.26997 w_k1_INTC_s110
                                 + 0.986171 w_k1_KO_s110
                                 + 1.01346 w_k1_MAT_s110
                                 + 1.16889 w_k1_MSFT_s110 + 0.9183 w_k1_PG_s110
                                 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.11461 w_k1_ADM_s110
                                 + 1.04717 w_k1_AXP_s110
                                 + 1.10549 w_k1_DIS_s110 + 1.0753 w_k1_FDX_s110
                                 + 0.839126 w_k1_IBM_s110
                                 + 1.31396 w_k1_INTC_s110
                                 + 0.930175 w_k1_KO_s110
                                 + 0.861286 w_k1_MAT_s110
                                 + 1.31189 w_k1_MSFT_s110
                                 + 1.01344 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.825982 w_k1_ADM_s110
                                 + 0.534406 w_k1_AXP_s110
                                 + 0.558112 w_k1_DIS_s110
                                 + 0.826786 w_k1_FDX_s110
                                 + 0.560088 w_k1_IBM_s110
                                 + 1.02264 w_k1_INTC_s110
                                 + 0.673345 w_k1_KO_s110
                                 + 0.441231 w_k1_MAT_s110
                                 + 0.942973 w_k1_MSFT_s110
                                 + 0.713876 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.60629 w_k1_ADM_s110
                                 + 1.38535 w_k1_AXP_s110
                                 + 1.48029 w_k1_DIS_s110 + 1.3579 w_k1_FDX_s110
                                 + 1.07875 w_k1_IBM_s110
                                 + 1.17893 w_k1_INTC_s110
                                 + 1.26071 w_k1_KO_s110 + 1.56764 w_k1_MAT_s110
                                 + 1.29917 w_k1_MSFT_s110
                                 + 1.17279 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.802066 w_k1_ADM_s111
                                 + 0.825269 w_k1_AXP_s111
                                 + 0.644932 w_k1_DIS_s111
                                 + 0.805693 w_k1_FDX_s111
                                 + 0.78313 w_k1_IBM_s111
                                 + 0.798618 w_k1_INTC_s111
                                 + 0.72168 w_k1_KO_s111
                                 + 0.907147 w_k1_MAT_s111
                                 + 0.782548 w_k1_MSFT_s111
                                 + 0.796915 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.98154 w_k1_ADM_s111
                                 + 1.21424 w_k1_AXP_s111
                                 + 1.52849 w_k1_DIS_s111 + 1.192 w_k1_FDX_s111
                                 + 1.42862 w_k1_IBM_s111
                                 + 1.38412 w_k1_INTC_s111 + 1.4895 w_k1_KO_s111
                                 + 1.52205 w_k1_MAT_s111
                                 + 1.19459 w_k1_MSFT_s111
                                 + 1.09729 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.01732 w_k1_ADM_s111
                                 + 1.25827 w_k1_AXP_s111
                                 + 1.02345 w_k1_DIS_s111
                                 + 1.16773 w_k1_FDX_s111
                                 + 0.731231 w_k1_IBM_s111
                                 + 0.868078 w_k1_INTC_s111
                                 + 0.76315 w_k1_KO_s111
                                 + 0.952326 w_k1_MAT_s111
                                 + 0.975644 w_k1_MSFT_s111
                                 + 0.977206 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.15182 w_k1_ADM_s111
                                 + 1.23829 w_k1_AXP_s111
                                 + 1.42559 w_k1_DIS_s111
                                 + 1.17847 w_k1_FDX_s111
                                 + 0.983256 w_k1_IBM_s111
                                 + 1.05683 w_k1_INTC_s111
                                 + 1.07935 w_k1_KO_s111 + 1.01159 w_k1_MAT_s111
                                 + 1.18591 w_k1_MSFT_s111
                                 + 1.10379 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.888259 w_k1_ADM_s111
                                 + 1.02034 w_k1_AXP_s111 + 1.0623 w_k1_DIS_s111
                                 + 0.911877 w_k1_FDX_s111
                                 + 0.881401 w_k1_IBM_s111
                                 + 1.0934 w_k1_INTC_s111
                                 + 0.955305 w_k1_KO_s111
                                 + 0.577798 w_k1_MAT_s111
                                 + 0.933165 w_k1_MSFT_s111
                                 + 0.916657 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.973228 w_k1_ADM_s111
                                 + 1.43796 w_k1_AXP_s111
                                 + 1.35557 w_k1_DIS_s111
                                 + 1.19053 w_k1_FDX_s111
                                 + 1.37076 w_k1_IBM_s111
                                 + 1.13942 w_k1_INTC_s111 + 1.2327 w_k1_KO_s111
                                 + 1.56382 w_k1_MAT_s111
                                 + 1.07729 w_k1_MSFT_s111 + 1.0471 w_k1_PG_s111
                                 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.802645 w_k1_ADM_s111
                                 + 0.769719 w_k1_AXP_s111
                                 + 0.857659 w_k1_DIS_s111
                                 + 0.800676 w_k1_FDX_s111
                                 + 0.961551 w_k1_IBM_s111
                                 + 0.794302 w_k1_INTC_s111
                                 + 0.845637 w_k1_KO_s111
                                 + 1.15441 w_k1_MAT_s111
                                 + 0.766487 w_k1_MSFT_s111
                                 + 0.900386 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.858192 w_k1_ADM_s111
                                 + 1.06464 w_k1_AXP_s111
                                 + 1.07856 w_k1_DIS_s111
                                 + 0.938162 w_k1_FDX_s111
                                 + 1.44171 w_k1_IBM_s111
                                 + 1.12192 w_k1_INTC_s111
                                 + 1.16079 w_k1_KO_s111 + 1.17058 w_k1_MAT_s111
                                 + 0.972441 w_k1_MSFT_s111
                                 + 1.05239 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.06297 w_k1_ADM_s112
                                 + 0.944435 w_k1_AXP_s112
                                 + 1.20432 w_k1_DIS_s112
                                 + 1.12062 w_k1_FDX_s112
                                 + 1.28534 w_k1_IBM_s112
                                 + 1.24957 w_k1_INTC_s112
                                 + 1.09218 w_k1_KO_s112
                                 + 0.641014 w_k1_MAT_s112
                                 + 1.0854 w_k1_MSFT_s112
                                 + 0.997284 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.877682 w_k1_ADM_s112
                                 + 0.984102 w_k1_AXP_s112
                                 + 1.00845 w_k1_DIS_s112
                                 + 1.05368 w_k1_FDX_s112
                                 + 0.775875 w_k1_IBM_s112
                                 + 0.954156 w_k1_INTC_s112
                                 + 0.849338 w_k1_KO_s112
                                 + 0.930622 w_k1_MAT_s112
                                 + 1.04993 w_k1_MSFT_s112
                                 + 0.918815 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.11603 w_k1_ADM_s112
                                 + 0.997618 w_k1_AXP_s112
                                 + 0.70944 w_k1_DIS_s112
                                 + 0.956747 w_k1_FDX_s112
                                 + 0.741009 w_k1_IBM_s112
                                 + 0.941498 w_k1_INTC_s112
                                 + 0.73643 w_k1_KO_s112
                                 + 0.766837 w_k1_MAT_s112
                                 + 0.82118 w_k1_MSFT_s112
                                 + 0.815633 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.963037 w_k1_ADM_s112
                                 + 0.689836 w_k1_AXP_s112
                                 + 0.661245 w_k1_DIS_s112
                                 + 0.730073 w_k1_FDX_s112
                                 + 0.851795 w_k1_IBM_s112
                                 + 0.653115 w_k1_INTC_s112
                                 + 0.820329 w_k1_KO_s112
                                 + 0.657156 w_k1_MAT_s112
                                 + 0.649806 w_k1_MSFT_s112
                                 + 0.838191 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.993193 w_k1_ADM_s112
                                 + 1.05125 w_k1_AXP_s112
                                 + 0.887914 w_k1_DIS_s112
                                 + 0.814458 w_k1_FDX_s112
                                 + 1.51008 w_k1_IBM_s112
                                 + 0.947444 w_k1_INTC_s112
                                 + 1.02235 w_k1_KO_s112 + 1.25115 w_k1_MAT_s112
                                 + 0.899356 w_k1_MSFT_s112
                                 + 0.890299 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.11719 w_k1_ADM_s112
                                 + 1.42275 w_k1_AXP_s112
                                 + 1.34257 w_k1_DIS_s112 + 1.1058 w_k1_FDX_s112
                                 + 1.56719 w_k1_IBM_s112
                                 + 1.39395 w_k1_INTC_s112
                                 + 1.27294 w_k1_KO_s112 + 1.41547 w_k1_MAT_s112
                                 + 1.23069 w_k1_MSFT_s112
                                 + 1.01926 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.50241 w_k1_ADM_s112
                                 + 0.844046 w_k1_AXP_s112
                                 + 1.67612 w_k1_DIS_s112 + 1.1743 w_k1_FDX_s112
                                 + 1.10721 w_k1_IBM_s112
                                 + 1.20485 w_k1_INTC_s112
                                 + 1.33188 w_k1_KO_s112
                                 + 0.690256 w_k1_MAT_s112
                                 + 1.29499 w_k1_MSFT_s112
                                 + 1.17414 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.60025 w_k1_ADM_s112
                                 + 1.46151 w_k1_AXP_s112
                                 + 1.85204 w_k1_DIS_s112
                                 + 1.40638 w_k1_FDX_s112
                                 + 1.07927 w_k1_IBM_s112
                                 + 1.19178 w_k1_INTC_s112 + 1.6034 w_k1_KO_s112
                                 + 1.10428 w_k1_MAT_s112
                                 + 1.53445 w_k1_MSFT_s112
                                 + 1.25496 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.1256 w_k1_ADM_s113
                                 + 1.09563 w_k1_AXP_s113
                                 + 0.796938 w_k1_DIS_s113
                                 + 0.989113 w_k1_FDX_s113
                                 + 1.03813 w_k1_IBM_s113
                                 + 1.10842 w_k1_INTC_s113
                                 + 0.989616 w_k1_KO_s113
                                 + 1.21013 w_k1_MAT_s113
                                 + 0.839604 w_k1_MSFT_s113
                                 + 1.03912 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.18292 w_k1_ADM_s113
                                 + 1.30974 w_k1_AXP_s113
                                 + 0.964161 w_k1_DIS_s113
                                 + 1.04427 w_k1_FDX_s113
                                 + 1.30057 w_k1_IBM_s113
                                 + 1.01461 w_k1_INTC_s113
                                 + 1.12702 w_k1_KO_s113 + 1.06873 w_k1_MAT_s113
                                 + 1.00807 w_k1_MSFT_s113
                                 + 1.07706 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.705314 w_k1_ADM_s113
                                 + 0.651897 w_k1_AXP_s113
                                 + 0.691662 w_k1_DIS_s113
                                 + 0.739348 w_k1_FDX_s113
                                 + 0.739485 w_k1_IBM_s113
                                 + 0.861352 w_k1_INTC_s113
                                 + 0.822744 w_k1_KO_s113
                                 + 0.507348 w_k1_MAT_s113
                                 + 0.839251 w_k1_MSFT_s113
                                 + 0.78493 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.08562 w_k1_ADM_s113
                                 + 1.74305 w_k1_AXP_s113
                                 + 1.94203 w_k1_DIS_s113
                                 + 1.36465 w_k1_FDX_s113
                                 + 1.49642 w_k1_IBM_s113
                                 + 1.82425 w_k1_INTC_s113
                                 + 1.55141 w_k1_KO_s113 + 1.50071 w_k1_MAT_s113
                                 + 1.46406 w_k1_MSFT_s113
                                 + 1.43244 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.910307 w_k1_ADM_s113
                                 + 0.606076 w_k1_AXP_s113
                                 + 0.815109 w_k1_DIS_s113
                                 + 0.775001 w_k1_FDX_s113
                                 + 0.965642 w_k1_IBM_s113
                                 + 1.07067 w_k1_INTC_s113
                                 + 0.929394 w_k1_KO_s113
                                 + 0.690743 w_k1_MAT_s113
                                 + 0.961398 w_k1_MSFT_s113
                                 + 0.908909 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.02318 w_k1_ADM_s113
                                 + 1.49605 w_k1_AXP_s113
                                 + 1.31234 w_k1_DIS_s113
                                 + 1.14984 w_k1_FDX_s113
                                 + 1.00235 w_k1_IBM_s113
                                 + 1.4514 w_k1_INTC_s113 + 1.03573 w_k1_KO_s113
                                 + 1.06181 w_k1_MAT_s113
                                 + 1.29864 w_k1_MSFT_s113 + 1.0907 w_k1_PG_s113
                                 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.01333 w_k1_ADM_s113
                                 + 1.16137 w_k1_AXP_s113
                                 + 1.14689 w_k1_DIS_s113
                                 + 1.14517 w_k1_FDX_s113
                                 + 0.757817 w_k1_IBM_s113
                                 + 1.33839 w_k1_INTC_s113
                                 + 0.944413 w_k1_KO_s113
                                 + 0.786183 w_k1_MAT_s113
                                 + 1.17959 w_k1_MSFT_s113
                                 + 1.00118 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.946193 w_k1_ADM_s113
                                 + 1.06764 w_k1_AXP_s113
                                 + 1.31979 w_k1_DIS_s113
                                 + 0.969985 w_k1_FDX_s113
                                 + 1.21409 w_k1_IBM_s113
                                 + 0.985043 w_k1_INTC_s113
                                 + 1.21006 w_k1_KO_s113 + 1.28037 w_k1_MAT_s113
                                 + 1.08731 w_k1_MSFT_s113
                                 + 1.11987 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.880429 w_k1_ADM_s114
                                 + 1.17567 w_k1_AXP_s114 + 1.2171 w_k1_DIS_s114
                                 + 1.09184 w_k1_FDX_s114
                                 + 1.75608 w_k1_IBM_s114 + 1.18 w_k1_INTC_s114
                                 + 1.28126 w_k1_KO_s114 + 1.85957 w_k1_MAT_s114
                                 + 1.09339 w_k1_MSFT_s114 + 1.0382 w_k1_PG_s114
                                 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.905735 w_k1_ADM_s114
                                 + 0.998333 w_k1_AXP_s114
                                 + 0.892956 w_k1_DIS_s114
                                 + 0.950246 w_k1_FDX_s114
                                 + 1.14924 w_k1_IBM_s114
                                 + 0.81009 w_k1_INTC_s114
                                 + 0.965041 w_k1_KO_s114
                                 + 1.23214 w_k1_MAT_s114
                                 + 0.970396 w_k1_MSFT_s114
                                 + 0.955188 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.999017 w_k1_ADM_s114
                                 + 1.06854 w_k1_AXP_s114
                                 + 1.58397 w_k1_DIS_s114
                                 + 1.29844 w_k1_FDX_s114
                                 + 1.35116 w_k1_IBM_s114
                                 + 1.47473 w_k1_INTC_s114
                                 + 1.40621 w_k1_KO_s114 + 1.27766 w_k1_MAT_s114
                                 + 1.53008 w_k1_MSFT_s114
                                 + 1.06414 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.660037 w_k1_ADM_s114
                                 + 0.764764 w_k1_AXP_s114
                                 + 0.58295 w_k1_DIS_s114
                                 + 0.745321 w_k1_FDX_s114
                                 + 0.669282 w_k1_IBM_s114
                                 + 0.912952 w_k1_INTC_s114
                                 + 0.676319 w_k1_KO_s114
                                 + 0.830646 w_k1_MAT_s114
                                 + 0.941025 w_k1_MSFT_s114
                                 + 0.779635 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.44838 w_k1_ADM_s114
                                 + 1.62213 w_k1_AXP_s114
                                 + 1.22755 w_k1_DIS_s114
                                 + 1.14944 w_k1_FDX_s114
                                 + 1.00073 w_k1_IBM_s114
                                 + 1.23174 w_k1_INTC_s114 + 1.1662 w_k1_KO_s114
                                 + 1.09917 w_k1_MAT_s114
                                 + 1.26296 w_k1_MSFT_s114
                                 + 1.11073 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.82892 w_k1_ADM_s114
                                 + 1.12879 w_k1_AXP_s114
                                 + 1.19402 w_k1_DIS_s114 + 1.2382 w_k1_FDX_s114
                                 + 0.810181 w_k1_IBM_s114
                                 + 1.47813 w_k1_INTC_s114
                                 + 1.14584 w_k1_KO_s114
                                 + 0.804028 w_k1_MAT_s114
                                 + 1.43042 w_k1_MSFT_s114
                                 + 1.12671 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.31031 w_k1_ADM_s114
                                 + 1.78588 w_k1_AXP_s114
                                 + 1.45631 w_k1_DIS_s114
                                 + 1.36968 w_k1_FDX_s114
                                 + 1.22436 w_k1_IBM_s114
                                 + 1.54744 w_k1_INTC_s114
                                 + 1.19836 w_k1_KO_s114 + 1.21168 w_k1_MAT_s114
                                 + 1.3839 w_k1_MSFT_s114 + 1.03897 w_k1_PG_s114
                                 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.964104 w_k1_ADM_s114
                                 + 0.801423 w_k1_AXP_s114
                                 + 0.751327 w_k1_DIS_s114
                                 + 0.80366 w_k1_FDX_s114
                                 + 0.902791 w_k1_IBM_s114
                                 + 0.960384 w_k1_INTC_s114
                                 + 0.954898 w_k1_KO_s114
                                 + 0.839275 w_k1_MAT_s114
                                 + 0.905339 w_k1_MSFT_s114
                                 + 0.882096 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.23233 w_k1_ADM_s115
                                 + 0.835928 w_k1_AXP_s115
                                 + 1.08092 w_k1_DIS_s115
                                 + 1.12872 w_k1_FDX_s115
                                 + 0.756709 w_k1_IBM_s115
                                 + 0.933744 w_k1_INTC_s115
                                 + 1.01265 w_k1_KO_s115 + 1.00443 w_k1_MAT_s115
                                 + 1.18176 w_k1_MSFT_s115
                                 + 1.03294 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.900865 w_k1_ADM_s115
                                 + 0.810169 w_k1_AXP_s115
                                 + 0.756677 w_k1_DIS_s115
                                 + 0.802664 w_k1_FDX_s115
                                 + 0.741455 w_k1_IBM_s115
                                 + 0.716902 w_k1_INTC_s115
                                 + 0.81419 w_k1_KO_s115
                                 + 0.650646 w_k1_MAT_s115
                                 + 0.832008 w_k1_MSFT_s115
                                 + 0.929897 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.922847 w_k1_ADM_s115
                                 + 0.937789 w_k1_AXP_s115
                                 + 1.24753 w_k1_DIS_s115
                                 + 0.937005 w_k1_FDX_s115
                                 + 1.2791 w_k1_IBM_s115
                                 + 0.991078 w_k1_INTC_s115
                                 + 1.2708 w_k1_KO_s115 + 1.46138 w_k1_MAT_s115
                                 + 1.039 w_k1_MSFT_s115 + 1.01238 w_k1_PG_s115
                                 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.03055 w_k1_ADM_s115
                                 + 1.49211 w_k1_AXP_s115
                                 + 1.38892 w_k1_DIS_s115
                                 + 1.19235 w_k1_FDX_s115
                                 + 1.41961 w_k1_IBM_s115
                                 + 1.08013 w_k1_INTC_s115
                                 + 1.36398 w_k1_KO_s115 + 1.68547 w_k1_MAT_s115
                                 + 1.08936 w_k1_MSFT_s115
                                 + 1.14619 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.29713 w_k1_ADM_s115
                                 + 1.43483 w_k1_AXP_s115
                                 + 1.42375 w_k1_DIS_s115
                                 + 1.12611 w_k1_FDX_s115 + 1.734 w_k1_IBM_s115
                                 + 1.00971 w_k1_INTC_s115
                                 + 1.59494 w_k1_KO_s115 + 2.31243 w_k1_MAT_s115
                                 + 1.09727 w_k1_MSFT_s115
                                 + 1.26353 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.01252 w_k1_ADM_s115
                                 + 1.02969 w_k1_AXP_s115
                                 + 0.759463 w_k1_DIS_s115
                                 + 0.776578 w_k1_FDX_s115
                                 + 1.15774 w_k1_IBM_s115
                                 + 0.677009 w_k1_INTC_s115
                                 + 0.996468 w_k1_KO_s115
                                 + 1.43737 w_k1_MAT_s115
                                 + 0.732437 w_k1_MSFT_s115
                                 + 0.969546 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.29424 w_k1_ADM_s115
                                 + 1.09085 w_k1_AXP_s115
                                 + 1.22473 w_k1_DIS_s115
                                 + 1.15713 w_k1_FDX_s115
                                 + 1.03801 w_k1_IBM_s115
                                 + 1.12138 w_k1_INTC_s115 + 1.1499 w_k1_KO_s115
                                 + 1.20875 w_k1_MAT_s115
                                 + 1.13616 w_k1_MSFT_s115
                                 + 0.988933 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.810989 w_k1_ADM_s115
                                 + 0.986989 w_k1_AXP_s115
                                 + 0.982863 w_k1_DIS_s115
                                 + 0.950257 w_k1_FDX_s115
                                 + 0.964998 w_k1_IBM_s115
                                 + 0.835228 w_k1_INTC_s115
                                 + 1.04965 w_k1_KO_s115 + 1.19403 w_k1_MAT_s115
                                 + 0.871865 w_k1_MSFT_s115
                                 + 0.969788 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.14291 w_k1_ADM_s116
                                 + 1.1063 w_k1_AXP_s116 + 1.17059 w_k1_DIS_s116
                                 + 1.00264 w_k1_FDX_s116
                                 + 1.47651 w_k1_IBM_s116
                                 + 1.19582 w_k1_INTC_s116
                                 + 1.27945 w_k1_KO_s116 + 1.15821 w_k1_MAT_s116
                                 + 1.06674 w_k1_MSFT_s116
                                 + 1.13837 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.983889 w_k1_ADM_s116
                                 + 1.36143 w_k1_AXP_s116
                                 + 1.16085 w_k1_DIS_s116
                                 + 1.11129 w_k1_FDX_s116
                                 + 1.25021 w_k1_IBM_s116
                                 + 1.35825 w_k1_INTC_s116 + 1.2273 w_k1_KO_s116
                                 + 1.00658 w_k1_MAT_s116
                                 + 0.982452 w_k1_MSFT_s116
                                 + 1.01725 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.00655 w_k1_ADM_s116
                                 + 1.01821 w_k1_AXP_s116
                                 + 1.20164 w_k1_DIS_s116
                                 + 1.13324 w_k1_FDX_s116
                                 + 0.801922 w_k1_IBM_s116
                                 + 1.3626 w_k1_INTC_s116
                                 + 0.940351 w_k1_KO_s116
                                 + 0.579456 w_k1_MAT_s116
                                 + 1.1695 w_k1_MSFT_s116 + 1.01794 w_k1_PG_s116
                                 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.857617 w_k1_ADM_s116
                                 + 1.06578 w_k1_AXP_s116
                                 + 0.926106 w_k1_DIS_s116
                                 + 0.962663 w_k1_FDX_s116
                                 + 0.825263 w_k1_IBM_s116
                                 + 1.11979 w_k1_INTC_s116
                                 + 0.858193 w_k1_KO_s116
                                 + 0.658749 w_k1_MAT_s116
                                 + 0.910584 w_k1_MSFT_s116
                                 + 0.993894 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.880631 w_k1_ADM_s116
                                 + 0.783475 w_k1_AXP_s116
                                 + 0.807529 w_k1_DIS_s116
                                 + 0.868724 w_k1_FDX_s116
                                 + 0.783906 w_k1_IBM_s116
                                 + 0.968475 w_k1_INTC_s116
                                 + 0.833914 w_k1_KO_s116
                                 + 0.629519 w_k1_MAT_s116
                                 + 0.919728 w_k1_MSFT_s116
                                 + 0.969288 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.725677 w_k1_ADM_s116
                                 + 1.20518 w_k1_AXP_s116
                                 + 0.970122 w_k1_DIS_s116
                                 + 0.905351 w_k1_FDX_s116
                                 + 1.13437 w_k1_IBM_s116
                                 + 0.905484 w_k1_INTC_s116
                                 + 0.922085 w_k1_KO_s116
                                 + 1.34582 w_k1_MAT_s116
                                 + 0.783156 w_k1_MSFT_s116
                                 + 1.06755 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.727822 w_k1_ADM_s116
                                 + 0.816016 w_k1_AXP_s116
                                 + 0.709119 w_k1_DIS_s116
                                 + 0.714149 w_k1_FDX_s116
                                 + 0.938176 w_k1_IBM_s116
                                 + 0.912885 w_k1_INTC_s116
                                 + 0.809913 w_k1_KO_s116
                                 + 0.756266 w_k1_MAT_s116
                                 + 0.676252 w_k1_MSFT_s116
                                 + 0.813629 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.2381 w_k1_ADM_s116
                                 + 1.6293 w_k1_AXP_s116 + 1.71454 w_k1_DIS_s116
                                 + 1.37637 w_k1_FDX_s116
                                 + 1.48272 w_k1_IBM_s116
                                 + 1.28285 w_k1_INTC_s116
                                 + 1.28571 w_k1_KO_s116 + 1.12001 w_k1_MAT_s116
                                 + 1.25416 w_k1_MSFT_s116
                                 + 1.39165 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.39657 w_k1_ADM_s117
                                 + 1.28389 w_k1_AXP_s117
                                 + 1.56464 w_k1_DIS_s117
                                 + 1.33598 w_k1_FDX_s117 + 1.2045 w_k1_IBM_s117
                                 + 1.22561 w_k1_INTC_s117
                                 + 1.33729 w_k1_KO_s117 + 1.61415 w_k1_MAT_s117
                                 + 1.36521 w_k1_MSFT_s117
                                 + 1.16424 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.898014 w_k1_ADM_s117
                                 + 0.730695 w_k1_AXP_s117
                                 + 0.802088 w_k1_DIS_s117
                                 + 0.816843 w_k1_FDX_s117
                                 + 1.04167 w_k1_IBM_s117
                                 + 0.964576 w_k1_INTC_s117
                                 + 0.858321 w_k1_KO_s117
                                 + 0.829504 w_k1_MAT_s117
                                 + 0.794897 w_k1_MSFT_s117
                                 + 0.884546 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.23722 w_k1_ADM_s117
                                 + 1.07556 w_k1_AXP_s117 + 1.2108 w_k1_DIS_s117
                                 + 1.22025 w_k1_FDX_s117
                                 + 0.986572 w_k1_IBM_s117
                                 + 1.00572 w_k1_INTC_s117
                                 + 1.01219 w_k1_KO_s117
                                 + 0.882134 w_k1_MAT_s117
                                 + 1.16364 w_k1_MSFT_s117
                                 + 1.10749 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.734599 w_k1_ADM_s117
                                 + 0.71034 w_k1_AXP_s117
                                 + 0.512503 w_k1_DIS_s117
                                 + 0.708864 w_k1_FDX_s117
                                 + 0.782413 w_k1_IBM_s117
                                 + 0.844338 w_k1_INTC_s117
                                 + 0.697376 w_k1_KO_s117
                                 + 0.541891 w_k1_MAT_s117
                                 + 0.836208 w_k1_MSFT_s117
                                 + 0.808621 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.36597 w_k1_ADM_s117
                                 + 1.40424 w_k1_AXP_s117
                                 + 1.13336 w_k1_DIS_s117
                                 + 1.08146 w_k1_FDX_s117
                                 + 1.50937 w_k1_IBM_s117
                                 + 1.00594 w_k1_INTC_s117
                                 + 1.11065 w_k1_KO_s117 + 1.9705 w_k1_MAT_s117
                                 + 0.955041 w_k1_MSFT_s117
                                 + 1.22993 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.00518 w_k1_ADM_s117
                                 + 0.761386 w_k1_AXP_s117
                                 + 0.858792 w_k1_DIS_s117
                                 + 0.910823 w_k1_FDX_s117
                                 + 0.829651 w_k1_IBM_s117
                                 + 1.04827 w_k1_INTC_s117
                                 + 0.768401 w_k1_KO_s117
                                 + 0.605398 w_k1_MAT_s117
                                 + 0.988528 w_k1_MSFT_s117
                                 + 0.852834 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.838771 w_k1_ADM_s117
                                 + 0.942708 w_k1_AXP_s117
                                 + 0.790241 w_k1_DIS_s117
                                 + 0.938514 w_k1_FDX_s117
                                 + 0.636849 w_k1_IBM_s117
                                 + 0.967755 w_k1_INTC_s117
                                 + 0.85073 w_k1_KO_s117
                                 + 0.779335 w_k1_MAT_s117
                                 + 0.951728 w_k1_MSFT_s117
                                 + 0.944294 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.919618 w_k1_ADM_s117
                                 + 0.975614 w_k1_AXP_s117
                                 + 0.95793 w_k1_DIS_s117
                                 + 0.967448 w_k1_FDX_s117
                                 + 1.11421 w_k1_IBM_s117
                                 + 1.01498 w_k1_INTC_s117
                                 + 1.10441 w_k1_KO_s117 + 1.43667 w_k1_MAT_s117
                                 + 1.25471 w_k1_MSFT_s117
                                 + 0.924099 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.00817938775739713 w_k1_ADM_s000
                                 + 0.0276012328809333 w_k1_ADM_s001
                                 + 0.00911718958099756 w_k1_ADM_s002
                                 + 0.0247007486673864 w_k1_ADM_s003
                                 + 0.0224920525063816 w_k1_ADM_s004
                                 + 0.014534139566794 w_k1_ADM_s005
                                 + 0.00894230975148726 w_k1_ADM_s006
                                 + 0.0191541286265075 w_k1_ADM_s007
                                 + 0.0865824339191338 w_k1_ADM_s010
                                 + 0.0565243167098482 w_k1_ADM_s011
                                 + 0.0586636158795466 w_k1_ADM_s012
                                 + 0.0742704804869925 w_k1_ADM_s013
                                 + 0.145513202897219 w_k1_ADM_s014
                                 + 0.0334246230784562 w_k1_ADM_s015
                                 + 0.0595832605567617 w_k1_ADM_s016
                                 + 0.0799968415042487 w_k1_ADM_s017
                                 + 0.0185098407195181 w_k1_ADM_s100
                                 + 0.0385828045359134 w_k1_ADM_s101
                                 + 0.0357035148507683 w_k1_ADM_s102
                                 + 0.01724429827843 w_k1_ADM_s103
                                 + 0.0578901725388374 w_k1_ADM_s104
                                 + 0.0181149434986552 w_k1_ADM_s105
                                 + 0.0183147607526262 w_k1_ADM_s106
                                 + 0.0316374136796341 w_k1_ADM_s107
                                 + 0.00902985046008672 w_k1_ADM_s110
                                 + 0.00566280758854239 w_k1_ADM_s111
                                 + 0.0103556019185892 w_k1_ADM_s112
                                 + 0.0118446151569207 w_k1_ADM_s113
                                 + 0.00253605362930306 w_k1_ADM_s114
                                 + 0.00968927755625288 w_k1_ADM_s115
                                 + 0.00497262445570918 w_k1_ADM_s116
                                 + 0.00699264218382877 w_k1_ADM_s117
                                 + 0.00827991595020691 w_k1_AXP_s000
                                 + 0.0349914567144478 w_k1_AXP_s001
                                 + 0.0097256132725046 w_k1_AXP_s002
                                 + 0.0250265483518622 w_k1_AXP_s003
                                 + 0.0225580714521616 w_k1_AXP_s004
                                 + 0.0166054230256971 w_k1_AXP_s005
                                 + 0.00967349922138543 w_k1_AXP_s006
                                 + 0.0174747488554664 w_k1_AXP_s007
                                 + 0.100966532466007 w_k1_AXP_s010
                                 + 0.0630927051018704 w_k1_AXP_s011
                                 + 0.0581632435335776 w_k1_AXP_s012
                                 + 0.0698458684331251 w_k1_AXP_s013
                                 + 0.139571145461129 w_k1_AXP_s014
                                 + 0.0367313581951315 w_k1_AXP_s015
                                 + 0.0616991398261181 w_k1_AXP_s016
                                 + 0.0662865772926969 w_k1_AXP_s017
                                 + 0.0205751366468602 w_k1_AXP_s100
                                 + 0.0406880946471107 w_k1_AXP_s101
                                 + 0.0324739458990227 w_k1_AXP_s102
                                 + 0.0143500516252851 w_k1_AXP_s103
                                 + 0.0463617429429507 w_k1_AXP_s104
                                 + 0.0200240572647666 w_k1_AXP_s105
                                 + 0.0180250017156662 w_k1_AXP_s106
                                 + 0.0330558185477465 w_k1_AXP_s107
                                 + 0.00846478242369554 w_k1_AXP_s110
                                 + 0.00673468646537549 w_k1_AXP_s111
                                 + 0.00986040741062916 w_k1_AXP_s112
                                 + 0.0130465410757742 w_k1_AXP_s113
                                 + 0.00286122067526569 w_k1_AXP_s114
                                 + 0.00970933877774154 w_k1_AXP_s115
                                 + 0.00577775480076396 w_k1_AXP_s116
                                 + 0.00653464218547056 w_k1_AXP_s117
                                 + 0.00773514819942403 w_k1_DIS_s000
                                 + 0.0309140300555577 w_k1_DIS_s001
                                 + 0.00999700154749096 w_k1_DIS_s002
                                 + 0.0243831350671203 w_k1_DIS_s003
                                 + 0.023629613934783 w_k1_DIS_s004
                                 + 0.0153283380843228 w_k1_DIS_s005
                                 + 0.00899722600497951 w_k1_DIS_s006
                                 + 0.0159534136081615 w_k1_DIS_s007
                                 + 0.093268223630428 w_k1_DIS_s010
                                 + 0.0558825091516062 w_k1_DIS_s011
                                 + 0.0661297871467757 w_k1_DIS_s012
                                 + 0.0670668730203611 w_k1_DIS_s013
                                 + 0.148057757780488 w_k1_DIS_s014
                                 + 0.0372386591667912 w_k1_DIS_s015
                                 + 0.0598338181597155 w_k1_DIS_s016
                                 + 0.0689781613979865 w_k1_DIS_s017
                                 + 0.0189462828496302 w_k1_DIS_s100
                                 + 0.0412649652548855 w_k1_DIS_s101
                                 + 0.0326829209550463 w_k1_DIS_s102
                                 + 0.0166950669378584 w_k1_DIS_s103
                                 + 0.048218160417627 w_k1_DIS_s104
                                 + 0.0217942106255694 w_k1_DIS_s105
                                 + 0.0181930763401073 w_k1_DIS_s106
                                 + 0.0344368238975333 w_k1_DIS_s107
                                 + 0.00833163183462513 w_k1_DIS_s110
                                 + 0.00656487699774464 w_k1_DIS_s111
                                 + 0.0100347903106722 w_k1_DIS_s112
                                 + 0.0126836482404099 w_k1_DIS_s113
                                 + 0.00270228179437525 w_k1_DIS_s114
                                 + 0.00995150363268286 w_k1_DIS_s115
                                 + 0.00572401066613478 w_k1_DIS_s116
                                 + 0.00672174353939443 w_k1_DIS_s117
                                 + 0.00790298688287104 w_k1_FDX_s000
                                 + 0.029993149813797 w_k1_FDX_s001
                                 + 0.00926354981359697 w_k1_FDX_s002
                                 + 0.0236776155516747 w_k1_FDX_s003
                                 + 0.022357208148434 w_k1_FDX_s004
                                 + 0.0150107835552844 w_k1_FDX_s005
                                 + 0.00886378222868956 w_k1_FDX_s006
                                 + 0.0165075445876028 w_k1_FDX_s007
                                 + 0.0950454084603579 w_k1_FDX_s010
                                 + 0.0584675366400014 w_k1_FDX_s011
                                 + 0.0628989000113497 w_k1_FDX_s012
                                 + 0.0643922000443091 w_k1_FDX_s013
                                 + 0.142167845329228 w_k1_FDX_s014
                                 + 0.0365926547358797 w_k1_FDX_s015
                                 + 0.062023951074403 w_k1_FDX_s016
                                 + 0.0673414833034433 w_k1_FDX_s017
                                 + 0.018805087506365 w_k1_FDX_s100
                                 + 0.0404035652595705 w_k1_FDX_s101
                                 + 0.0323884764096898 w_k1_FDX_s102
                                 + 0.0159084854412649 w_k1_FDX_s103
                                 + 0.0502532095911895 w_k1_FDX_s104
                                 + 0.0207463344450988 w_k1_FDX_s105
                                 + 0.0182415204107968 w_k1_FDX_s106
                                 + 0.0335903487698286 w_k1_FDX_s107
                                 + 0.00880673039615998 w_k1_FDX_s110
                                 + 0.00618208909080464 w_k1_FDX_s111
                                 + 0.0097321614495647 w_k1_FDX_s112
                                 + 0.0119902838965466 w_k1_FDX_s113
                                 + 0.00272056768221784 w_k1_FDX_s114
                                 + 0.00926753280970344 w_k1_FDX_s115
                                 + 0.00534054436026973 w_k1_FDX_s116
                                 + 0.00683878939946867 w_k1_FDX_s117
                                 + 0.00825408829981891 w_k1_IBM_s000
                                 + 0.0303619369950099 w_k1_IBM_s001
                                 + 0.00828365260562319 w_k1_IBM_s002
                                 + 0.0257883674326457 w_k1_IBM_s003
                                 + 0.0239468444525894 w_k1_IBM_s004
                                 + 0.0160267126238854 w_k1_IBM_s005
                                 + 0.00951857057849777 w_k1_IBM_s006
                                 + 0.0164054034599903 w_k1_IBM_s007
                                 + 0.0943905804039196 w_k1_IBM_s010
                                 + 0.0564623256247443 w_k1_IBM_s011
                                 + 0.0626682819718824 w_k1_IBM_s012
                                 + 0.06471626347436 w_k1_IBM_s013
                                 + 0.150181778038987 w_k1_IBM_s014
                                 + 0.0359055581311317 w_k1_IBM_s015
                                 + 0.0633745508763056 w_k1_IBM_s016
                                 + 0.0692780563442751 w_k1_IBM_s017
                                 + 0.0197098857984319 w_k1_IBM_s100
                                 + 0.0392525469913286 w_k1_IBM_s101
                                 + 0.0323796333077077 w_k1_IBM_s102
                                 + 0.0160941847152744 w_k1_IBM_s103
                                 + 0.0509715225232954 w_k1_IBM_s104
                                 + 0.0213620876321175 w_k1_IBM_s105
                                 + 0.0179797536730698 w_k1_IBM_s106
                                 + 0.0350334492893028 w_k1_IBM_s107
                                 + 0.008759627018236 w_k1_IBM_s110
                                 + 0.00624166031348019 w_k1_IBM_s111
                                 + 0.010052683084481 w_k1_IBM_s112
                                 + 0.0123635440122339 w_k1_IBM_s113
                                 + 0.00271925081725019 w_k1_IBM_s114
                                 + 0.0100063791560558 w_k1_IBM_s115
                                 + 0.00548701396302632 w_k1_IBM_s116
                                 + 0.0064572198166666 w_k1_IBM_s117
                                 + 0.00760884126037209 w_k1_INTC_s000
                                 + 0.0287889679259999 w_k1_INTC_s001
                                 + 0.00905497917167177 w_k1_INTC_s002
                                 + 0.025159316200661 w_k1_INTC_s003
                                 + 0.0227123845163895 w_k1_INTC_s004
                                 + 0.0150432387053919 w_k1_INTC_s005
                                 + 0.00964729087962541 w_k1_INTC_s006
                                 + 0.0167036919962451 w_k1_INTC_s007
                                 + 0.0887278912935954 w_k1_INTC_s010
                                 + 0.062163967174527 w_k1_INTC_s011
                                 + 0.0629654523565961 w_k1_INTC_s012
                                 + 0.0704033138474612 w_k1_INTC_s013
                                 + 0.120746076247787 w_k1_INTC_s014
                                 + 0.0388551183972968 w_k1_INTC_s015
                                 + 0.0633997121325274 w_k1_INTC_s016
                                 + 0.0636279104010586 w_k1_INTC_s017
                                 + 0.0180447828793436 w_k1_INTC_s100
                                 + 0.0358880657754199 w_k1_INTC_s101
                                 + 0.0339979538625955 w_k1_INTC_s102
                                 + 0.0133429966695024 w_k1_INTC_s103
                                 + 0.051996174456443 w_k1_INTC_s104
                                 + 0.0220983886768561 w_k1_INTC_s105
                                 + 0.021255833087241 w_k1_INTC_s106
                                 + 0.0302451563913839 w_k1_INTC_s107
                                 + 0.0103753960572367 w_k1_INTC_s110
                                 + 0.00613804368021723 w_k1_INTC_s111
                                 + 0.00984853617059849 w_k1_INTC_s112
                                 + 0.0139037041902356 w_k1_INTC_s113
                                 + 0.0030378188361273 w_k1_INTC_s114
                                 + 0.00845041878291418 w_k1_INTC_s115
                                 + 0.00610927068710706 w_k1_INTC_s116
                                 + 0.00672283086195719 w_k1_INTC_s117
                                 + 0.00701342695892431 w_k1_KO_s000
                                 + 0.0283309974546458 w_k1_KO_s001
                                 + 0.00956691552325384 w_k1_KO_s002
                                 + 0.025290108125192 w_k1_KO_s003
                                 + 0.023458017538708 w_k1_KO_s004
                                 + 0.0166883127252017 w_k1_KO_s005
                                 + 0.00891011194409824 w_k1_KO_s006
                                 + 0.0157827458659634 w_k1_KO_s007
                                 + 0.0995773431019517 w_k1_KO_s010
                                 + 0.0529154656239249 w_k1_KO_s011
                                 + 0.0588172450643714 w_k1_KO_s012
                                 + 0.066385376059128 w_k1_KO_s013
                                 + 0.1417965315997 w_k1_KO_s014
                                 + 0.0386240552553322 w_k1_KO_s015
                                 + 0.0543683376866524 w_k1_KO_s016
                                 + 0.0679590912056841 w_k1_KO_s017
                                 + 0.0176301566242525 w_k1_KO_s100
                                 + 0.0388996279186393 w_k1_KO_s101
                                 + 0.0345779106433903 w_k1_KO_s102
                                 + 0.0168174159133889 w_k1_KO_s103
                                 + 0.0485573035772234 w_k1_KO_s104
                                 + 0.023346645633192 w_k1_KO_s105
                                 + 0.0190431976767767 w_k1_KO_s106
                                 + 0.0339335120951409 w_k1_KO_s107
                                 + 0.00877780509577886 w_k1_KO_s110
                                 + 0.00599587297097594 w_k1_KO_s111
                                 + 0.00959646782299162 w_k1_KO_s112
                                 + 0.0125933925731603 w_k1_KO_s113
                                 + 0.00277026820097386 w_k1_KO_s114
                                 + 0.0103822120093831 w_k1_KO_s115
                                 + 0.005287370390181 w_k1_KO_s116
                                 + 0.00647764716260393 w_k1_KO_s117
                                 + 0.00887276872958977 w_k1_MAT_s000
                                 + 0.0363564087103067 w_k1_MAT_s001
                                 + 0.00996473479944591 w_k1_MAT_s002
                                 + 0.0267651889965054 w_k1_MAT_s003
                                 + 0.0251432945964937 w_k1_MAT_s004
                                 + 0.0163277181887813 w_k1_MAT_s005
                                 + 0.00850707194298505 w_k1_MAT_s006
                                 + 0.0133466906037081 w_k1_MAT_s007
                                 + 0.103784558456861 w_k1_MAT_s010
                                 + 0.053394655318607 w_k1_MAT_s011
                                 + 0.0703265908654063 w_k1_MAT_s012
                                 + 0.0532609388557794 w_k1_MAT_s013
                                 + 0.147545164380309 w_k1_MAT_s014
                                 + 0.0331968444195174 w_k1_MAT_s015
                                 + 0.0676821083535684 w_k1_MAT_s016
                                 + 0.0735791290942171 w_k1_MAT_s017
                                 + 0.0195364348443405 w_k1_MAT_s100
                                 + 0.0472410353313158 w_k1_MAT_s101
                                 + 0.0405390260179459 w_k1_MAT_s102
                                 + 0.0173970149404617 w_k1_MAT_s103
                                 + 0.0459744788294695 w_k1_MAT_s104
                                 + 0.0252836074147258 w_k1_MAT_s105
                                 + 0.0164517952511533 w_k1_MAT_s106
                                 + 0.0332395585925015 w_k1_MAT_s107
                                 + 0.0105746034630706 w_k1_MAT_s110
                                 + 0.0064303062356553 w_k1_MAT_s111
                                 + 0.00886508985757534 w_k1_MAT_s112
                                 + 0.0116221840541432 w_k1_MAT_s113
                                 + 0.00279356965714593 w_k1_MAT_s114
                                 + 0.0116582175211608 w_k1_MAT_s115
                                 + 0.00447642265240811 w_k1_MAT_s116
                                 + 0.00687199211750165 w_k1_MAT_s117
                                 + 0.00756986586225112 w_k1_MSFT_s000
                                 + 0.0305273685719172 w_k1_MSFT_s001
                                 + 0.00972509540010612 w_k1_MSFT_s002
                                 + 0.0225466818013637 w_k1_MSFT_s003
                                 + 0.0228340178686883 w_k1_MSFT_s004
                                 + 0.0160345713155624 w_k1_MSFT_s005
                                 + 0.00901572100732164 w_k1_MSFT_s006
                                 + 0.0154220675151018 w_k1_MSFT_s007
                                 + 0.0961903170716689 w_k1_MSFT_s010
                                 + 0.0561308577620406 w_k1_MSFT_s011
                                 + 0.0636538587728449 w_k1_MSFT_s012
                                 + 0.0650295372137213 w_k1_MSFT_s013
                                 + 0.142783241853918 w_k1_MSFT_s014
                                 + 0.0404457049077219 w_k1_MSFT_s015
                                 + 0.0600527770866365 w_k1_MSFT_s016
                                 + 0.066881919306532 w_k1_MSFT_s017
                                 + 0.0178520653349964 w_k1_MSFT_s100
                                 + 0.0433579206753537 w_k1_MSFT_s101
                                 + 0.0347337318258389 w_k1_MSFT_s102
                                 + 0.0164940019363253 w_k1_MSFT_s103
                                 + 0.0497928997157278 w_k1_MSFT_s104
                                 + 0.0210802763707446 w_k1_MSFT_s105
                                 + 0.0195889400994215 w_k1_MSFT_s106
                                 + 0.0337503605513818 w_k1_MSFT_s107
                                 + 0.0103453667695329 w_k1_MSFT_s110
                                 + 0.00589030707474862 w_k1_MSFT_s111
                                 + 0.00968079080039493 w_k1_MSFT_s112
                                 + 0.0125943450903243 w_k1_MSFT_s113
                                 + 0.00302195751884 w_k1_MSFT_s114
                                 + 0.00919417200799576 w_k1_MSFT_s115
                                 + 0.0051832898599539 w_k1_MSFT_s116
                                 + 0.0071293676531158 w_k1_MSFT_s117
                                 + 0.00808855816289071 w_k1_PG_s000
                                 + 0.0280901059107943 w_k1_PG_s001
                                 + 0.00885273334053088 w_k1_PG_s002
                                 + 0.0247956589260832 w_k1_PG_s003
                                 + 0.0221981190036045 w_k1_PG_s004
                                 + 0.0159715876608942 w_k1_PG_s005
                                 + 0.00905596030799559 w_k1_PG_s006
                                 + 0.0154669342997386 w_k1_PG_s007
                                 + 0.0943990834513555 w_k1_PG_s010
                                 + 0.0574665478858542 w_k1_PG_s011
                                 + 0.0631647614425802 w_k1_PG_s012
                                 + 0.0617977362357765 w_k1_PG_s013
                                 + 0.142815288232218 w_k1_PG_s014
                                 + 0.0381747771207439 w_k1_PG_s015
                                 + 0.0592630265181923 w_k1_PG_s016
                                 + 0.0651601344999942 w_k1_PG_s017
                                 + 0.0170552956002461 w_k1_PG_s100
                                 + 0.0382593569461533 w_k1_PG_s101
                                 + 0.0314879738210818 w_k1_PG_s102
                                 + 0.0172414259626248 w_k1_PG_s103
                                 + 0.0448704027152782 w_k1_PG_s104
                                 + 0.0196130566797491 w_k1_PG_s105
                                 + 0.0175696015216936 w_k1_PG_s106
                                 + 0.0329093643863535 w_k1_PG_s107
                                 + 0.0085428293284132 w_k1_PG_s110
                                 + 0.00591230647735217 w_k1_PG_s111
                                 + 0.00911800405869962 w_k1_PG_s112
                                 + 0.012382112171557 w_k1_PG_s113
                                 + 0.002588105896071 w_k1_PG_s114
                                 + 0.00963653935643241 w_k1_PG_s115
                                 + 0.00541740516862198 w_k1_PG_s116
                                 + 0.00658130833309671 w_k1_PG_s117 >= 1.07
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
