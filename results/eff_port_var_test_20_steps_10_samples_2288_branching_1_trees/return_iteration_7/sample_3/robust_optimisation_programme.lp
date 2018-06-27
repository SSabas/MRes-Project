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
 w_k1_ADM_s0_return_constraint:  1.08361 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.940257 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.06444 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.801489 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.769711 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.03663 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.05832 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.42431 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.32202 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.827931 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.991968 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.954117 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.30473 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.35582 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.938879 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.3998 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.30666 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.918351 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.835805 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.08703 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.21865 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.21625 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.743844 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.978788 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.996673 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.25163 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.16676 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.913241 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.12643 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.01933 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.05934 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.17332 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.01972 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.905468 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.915486 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.08476 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.970951 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.13304 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.14651 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.809056 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.24554 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.815194 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.765915 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.11902 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.957928 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.35043 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.51279 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.64412 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.969357 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.781066 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.42813 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.1802 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.812919 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.3465 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.02218 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.701571 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.661022 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.0576 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.25545 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.30745 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.627497 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.12769 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.891958 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.36724 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.81486 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.804753 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.38971 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.07697 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.865487 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.67565 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.00205 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.04381 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.709561 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.23174 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.02981 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.37928 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.21155 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.774287 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.15999 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.776356 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.666038 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.16289 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.26825 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.79381 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.81565 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.612616 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.01935 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.862558 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.39713 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.32357 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.838198 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.50239 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.17782 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.621592 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.650525 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.804748 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.56893 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.29162 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.665685 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.24744 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.870789 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.60408 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.9878 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.955455 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.22317 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.22673 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.05002 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.91227 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.943816 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.883758 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.68912 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.30026 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.02091 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.25748 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.10612 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.848519 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.11523 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.866616 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.80655 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.11712 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.15399 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.48726 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.32793 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.712079 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.97877 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.833317 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.22436 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.26612 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.823163 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.50455 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.12614 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.829109 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.715915 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.964971 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.25539 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.11384 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.780098 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.956044 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.934807 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.3368 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.287 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.859828 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.03727 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.1711 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.12023 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.43952 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.910731 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.843518 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.788295 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.27134 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.01684 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.13369 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.10983 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.852203 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.21625 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.753762 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.935285 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.10862 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.84844 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.924335 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.98846 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.772937 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.974847 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.16174 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.36831 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.922825 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.55464 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.9212 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.988739 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.618675 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.00037 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.94737 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.2291 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.61865 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.559848 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.67786 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.816324 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.03194 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.59196 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.0834 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.24983 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.633684 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.616099 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.29794 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.17035 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.68223 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.762259 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.740368 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.876354 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.36803 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.03661 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.815112 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.1938 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.964203 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.946916 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.09556 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.04139 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.21901 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.14716 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.781605 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.01732 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.837154 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.01551 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.25909 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.71798 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.36594 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.921131 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.821046 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.723568 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.890176 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.2592 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.18648 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.873563 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.046 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.962308 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.10566 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.32114 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.769095 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.01783 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.07993 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.05772 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.50306 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.0762 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.92685 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.984834 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.1518 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.08638 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.1551 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.09984 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.852645 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.1421 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.830627 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.819192 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.07765 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.08227 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.29214 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.67543 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.706575 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.0759 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.01713 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.30152 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.23411 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.09002 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.21817 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.94738 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.670184 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.84562 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.913482 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.25849 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.31724 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.705967 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.30163 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.941977 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.16583 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.63866 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.915028 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.12922 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.890284 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.919031 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.34857 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.970393 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.19948 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.806861 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.04787 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.926064 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.21742 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.20265 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.810533 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.10795 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.732369 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.812603 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.00135 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.955032 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.06555 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 2.1505 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.684741 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.04624 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.09627 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.73266 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.11347 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.43545 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.0978 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.936285 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.488312 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.882311 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.850814 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.45461 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.70185 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.629003 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.71851 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.875928 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.09415 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 2.34683 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.01647 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.54768 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.890778 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.654419 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.75623 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.05853 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.94708 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.712317 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.920267 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.09001 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 2.00586 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.06114 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.862995 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.18921 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.886044 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.836367 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.0532 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.07171 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.32175 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.18959 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.758076 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.974557 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.876254 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.974605 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.15513 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.86221 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.459 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.13456 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.857852 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.782728 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.951164 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.35029 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.14064 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.74451 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.11255 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.970954 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.10324 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.3095 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.885583 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.06096 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.14106 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.21817 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.32632 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.02982 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.84683 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.848906 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.1469 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.14133 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.15776 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.05877 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.89896 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.13031 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.893482 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.888876 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.05054 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.962568 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.25525 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.23015 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.818713 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.02755 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.92529 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.04397 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.03894 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.95471 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.26169 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.04685 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.80011 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.815397 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.997495 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.08994 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.14392 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.768449 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.00241 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.936032 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.10314 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.29114 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.921783 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.01668 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.918367 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.991771 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.27102 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.02128 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.0238 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.861653 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.03918 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.978513 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.17664 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.602000545099998 z_k1_s0000
                                 - 0.460998944380943 z_k1_s0001
                                 - 0.495999082375101 z_k1_s0002
                                 - 0.705999910340053 z_k1_s0003
                                 - 0.847997988802706 z_k1_s0004
                                 - 0.240999970688601 z_k1_s0005
                                 - 0.405000094889936 z_k1_s0006
                                 - 0.81200137306495 z_k1_s0007
                                 - 0.586000152344172 z_k1_s0010
                                 - 0.33200072968284 z_k1_s0011
                                 - 0.277000123295778 z_k1_s0012
                                 - 0.368000282154642 z_k1_s0013
                                 - 0.136000009248485 z_k1_s0014
                                 - 0.408999393025427 z_k1_s0015
                                 - 0.231999999706244 z_k1_s0016
                                 - 0.392000894468885 z_k1_s0017
                                 - 0.551000349076482 z_k1_s0020
                                 - 0.511000466029213 z_k1_s0021
                                 - 0.136000063560797 z_k1_s0022
                                 - 0.173999952455703 z_k1_s0023
                                 - 0.12600009279898 z_k1_s0024
                                 - 0.154000010932069 z_k1_s0025
                                 - 0.175999946608067 z_k1_s0026
                                 - 0.477999294214992 z_k1_s0027
                                 - 1.18199973273368 z_k1_s0030
                                 - 0.396000809775457 z_k1_s0031
                                 - 0.441000806681575 z_k1_s0032
                                 - 0.451000883494195 z_k1_s0033
                                 - 0.851000468492276 z_k1_s0034
                                 - 1.69600242539994 z_k1_s0035
                                 - 0.573000844106278 z_k1_s0036
                                 - 1.38500052477839 z_k1_s0037
                                 - 0.948999378403578 z_k1_s0040
                                 - 0.932000448925407 z_k1_s0041
                                 - 1.03099470753264 z_k1_s0042
                                 - 1.3079959915823 z_k1_s0043
                                 - 1.21500036949866 z_k1_s0044
                                 - 1.5090005830265 z_k1_s0045
                                 - 0.487999134775065 z_k1_s0046
                                 - 0.946000060141782 z_k1_s0047
                                 - 1.44900326598494 z_k1_s0050
                                 - 0.951000130476289 z_k1_s0051
                                 - 1.22800198716296 z_k1_s0052
                                 - 1.78799841679863 z_k1_s0053
                                 - 0.687000878746913 z_k1_s0054
                                 - 1.66000110268417 z_k1_s0055
                                 - 0.84000118876143 z_k1_s0056
                                 - 0.74600061944617 z_k1_s0057
                                 - 0.195000122073399 z_k1_s0060
                                 - 0.196000104814797 z_k1_s0061
                                 - 0.307999799351735 z_k1_s0062
                                 - 0.221000138349852 z_k1_s0063
                                 - 0.202999984004581 z_k1_s0064
                                 - 0.812999221259552 z_k1_s0065
                                 - 0.195000122073399 z_k1_s0066
                                 - 0.194000139332001 z_k1_s0067
                                 - 0.188000123126455 z_k1_s0070
                                 - 0.266000467734323 z_k1_s0071
                                 - 0.447001444268017 z_k1_s0072
                                 - 0.182000096618158 z_k1_s0073
                                 - 0.338000255233743 z_k1_s0074
                                 - 0.276999808866001 z_k1_s0075
                                 - 0.576000952996112 z_k1_s0076
                                 - 0.378999728906908 z_k1_s0077
                                 - 0.230000162418582 z_k1_s0100
                                 - 0.297000099350017 z_k1_s0101
                                 - 0.43900012170927 z_k1_s0102
                                 - 0.340000088592598 z_k1_s0103
                                 - 0.276000394022415 z_k1_s0104
                                 - 0.334999684657176 z_k1_s0105
                                 - 0.212000201651941 z_k1_s0106
                                 - 0.359999215332823 z_k1_s0107
                                 - 0.364002659609459 z_k1_s0110
                                 - 0.342002135285169 z_k1_s0111
                                 - 0.271001175478465 z_k1_s0112
                                 - 0.348000709002463 z_k1_s0113
                                 - 0.195000828837929 z_k1_s0114
                                 - 0.414002281975333 z_k1_s0115
                                 - 0.691004045080734 z_k1_s0116
                                 - 0.252001088818331 z_k1_s0117
                                 - 0.455000739399429 z_k1_s0120
                                 - 0.730998946245979 z_k1_s0121
                                 - 0.380000654819375 z_k1_s0122
                                 - 0.830002293701875 z_k1_s0123
                                 - 0.866003510958565 z_k1_s0124
                                 - 0.491000825253942 z_k1_s0125
                                 - 0.665999891205225 z_k1_s0126
                                 - 1.23800288958563 z_k1_s0127
                                 - 0.28399993462827 z_k1_s0130
                                 - 0.504999266532823 z_k1_s0131
                                 - 0.26299995845364 z_k1_s0132
                                 - 0.345999350610646 z_k1_s0133
                                 - 0.793000577127333 z_k1_s0134
                                 - 0.408999953334363 z_k1_s0135
                                 - 0.347000537351988 z_k1_s0136
                                 - 0.42400089945885 z_k1_s0137
                                 - 0.573999204382558 z_k1_s0140
                                 - 0.303001319727968 z_k1_s0141
                                 - 0.250000342423364 z_k1_s0142
                                 - 0.526999760217548 z_k1_s0143
                                 - 0.174000079006512 z_k1_s0144
                                 - 0.560001791229298 z_k1_s0145
                                 - 0.23800012114685 z_k1_s0146
                                 - 0.219000339792905 z_k1_s0147
                                 - 0.286999626490731 z_k1_s0150
                                 - 0.953000826204654 z_k1_s0151
                                 - 0.381999692530667 z_k1_s0152
                                 - 0.254999767965526 z_k1_s0153
                                 - 0.286999626490731 z_k1_s0154
                                 - 0.750000521367913 z_k1_s0155
                                 - 0.304999802723387 z_k1_s0156
                                 - 0.873997824330371 z_k1_s0157
                                 - 0.220000110950102 z_k1_s0160
                                 - 0.12499991671257 z_k1_s0161
                                 - 0.169999902825093 z_k1_s0162
                                 - 0.0870000804575377 z_k1_s0163
                                 - 0.297999308905204 z_k1_s0164
                                 - 0.394999833387712 z_k1_s0165
                                 - 0.301999164595217 z_k1_s0166
                                 - 0.414999111837778 z_k1_s0167
                                 - 0.594000180249013 z_k1_s0170
                                 - 0.273000569165091 z_k1_s0171
                                 - 0.24999995129534 z_k1_s0172
                                 - 0.232000183875742 z_k1_s0173
                                 - 0.129000149207646 z_k1_s0174
                                 - 0.295999159834255 z_k1_s0175
                                 - 0.628001402108735 z_k1_s0176
                                 - 0.132000152677591 z_k1_s0177
                                 - 0.0920000771129434 z_k1_s1000
                                 - 0.159000182532187 z_k1_s1001
                                 - 0.0750000180811722 z_k1_s1002
                                 - 0.1329998741305 z_k1_s1003
                                 - 0.262000255430364 z_k1_s1004
                                 - 0.111000088560179 z_k1_s1005
                                 - 0.0680000301269392 z_k1_s1006
                                 - 0.130000512007705 z_k1_s1007
                                 - 0.209999697223933 z_k1_s1010
                                 - 0.216999432131948 z_k1_s1011
                                 - 0.186999256814714 z_k1_s1012
                                 - 0.132999706242044 z_k1_s1013
                                 - 0.154999375809005 z_k1_s1014
                                 - 0.248999313137656 z_k1_s1015
                                 - 0.289998634739855 z_k1_s1016
                                 - 0.0889997551094421 z_k1_s1017
                                 - 0.302001694553601 z_k1_s1020
                                 - 0.4150005487723 z_k1_s1021
                                 - 0.313000856990131 z_k1_s1022
                                 - 0.386000442255617 z_k1_s1023
                                 - 0.176000178197357 z_k1_s1024
                                 - 0.3309989720676 z_k1_s1025
                                 - 0.57300034869004 z_k1_s1026
                                 - 0.332999076965395 z_k1_s1027
                                 - 0.0740000112801758 z_k1_s1030
                                 - 0.050000018616341 z_k1_s1031
                                 - 0.158000189003708 z_k1_s1032
                                 - 0.211999867397171 z_k1_s1033
                                 - 0.0590000667153067 z_k1_s1034
                                 - 0.145999582471459 z_k1_s1035
                                 - 0.106999645242755 z_k1_s1036
                                 - 0.211000155853001 z_k1_s1037
                                 - 0.0530000785768797 z_k1_s1040
                                 - 0.0570000606807611 z_k1_s1041
                                 - 0.0510000875249389 z_k1_s1042
                                 - 0.0510000875249389 z_k1_s1043
                                 - 0.0870001560601995 z_k1_s1044
                                 - 0.105000247727912 z_k1_s1045
                                 - 0.126999977099014 z_k1_s1046
                                 - 0.0430000659170941 z_k1_s1047
                                 - 0.117000467788679 z_k1_s1050
                                 - 0.346000989999375 z_k1_s1051
                                 - 0.11600041559545 z_k1_s1052
                                 - 0.176001261181443 z_k1_s1053
                                 - 0.225001075440365 z_k1_s1054
                                 - 0.11300041141628 z_k1_s1055
                                 - 0.182001269539784 z_k1_s1056
                                 - 0.249001108873728 z_k1_s1057
                                 - 0.13500012690793 z_k1_s1060
                                 - 0.0479999961783419 z_k1_s1061
                                 - 0.12300001773827 z_k1_s1062
                                 - 0.13300025554642 z_k1_s1063
                                 - 0.0730000620983579 z_k1_s1064
                                 - 0.193999856074878 z_k1_s1065
                                 - 0.12300001773827 z_k1_s1066
                                 - 0.0520000032015421 z_k1_s1067
                                 - 0.312998772737982 z_k1_s1070
                                 - 0.190999876024122 z_k1_s1071
                                 - 0.238998675765586 z_k1_s1072
                                 - 0.202999143210383 z_k1_s1073
                                 - 0.192000391955117 z_k1_s1074
                                 - 0.108999805995173 z_k1_s1075
                                 - 0.170999597183456 z_k1_s1076
                                 - 0.312998772737982 z_k1_s1077
                                 - 0.102999836866365 z_k1_s1100
                                 - 0.406999157151783 z_k1_s1101
                                 - 0.388999357621543 z_k1_s1102
                                 - 0.215999808360361 z_k1_s1103
                                 - 0.392999313072708 z_k1_s1104
                                 - 0.159999940044618 z_k1_s1105
                                 - 0.418000264641086 z_k1_s1106
                                 - 0.374000754678278 z_k1_s1107
                                 - 0.438001884113993 z_k1_s1110
                                 - 0.271000278726825 z_k1_s1111
                                 - 0.284000112894282 z_k1_s1112
                                 - 0.191000178525523 z_k1_s1113
                                 - 0.164000345992865 z_k1_s1114
                                 - 0.515001491722063 z_k1_s1115
                                 - 0.166000163187639 z_k1_s1116
                                 - 0.527001928492848 z_k1_s1117
                                 - 0.338999761934318 z_k1_s1120
                                 - 0.241999612897197 z_k1_s1121
                                 - 0.151999827604808 z_k1_s1122
                                 - 0.251999398818689 z_k1_s1123
                                 - 0.248999976641929 z_k1_s1124
                                 - 0.147999913236211 z_k1_s1125
                                 - 0.122999934832793 z_k1_s1126
                                 - 0.206999506172494 z_k1_s1127
                                 - 0.188999877202978 z_k1_s1130
                                 - 0.363999844761265 z_k1_s1131
                                 - 0.091999876379247 z_k1_s1132
                                 - 0.217999940781187 z_k1_s1133
                                 - 0.334999561783124 z_k1_s1134
                                 - 0.2030000062476 z_k1_s1135
                                 - 0.362999381072505 z_k1_s1136
                                 - 0.429999732228993 z_k1_s1137
                                 - 0.480000788946748 z_k1_s1140
                                 - 0.524000930018622 z_k1_s1141
                                 - 0.734999477003733 z_k1_s1142
                                 - 0.383000705631774 z_k1_s1143
                                 - 0.485000416574871 z_k1_s1144
                                 - 1.36000007457968 z_k1_s1145
                                 - 1.48700028086642 z_k1_s1146
                                 - 0.439000778032203 z_k1_s1147
                                 - 0.242999945751189 z_k1_s1150
                                 - 0.118000002593156 z_k1_s1151
                                 - 0.176999918139776 z_k1_s1152
                                 - 0.177999762649116 z_k1_s1153
                                 - 0.129000007195058 z_k1_s1154
                                 - 0.115999970574644 z_k1_s1155
                                 - 0.209000087436142 z_k1_s1156
                                 - 0.544999292549252 z_k1_s1157
                                 - 0.456000009746616 z_k1_s1160
                                 - 0.890999889899911 z_k1_s1161
                                 - 0.262999435663837 z_k1_s1162
                                 - 0.224999369418424 z_k1_s1163
                                 - 0.481997926771662 z_k1_s1164
                                 - 0.513998685103156 z_k1_s1165
                                 - 0.839997331749447 z_k1_s1166
                                 - 0.254999442380548 z_k1_s1167
                                 - 0.237999192965413 z_k1_s1170
                                 - 0.502000224218404 z_k1_s1171
                                 - 0.210999491414568 z_k1_s1172
                                 - 0.229999329705771 z_k1_s1173
                                 - 0.903998896700223 z_k1_s1174
                                 - 0.271999264017104 z_k1_s1175
                                 - 0.607998738507769 z_k1_s1176
                                 - 0.295999179895137 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.14495 w_k1_ADM_s000
                                 + 1.04714 w_k1_AXP_s000
                                 + 0.976964 w_k1_DIS_s000
                                 + 1.03539 w_k1_FDX_s000 + 1.325 w_k1_IBM_s000
                                 + 1.13511 w_k1_INTC_s000 + 1.0838 w_k1_KO_s000
                                 + 0.983922 w_k1_MAT_s000
                                 + 1.12322 w_k1_MSFT_s000
                                 + 1.15435 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.15769 w_k1_ADM_s000
                                 + 1.05085 w_k1_AXP_s000
                                 + 0.881174 w_k1_DIS_s000
                                 + 1.14153 w_k1_FDX_s000
                                 + 0.799933 w_k1_IBM_s000
                                 + 1.10922 w_k1_INTC_s000
                                 + 0.752116 w_k1_KO_s000
                                 + 0.760013 w_k1_MAT_s000
                                 + 0.927054 w_k1_MSFT_s000
                                 + 0.99563 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.13826 w_k1_ADM_s000
                                 + 1.25378 w_k1_AXP_s000
                                 + 1.14526 w_k1_DIS_s000
                                 + 1.18726 w_k1_FDX_s000
                                 + 0.87251 w_k1_IBM_s000
                                 + 1.06906 w_k1_INTC_s000
                                 + 1.05745 w_k1_KO_s000 + 0.99661 w_k1_MAT_s000
                                 + 1.15771 w_k1_MSFT_s000
                                 + 1.10483 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.11728 w_k1_ADM_s000
                                 + 0.745309 w_k1_AXP_s000
                                 + 0.914827 w_k1_DIS_s000
                                 + 0.957114 w_k1_FDX_s000
                                 + 0.812609 w_k1_IBM_s000
                                 + 1.0063 w_k1_INTC_s000
                                 + 0.920035 w_k1_KO_s000
                                 + 0.749279 w_k1_MAT_s000
                                 + 1.02525 w_k1_MSFT_s000
                                 + 1.05408 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.936515 w_k1_ADM_s000
                                 + 0.859146 w_k1_AXP_s000
                                 + 0.665609 w_k1_DIS_s000
                                 + 0.810775 w_k1_FDX_s000
                                 + 1.03972 w_k1_IBM_s000
                                 + 0.788288 w_k1_INTC_s000
                                 + 0.912935 w_k1_KO_s000
                                 + 0.908279 w_k1_MAT_s000
                                 + 0.832307 w_k1_MSFT_s000
                                 + 0.922546 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.57698 w_k1_ADM_s000
                                 + 1.61799 w_k1_AXP_s000
                                 + 1.55839 w_k1_DIS_s000
                                 + 1.27599 w_k1_FDX_s000
                                 + 1.59884 w_k1_IBM_s000
                                 + 1.30661 w_k1_INTC_s000
                                 + 1.32333 w_k1_KO_s000 + 1.13725 w_k1_MAT_s000
                                 + 1.25172 w_k1_MSFT_s000
                                 + 1.36893 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.12656 w_k1_ADM_s000
                                 + 0.716985 w_k1_AXP_s000
                                 + 0.628249 w_k1_DIS_s000
                                 + 0.931183 w_k1_FDX_s000
                                 + 0.480423 w_k1_IBM_s000
                                 + 0.971649 w_k1_INTC_s000
                                 + 0.597313 w_k1_KO_s000
                                 + 0.424855 w_k1_MAT_s000
                                 + 0.921763 w_k1_MSFT_s000
                                 + 0.959118 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.18541 w_k1_ADM_s000
                                 + 1.1079 w_k1_AXP_s000 + 1.31527 w_k1_DIS_s000
                                 + 1.08786 w_k1_FDX_s000
                                 + 1.47169 w_k1_IBM_s000
                                 + 0.967995 w_k1_INTC_s000
                                 + 1.22848 w_k1_KO_s000 + 1.28915 w_k1_MAT_s000
                                 + 1.03016 w_k1_MSFT_s000
                                 + 1.19878 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.11854 w_k1_ADM_s001
                                 + 1.03303 w_k1_AXP_s001
                                 + 1.40722 w_k1_DIS_s001
                                 + 1.21293 w_k1_FDX_s001
                                 + 1.10145 w_k1_IBM_s001
                                 + 1.00129 w_k1_INTC_s001
                                 + 1.17166 w_k1_KO_s001 + 1.30428 w_k1_MAT_s001
                                 + 1.27428 w_k1_MSFT_s001
                                 + 0.997636 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.928764 w_k1_ADM_s001
                                 + 1.15305 w_k1_AXP_s001
                                 + 1.39928 w_k1_DIS_s001
                                 + 1.16101 w_k1_FDX_s001
                                 + 1.66123 w_k1_IBM_s001
                                 + 1.22136 w_k1_INTC_s001
                                 + 1.23287 w_k1_KO_s001 + 1.61537 w_k1_MAT_s001
                                 + 1.20703 w_k1_MSFT_s001
                                 + 1.02584 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.97434 w_k1_ADM_s001
                                 + 0.93939 w_k1_AXP_s001
                                 + 0.881943 w_k1_DIS_s001
                                 + 1.0854 w_k1_FDX_s001
                                 + 0.709936 w_k1_IBM_s001
                                 + 1.16574 w_k1_INTC_s001
                                 + 0.840153 w_k1_KO_s001
                                 + 0.74124 w_k1_MAT_s001
                                 + 1.09505 w_k1_MSFT_s001
                                 + 0.79463 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.02541 w_k1_ADM_s001
                                 + 0.836646 w_k1_AXP_s001
                                 + 0.757353 w_k1_DIS_s001
                                 + 0.877053 w_k1_FDX_s001
                                 + 0.689311 w_k1_IBM_s001
                                 + 0.932972 w_k1_INTC_s001
                                 + 0.829109 w_k1_KO_s001
                                 + 0.755051 w_k1_MAT_s001
                                 + 0.874449 w_k1_MSFT_s001
                                 + 0.860759 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.38225 w_k1_ADM_s001
                                 + 1.40267 w_k1_AXP_s001
                                 + 1.12777 w_k1_DIS_s001
                                 + 1.33596 w_k1_FDX_s001
                                 + 0.79481 w_k1_IBM_s001
                                 + 1.00348 w_k1_INTC_s001
                                 + 0.856261 w_k1_KO_s001
                                 + 0.937542 w_k1_MAT_s001
                                 + 1.11998 w_k1_MSFT_s001
                                 + 1.00171 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.931272 w_k1_ADM_s001
                                 + 0.857039 w_k1_AXP_s001
                                 + 0.974849 w_k1_DIS_s001
                                 + 0.891147 w_k1_FDX_s001
                                 + 1.32247 w_k1_IBM_s001
                                 + 1.05791 w_k1_INTC_s001
                                 + 1.26995 w_k1_KO_s001 + 1.2493 w_k1_MAT_s001
                                 + 1.02897 w_k1_MSFT_s001
                                 + 0.973246 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.01659 w_k1_ADM_s001
                                 + 1.156 w_k1_AXP_s001 + 0.98303 w_k1_DIS_s001
                                 + 1.08097 w_k1_FDX_s001 + 1.0168 w_k1_IBM_s001
                                 + 0.963215 w_k1_INTC_s001
                                 + 0.925118 w_k1_KO_s001
                                 + 1.06063 w_k1_MAT_s001
                                 + 1.0015 w_k1_MSFT_s001
                                 + 0.949501 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.660583 w_k1_ADM_s001
                                 + 0.567253 w_k1_AXP_s001
                                 + 0.707641 w_k1_DIS_s001
                                 + 0.797181 w_k1_FDX_s001
                                 + 0.961474 w_k1_IBM_s001
                                 + 0.96468 w_k1_INTC_s001
                                 + 0.807324 w_k1_KO_s001
                                 + 0.726656 w_k1_MAT_s001
                                 + 0.903895 w_k1_MSFT_s001
                                 + 0.795908 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.789555 w_k1_ADM_s002
                                 + 0.662948 w_k1_AXP_s002
                                 + 0.747394 w_k1_DIS_s002
                                 + 0.83969 w_k1_FDX_s002
                                 + 0.882032 w_k1_IBM_s002
                                 + 0.741083 w_k1_INTC_s002
                                 + 0.811635 w_k1_KO_s002
                                 + 0.830438 w_k1_MAT_s002
                                 + 0.821839 w_k1_MSFT_s002
                                 + 0.895473 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.22571 w_k1_ADM_s002
                                 + 1.26463 w_k1_AXP_s002
                                 + 1.11117 w_k1_DIS_s002
                                 + 1.23786 w_k1_FDX_s002
                                 + 0.745882 w_k1_IBM_s002
                                 + 1.0197 w_k1_INTC_s002
                                 + 0.841208 w_k1_KO_s002
                                 + 0.923026 w_k1_MAT_s002
                                 + 1.00606 w_k1_MSFT_s002
                                 + 1.11925 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.21886 w_k1_ADM_s002
                                 + 1.53459 w_k1_AXP_s002
                                 + 1.02325 w_k1_DIS_s002
                                 + 1.06367 w_k1_FDX_s002
                                 + 1.22453 w_k1_IBM_s002
                                 + 1.17202 w_k1_INTC_s002
                                 + 1.12447 w_k1_KO_s002 + 1.20408 w_k1_MAT_s002
                                 + 0.9083 w_k1_MSFT_s002 + 1.08981 w_k1_PG_s002
                                 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.00862 w_k1_ADM_s002
                                 + 1.04228 w_k1_AXP_s002
                                 + 0.866964 w_k1_DIS_s002
                                 + 0.885775 w_k1_FDX_s002
                                 + 1.20637 w_k1_IBM_s002
                                 + 0.963653 w_k1_INTC_s002
                                 + 1.02054 w_k1_KO_s002 + 1.19617 w_k1_MAT_s002
                                 + 0.915462 w_k1_MSFT_s002
                                 + 1.10204 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.46349 w_k1_ADM_s002
                                 + 1.83542 w_k1_AXP_s002
                                 + 2.43834 w_k1_DIS_s002
                                 + 1.91541 w_k1_FDX_s002 + 1.2055 w_k1_IBM_s002
                                 + 1.54191 w_k1_INTC_s002
                                 + 1.39344 w_k1_KO_s002 + 1.42432 w_k1_MAT_s002
                                 + 1.93105 w_k1_MSFT_s002
                                 + 1.21977 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.973017 w_k1_ADM_s002
                                 + 1.20088 w_k1_AXP_s002
                                 + 0.801654 w_k1_DIS_s002
                                 + 0.869439 w_k1_FDX_s002
                                 + 1.80076 w_k1_IBM_s002
                                 + 0.893156 w_k1_INTC_s002
                                 + 1.25369 w_k1_KO_s002 + 1.83473 w_k1_MAT_s002
                                 + 0.844161 w_k1_MSFT_s002
                                 + 1.21998 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.84329 w_k1_ADM_s002
                                 + 0.696944 w_k1_AXP_s002
                                 + 0.505357 w_k1_DIS_s002
                                 + 0.645901 w_k1_FDX_s002
                                 + 1.11157 w_k1_IBM_s002
                                 + 0.829952 w_k1_INTC_s002
                                 + 0.784617 w_k1_KO_s002
                                 + 0.948181 w_k1_MAT_s002
                                 + 0.774483 w_k1_MSFT_s002
                                 + 0.791261 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.12052 w_k1_ADM_s002
                                 + 1.03221 w_k1_AXP_s002
                                 + 1.22516 w_k1_DIS_s002
                                 + 1.11185 w_k1_FDX_s002
                                 + 1.10491 w_k1_IBM_s002
                                 + 1.03294 w_k1_INTC_s002
                                 + 1.21222 w_k1_KO_s002 + 1.25258 w_k1_MAT_s002
                                 + 1.1663 w_k1_MSFT_s002 + 1.25992 w_k1_PG_s002
                                 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.14392 w_k1_ADM_s003
                                 + 1.26935 w_k1_AXP_s003
                                 + 1.28586 w_k1_DIS_s003 + 1.2325 w_k1_FDX_s003
                                 + 0.890687 w_k1_IBM_s003
                                 + 1.02994 w_k1_INTC_s003
                                 + 1.15643 w_k1_KO_s003 + 1.09395 w_k1_MAT_s003
                                 + 1.01581 w_k1_MSFT_s003 + 1.0067 w_k1_PG_s003
                                 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.3518 w_k1_ADM_s003
                                 + 1.57473 w_k1_AXP_s003 + 1.7303 w_k1_DIS_s003
                                 + 1.63343 w_k1_FDX_s003
                                 + 1.01399 w_k1_IBM_s003
                                 + 1.4212 w_k1_INTC_s003 + 1.40674 w_k1_KO_s003
                                 + 1.28263 w_k1_MAT_s003
                                 + 1.35014 w_k1_MSFT_s003
                                 + 1.37316 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.99137 w_k1_ADM_s003
                                 + 1.13092 w_k1_AXP_s003
                                 + 1.17362 w_k1_DIS_s003
                                 + 0.907405 w_k1_FDX_s003
                                 + 1.93747 w_k1_IBM_s003
                                 + 1.10637 w_k1_INTC_s003
                                 + 1.36506 w_k1_KO_s003 + 2.1135 w_k1_MAT_s003
                                 + 1.01223 w_k1_MSFT_s003
                                 + 1.00935 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.976395 w_k1_ADM_s003
                                 + 1.21323 w_k1_AXP_s003
                                 + 0.884569 w_k1_DIS_s003
                                 + 1.1967 w_k1_FDX_s003
                                 + 0.726732 w_k1_IBM_s003
                                 + 0.92625 w_k1_INTC_s003
                                 + 0.833421 w_k1_KO_s003
                                 + 1.13477 w_k1_MAT_s003
                                 + 0.949765 w_k1_MSFT_s003
                                 + 0.932797 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.833464 w_k1_ADM_s003
                                 + 0.925121 w_k1_AXP_s003
                                 + 1.03867 w_k1_DIS_s003
                                 + 1.02044 w_k1_FDX_s003
                                 + 0.758052 w_k1_IBM_s003
                                 + 0.972086 w_k1_INTC_s003
                                 + 1.03249 w_k1_KO_s003
                                 + 0.788669 w_k1_MAT_s003
                                 + 0.992931 w_k1_MSFT_s003
                                 + 0.881303 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.937725 w_k1_ADM_s003
                                 + 1.07299 w_k1_AXP_s003
                                 + 0.947233 w_k1_DIS_s003
                                 + 0.913799 w_k1_FDX_s003
                                 + 1.28166 w_k1_IBM_s003
                                 + 1.00518 w_k1_INTC_s003
                                 + 1.04826 w_k1_KO_s003 + 1.41085 w_k1_MAT_s003
                                 + 0.891294 w_k1_MSFT_s003
                                 + 0.926035 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.886274 w_k1_ADM_s003
                                 + 0.913579 w_k1_AXP_s003
                                 + 0.72132 w_k1_DIS_s003
                                 + 0.974375 w_k1_FDX_s003
                                 + 0.621715 w_k1_IBM_s003
                                 + 0.70515 w_k1_INTC_s003
                                 + 0.754599 w_k1_KO_s003
                                 + 0.865236 w_k1_MAT_s003
                                 + 0.784937 w_k1_MSFT_s003
                                 + 0.868059 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.736914 w_k1_ADM_s003
                                 + 0.807354 w_k1_AXP_s003
                                 + 0.692645 w_k1_DIS_s003
                                 + 0.718768 w_k1_FDX_s003
                                 + 0.884408 w_k1_IBM_s003
                                 + 0.779943 w_k1_INTC_s003
                                 + 0.903185 w_k1_KO_s003
                                 + 0.83162 w_k1_MAT_s003
                                 + 0.756418 w_k1_MSFT_s003
                                 + 0.757945 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.06533 w_k1_ADM_s004
                                 + 0.994117 w_k1_AXP_s004
                                 + 1.14746 w_k1_DIS_s004
                                 + 1.23803 w_k1_FDX_s004
                                 + 0.72404 w_k1_IBM_s004
                                 + 0.9099 w_k1_INTC_s004
                                 + 0.944778 w_k1_KO_s004
                                 + 1.05467 w_k1_MAT_s004
                                 + 1.04441 w_k1_MSFT_s004
                                 + 0.96433 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.00237 w_k1_ADM_s004
                                 + 1.22295 w_k1_AXP_s004
                                 + 1.18446 w_k1_DIS_s004
                                 + 1.05483 w_k1_FDX_s004
                                 + 1.88072 w_k1_IBM_s004
                                 + 1.14963 w_k1_INTC_s004
                                 + 1.19532 w_k1_KO_s004 + 1.87884 w_k1_MAT_s004
                                 + 1.0952 w_k1_MSFT_s004
                                 + 0.970432 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.850596 w_k1_ADM_s004
                                 + 0.888449 w_k1_AXP_s004
                                 + 0.687827 w_k1_DIS_s004
                                 + 0.924412 w_k1_FDX_s004
                                 + 0.674016 w_k1_IBM_s004
                                 + 0.748228 w_k1_INTC_s004
                                 + 0.735548 w_k1_KO_s004
                                 + 0.771421 w_k1_MAT_s004
                                 + 0.685307 w_k1_MSFT_s004
                                 + 0.785055 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.10104 w_k1_ADM_s004
                                 + 1.12093 w_k1_AXP_s004
                                 + 1.27806 w_k1_DIS_s004
                                 + 1.12858 w_k1_FDX_s004
                                 + 1.18764 w_k1_IBM_s004
                                 + 1.0016 w_k1_INTC_s004 + 1.10685 w_k1_KO_s004
                                 + 1.23347 w_k1_MAT_s004
                                 + 0.873801 w_k1_MSFT_s004
                                 + 1.09176 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.815498 w_k1_ADM_s004
                                 + 0.628237 w_k1_AXP_s004
                                 + 0.889335 w_k1_DIS_s004
                                 + 0.870018 w_k1_FDX_s004
                                 + 0.672476 w_k1_IBM_s004
                                 + 0.774179 w_k1_INTC_s004
                                 + 0.851018 w_k1_KO_s004
                                 + 0.688172 w_k1_MAT_s004
                                 + 0.896512 w_k1_MSFT_s004
                                 + 0.802623 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.98028 w_k1_ADM_s004
                                 + 1.06944 w_k1_AXP_s004
                                 + 0.773866 w_k1_DIS_s004
                                 + 0.905152 w_k1_FDX_s004
                                 + 1.06404 w_k1_IBM_s004
                                 + 1.08668 w_k1_INTC_s004 + 0.7918 w_k1_KO_s004
                                 + 1.12979 w_k1_MAT_s004
                                 + 0.847621 w_k1_MSFT_s004
                                 + 0.819946 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.18001 w_k1_ADM_s004
                                 + 1.51919 w_k1_AXP_s004
                                 + 1.73529 w_k1_DIS_s004
                                 + 1.35801 w_k1_FDX_s004
                                 + 1.34121 w_k1_IBM_s004
                                 + 1.53504 w_k1_INTC_s004
                                 + 1.34236 w_k1_KO_s004 + 1.38981 w_k1_MAT_s004
                                 + 1.29353 w_k1_MSFT_s004
                                 + 1.16709 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.868776 w_k1_ADM_s004
                                 + 0.7878 w_k1_AXP_s004 + 1.03047 w_k1_DIS_s004
                                 + 0.933283 w_k1_FDX_s004
                                 + 1.15389 w_k1_IBM_s004
                                 + 0.928424 w_k1_INTC_s004
                                 + 1.02938 w_k1_KO_s004 + 1.16731 w_k1_MAT_s004
                                 + 0.97433 w_k1_MSFT_s004
                                 + 0.994878 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.821743 w_k1_ADM_s005
                                 + 0.852411 w_k1_AXP_s005
                                 + 0.702765 w_k1_DIS_s005
                                 + 0.821193 w_k1_FDX_s005
                                 + 0.997648 w_k1_IBM_s005
                                 + 0.936869 w_k1_INTC_s005
                                 + 0.989213 w_k1_KO_s005
                                 + 1.14371 w_k1_MAT_s005
                                 + 0.888473 w_k1_MSFT_s005
                                 + 0.856253 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.16692 w_k1_ADM_s005
                                 + 1.13808 w_k1_AXP_s005
                                 + 1.38504 w_k1_DIS_s005
                                 + 1.18586 w_k1_FDX_s005
                                 + 1.26887 w_k1_IBM_s005
                                 + 1.02678 w_k1_INTC_s005 + 1.4025 w_k1_KO_s005
                                 + 1.44054 w_k1_MAT_s005
                                 + 1.24978 w_k1_MSFT_s005
                                 + 1.31171 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.992039 w_k1_ADM_s005
                                 + 0.843292 w_k1_AXP_s005
                                 + 0.960043 w_k1_DIS_s005
                                 + 0.978621 w_k1_FDX_s005
                                 + 0.761799 w_k1_IBM_s005
                                 + 0.985468 w_k1_INTC_s005
                                 + 0.903035 w_k1_KO_s005
                                 + 0.633935 w_k1_MAT_s005
                                 + 1.03099 w_k1_MSFT_s005
                                 + 0.950463 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.08045 w_k1_ADM_s005
                                 + 1.00294 w_k1_AXP_s005
                                 + 0.99702 w_k1_DIS_s005
                                 + 0.925231 w_k1_FDX_s005
                                 + 1.5666 w_k1_IBM_s005
                                 + 0.758825 w_k1_INTC_s005
                                 + 1.27555 w_k1_KO_s005 + 1.54551 w_k1_MAT_s005
                                 + 0.857869 w_k1_MSFT_s005
                                 + 1.06334 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.08597 w_k1_ADM_s005
                                 + 1.39001 w_k1_AXP_s005 + 1.3258 w_k1_DIS_s005
                                 + 1.34766 w_k1_FDX_s005
                                 + 0.902758 w_k1_IBM_s005
                                 + 1.10378 w_k1_INTC_s005
                                 + 1.05385 w_k1_KO_s005 + 1.09365 w_k1_MAT_s005
                                 + 1.089 w_k1_MSFT_s005 + 1.23191 w_k1_PG_s005
                                 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.02649 w_k1_ADM_s005
                                 + 1.2382 w_k1_AXP_s005
                                 + 0.973267 w_k1_DIS_s005
                                 + 1.07515 w_k1_FDX_s005
                                 + 0.935605 w_k1_IBM_s005
                                 + 1.07751 w_k1_INTC_s005
                                 + 0.990939 w_k1_KO_s005
                                 + 1.01871 w_k1_MAT_s005
                                 + 0.950743 w_k1_MSFT_s005
                                 + 1.04373 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.910687 w_k1_ADM_s005
                                 + 0.943734 w_k1_AXP_s005
                                 + 0.872746 w_k1_DIS_s005
                                 + 1.01624 w_k1_FDX_s005
                                 + 0.546719 w_k1_IBM_s005
                                 + 0.886936 w_k1_INTC_s005
                                 + 0.844102 w_k1_KO_s005
                                 + 0.580669 w_k1_MAT_s005
                                 + 0.877675 w_k1_MSFT_s005
                                 + 0.9422 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.793184 w_k1_ADM_s005
                                 + 0.767557 w_k1_AXP_s005
                                 + 0.571089 w_k1_DIS_s005
                                 + 0.675947 w_k1_FDX_s005
                                 + 0.716891 w_k1_IBM_s005
                                 + 0.848929 w_k1_INTC_s005
                                 + 0.744384 w_k1_KO_s005
                                 + 0.736902 w_k1_MAT_s005
                                 + 0.719616 w_k1_MSFT_s005
                                 + 0.796436 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.26697 w_k1_ADM_s006
                                 + 1.68163 w_k1_AXP_s006
                                 + 1.64385 w_k1_DIS_s006
                                 + 1.44506 w_k1_FDX_s006
                                 + 1.21323 w_k1_IBM_s006
                                 + 1.29841 w_k1_INTC_s006
                                 + 1.22437 w_k1_KO_s006 + 1.25599 w_k1_MAT_s006
                                 + 1.28373 w_k1_MSFT_s006
                                 + 1.37137 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.934741 w_k1_ADM_s006
                                 + 1.11558 w_k1_AXP_s006 + 1.1931 w_k1_DIS_s006
                                 + 1.13017 w_k1_FDX_s006 + 1.2795 w_k1_IBM_s006
                                 + 1.11238 w_k1_INTC_s006
                                 + 1.17129 w_k1_KO_s006 + 1.30618 w_k1_MAT_s006
                                 + 1.22748 w_k1_MSFT_s006
                                 + 1.24052 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.96616 w_k1_ADM_s006
                                 + 1.35764 w_k1_AXP_s006
                                 + 1.01237 w_k1_DIS_s006
                                 + 1.10531 w_k1_FDX_s006
                                 + 0.914215 w_k1_IBM_s006
                                 + 1.13585 w_k1_INTC_s006
                                 + 0.85457 w_k1_KO_s006 + 1.207 w_k1_MAT_s006
                                 + 1.03738 w_k1_MSFT_s006
                                 + 1.21976 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.00855 w_k1_ADM_s006
                                 + 1.24326 w_k1_AXP_s006
                                 + 1.04742 w_k1_DIS_s006
                                 + 0.998713 w_k1_FDX_s006
                                 + 1.4028 w_k1_IBM_s006
                                 + 1.06198 w_k1_INTC_s006
                                 + 1.12069 w_k1_KO_s006 + 1.28295 w_k1_MAT_s006
                                 + 1.09422 w_k1_MSFT_s006
                                 + 1.08795 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.689886 w_k1_ADM_s006
                                 + 0.892823 w_k1_AXP_s006
                                 + 0.859992 w_k1_DIS_s006
                                 + 0.824267 w_k1_FDX_s006
                                 + 1.39126 w_k1_IBM_s006
                                 + 1.04765 w_k1_INTC_s006
                                 + 1.08711 w_k1_KO_s006 + 1.49279 w_k1_MAT_s006
                                 + 1.01611 w_k1_MSFT_s006
                                 + 1.10344 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.012 w_k1_ADM_s006
                                 + 0.958372 w_k1_AXP_s006
                                 + 0.719468 w_k1_DIS_s006
                                 + 0.913026 w_k1_FDX_s006
                                 + 0.77278 w_k1_IBM_s006
                                 + 0.911907 w_k1_INTC_s006
                                 + 0.794586 w_k1_KO_s006
                                 + 1.04133 w_k1_MAT_s006
                                 + 0.893286 w_k1_MSFT_s006
                                 + 0.959754 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.12527 w_k1_ADM_s006
                                 + 0.804357 w_k1_AXP_s006
                                 + 1.03554 w_k1_DIS_s006
                                 + 0.958515 w_k1_FDX_s006 + 1.081 w_k1_IBM_s006
                                 + 1.11494 w_k1_INTC_s006
                                 + 0.988661 w_k1_KO_s006
                                 + 0.765451 w_k1_MAT_s006
                                 + 1.12201 w_k1_MSFT_s006
                                 + 1.12969 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.19551 w_k1_ADM_s006
                                 + 1.17489 w_k1_AXP_s006
                                 + 1.30421 w_k1_DIS_s006
                                 + 1.23135 w_k1_FDX_s006
                                 + 0.854648 w_k1_IBM_s006
                                 + 1.22845 w_k1_INTC_s006
                                 + 1.03516 w_k1_KO_s006
                                 + 0.761245 w_k1_MAT_s006
                                 + 1.25491 w_k1_MSFT_s006
                                 + 1.01722 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.38926 w_k1_ADM_s007
                                 + 1.74577 w_k1_AXP_s007
                                 + 1.47587 w_k1_DIS_s007
                                 + 1.38325 w_k1_FDX_s007 + 1.4667 w_k1_IBM_s007
                                 + 1.10163 w_k1_INTC_s007
                                 + 1.29608 w_k1_KO_s007 + 2.11864 w_k1_MAT_s007
                                 + 1.08232 w_k1_MSFT_s007
                                 + 1.17619 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.09669 w_k1_ADM_s007
                                 + 1.04385 w_k1_AXP_s007
                                 + 1.09084 w_k1_DIS_s007
                                 + 1.07518 w_k1_FDX_s007
                                 + 0.943725 w_k1_IBM_s007
                                 + 0.991839 w_k1_INTC_s007
                                 + 0.874813 w_k1_KO_s007
                                 + 0.690279 w_k1_MAT_s007
                                 + 1.01433 w_k1_MSFT_s007
                                 + 1.04647 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.16453 w_k1_ADM_s007
                                 + 1.31534 w_k1_AXP_s007
                                 + 1.34075 w_k1_DIS_s007
                                 + 1.20078 w_k1_FDX_s007 + 1.1797 w_k1_IBM_s007
                                 + 0.933691 w_k1_INTC_s007
                                 + 1.12248 w_k1_KO_s007 + 1.22983 w_k1_MAT_s007
                                 + 1.0903 w_k1_MSFT_s007 + 1.22177 w_k1_PG_s007
                                 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.12711 w_k1_ADM_s007
                                 + 1.13764 w_k1_AXP_s007
                                 + 0.946653 w_k1_DIS_s007
                                 + 1.1365 w_k1_FDX_s007
                                 + 0.624488 w_k1_IBM_s007
                                 + 1.0055 w_k1_INTC_s007
                                 + 0.710893 w_k1_KO_s007
                                 + 0.698172 w_k1_MAT_s007
                                 + 1.01697 w_k1_MSFT_s007
                                 + 0.966729 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.8884 w_k1_ADM_s007
                                 + 1.01837 w_k1_AXP_s007
                                 + 0.92545 w_k1_DIS_s007
                                 + 0.963039 w_k1_FDX_s007
                                 + 0.723773 w_k1_IBM_s007
                                 + 0.824211 w_k1_INTC_s007
                                 + 0.830616 w_k1_KO_s007
                                 + 0.701439 w_k1_MAT_s007
                                 + 0.942736 w_k1_MSFT_s007
                                 + 1.00866 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.01353 w_k1_ADM_s007
                                 + 1.00619 w_k1_AXP_s007
                                 + 0.888453 w_k1_DIS_s007
                                 + 0.853446 w_k1_FDX_s007
                                 + 1.02612 w_k1_IBM_s007
                                 + 0.793753 w_k1_INTC_s007
                                 + 0.968376 w_k1_KO_s007
                                 + 1.00458 w_k1_MAT_s007
                                 + 0.842852 w_k1_MSFT_s007
                                 + 1.05525 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.805642 w_k1_ADM_s007
                                 + 1.06849 w_k1_AXP_s007
                                 + 1.00923 w_k1_DIS_s007
                                 + 0.98619 w_k1_FDX_s007
                                 + 1.31551 w_k1_IBM_s007
                                 + 0.815214 w_k1_INTC_s007
                                 + 0.960861 w_k1_KO_s007
                                 + 1.62045 w_k1_MAT_s007
                                 + 1.00158 w_k1_MSFT_s007
                                 + 1.05322 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.748905 w_k1_ADM_s007
                                 + 0.858454 w_k1_AXP_s007
                                 + 0.680244 w_k1_DIS_s007
                                 + 0.788153 w_k1_FDX_s007
                                 + 0.711238 w_k1_IBM_s007
                                 + 0.827317 w_k1_INTC_s007
                                 + 0.67208 w_k1_KO_s007 + 1.02052 w_k1_MAT_s007
                                 + 0.869362 w_k1_MSFT_s007
                                 + 0.832775 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.05452 w_k1_ADM_s010
                                 + 0.93724 w_k1_AXP_s010
                                 + 1.23786 w_k1_DIS_s010
                                 + 1.02863 w_k1_FDX_s010
                                 + 0.824737 w_k1_IBM_s010
                                 + 1.22924 w_k1_INTC_s010
                                 + 1.03014 w_k1_KO_s010
                                 + 0.838607 w_k1_MAT_s010
                                 + 1.23133 w_k1_MSFT_s010 + 1.1403 w_k1_PG_s010
                                 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.744857 w_k1_ADM_s010
                                 + 1.08849 w_k1_AXP_s010
                                 + 0.804579 w_k1_DIS_s010
                                 + 0.759371 w_k1_FDX_s010
                                 + 1.22062 w_k1_IBM_s010
                                 + 1.20138 w_k1_INTC_s010
                                 + 0.928163 w_k1_KO_s010
                                 + 1.31436 w_k1_MAT_s010
                                 + 0.944895 w_k1_MSFT_s010
                                 + 0.960274 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.00983 w_k1_ADM_s010
                                 + 1.27884 w_k1_AXP_s010
                                 + 1.22707 w_k1_DIS_s010
                                 + 1.05222 w_k1_FDX_s010
                                 + 0.988848 w_k1_IBM_s010
                                 + 1.36384 w_k1_INTC_s010
                                 + 1.00812 w_k1_KO_s010
                                 + 0.961927 w_k1_MAT_s010
                                 + 1.12622 w_k1_MSFT_s010
                                 + 1.01849 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.799348 w_k1_ADM_s010
                                 + 0.829883 w_k1_AXP_s010
                                 + 0.639306 w_k1_DIS_s010
                                 + 0.815008 w_k1_FDX_s010
                                 + 0.663545 w_k1_IBM_s010
                                 + 0.90353 w_k1_INTC_s010
                                 + 0.649177 w_k1_KO_s010
                                 + 0.55246 w_k1_MAT_s010
                                 + 0.713878 w_k1_MSFT_s010
                                 + 0.85115 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.11326 w_k1_ADM_s010
                                 + 0.995191 w_k1_AXP_s010
                                 + 0.785117 w_k1_DIS_s010
                                 + 0.952976 w_k1_FDX_s010
                                 + 0.875854 w_k1_IBM_s010
                                 + 0.928396 w_k1_INTC_s010
                                 + 0.857231 w_k1_KO_s010
                                 + 0.958848 w_k1_MAT_s010
                                 + 0.857965 w_k1_MSFT_s010
                                 + 0.931391 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.988601 w_k1_ADM_s010
                                 + 0.819032 w_k1_AXP_s010
                                 + 0.891854 w_k1_DIS_s010
                                 + 0.929727 w_k1_FDX_s010
                                 + 0.766708 w_k1_IBM_s010
                                 + 0.9661 w_k1_INTC_s010
                                 + 0.780939 w_k1_KO_s010
                                 + 0.500522 w_k1_MAT_s010
                                 + 1.03398 w_k1_MSFT_s010
                                 + 0.901671 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.34195 w_k1_ADM_s010
                                 + 1.76992 w_k1_AXP_s010
                                 + 1.63725 w_k1_DIS_s010
                                 + 1.18665 w_k1_FDX_s010
                                 + 1.64962 w_k1_IBM_s010 + 1.377 w_k1_INTC_s010
                                 + 1.38942 w_k1_KO_s010 + 1.82223 w_k1_MAT_s010
                                 + 1.37464 w_k1_MSFT_s010
                                 + 1.37519 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.15257 w_k1_ADM_s010
                                 + 1.25834 w_k1_AXP_s010
                                 + 1.31335 w_k1_DIS_s010 + 1.1178 w_k1_FDX_s010
                                 + 1.24408 w_k1_IBM_s010
                                 + 1.28294 w_k1_INTC_s010
                                 + 1.22072 w_k1_KO_s010 + 0.98959 w_k1_MAT_s010
                                 + 1.19559 w_k1_MSFT_s010
                                 + 1.13292 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.38767 w_k1_ADM_s011
                                 + 1.34269 w_k1_AXP_s011
                                 + 1.40916 w_k1_DIS_s011
                                 + 1.23199 w_k1_FDX_s011
                                 + 1.22247 w_k1_IBM_s011
                                 + 1.15294 w_k1_INTC_s011
                                 + 1.22572 w_k1_KO_s011 + 1.07867 w_k1_MAT_s011
                                 + 1.19242 w_k1_MSFT_s011
                                 + 1.05599 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.993629 w_k1_ADM_s011
                                 + 0.989409 w_k1_AXP_s011
                                 + 0.687578 w_k1_DIS_s011
                                 + 0.838377 w_k1_FDX_s011
                                 + 0.82259 w_k1_IBM_s011
                                 + 0.786625 w_k1_INTC_s011
                                 + 0.809893 w_k1_KO_s011
                                 + 1.06472 w_k1_MAT_s011
                                 + 0.79917 w_k1_MSFT_s011
                                 + 0.818636 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.87283 w_k1_ADM_s011
                                 + 1.47606 w_k1_AXP_s011
                                 + 2.06457 w_k1_DIS_s011
                                 + 1.43032 w_k1_FDX_s011
                                 + 1.46141 w_k1_IBM_s011
                                 + 1.04465 w_k1_INTC_s011
                                 + 1.61383 w_k1_KO_s011 + 1.23479 w_k1_MAT_s011
                                 + 1.27648 w_k1_MSFT_s011
                                 + 1.16487 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.865422 w_k1_ADM_s011
                                 + 0.624106 w_k1_AXP_s011
                                 + 0.59937 w_k1_DIS_s011
                                 + 0.702667 w_k1_FDX_s011
                                 + 0.626793 w_k1_IBM_s011
                                 + 0.668855 w_k1_INTC_s011
                                 + 0.578775 w_k1_KO_s011
                                 + 0.534627 w_k1_MAT_s011
                                 + 0.751906 w_k1_MSFT_s011
                                 + 0.67951 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.1354 w_k1_ADM_s011
                                 + 0.973241 w_k1_AXP_s011
                                 + 1.29472 w_k1_DIS_s011
                                 + 1.16913 w_k1_FDX_s011
                                 + 0.65835 w_k1_IBM_s011
                                 + 1.0818 w_k1_INTC_s011
                                 + 0.864056 w_k1_KO_s011
                                 + 0.735305 w_k1_MAT_s011
                                 + 1.23464 w_k1_MSFT_s011
                                 + 0.917867 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.18678 w_k1_ADM_s011
                                 + 0.945502 w_k1_AXP_s011
                                 + 1.26227 w_k1_DIS_s011
                                 + 0.915433 w_k1_FDX_s011
                                 + 1.30935 w_k1_IBM_s011
                                 + 0.871166 w_k1_INTC_s011
                                 + 1.12941 w_k1_KO_s011 + 1.00381 w_k1_MAT_s011
                                 + 1.00405 w_k1_MSFT_s011
                                 + 1.09301 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.06913 w_k1_ADM_s011
                                 + 0.920234 w_k1_AXP_s011
                                 + 0.993929 w_k1_DIS_s011
                                 + 0.990014 w_k1_FDX_s011
                                 + 0.758114 w_k1_IBM_s011
                                 + 0.865031 w_k1_INTC_s011
                                 + 0.725387 w_k1_KO_s011
                                 + 0.698774 w_k1_MAT_s011
                                 + 0.843634 w_k1_MSFT_s011
                                 + 0.86361 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.12375 w_k1_ADM_s011
                                 + 1.53106 w_k1_AXP_s011
                                 + 0.965746 w_k1_DIS_s011
                                 + 1.07166 w_k1_FDX_s011
                                 + 1.18455 w_k1_IBM_s011
                                 + 0.881497 w_k1_INTC_s011
                                 + 1.00788 w_k1_KO_s011 + 1.37551 w_k1_MAT_s011
                                 + 0.842674 w_k1_MSFT_s011
                                 + 0.991899 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.30391 w_k1_ADM_s012
                                 + 1.40129 w_k1_AXP_s012
                                 + 1.49591 w_k1_DIS_s012 + 1.3792 w_k1_FDX_s012
                                 + 0.918795 w_k1_IBM_s012
                                 + 1.5172 w_k1_INTC_s012 + 1.32979 w_k1_KO_s012
                                 + 0.875903 w_k1_MAT_s012
                                 + 1.30782 w_k1_MSFT_s012
                                 + 1.15867 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.971296 w_k1_ADM_s012
                                 + 0.958714 w_k1_AXP_s012
                                 + 1.00458 w_k1_DIS_s012
                                 + 1.15912 w_k1_FDX_s012
                                 + 0.762634 w_k1_IBM_s012
                                 + 1.09294 w_k1_INTC_s012
                                 + 1.04804 w_k1_KO_s012 + 1.05838 w_k1_MAT_s012
                                 + 1.04225 w_k1_MSFT_s012
                                 + 1.04567 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.11888 w_k1_ADM_s012
                                 + 1.27657 w_k1_AXP_s012
                                 + 0.844014 w_k1_DIS_s012
                                 + 1.11756 w_k1_FDX_s012
                                 + 1.08475 w_k1_IBM_s012
                                 + 1.14324 w_k1_INTC_s012 + 1.1605 w_k1_KO_s012
                                 + 1.34496 w_k1_MAT_s012
                                 + 0.924721 w_k1_MSFT_s012
                                 + 1.05271 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.972528 w_k1_ADM_s012
                                 + 0.958953 w_k1_AXP_s012
                                 + 0.781758 w_k1_DIS_s012
                                 + 0.802327 w_k1_FDX_s012
                                 + 1.22744 w_k1_IBM_s012
                                 + 1.00176 w_k1_INTC_s012
                                 + 1.04934 w_k1_KO_s012 + 1.20164 w_k1_MAT_s012
                                 + 0.790344 w_k1_MSFT_s012
                                 + 0.872242 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.933551 w_k1_ADM_s012
                                 + 1.13527 w_k1_AXP_s012
                                 + 1.14787 w_k1_DIS_s012
                                 + 1.06366 w_k1_FDX_s012
                                 + 1.09557 w_k1_IBM_s012
                                 + 1.21988 w_k1_INTC_s012
                                 + 1.20767 w_k1_KO_s012
                                 + 0.837866 w_k1_MAT_s012
                                 + 1.06374 w_k1_MSFT_s012
                                 + 1.05349 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.837605 w_k1_ADM_s012
                                 + 0.684348 w_k1_AXP_s012
                                 + 0.625432 w_k1_DIS_s012
                                 + 0.812476 w_k1_FDX_s012
                                 + 0.617473 w_k1_IBM_s012
                                 + 0.932899 w_k1_INTC_s012
                                 + 0.795632 w_k1_KO_s012
                                 + 0.626515 w_k1_MAT_s012
                                 + 0.86821 w_k1_MSFT_s012
                                 + 0.805151 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.15321 w_k1_ADM_s012
                                 + 1.5277 w_k1_AXP_s012 + 1.43433 w_k1_DIS_s012
                                 + 1.20318 w_k1_FDX_s012
                                 + 1.58861 w_k1_IBM_s012
                                 + 1.44933 w_k1_INTC_s012
                                 + 1.48279 w_k1_KO_s012 + 1.34744 w_k1_MAT_s012
                                 + 1.13831 w_k1_MSFT_s012
                                 + 1.09776 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.848459 w_k1_ADM_s012
                                 + 0.826367 w_k1_AXP_s012
                                 + 0.803479 w_k1_DIS_s012
                                 + 0.855924 w_k1_FDX_s012
                                 + 0.882569 w_k1_IBM_s012
                                 + 0.936948 w_k1_INTC_s012
                                 + 0.968538 w_k1_KO_s012
                                 + 0.844133 w_k1_MAT_s012
                                 + 0.928687 w_k1_MSFT_s012
                                 + 1.00029 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.85425 w_k1_ADM_s013
                                 + 0.848255 w_k1_AXP_s013
                                 + 1.03354 w_k1_DIS_s013
                                 + 0.851307 w_k1_FDX_s013
                                 + 1.25167 w_k1_IBM_s013
                                 + 0.975754 w_k1_INTC_s013
                                 + 1.03776 w_k1_KO_s013
                                 + 0.983374 w_k1_MAT_s013
                                 + 0.961237 w_k1_MSFT_s013
                                 + 0.916894 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.26784 w_k1_ADM_s013
                                 + 1.3423 w_k1_AXP_s013 + 1.08772 w_k1_DIS_s013
                                 + 1.10471 w_k1_FDX_s013 + 1.178 w_k1_IBM_s013
                                 + 0.986184 w_k1_INTC_s013
                                 + 1.03657 w_k1_KO_s013 + 1.6342 w_k1_MAT_s013
                                 + 0.94323 w_k1_MSFT_s013
                                 + 1.07672 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.98306 w_k1_ADM_s013
                                 + 1.15075 w_k1_AXP_s013
                                 + 1.07817 w_k1_DIS_s013
                                 + 1.14921 w_k1_FDX_s013
                                 + 0.851007 w_k1_IBM_s013
                                 + 1.13482 w_k1_INTC_s013
                                 + 1.03491 w_k1_KO_s013 + 1.48838 w_k1_MAT_s013
                                 + 1.01112 w_k1_MSFT_s013
                                 + 1.04534 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.17202 w_k1_ADM_s013
                                 + 1.27946 w_k1_AXP_s013
                                 + 1.36113 w_k1_DIS_s013
                                 + 1.16019 w_k1_FDX_s013
                                 + 1.56666 w_k1_IBM_s013
                                 + 1.18853 w_k1_INTC_s013
                                 + 1.39963 w_k1_KO_s013 + 2.19893 w_k1_MAT_s013
                                 + 1.17454 w_k1_MSFT_s013
                                 + 1.30956 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.963151 w_k1_ADM_s013
                                 + 0.760101 w_k1_AXP_s013
                                 + 0.686307 w_k1_DIS_s013
                                 + 0.770367 w_k1_FDX_s013
                                 + 0.802638 w_k1_IBM_s013
                                 + 0.779695 w_k1_INTC_s013
                                 + 0.839202 w_k1_KO_s013
                                 + 0.762189 w_k1_MAT_s013
                                 + 0.766623 w_k1_MSFT_s013
                                 + 0.842155 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.774664 w_k1_ADM_s013
                                 + 0.913642 w_k1_AXP_s013
                                 + 0.862146 w_k1_DIS_s013
                                 + 0.824894 w_k1_FDX_s013
                                 + 1.31723 w_k1_IBM_s013
                                 + 0.97208 w_k1_INTC_s013 + 1.0318 w_k1_KO_s013
                                 + 1.42996 w_k1_MAT_s013
                                 + 0.810819 w_k1_MSFT_s013
                                 + 1.04623 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.882018 w_k1_ADM_s013
                                 + 0.81641 w_k1_AXP_s013
                                 + 1.15105 w_k1_DIS_s013
                                 + 1.04486 w_k1_FDX_s013
                                 + 0.790536 w_k1_IBM_s013
                                 + 0.972797 w_k1_INTC_s013
                                 + 1.04635 w_k1_KO_s013 + 1.18324 w_k1_MAT_s013
                                 + 1.2112 w_k1_MSFT_s013 + 1.02895 w_k1_PG_s013
                                 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.36642 w_k1_ADM_s013
                                 + 1.20408 w_k1_AXP_s013
                                 + 1.72921 w_k1_DIS_s013
                                 + 1.35096 w_k1_FDX_s013
                                 + 1.22181 w_k1_IBM_s013
                                 + 1.41303 w_k1_INTC_s013
                                 + 1.48526 w_k1_KO_s013 + 1.74202 w_k1_MAT_s013
                                 + 1.31871 w_k1_MSFT_s013
                                 + 1.21757 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.30825 w_k1_ADM_s014
                                 + 1.04156 w_k1_AXP_s014 + 0.9186 w_k1_DIS_s014
                                 + 1.06011 w_k1_FDX_s014
                                 + 0.818812 w_k1_IBM_s014
                                 + 1.05812 w_k1_INTC_s014
                                 + 0.987659 w_k1_KO_s014
                                 + 0.918652 w_k1_MAT_s014
                                 + 1.10808 w_k1_MSFT_s014
                                 + 1.05065 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.03606 w_k1_ADM_s014
                                 + 1.1683 w_k1_AXP_s014 + 1.25607 w_k1_DIS_s014
                                 + 1.06501 w_k1_FDX_s014
                                 + 1.50031 w_k1_IBM_s014
                                 + 1.23881 w_k1_INTC_s014
                                 + 1.28975 w_k1_KO_s014 + 1.221 w_k1_MAT_s014
                                 + 1.06634 w_k1_MSFT_s014
                                 + 1.25061 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.771892 w_k1_ADM_s014
                                 + 0.682778 w_k1_AXP_s014
                                 + 0.908185 w_k1_DIS_s014
                                 + 0.944625 w_k1_FDX_s014
                                 + 0.578423 w_k1_IBM_s014
                                 + 1.04506 w_k1_INTC_s014
                                 + 0.792326 w_k1_KO_s014
                                 + 0.454471 w_k1_MAT_s014
                                 + 0.985676 w_k1_MSFT_s014
                                 + 0.928004 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.28622 w_k1_ADM_s014
                                 + 1.31461 w_k1_AXP_s014
                                 + 1.19608 w_k1_DIS_s014
                                 + 1.19709 w_k1_FDX_s014
                                 + 1.14997 w_k1_IBM_s014
                                 + 1.19108 w_k1_INTC_s014
                                 + 1.07478 w_k1_KO_s014 + 1.34068 w_k1_MAT_s014
                                 + 1.11805 w_k1_MSFT_s014
                                 + 1.10157 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.1751 w_k1_ADM_s014
                                 + 1.27182 w_k1_AXP_s014
                                 + 0.82359 w_k1_DIS_s014
                                 + 0.881215 w_k1_FDX_s014
                                 + 1.33753 w_k1_IBM_s014
                                 + 0.757837 w_k1_INTC_s014
                                 + 0.969322 w_k1_KO_s014
                                 + 1.80709 w_k1_MAT_s014
                                 + 0.896409 w_k1_MSFT_s014
                                 + 1.11572 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.19848 w_k1_ADM_s014
                                 + 1.13185 w_k1_AXP_s014
                                 + 1.35809 w_k1_DIS_s014
                                 + 1.08072 w_k1_FDX_s014
                                 + 1.08576 w_k1_IBM_s014
                                 + 1.33497 w_k1_INTC_s014
                                 + 1.10672 w_k1_KO_s014
                                 + 0.771506 w_k1_MAT_s014
                                 + 1.20511 w_k1_MSFT_s014
                                 + 1.18428 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.687433 w_k1_ADM_s014
                                 + 0.790264 w_k1_AXP_s014
                                 + 0.662985 w_k1_DIS_s014
                                 + 0.725809 w_k1_FDX_s014
                                 + 0.936835 w_k1_IBM_s014
                                 + 0.788404 w_k1_INTC_s014
                                 + 0.799179 w_k1_KO_s014
                                 + 0.817113 w_k1_MAT_s014
                                 + 0.819927 w_k1_MSFT_s014
                                 + 0.829993 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.891032 w_k1_ADM_s014
                                 + 0.841111 w_k1_AXP_s014
                                 + 0.85198 w_k1_DIS_s014
                                 + 0.901198 w_k1_FDX_s014
                                 + 0.900556 w_k1_IBM_s014
                                 + 0.93049 w_k1_INTC_s014
                                 + 0.866269 w_k1_KO_s014
                                 + 0.965091 w_k1_MAT_s014
                                 + 0.94981 w_k1_MSFT_s014
                                 + 0.984264 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.27398 w_k1_ADM_s015
                                 + 1.05537 w_k1_AXP_s015
                                 + 1.26665 w_k1_DIS_s015
                                 + 1.03557 w_k1_FDX_s015
                                 + 1.29268 w_k1_IBM_s015
                                 + 1.0774 w_k1_INTC_s015 + 1.26888 w_k1_KO_s015
                                 + 1.34166 w_k1_MAT_s015
                                 + 1.11594 w_k1_MSFT_s015
                                 + 1.12861 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.07711 w_k1_ADM_s015
                                 + 1.21202 w_k1_AXP_s015
                                 + 1.39609 w_k1_DIS_s015
                                 + 1.20515 w_k1_FDX_s015
                                 + 0.930378 w_k1_IBM_s015
                                 + 1.14631 w_k1_INTC_s015
                                 + 1.07598 w_k1_KO_s015 + 1.12385 w_k1_MAT_s015
                                 + 1.13903 w_k1_MSFT_s015
                                 + 1.09643 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.918842 w_k1_ADM_s015
                                 + 0.987322 w_k1_AXP_s015
                                 + 0.943371 w_k1_DIS_s015
                                 + 0.961781 w_k1_FDX_s015
                                 + 1.1553 w_k1_IBM_s015
                                 + 1.05256 w_k1_INTC_s015
                                 + 0.936125 w_k1_KO_s015
                                 + 1.42414 w_k1_MAT_s015
                                 + 0.904737 w_k1_MSFT_s015
                                 + 0.92093 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.94484 w_k1_ADM_s015
                                 + 1.10226 w_k1_AXP_s015
                                 + 1.04821 w_k1_DIS_s015
                                 + 0.899372 w_k1_FDX_s015
                                 + 1.5431 w_k1_IBM_s015
                                 + 1.04002 w_k1_INTC_s015
                                 + 1.28877 w_k1_KO_s015 + 1.837 w_k1_MAT_s015
                                 + 1.05378 w_k1_MSFT_s015
                                 + 1.00827 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.710236 w_k1_ADM_s015
                                 + 0.630212 w_k1_AXP_s015
                                 + 0.75338 w_k1_DIS_s015
                                 + 0.758278 w_k1_FDX_s015
                                 + 1.09032 w_k1_IBM_s015
                                 + 0.8228 w_k1_INTC_s015
                                 + 0.856379 w_k1_KO_s015
                                 + 1.48637 w_k1_MAT_s015
                                 + 0.954756 w_k1_MSFT_s015
                                 + 0.837548 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.982915 w_k1_ADM_s015
                                 + 0.852173 w_k1_AXP_s015
                                 + 1.00967 w_k1_DIS_s015 + 1.0815 w_k1_FDX_s015
                                 + 0.730765 w_k1_IBM_s015
                                 + 1.06373 w_k1_INTC_s015
                                 + 0.958954 w_k1_KO_s015
                                 + 0.699102 w_k1_MAT_s015
                                 + 0.951461 w_k1_MSFT_s015
                                 + 0.937133 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.12403 w_k1_ADM_s015
                                 + 1.43073 w_k1_AXP_s015
                                 + 1.51184 w_k1_DIS_s015
                                 + 1.26698 w_k1_FDX_s015
                                 + 1.44246 w_k1_IBM_s015
                                 + 1.22818 w_k1_INTC_s015
                                 + 1.45228 w_k1_KO_s015 + 2.17366 w_k1_MAT_s015
                                 + 1.31777 w_k1_MSFT_s015
                                 + 1.26626 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.829573 w_k1_ADM_s015
                                 + 0.80538 w_k1_AXP_s015
                                 + 0.799696 w_k1_DIS_s015
                                 + 0.872083 w_k1_FDX_s015
                                 + 0.84586 w_k1_IBM_s015
                                 + 0.956931 w_k1_INTC_s015
                                 + 0.863164 w_k1_KO_s015
                                 + 0.881785 w_k1_MAT_s015
                                 + 0.974295 w_k1_MSFT_s015
                                 + 0.837325 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.768324 w_k1_ADM_s016
                                 + 0.597741 w_k1_AXP_s016
                                 + 0.800485 w_k1_DIS_s016
                                 + 0.756483 w_k1_FDX_s016
                                 + 0.78963 w_k1_IBM_s016
                                 + 0.561776 w_k1_INTC_s016
                                 + 0.891217 w_k1_KO_s016
                                 + 0.638508 w_k1_MAT_s016
                                 + 0.734715 w_k1_MSFT_s016
                                 + 0.910096 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.03452 w_k1_ADM_s016
                                 + 0.831678 w_k1_AXP_s016
                                 + 0.953582 w_k1_DIS_s016
                                 + 1.02857 w_k1_FDX_s016 + 0.6274 w_k1_IBM_s016
                                 + 0.875611 w_k1_INTC_s016
                                 + 0.755697 w_k1_KO_s016
                                 + 0.591547 w_k1_MAT_s016
                                 + 1.07205 w_k1_MSFT_s016
                                 + 0.965483 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.961031 w_k1_ADM_s016
                                 + 1.09921 w_k1_AXP_s016
                                 + 0.618072 w_k1_DIS_s016
                                 + 0.900067 w_k1_FDX_s016
                                 + 0.826232 w_k1_IBM_s016
                                 + 0.900465 w_k1_INTC_s016
                                 + 0.809477 w_k1_KO_s016
                                 + 0.932972 w_k1_MAT_s016
                                 + 0.770437 w_k1_MSFT_s016
                                 + 0.951724 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.879929 w_k1_ADM_s016
                                 + 0.670957 w_k1_AXP_s016
                                 + 0.837336 w_k1_DIS_s016
                                 + 0.806556 w_k1_FDX_s016
                                 + 1.05992 w_k1_IBM_s016
                                 + 0.774321 w_k1_INTC_s016
                                 + 1.04813 w_k1_KO_s016
                                 + 0.862528 w_k1_MAT_s016
                                 + 0.948115 w_k1_MSFT_s016
                                 + 1.01426 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.01579 w_k1_ADM_s016
                                 + 1.02568 w_k1_AXP_s016
                                 + 0.815177 w_k1_DIS_s016
                                 + 0.934224 w_k1_FDX_s016
                                 + 0.775417 w_k1_IBM_s016
                                 + 0.902841 w_k1_INTC_s016
                                 + 0.834688 w_k1_KO_s016
                                 + 0.627008 w_k1_MAT_s016
                                 + 0.762072 w_k1_MSFT_s016
                                 + 1.0789 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.966033 w_k1_ADM_s016
                                 + 1.35255 w_k1_AXP_s016
                                 + 1.04015 w_k1_DIS_s016
                                 + 1.02444 w_k1_FDX_s016
                                 + 1.36899 w_k1_IBM_s016
                                 + 1.11276 w_k1_INTC_s016
                                 + 1.10172 w_k1_KO_s016 + 1.09968 w_k1_MAT_s016
                                 + 0.965609 w_k1_MSFT_s016 + 1.086 w_k1_PG_s016
                                 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.60348 w_k1_ADM_s016
                                 + 1.46737 w_k1_AXP_s016
                                 + 1.59594 w_k1_DIS_s016
                                 + 1.36201 w_k1_FDX_s016
                                 + 1.26795 w_k1_IBM_s016
                                 + 1.26223 w_k1_INTC_s016
                                 + 1.37499 w_k1_KO_s016 + 1.25054 w_k1_MAT_s016
                                 + 1.32858 w_k1_MSFT_s016
                                 + 1.38793 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.11067 w_k1_ADM_s016
                                 + 1.0152 w_k1_AXP_s016 + 1.04772 w_k1_DIS_s016
                                 + 1.12405 w_k1_FDX_s016
                                 + 0.904815 w_k1_IBM_s016
                                 + 1.21896 w_k1_INTC_s016
                                 + 1.11858 w_k1_KO_s016
                                 + 0.610392 w_k1_MAT_s016 + 1.09 w_k1_MSFT_s016
                                 + 1.04699 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.19132 w_k1_ADM_s017
                                 + 0.88767 w_k1_AXP_s017
                                 + 0.935516 w_k1_DIS_s017
                                 + 0.993301 w_k1_FDX_s017
                                 + 0.919744 w_k1_IBM_s017
                                 + 0.851924 w_k1_INTC_s017
                                 + 0.962112 w_k1_KO_s017
                                 + 0.838865 w_k1_MAT_s017
                                 + 0.943614 w_k1_MSFT_s017
                                 + 1.02844 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.0434 w_k1_ADM_s017
                                 + 1.16168 w_k1_AXP_s017
                                 + 1.58891 w_k1_DIS_s017 + 1.3165 w_k1_FDX_s017
                                 + 1.08852 w_k1_IBM_s017
                                 + 1.08315 w_k1_INTC_s017
                                 + 1.15496 w_k1_KO_s017 + 1.02004 w_k1_MAT_s017
                                 + 1.28476 w_k1_MSFT_s017
                                 + 1.09705 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.29469 w_k1_ADM_s017
                                 + 1.39074 w_k1_AXP_s017
                                 + 1.60351 w_k1_DIS_s017
                                 + 1.20444 w_k1_FDX_s017
                                 + 1.54413 w_k1_IBM_s017
                                 + 1.03205 w_k1_INTC_s017
                                 + 1.33196 w_k1_KO_s017 + 1.47007 w_k1_MAT_s017
                                 + 1.13977 w_k1_MSFT_s017 + 1.222 w_k1_PG_s017
                                 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.18997 w_k1_ADM_s017
                                 + 1.38366 w_k1_AXP_s017
                                 + 1.03696 w_k1_DIS_s017 + 1.2064 w_k1_FDX_s017
                                 + 1.02302 w_k1_IBM_s017
                                 + 1.05355 w_k1_INTC_s017
                                 + 1.09491 w_k1_KO_s017 + 1.22541 w_k1_MAT_s017
                                 + 1.11961 w_k1_MSFT_s017
                                 + 1.31643 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.1391 w_k1_ADM_s017
                                 + 1.09747 w_k1_AXP_s017
                                 + 0.775706 w_k1_DIS_s017
                                 + 0.973103 w_k1_FDX_s017
                                 + 0.902825 w_k1_IBM_s017
                                 + 0.778703 w_k1_INTC_s017
                                 + 0.831818 w_k1_KO_s017
                                 + 0.951963 w_k1_MAT_s017
                                 + 0.887223 w_k1_MSFT_s017
                                 + 0.998699 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.904626 w_k1_ADM_s017
                                 + 0.940797 w_k1_AXP_s017
                                 + 1.06224 w_k1_DIS_s017
                                 + 0.990658 w_k1_FDX_s017
                                 + 1.25189 w_k1_IBM_s017
                                 + 0.961662 w_k1_INTC_s017
                                 + 1.13508 w_k1_KO_s017 + 1.11907 w_k1_MAT_s017
                                 + 0.996505 w_k1_MSFT_s017
                                 + 1.04009 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.984674 w_k1_ADM_s017
                                 + 0.741031 w_k1_AXP_s017
                                 + 0.705567 w_k1_DIS_s017
                                 + 0.856538 w_k1_FDX_s017
                                 + 0.675383 w_k1_IBM_s017
                                 + 0.820335 w_k1_INTC_s017
                                 + 0.747471 w_k1_KO_s017
                                 + 0.500792 w_k1_MAT_s017
                                 + 0.919539 w_k1_MSFT_s017
                                 + 0.909898 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.800167 w_k1_ADM_s017
                                 + 0.805314 w_k1_AXP_s017
                                 + 0.750127 w_k1_DIS_s017
                                 + 0.763859 w_k1_FDX_s017 + 1.116 w_k1_IBM_s017
                                 + 0.633596 w_k1_INTC_s017
                                 + 0.97747 w_k1_KO_s017 + 1.15815 w_k1_MAT_s017
                                 + 0.76948 w_k1_MSFT_s017
                                 + 0.979342 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.23238 w_k1_ADM_s100
                                 + 1.26685 w_k1_AXP_s100
                                 + 1.50784 w_k1_DIS_s100
                                 + 1.39615 w_k1_FDX_s100
                                 + 0.896776 w_k1_IBM_s100
                                 + 1.10187 w_k1_INTC_s100
                                 + 1.05492 w_k1_KO_s100
                                 + 0.908263 w_k1_MAT_s100
                                 + 1.4073 w_k1_MSFT_s100 + 1.02178 w_k1_PG_s100
                                 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.36085 w_k1_ADM_s100
                                 + 1.58964 w_k1_AXP_s100
                                 + 1.27166 w_k1_DIS_s100
                                 + 1.12621 w_k1_FDX_s100
                                 + 1.11816 w_k1_IBM_s100
                                 + 1.08808 w_k1_INTC_s100
                                 + 1.12998 w_k1_KO_s100 + 1.17597 w_k1_MAT_s100
                                 + 1.30621 w_k1_MSFT_s100
                                 + 1.13807 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.933835 w_k1_ADM_s100
                                 + 0.722199 w_k1_AXP_s100
                                 + 0.721467 w_k1_DIS_s100
                                 + 0.865344 w_k1_FDX_s100
                                 + 0.623094 w_k1_IBM_s100
                                 + 0.879435 w_k1_INTC_s100
                                 + 0.712432 w_k1_KO_s100
                                 + 0.446462 w_k1_MAT_s100
                                 + 0.960551 w_k1_MSFT_s100
                                 + 0.848757 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.891179 w_k1_ADM_s100
                                 + 1.05209 w_k1_AXP_s100
                                 + 1.04671 w_k1_DIS_s100
                                 + 0.984228 w_k1_FDX_s100
                                 + 1.53515 w_k1_IBM_s100
                                 + 1.03886 w_k1_INTC_s100
                                 + 1.17373 w_k1_KO_s100 + 1.76054 w_k1_MAT_s100
                                 + 1.0456 w_k1_MSFT_s100 + 1.07545 w_k1_PG_s100
                                 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.18606 w_k1_ADM_s100
                                 + 1.04963 w_k1_AXP_s100
                                 + 1.00684 w_k1_DIS_s100
                                 + 0.968282 w_k1_FDX_s100
                                 + 0.962237 w_k1_IBM_s100
                                 + 0.89624 w_k1_INTC_s100
                                 + 0.97231 w_k1_KO_s100
                                 + 0.744643 w_k1_MAT_s100
                                 + 1.05115 w_k1_MSFT_s100
                                 + 0.951343 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.00834 w_k1_ADM_s100
                                 + 0.781874 w_k1_AXP_s100
                                 + 0.61747 w_k1_DIS_s100
                                 + 0.835393 w_k1_FDX_s100
                                 + 0.782289 w_k1_IBM_s100
                                 + 0.627992 w_k1_INTC_s100
                                 + 0.689764 w_k1_KO_s100
                                 + 0.485605 w_k1_MAT_s100
                                 + 0.83424 w_k1_MSFT_s100
                                 + 0.854578 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.824329 w_k1_ADM_s100
                                 + 0.665999 w_k1_AXP_s100
                                 + 0.600736 w_k1_DIS_s100
                                 + 0.633447 w_k1_FDX_s100
                                 + 1.09203 w_k1_IBM_s100
                                 + 0.601516 w_k1_INTC_s100
                                 + 0.741333 w_k1_KO_s100
                                 + 0.775574 w_k1_MAT_s100
                                 + 0.651575 w_k1_MSFT_s100
                                 + 0.783772 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.03547 w_k1_ADM_s100
                                 + 0.936502 w_k1_AXP_s100
                                 + 1.06196 w_k1_DIS_s100
                                 + 1.09982 w_k1_FDX_s100
                                 + 0.698249 w_k1_IBM_s100
                                 + 0.868077 w_k1_INTC_s100
                                 + 0.834725 w_k1_KO_s100
                                 + 0.621801 w_k1_MAT_s100
                                 + 1.08031 w_k1_MSFT_s100
                                 + 0.974623 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.05592 w_k1_ADM_s101
                                 + 1.07003 w_k1_AXP_s101
                                 + 1.21437 w_k1_DIS_s101
                                 + 1.05238 w_k1_FDX_s101
                                 + 1.35782 w_k1_IBM_s101
                                 + 0.948863 w_k1_INTC_s101
                                 + 1.23914 w_k1_KO_s101 + 1.43507 w_k1_MAT_s101
                                 + 0.827452 w_k1_MSFT_s101
                                 + 1.10681 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.2263 w_k1_ADM_s101
                                 + 1.40341 w_k1_AXP_s101
                                 + 1.30875 w_k1_DIS_s101
                                 + 1.23927 w_k1_FDX_s101
                                 + 0.975037 w_k1_IBM_s101
                                 + 1.2297 w_k1_INTC_s101 + 1.2227 w_k1_KO_s101
                                 + 1.24657 w_k1_MAT_s101
                                 + 1.23725 w_k1_MSFT_s101
                                 + 1.24418 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.851171 w_k1_ADM_s101
                                 + 0.72526 w_k1_AXP_s101
                                 + 0.696143 w_k1_DIS_s101
                                 + 0.7181 w_k1_FDX_s101
                                 + 0.966043 w_k1_IBM_s101
                                 + 0.74419 w_k1_INTC_s101
                                 + 0.85679 w_k1_KO_s101
                                 + 0.758716 w_k1_MAT_s101
                                 + 0.74629 w_k1_MSFT_s101
                                 + 0.863161 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.842377 w_k1_ADM_s101
                                 + 1.27096 w_k1_AXP_s101
                                 + 1.28921 w_k1_DIS_s101
                                 + 1.12578 w_k1_FDX_s101
                                 + 1.83286 w_k1_IBM_s101
                                 + 1.15994 w_k1_INTC_s101 + 1.3736 w_k1_KO_s101
                                 + 2.46977 w_k1_MAT_s101
                                 + 1.13258 w_k1_MSFT_s101
                                 + 1.29902 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.93815 w_k1_ADM_s101
                                 + 0.96794 w_k1_AXP_s101
                                 + 1.09232 w_k1_DIS_s101
                                 + 0.943022 w_k1_FDX_s101
                                 + 1.05197 w_k1_IBM_s101
                                 + 0.874278 w_k1_INTC_s101
                                 + 0.984615 w_k1_KO_s101
                                 + 1.26339 w_k1_MAT_s101
                                 + 0.874525 w_k1_MSFT_s101
                                 + 1.13584 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.775281 w_k1_ADM_s101
                                 + 0.863512 w_k1_AXP_s101
                                 + 0.906927 w_k1_DIS_s101
                                 + 0.956345 w_k1_FDX_s101
                                 + 0.819887 w_k1_IBM_s101
                                 + 0.9704 w_k1_INTC_s101
                                 + 0.848573 w_k1_KO_s101
                                 + 0.63056 w_k1_MAT_s101
                                 + 0.887984 w_k1_MSFT_s101
                                 + 0.955704 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.809414 w_k1_ADM_s101
                                 + 0.82772 w_k1_AXP_s101
                                 + 0.790801 w_k1_DIS_s101
                                 + 0.840347 w_k1_FDX_s101
                                 + 0.653136 w_k1_IBM_s101
                                 + 0.912579 w_k1_INTC_s101
                                 + 0.771759 w_k1_KO_s101
                                 + 0.825314 w_k1_MAT_s101
                                 + 0.864048 w_k1_MSFT_s101
                                 + 0.963417 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.989284 w_k1_ADM_s101
                                 + 1.05516 w_k1_AXP_s101
                                 + 0.921256 w_k1_DIS_s101
                                 + 0.934402 w_k1_FDX_s101
                                 + 1.10157 w_k1_IBM_s101
                                 + 0.986298 w_k1_INTC_s101
                                 + 1.07457 w_k1_KO_s101 + 1.40533 w_k1_MAT_s101
                                 + 0.825602 w_k1_MSFT_s101
                                 + 1.11817 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.792272 w_k1_ADM_s102
                                 + 1.11318 w_k1_AXP_s102 + 1.2081 w_k1_DIS_s102
                                 + 0.985887 w_k1_FDX_s102
                                 + 1.10428 w_k1_IBM_s102
                                 + 1.00056 w_k1_INTC_s102
                                 + 0.997598 w_k1_KO_s102
                                 + 1.02519 w_k1_MAT_s102
                                 + 1.07003 w_k1_MSFT_s102
                                 + 0.942475 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.10078 w_k1_ADM_s102
                                 + 1.11097 w_k1_AXP_s102 + 1.0958 w_k1_DIS_s102
                                 + 1.0584 w_k1_FDX_s102
                                 + 0.870088 w_k1_IBM_s102
                                 + 1.03627 w_k1_INTC_s102
                                 + 0.866858 w_k1_KO_s102
                                 + 1.08556 w_k1_MAT_s102
                                 + 1.06525 w_k1_MSFT_s102
                                 + 1.00485 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.867917 w_k1_ADM_s102
                                 + 0.788128 w_k1_AXP_s102
                                 + 0.996061 w_k1_DIS_s102
                                 + 0.89398 w_k1_FDX_s102
                                 + 0.867355 w_k1_IBM_s102
                                 + 0.800002 w_k1_INTC_s102
                                 + 0.882767 w_k1_KO_s102
                                 + 0.769164 w_k1_MAT_s102
                                 + 0.921513 w_k1_MSFT_s102
                                 + 0.933471 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.765414 w_k1_ADM_s102
                                 + 0.935879 w_k1_AXP_s102
                                 + 0.923663 w_k1_DIS_s102
                                 + 0.891138 w_k1_FDX_s102
                                 + 1.35265 w_k1_IBM_s102
                                 + 0.862186 w_k1_INTC_s102
                                 + 0.955184 w_k1_KO_s102
                                 + 1.09172 w_k1_MAT_s102
                                 + 0.791573 w_k1_MSFT_s102
                                 + 0.937109 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.722437 w_k1_ADM_s102
                                 + 0.711516 w_k1_AXP_s102
                                 + 0.952531 w_k1_DIS_s102
                                 + 0.760306 w_k1_FDX_s102
                                 + 1.15875 w_k1_IBM_s102
                                 + 0.820598 w_k1_INTC_s102
                                 + 0.854024 w_k1_KO_s102
                                 + 0.861783 w_k1_MAT_s102
                                 + 0.846387 w_k1_MSFT_s102
                                 + 0.857571 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.747819 w_k1_ADM_s102
                                 + 0.630299 w_k1_AXP_s102
                                 + 0.558611 w_k1_DIS_s102
                                 + 0.678365 w_k1_FDX_s102
                                 + 0.797814 w_k1_IBM_s102
                                 + 0.70507 w_k1_INTC_s102
                                 + 0.705993 w_k1_KO_s102
                                 + 0.774995 w_k1_MAT_s102
                                 + 0.722846 w_k1_MSFT_s102
                                 + 0.798909 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.40849 w_k1_ADM_s102
                                 + 1.41818 w_k1_AXP_s102
                                 + 1.70735 w_k1_DIS_s102
                                 + 1.29646 w_k1_FDX_s102
                                 + 1.28266 w_k1_IBM_s102
                                 + 1.11252 w_k1_INTC_s102
                                 + 1.25709 w_k1_KO_s102 + 1.34572 w_k1_MAT_s102
                                 + 1.18093 w_k1_MSFT_s102
                                 + 1.20546 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.729185 w_k1_ADM_s102
                                 + 0.838666 w_k1_AXP_s102
                                 + 0.934738 w_k1_DIS_s102
                                 + 0.960823 w_k1_FDX_s102
                                 + 0.704462 w_k1_IBM_s102
                                 + 0.964812 w_k1_INTC_s102
                                 + 0.740446 w_k1_KO_s102
                                 + 0.644859 w_k1_MAT_s102
                                 + 0.903927 w_k1_MSFT_s102
                                 + 0.92926 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.984308 w_k1_ADM_s103
                                 + 1.25805 w_k1_AXP_s103 + 1.0455 w_k1_DIS_s103
                                 + 1.04318 w_k1_FDX_s103
                                 + 1.76399 w_k1_IBM_s103
                                 + 1.29241 w_k1_INTC_s103
                                 + 1.39428 w_k1_KO_s103 + 2.12227 w_k1_MAT_s103
                                 + 1.07788 w_k1_MSFT_s103
                                 + 1.12828 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.34925 w_k1_ADM_s103
                                 + 1.88886 w_k1_AXP_s103
                                 + 1.55949 w_k1_DIS_s103
                                 + 1.55742 w_k1_FDX_s103
                                 + 1.15734 w_k1_IBM_s103
                                 + 1.47512 w_k1_INTC_s103
                                 + 1.19805 w_k1_KO_s103 + 1.64503 w_k1_MAT_s103
                                 + 1.45381 w_k1_MSFT_s103
                                 + 1.37504 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.05793 w_k1_ADM_s103
                                 + 1.181 w_k1_AXP_s103 + 1.24288 w_k1_DIS_s103
                                 + 1.09317 w_k1_FDX_s103
                                 + 0.993182 w_k1_IBM_s103
                                 + 1.01381 w_k1_INTC_s103
                                 + 1.24472 w_k1_KO_s103 + 1.63205 w_k1_MAT_s103
                                 + 1.33204 w_k1_MSFT_s103
                                 + 1.07524 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.849272 w_k1_ADM_s103
                                 + 0.89275 w_k1_AXP_s103
                                 + 0.638962 w_k1_DIS_s103
                                 + 0.827364 w_k1_FDX_s103
                                 + 0.985254 w_k1_IBM_s103
                                 + 0.912602 w_k1_INTC_s103
                                 + 0.890891 w_k1_KO_s103
                                 + 1.02419 w_k1_MAT_s103
                                 + 0.915307 w_k1_MSFT_s103
                                 + 0.955804 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.961831 w_k1_ADM_s103
                                 + 0.812029 w_k1_AXP_s103
                                 + 0.78558 w_k1_DIS_s103
                                 + 0.786468 w_k1_FDX_s103
                                 + 1.22158 w_k1_IBM_s103
                                 + 1.09656 w_k1_INTC_s103
                                 + 1.15745 w_k1_KO_s103 + 1.01389 w_k1_MAT_s103
                                 + 1.15994 w_k1_MSFT_s103
                                 + 0.924226 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.21326 w_k1_ADM_s103
                                 + 1.27955 w_k1_AXP_s103
                                 + 1.01318 w_k1_DIS_s103
                                 + 1.13529 w_k1_FDX_s103 + 1.2458 w_k1_IBM_s103
                                 + 1.10648 w_k1_INTC_s103
                                 + 1.19309 w_k1_KO_s103 + 1.36486 w_k1_MAT_s103
                                 + 1.19657 w_k1_MSFT_s103
                                 + 1.13548 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.36529 w_k1_ADM_s103
                                 + 1.44665 w_k1_AXP_s103
                                 + 1.45257 w_k1_DIS_s103
                                 + 1.23644 w_k1_FDX_s103
                                 + 0.918272 w_k1_IBM_s103
                                 + 1.20075 w_k1_INTC_s103
                                 + 1.27202 w_k1_KO_s103 + 1.44453 w_k1_MAT_s103
                                 + 1.46968 w_k1_MSFT_s103
                                 + 1.15535 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.28775 w_k1_ADM_s103
                                 + 1.00349 w_k1_AXP_s103
                                 + 0.892264 w_k1_DIS_s103
                                 + 0.939506 w_k1_FDX_s103
                                 + 0.89414 w_k1_IBM_s103
                                 + 0.910476 w_k1_INTC_s103
                                 + 1.03073 w_k1_KO_s103 + 1.13516 w_k1_MAT_s103
                                 + 1.07531 w_k1_MSFT_s103
                                 + 1.03235 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.777733 w_k1_ADM_s104
                                 + 0.730908 w_k1_AXP_s104
                                 + 0.532583 w_k1_DIS_s104
                                 + 0.704818 w_k1_FDX_s104
                                 + 0.766417 w_k1_IBM_s104
                                 + 0.736437 w_k1_INTC_s104
                                 + 0.617083 w_k1_KO_s104
                                 + 0.810398 w_k1_MAT_s104
                                 + 0.712651 w_k1_MSFT_s104
                                 + 0.694845 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.977907 w_k1_ADM_s104
                                 + 1.15858 w_k1_AXP_s104
                                 + 0.933409 w_k1_DIS_s104
                                 + 0.995762 w_k1_FDX_s104
                                 + 1.49797 w_k1_IBM_s104
                                 + 1.00528 w_k1_INTC_s104
                                 + 1.12226 w_k1_KO_s104 + 1.50282 w_k1_MAT_s104
                                 + 0.91721 w_k1_MSFT_s104
                                 + 1.04983 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.808964 w_k1_ADM_s104
                                 + 0.822387 w_k1_AXP_s104
                                 + 0.676426 w_k1_DIS_s104
                                 + 0.875825 w_k1_FDX_s104
                                 + 0.684466 w_k1_IBM_s104
                                 + 0.823525 w_k1_INTC_s104
                                 + 0.625472 w_k1_KO_s104
                                 + 0.522139 w_k1_MAT_s104
                                 + 0.87528 w_k1_MSFT_s104
                                 + 0.798009 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.08095 w_k1_ADM_s104
                                 + 1.18788 w_k1_AXP_s104
                                 + 1.17556 w_k1_DIS_s104
                                 + 1.26987 w_k1_FDX_s104
                                 + 0.79384 w_k1_IBM_s104
                                 + 0.959954 w_k1_INTC_s104
                                 + 1.00439 w_k1_KO_s104
                                 + 0.981517 w_k1_MAT_s104
                                 + 1.08601 w_k1_MSFT_s104
                                 + 1.05666 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.10211 w_k1_ADM_s104
                                 + 0.91866 w_k1_AXP_s104
                                 + 0.863104 w_k1_DIS_s104
                                 + 1.11156 w_k1_FDX_s104
                                 + 0.805334 w_k1_IBM_s104
                                 + 1.03631 w_k1_INTC_s104
                                 + 0.808138 w_k1_KO_s104
                                 + 0.542785 w_k1_MAT_s104
                                 + 0.881529 w_k1_MSFT_s104
                                 + 0.905999 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.06376 w_k1_ADM_s104
                                 + 0.885318 w_k1_AXP_s104
                                 + 1.20889 w_k1_DIS_s104
                                 + 1.09969 w_k1_FDX_s104
                                 + 1.21036 w_k1_IBM_s104
                                 + 1.04335 w_k1_INTC_s104
                                 + 1.08151 w_k1_KO_s104 + 1.07867 w_k1_MAT_s104
                                 + 1.17137 w_k1_MSFT_s104
                                 + 0.951362 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.936465 w_k1_ADM_s104
                                 + 0.868476 w_k1_AXP_s104
                                 + 0.799375 w_k1_DIS_s104
                                 + 0.926153 w_k1_FDX_s104
                                 + 1.04651 w_k1_IBM_s104
                                 + 0.937525 w_k1_INTC_s104
                                 + 0.838818 w_k1_KO_s104
                                 + 0.680968 w_k1_MAT_s104
                                 + 0.861552 w_k1_MSFT_s104
                                 + 0.938045 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.58783 w_k1_ADM_s104
                                 + 1.40661 w_k1_AXP_s104
                                 + 1.50199 w_k1_DIS_s104 + 1.4061 w_k1_FDX_s104
                                 + 1.43608 w_k1_IBM_s104
                                 + 0.93817 w_k1_INTC_s104
                                 + 1.35972 w_k1_KO_s104 + 1.25628 w_k1_MAT_s104
                                 + 1.18136 w_k1_MSFT_s104
                                 + 1.22298 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.15967 w_k1_ADM_s105
                                 + 1.47402 w_k1_AXP_s105
                                 + 1.25373 w_k1_DIS_s105
                                 + 1.25641 w_k1_FDX_s105
                                 + 1.16638 w_k1_IBM_s105
                                 + 1.29895 w_k1_INTC_s105
                                 + 1.25675 w_k1_KO_s105 + 1.63926 w_k1_MAT_s105
                                 + 1.14647 w_k1_MSFT_s105
                                 + 1.16784 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.998795 w_k1_ADM_s105
                                 + 1.06893 w_k1_AXP_s105
                                 + 0.804453 w_k1_DIS_s105
                                 + 1.05856 w_k1_FDX_s105
                                 + 0.666171 w_k1_IBM_s105
                                 + 1.12459 w_k1_INTC_s105
                                 + 0.92973 w_k1_KO_s105
                                 + 0.943878 w_k1_MAT_s105
                                 + 1.05423 w_k1_MSFT_s105
                                 + 0.983862 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.896657 w_k1_ADM_s105
                                 + 1.1024 w_k1_AXP_s105 + 1.10866 w_k1_DIS_s105
                                 + 0.907471 w_k1_FDX_s105
                                 + 1.31247 w_k1_IBM_s105
                                 + 1.18107 w_k1_INTC_s105
                                 + 1.22245 w_k1_KO_s105 + 1.38169 w_k1_MAT_s105
                                 + 0.874671 w_k1_MSFT_s105
                                 + 0.952723 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.73015 w_k1_ADM_s105
                                 + 0.798284 w_k1_AXP_s105
                                 + 0.64492 w_k1_DIS_s105
                                 + 0.717496 w_k1_FDX_s105
                                 + 1.11205 w_k1_IBM_s105
                                 + 1.10868 w_k1_INTC_s105
                                 + 0.954112 w_k1_KO_s105
                                 + 1.16793 w_k1_MAT_s105
                                 + 0.958077 w_k1_MSFT_s105
                                 + 0.862693 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.35257 w_k1_ADM_s105
                                 + 1.56083 w_k1_AXP_s105
                                 + 1.67927 w_k1_DIS_s105 + 1.3495 w_k1_FDX_s105
                                 + 1.07154 w_k1_IBM_s105
                                 + 1.48035 w_k1_INTC_s105
                                 + 1.52833 w_k1_KO_s105 + 1.3857 w_k1_MAT_s105
                                 + 1.45841 w_k1_MSFT_s105
                                 + 1.25661 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.18451 w_k1_ADM_s105
                                 + 1.7524 w_k1_AXP_s105 + 1.4539 w_k1_DIS_s105
                                 + 1.23934 w_k1_FDX_s105
                                 + 1.59532 w_k1_IBM_s105
                                 + 1.61831 w_k1_INTC_s105
                                 + 1.59084 w_k1_KO_s105 + 1.93441 w_k1_MAT_s105
                                 + 1.3513 w_k1_MSFT_s105 + 1.26572 w_k1_PG_s105
                                 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.771349 w_k1_ADM_s105
                                 + 1.01088 w_k1_AXP_s105
                                 + 0.751637 w_k1_DIS_s105
                                 + 0.856792 w_k1_FDX_s105
                                 + 0.951873 w_k1_IBM_s105
                                 + 1.1449 w_k1_INTC_s105
                                 + 0.907255 w_k1_KO_s105
                                 + 1.34468 w_k1_MAT_s105
                                 + 0.858564 w_k1_MSFT_s105
                                 + 0.884593 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.04227 w_k1_ADM_s105
                                 + 1.10762 w_k1_AXP_s105
                                 + 0.940009 w_k1_DIS_s105
                                 + 0.977608 w_k1_FDX_s105
                                 + 0.993529 w_k1_IBM_s105
                                 + 1.22639 w_k1_INTC_s105
                                 + 0.957474 w_k1_KO_s105
                                 + 0.900968 w_k1_MAT_s105
                                 + 1.0158 w_k1_MSFT_s105
                                 + 0.936884 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.896944 w_k1_ADM_s106
                                 + 0.621962 w_k1_AXP_s106
                                 + 0.745229 w_k1_DIS_s106
                                 + 0.866097 w_k1_FDX_s106
                                 + 0.68117 w_k1_IBM_s106
                                 + 0.810155 w_k1_INTC_s106
                                 + 0.816161 w_k1_KO_s106
                                 + 0.708164 w_k1_MAT_s106
                                 + 0.859417 w_k1_MSFT_s106
                                 + 0.98408 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.16885 w_k1_ADM_s106
                                 + 1.42198 w_k1_AXP_s106
                                 + 1.76841 w_k1_DIS_s106
                                 + 1.42778 w_k1_FDX_s106
                                 + 1.24072 w_k1_IBM_s106
                                 + 1.65094 w_k1_INTC_s106
                                 + 1.51185 w_k1_KO_s106 + 1.46317 w_k1_MAT_s106
                                 + 1.53576 w_k1_MSFT_s106
                                 + 1.41835 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.973335 w_k1_ADM_s106
                                 + 1.08081 w_k1_AXP_s106
                                 + 1.02081 w_k1_DIS_s106
                                 + 1.15389 w_k1_FDX_s106
                                 + 0.762852 w_k1_IBM_s106
                                 + 1.04238 w_k1_INTC_s106
                                 + 0.877738 w_k1_KO_s106
                                 + 0.762353 w_k1_MAT_s106
                                 + 0.916894 w_k1_MSFT_s106
                                 + 0.966476 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.851128 w_k1_ADM_s106
                                 + 1.0605 w_k1_AXP_s106
                                 + 0.900338 w_k1_DIS_s106
                                 + 0.909952 w_k1_FDX_s106
                                 + 1.27822 w_k1_IBM_s106
                                 + 1.17859 w_k1_INTC_s106
                                 + 1.26877 w_k1_KO_s106 + 1.02689 w_k1_MAT_s106
                                 + 0.989239 w_k1_MSFT_s106
                                 + 1.09015 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.815581 w_k1_ADM_s106
                                 + 0.708948 w_k1_AXP_s106
                                 + 0.514653 w_k1_DIS_s106
                                 + 0.731388 w_k1_FDX_s106
                                 + 0.83432 w_k1_IBM_s106
                                 + 0.816485 w_k1_INTC_s106
                                 + 0.729957 w_k1_KO_s106
                                 + 0.777433 w_k1_MAT_s106
                                 + 0.732716 w_k1_MSFT_s106
                                 + 0.849763 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.03979 w_k1_ADM_s106
                                 + 1.30334 w_k1_AXP_s106
                                 + 1.04186 w_k1_DIS_s106
                                 + 1.16902 w_k1_FDX_s106
                                 + 1.15822 w_k1_IBM_s106
                                 + 1.0675 w_k1_INTC_s106 + 1.14605 w_k1_KO_s106
                                 + 1.34489 w_k1_MAT_s106
                                 + 1.05787 w_k1_MSFT_s106
                                 + 1.11141 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.820163 w_k1_ADM_s106
                                 + 0.816184 w_k1_AXP_s106
                                 + 0.887535 w_k1_DIS_s106
                                 + 0.955479 w_k1_FDX_s106
                                 + 0.946498 w_k1_IBM_s106
                                 + 0.868088 w_k1_INTC_s106
                                 + 0.953316 w_k1_KO_s106
                                 + 0.999241 w_k1_MAT_s106
                                 + 0.828255 w_k1_MSFT_s106
                                 + 1.00808 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.53157 w_k1_ADM_s106
                                 + 1.25391 w_k1_AXP_s106
                                 + 1.45963 w_k1_DIS_s106 + 1.2601 w_k1_FDX_s106
                                 + 1.23423 w_k1_IBM_s106
                                 + 1.3001 w_k1_INTC_s106 + 1.47021 w_k1_KO_s106
                                 + 0.910912 w_k1_MAT_s106
                                 + 1.21263 w_k1_MSFT_s106
                                 + 1.20339 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.37282 w_k1_ADM_s107
                                 + 1.32149 w_k1_AXP_s107
                                 + 1.41408 w_k1_DIS_s107
                                 + 1.33016 w_k1_FDX_s107
                                 + 1.30633 w_k1_IBM_s107
                                 + 1.18793 w_k1_INTC_s107
                                 + 1.20047 w_k1_KO_s107
                                 + 0.948818 w_k1_MAT_s107
                                 + 1.15736 w_k1_MSFT_s107
                                 + 1.21921 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.859276 w_k1_ADM_s107
                                 + 0.841189 w_k1_AXP_s107
                                 + 0.624004 w_k1_DIS_s107
                                 + 0.924798 w_k1_FDX_s107
                                 + 0.675677 w_k1_IBM_s107
                                 + 1.00437 w_k1_INTC_s107
                                 + 0.667813 w_k1_KO_s107
                                 + 0.51056 w_k1_MAT_s107
                                 + 0.794854 w_k1_MSFT_s107
                                 + 0.875511 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.967637 w_k1_ADM_s107
                                 + 0.894097 w_k1_AXP_s107
                                 + 0.747219 w_k1_DIS_s107
                                 + 0.795633 w_k1_FDX_s107 + 1.08 w_k1_IBM_s107
                                 + 0.826934 w_k1_INTC_s107
                                 + 0.868529 w_k1_KO_s107
                                 + 0.834824 w_k1_MAT_s107
                                 + 0.77864 w_k1_MSFT_s107
                                 + 0.874926 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.14553 w_k1_ADM_s107
                                 + 1.31321 w_k1_AXP_s107
                                 + 1.05508 w_k1_DIS_s107 + 1.0264 w_k1_FDX_s107
                                 + 1.36698 w_k1_IBM_s107
                                 + 0.937559 w_k1_INTC_s107
                                 + 1.23466 w_k1_KO_s107 + 1.51115 w_k1_MAT_s107
                                 + 1.13825 w_k1_MSFT_s107 + 1.2293 w_k1_PG_s107
                                 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.10986 w_k1_ADM_s107
                                 + 0.933004 w_k1_AXP_s107
                                 + 0.914564 w_k1_DIS_s107
                                 + 1.01898 w_k1_FDX_s107
                                 + 0.610301 w_k1_IBM_s107
                                 + 1.0642 w_k1_INTC_s107
                                 + 0.886413 w_k1_KO_s107
                                 + 0.588901 w_k1_MAT_s107
                                 + 0.948683 w_k1_MSFT_s107
                                 + 1.04604 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.949051 w_k1_ADM_s107
                                 + 1.00216 w_k1_AXP_s107
                                 + 1.06257 w_k1_DIS_s107
                                 + 1.01597 w_k1_FDX_s107
                                 + 1.09926 w_k1_IBM_s107
                                 + 0.915155 w_k1_INTC_s107
                                 + 1.06637 w_k1_KO_s107
                                 + 0.980155 w_k1_MAT_s107
                                 + 0.930828 w_k1_MSFT_s107
                                 + 1.04893 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.16811 w_k1_ADM_s107
                                 + 1.03227 w_k1_AXP_s107
                                 + 1.26829 w_k1_DIS_s107
                                 + 1.18111 w_k1_FDX_s107
                                 + 1.11176 w_k1_IBM_s107
                                 + 1.2491 w_k1_INTC_s107 + 1.2591 w_k1_KO_s107
                                 + 0.608151 w_k1_MAT_s107
                                 + 1.2457 w_k1_MSFT_s107 + 1.14199 w_k1_PG_s107
                                 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.14608 w_k1_ADM_s107
                                 + 0.868121 w_k1_AXP_s107
                                 + 0.825708 w_k1_DIS_s107
                                 + 0.854129 w_k1_FDX_s107
                                 + 1.05256 w_k1_IBM_s107
                                 + 0.967945 w_k1_INTC_s107
                                 + 0.957345 w_k1_KO_s107
                                 + 0.615564 w_k1_MAT_s107
                                 + 1.03564 w_k1_MSFT_s107
                                 + 1.03394 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.32105 w_k1_ADM_s110
                                 + 1.37375 w_k1_AXP_s110 + 1.8823 w_k1_DIS_s110
                                 + 1.50923 w_k1_FDX_s110
                                 + 1.48502 w_k1_IBM_s110
                                 + 1.69286 w_k1_INTC_s110
                                 + 1.50937 w_k1_KO_s110
                                 + 0.906894 w_k1_MAT_s110
                                 + 1.4635 w_k1_MSFT_s110 + 1.33396 w_k1_PG_s110
                                 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.36442 w_k1_ADM_s110
                                 + 1.28996 w_k1_AXP_s110
                                 + 1.14451 w_k1_DIS_s110
                                 + 1.25106 w_k1_FDX_s110
                                 + 0.898314 w_k1_IBM_s110
                                 + 1.35362 w_k1_INTC_s110
                                 + 0.98382 w_k1_KO_s110 + 1.07082 w_k1_MAT_s110
                                 + 1.02855 w_k1_MSFT_s110
                                 + 0.975314 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.0594 w_k1_ADM_s110
                                 + 1.30627 w_k1_AXP_s110
                                 + 1.24221 w_k1_DIS_s110
                                 + 1.10465 w_k1_FDX_s110 + 1.3394 w_k1_IBM_s110
                                 + 1.12621 w_k1_INTC_s110 + 1.1794 w_k1_KO_s110
                                 + 1.54229 w_k1_MAT_s110
                                 + 0.990234 w_k1_MSFT_s110
                                 + 1.1185 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.806896 w_k1_ADM_s110
                                 + 0.872067 w_k1_AXP_s110
                                 + 0.931162 w_k1_DIS_s110
                                 + 0.832994 w_k1_FDX_s110
                                 + 1.28989 w_k1_IBM_s110
                                 + 0.864727 w_k1_INTC_s110
                                 + 0.992808 w_k1_KO_s110
                                 + 1.02256 w_k1_MAT_s110
                                 + 0.823347 w_k1_MSFT_s110
                                 + 0.918958 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.837815 w_k1_ADM_s110
                                 + 0.764946 w_k1_AXP_s110
                                 + 0.687819 w_k1_DIS_s110
                                 + 0.769983 w_k1_FDX_s110
                                 + 0.761373 w_k1_IBM_s110
                                 + 0.884654 w_k1_INTC_s110
                                 + 0.708915 w_k1_KO_s110
                                 + 0.64071 w_k1_MAT_s110
                                 + 0.811071 w_k1_MSFT_s110
                                 + 0.826879 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.1863 w_k1_ADM_s110
                                 + 1.47382 w_k1_AXP_s110
                                 + 1.50959 w_k1_DIS_s110
                                 + 1.39194 w_k1_FDX_s110
                                 + 1.57529 w_k1_IBM_s110
                                 + 1.23963 w_k1_INTC_s110
                                 + 1.30674 w_k1_KO_s110 + 2.05492 w_k1_MAT_s110
                                 + 1.20491 w_k1_MSFT_s110
                                 + 1.20944 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.921813 w_k1_ADM_s110
                                 + 1.01017 w_k1_AXP_s110
                                 + 0.921999 w_k1_DIS_s110
                                 + 0.975995 w_k1_FDX_s110
                                 + 0.964901 w_k1_IBM_s110
                                 + 1.14156 w_k1_INTC_s110
                                 + 0.963227 w_k1_KO_s110
                                 + 1.13561 w_k1_MAT_s110
                                 + 0.879894 w_k1_MSFT_s110
                                 + 0.965568 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.00734 w_k1_ADM_s110
                                 + 0.976905 w_k1_AXP_s110
                                 + 0.775693 w_k1_DIS_s110
                                 + 0.94492 w_k1_FDX_s110
                                 + 0.746611 w_k1_IBM_s110
                                 + 0.917084 w_k1_INTC_s110
                                 + 0.671664 w_k1_KO_s110
                                 + 0.583221 w_k1_MAT_s110
                                 + 0.829256 w_k1_MSFT_s110
                                 + 0.831061 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.951897 w_k1_ADM_s111
                                 + 1.06748 w_k1_AXP_s111
                                 + 1.05218 w_k1_DIS_s111
                                 + 1.11002 w_k1_FDX_s111
                                 + 0.855793 w_k1_IBM_s111
                                 + 1.07043 w_k1_INTC_s111
                                 + 1.03426 w_k1_KO_s111
                                 + 0.907124 w_k1_MAT_s111
                                 + 1.1919 w_k1_MSFT_s111
                                 + 0.974476 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.15121 w_k1_ADM_s111
                                 + 0.938808 w_k1_AXP_s111
                                 + 0.973663 w_k1_DIS_s111
                                 + 0.937715 w_k1_FDX_s111
                                 + 1.39362 w_k1_IBM_s111
                                 + 0.880973 w_k1_INTC_s111
                                 + 1.26594 w_k1_KO_s111 + 1.36082 w_k1_MAT_s111
                                 + 1.10307 w_k1_MSFT_s111
                                 + 1.10053 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.39792 w_k1_ADM_s111
                                 + 1.34648 w_k1_AXP_s111
                                 + 1.57458 w_k1_DIS_s111
                                 + 1.32233 w_k1_FDX_s111
                                 + 1.19671 w_k1_IBM_s111
                                 + 1.32413 w_k1_INTC_s111
                                 + 1.65225 w_k1_KO_s111 + 1.28039 w_k1_MAT_s111
                                 + 1.58515 w_k1_MSFT_s111
                                 + 1.26538 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.901962 w_k1_ADM_s111
                                 + 1.09006 w_k1_AXP_s111
                                 + 1.03536 w_k1_DIS_s111
                                 + 0.949776 w_k1_FDX_s111
                                 + 1.6116 w_k1_IBM_s111
                                 + 1.05493 w_k1_INTC_s111
                                 + 1.22353 w_k1_KO_s111 + 1.55109 w_k1_MAT_s111
                                 + 1.14459 w_k1_MSFT_s111
                                 + 1.18826 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.02952 w_k1_ADM_s111
                                 + 1.36617 w_k1_AXP_s111
                                 + 1.56808 w_k1_DIS_s111 + 1.1281 w_k1_FDX_s111
                                 + 2.21358 w_k1_IBM_s111
                                 + 1.38859 w_k1_INTC_s111
                                 + 2.05688 w_k1_KO_s111 + 2.35566 w_k1_MAT_s111
                                 + 1.48056 w_k1_MSFT_s111
                                 + 1.22451 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.04668 w_k1_ADM_s111
                                 + 0.782359 w_k1_AXP_s111
                                 + 0.795649 w_k1_DIS_s111
                                 + 0.873277 w_k1_FDX_s111
                                 + 0.937452 w_k1_IBM_s111
                                 + 0.852303 w_k1_INTC_s111
                                 + 1.03355 w_k1_KO_s111
                                 + 0.691993 w_k1_MAT_s111
                                 + 0.894226 w_k1_MSFT_s111
                                 + 0.974684 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.27349 w_k1_ADM_s111
                                 + 1.02403 w_k1_AXP_s111
                                 + 1.32465 w_k1_DIS_s111
                                 + 1.41895 w_k1_FDX_s111
                                 + 0.645603 w_k1_IBM_s111
                                 + 1.35955 w_k1_INTC_s111
                                 + 0.975446 w_k1_KO_s111
                                 + 0.759357 w_k1_MAT_s111
                                 + 1.6347 w_k1_MSFT_s111 + 1.12527 w_k1_PG_s111
                                 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.09231 w_k1_ADM_s111
                                 + 0.764334 w_k1_AXP_s111
                                 + 0.678107 w_k1_DIS_s111
                                 + 0.856564 w_k1_FDX_s111
                                 + 0.704483 w_k1_IBM_s111
                                 + 0.813979 w_k1_INTC_s111
                                 + 0.821846 w_k1_KO_s111
                                 + 0.752339 w_k1_MAT_s111
                                 + 1.11752 w_k1_MSFT_s111
                                 + 0.958643 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.985686 w_k1_ADM_s112
                                 + 1.03851 w_k1_AXP_s112
                                 + 1.00689 w_k1_DIS_s112
                                 + 1.17524 w_k1_FDX_s112
                                 + 0.852145 w_k1_IBM_s112
                                 + 1.05574 w_k1_INTC_s112
                                 + 1.03118 w_k1_KO_s112 + 1.36642 w_k1_MAT_s112
                                 + 1.05188 w_k1_MSFT_s112
                                 + 1.02416 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.979199 w_k1_ADM_s112
                                 + 0.997072 w_k1_AXP_s112
                                 + 1.0774 w_k1_DIS_s112
                                 + 0.962159 w_k1_FDX_s112
                                 + 0.925719 w_k1_IBM_s112
                                 + 0.973016 w_k1_INTC_s112
                                 + 1.08611 w_k1_KO_s112
                                 + 0.956879 w_k1_MAT_s112
                                 + 1.01923 w_k1_MSFT_s112 + 1.0089 w_k1_PG_s112
                                 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.769708 w_k1_ADM_s112
                                 + 0.686236 w_k1_AXP_s112
                                 + 0.622166 w_k1_DIS_s112
                                 + 0.810142 w_k1_FDX_s112
                                 + 0.659377 w_k1_IBM_s112
                                 + 0.807659 w_k1_INTC_s112
                                 + 0.805362 w_k1_KO_s112
                                 + 0.660615 w_k1_MAT_s112
                                 + 0.780469 w_k1_MSFT_s112
                                 + 0.833676 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.23066 w_k1_ADM_s112
                                 + 1.2878 w_k1_AXP_s112 + 1.4652 w_k1_DIS_s112
                                 + 1.34782 w_k1_FDX_s112
                                 + 1.25686 w_k1_IBM_s112
                                 + 0.99661 w_k1_INTC_s112
                                 + 1.38479 w_k1_KO_s112 + 1.65912 w_k1_MAT_s112
                                 + 1.06737 w_k1_MSFT_s112
                                 + 1.15931 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.973948 w_k1_ADM_s112
                                 + 1.00769 w_k1_AXP_s112
                                 + 0.944119 w_k1_DIS_s112
                                 + 0.93943 w_k1_FDX_s112
                                 + 1.37935 w_k1_IBM_s112
                                 + 0.878159 w_k1_INTC_s112
                                 + 1.24006 w_k1_KO_s112 + 1.55989 w_k1_MAT_s112
                                 + 0.786482 w_k1_MSFT_s112
                                 + 0.979582 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.996917 w_k1_ADM_s112
                                 + 0.894275 w_k1_AXP_s112
                                 + 0.708171 w_k1_DIS_s112
                                 + 0.865856 w_k1_FDX_s112
                                 + 0.835079 w_k1_IBM_s112
                                 + 0.69245 w_k1_INTC_s112
                                 + 0.818842 w_k1_KO_s112
                                 + 1.07866 w_k1_MAT_s112
                                 + 0.773695 w_k1_MSFT_s112
                                 + 0.943403 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.790345 w_k1_ADM_s112
                                 + 0.756048 w_k1_AXP_s112
                                 + 0.768689 w_k1_DIS_s112
                                 + 0.793282 w_k1_FDX_s112
                                 + 0.875687 w_k1_IBM_s112
                                 + 0.701971 w_k1_INTC_s112
                                 + 0.939098 w_k1_KO_s112
                                 + 1.00247 w_k1_MAT_s112
                                 + 0.752605 w_k1_MSFT_s112
                                 + 0.882827 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.729833 w_k1_ADM_s112
                                 + 0.858742 w_k1_AXP_s112
                                 + 0.901887 w_k1_DIS_s112
                                 + 0.870613 w_k1_FDX_s112
                                 + 1.06908 w_k1_IBM_s112
                                 + 0.78882 w_k1_INTC_s112
                                 + 1.04549 w_k1_KO_s112 + 1.375 w_k1_MAT_s112
                                 + 0.824258 w_k1_MSFT_s112
                                 + 0.980497 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.679967 w_k1_ADM_s113
                                 + 0.612345 w_k1_AXP_s113
                                 + 0.610883 w_k1_DIS_s113
                                 + 0.693956 w_k1_FDX_s113
                                 + 0.600375 w_k1_IBM_s113
                                 + 0.859937 w_k1_INTC_s113
                                 + 0.611564 w_k1_KO_s113
                                 + 0.603887 w_k1_MAT_s113
                                 + 0.816623 w_k1_MSFT_s113
                                 + 0.795323 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.12819 w_k1_ADM_s113
                                 + 1.29577 w_k1_AXP_s113
                                 + 1.38282 w_k1_DIS_s113
                                 + 1.20568 w_k1_FDX_s113
                                 + 1.21781 w_k1_IBM_s113
                                 + 1.12906 w_k1_INTC_s113 + 1.165 w_k1_KO_s113
                                 + 1.41297 w_k1_MAT_s113
                                 + 1.27833 w_k1_MSFT_s113
                                 + 1.20815 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.1994 w_k1_ADM_s113
                                 + 0.744913 w_k1_AXP_s113
                                 + 1.17994 w_k1_DIS_s113
                                 + 1.05185 w_k1_FDX_s113
                                 + 0.76302 w_k1_IBM_s113
                                 + 1.10104 w_k1_INTC_s113
                                 + 1.10145 w_k1_KO_s113
                                 + 0.730639 w_k1_MAT_s113
                                 + 1.27362 w_k1_MSFT_s113
                                 + 0.985596 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.55099 w_k1_ADM_s113
                                 + 1.6637 w_k1_AXP_s113 + 1.76157 w_k1_DIS_s113
                                 + 1.27047 w_k1_FDX_s113
                                 + 1.81216 w_k1_IBM_s113 + 1.557 w_k1_INTC_s113
                                 + 1.54871 w_k1_KO_s113 + 1.65284 w_k1_MAT_s113
                                 + 1.39032 w_k1_MSFT_s113
                                 + 1.19237 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.05964 w_k1_ADM_s113
                                 + 1.08825 w_k1_AXP_s113
                                 + 0.990486 w_k1_DIS_s113
                                 + 0.956961 w_k1_FDX_s113
                                 + 1.19962 w_k1_IBM_s113
                                 + 1.07598 w_k1_INTC_s113
                                 + 1.25297 w_k1_KO_s113 + 1.16231 w_k1_MAT_s113
                                 + 1.23026 w_k1_MSFT_s113
                                 + 1.09317 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.926726 w_k1_ADM_s113
                                 + 0.741244 w_k1_AXP_s113
                                 + 0.960266 w_k1_DIS_s113
                                 + 0.970917 w_k1_FDX_s113
                                 + 0.651416 w_k1_IBM_s113
                                 + 0.890766 w_k1_INTC_s113
                                 + 0.736159 w_k1_KO_s113
                                 + 0.719922 w_k1_MAT_s113
                                 + 1.03844 w_k1_MSFT_s113
                                 + 0.972799 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.830833 w_k1_ADM_s113
                                 + 0.784541 w_k1_AXP_s113
                                 + 0.719289 w_k1_DIS_s113
                                 + 0.814024 w_k1_FDX_s113
                                 + 0.836211 w_k1_IBM_s113
                                 + 0.808748 w_k1_INTC_s113
                                 + 0.8343 w_k1_KO_s113 + 0.856401 w_k1_MAT_s113
                                 + 0.863136 w_k1_MSFT_s113
                                 + 0.856359 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.18484 w_k1_ADM_s113
                                 + 0.928928 w_k1_AXP_s113
                                 + 1.2136 w_k1_DIS_s113
                                 + 0.951364 w_k1_FDX_s113
                                 + 0.991392 w_k1_IBM_s113
                                 + 1.04611 w_k1_INTC_s113
                                 + 1.07959 w_k1_KO_s113
                                 + 0.717848 w_k1_MAT_s113
                                 + 1.20822 w_k1_MSFT_s113
                                 + 0.988814 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.846649 w_k1_ADM_s114
                                 + 0.707518 w_k1_AXP_s114
                                 + 0.836219 w_k1_DIS_s114
                                 + 0.914441 w_k1_FDX_s114
                                 + 0.624601 w_k1_IBM_s114
                                 + 0.827305 w_k1_INTC_s114
                                 + 0.841745 w_k1_KO_s114
                                 + 0.726627 w_k1_MAT_s114
                                 + 1.11796 w_k1_MSFT_s114
                                 + 0.922819 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.27189 w_k1_ADM_s114
                                 + 1.63711 w_k1_AXP_s114
                                 + 1.43927 w_k1_DIS_s114
                                 + 1.32369 w_k1_FDX_s114
                                 + 1.72746 w_k1_IBM_s114
                                 + 1.16716 w_k1_INTC_s114
                                 + 1.38299 w_k1_KO_s114 + 1.88601 w_k1_MAT_s114
                                 + 1.21389 w_k1_MSFT_s114
                                 + 1.27632 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.760313 w_k1_ADM_s114
                                 + 0.857302 w_k1_AXP_s114
                                 + 0.778009 w_k1_DIS_s114
                                 + 0.824765 w_k1_FDX_s114
                                 + 0.757655 w_k1_IBM_s114
                                 + 0.855818 w_k1_INTC_s114
                                 + 0.741556 w_k1_KO_s114
                                 + 0.705568 w_k1_MAT_s114
                                 + 0.798697 w_k1_MSFT_s114
                                 + 0.753177 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.735042 w_k1_ADM_s114
                                 + 1.20949 w_k1_AXP_s114
                                 + 0.924458 w_k1_DIS_s114
                                 + 1.01552 w_k1_FDX_s114
                                 + 0.823726 w_k1_IBM_s114
                                 + 0.910197 w_k1_INTC_s114
                                 + 0.84932 w_k1_KO_s114 + 1.2458 w_k1_MAT_s114
                                 + 0.950241 w_k1_MSFT_s114
                                 + 0.917207 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.690416 w_k1_ADM_s114
                                 + 0.717185 w_k1_AXP_s114
                                 + 0.716461 w_k1_DIS_s114
                                 + 0.749818 w_k1_FDX_s114
                                 + 0.922349 w_k1_IBM_s114
                                 + 0.591035 w_k1_INTC_s114
                                 + 0.76705 w_k1_KO_s114 + 1.15741 w_k1_MAT_s114
                                 + 0.797809 w_k1_MSFT_s114
                                 + 0.851096 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.678836 w_k1_ADM_s114
                                 + 1.07454 w_k1_AXP_s114
                                 + 1.03231 w_k1_DIS_s114
                                 + 0.86875 w_k1_FDX_s114 + 1.3721 w_k1_IBM_s114
                                 + 0.759617 w_k1_INTC_s114
                                 + 1.08925 w_k1_KO_s114 + 1.65902 w_k1_MAT_s114
                                 + 0.881533 w_k1_MSFT_s114
                                 + 0.996448 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.14532 w_k1_ADM_s114
                                 + 1.24008 w_k1_AXP_s114
                                 + 1.35806 w_k1_DIS_s114
                                 + 1.22685 w_k1_FDX_s114
                                 + 0.968045 w_k1_IBM_s114
                                 + 0.850927 w_k1_INTC_s114
                                 + 1.17424 w_k1_KO_s114 + 1.09978 w_k1_MAT_s114
                                 + 1.13911 w_k1_MSFT_s114
                                 + 1.08949 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.12267 w_k1_ADM_s114
                                 + 0.967863 w_k1_AXP_s114
                                 + 0.866622 w_k1_DIS_s114
                                 + 1.0904 w_k1_FDX_s114
                                 + 0.590288 w_k1_IBM_s114
                                 + 0.74562 w_k1_INTC_s114
                                 + 0.766912 w_k1_KO_s114
                                 + 0.640099 w_k1_MAT_s114
                                 + 0.975731 w_k1_MSFT_s114
                                 + 0.896986 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.804666 w_k1_ADM_s115
                                 + 0.877782 w_k1_AXP_s115
                                 + 1.20367 w_k1_DIS_s115
                                 + 0.964395 w_k1_FDX_s115
                                 + 1.11392 w_k1_IBM_s115
                                 + 1.04196 w_k1_INTC_s115
                                 + 1.05137 w_k1_KO_s115 + 1.00037 w_k1_MAT_s115
                                 + 1.16462 w_k1_MSFT_s115 + 1.0596 w_k1_PG_s115
                                 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.948558 w_k1_ADM_s115
                                 + 0.796508 w_k1_AXP_s115
                                 + 0.685296 w_k1_DIS_s115
                                 + 0.866256 w_k1_FDX_s115
                                 + 0.748482 w_k1_IBM_s115
                                 + 0.870689 w_k1_INTC_s115
                                 + 0.757895 w_k1_KO_s115
                                 + 0.634862 w_k1_MAT_s115
                                 + 0.929734 w_k1_MSFT_s115
                                 + 0.82896 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.11221 w_k1_ADM_s115
                                 + 0.930784 w_k1_AXP_s115
                                 + 1.05165 w_k1_DIS_s115
                                 + 0.972763 w_k1_FDX_s115
                                 + 0.911949 w_k1_IBM_s115
                                 + 1.10022 w_k1_INTC_s115
                                 + 0.952038 w_k1_KO_s115
                                 + 0.944583 w_k1_MAT_s115
                                 + 1.20498 w_k1_MSFT_s115
                                 + 0.978932 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.69737 w_k1_ADM_s115
                                 + 1.5045 w_k1_AXP_s115 + 1.71493 w_k1_DIS_s115
                                 + 1.41367 w_k1_FDX_s115
                                 + 1.33897 w_k1_IBM_s115
                                 + 1.46212 w_k1_INTC_s115
                                 + 1.35648 w_k1_KO_s115 + 1.03878 w_k1_MAT_s115
                                 + 1.53355 w_k1_MSFT_s115
                                 + 1.31905 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.925463 w_k1_ADM_s115
                                 + 0.615113 w_k1_AXP_s115
                                 + 0.665138 w_k1_DIS_s115
                                 + 0.742548 w_k1_FDX_s115
                                 + 0.687892 w_k1_IBM_s115
                                 + 0.742217 w_k1_INTC_s115
                                 + 0.643625 w_k1_KO_s115
                                 + 0.433561 w_k1_MAT_s115
                                 + 0.759599 w_k1_MSFT_s115
                                 + 0.830697 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.823937 w_k1_ADM_s115
                                 + 0.851994 w_k1_AXP_s115
                                 + 0.632419 w_k1_DIS_s115
                                 + 0.728886 w_k1_FDX_s115
                                 + 1.15912 w_k1_IBM_s115
                                 + 0.99928 w_k1_INTC_s115
                                 + 0.795503 w_k1_KO_s115
                                 + 0.799137 w_k1_MAT_s115
                                 + 0.798391 w_k1_MSFT_s115
                                 + 0.811145 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.16211 w_k1_ADM_s115
                                 + 1.1918 w_k1_AXP_s115 + 1.0136 w_k1_DIS_s115
                                 + 0.986408 w_k1_FDX_s115
                                 + 1.49924 w_k1_IBM_s115
                                 + 1.0151 w_k1_INTC_s115 + 1.00612 w_k1_KO_s115
                                 + 1.45489 w_k1_MAT_s115
                                 + 0.981677 w_k1_MSFT_s115
                                 + 1.13335 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.19918 w_k1_ADM_s115
                                 + 1.19562 w_k1_AXP_s115
                                 + 1.04212 w_k1_DIS_s115
                                 + 1.18415 w_k1_FDX_s115
                                 + 0.866258 w_k1_IBM_s115
                                 + 1.00473 w_k1_INTC_s115
                                 + 0.829336 w_k1_KO_s115
                                 + 1.03784 w_k1_MAT_s115
                                 + 1.17169 w_k1_MSFT_s115
                                 + 0.928975 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.09154 w_k1_ADM_s116
                                 + 0.827138 w_k1_AXP_s116
                                 + 1.06599 w_k1_DIS_s116
                                 + 0.988003 w_k1_FDX_s116
                                 + 0.815652 w_k1_IBM_s116
                                 + 0.855917 w_k1_INTC_s116
                                 + 1.04511 w_k1_KO_s116
                                 + 0.834813 w_k1_MAT_s116
                                 + 0.953034 w_k1_MSFT_s116
                                 + 1.06109 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.23354 w_k1_ADM_s116
                                 + 1.15761 w_k1_AXP_s116
                                 + 1.22945 w_k1_DIS_s116
                                 + 1.02882 w_k1_FDX_s116
                                 + 1.32356 w_k1_IBM_s116
                                 + 0.938025 w_k1_INTC_s116
                                 + 1.2578 w_k1_KO_s116 + 1.43749 w_k1_MAT_s116
                                 + 0.970808 w_k1_MSFT_s116
                                 + 1.24234 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.715664 w_k1_ADM_s116
                                 + 0.561258 w_k1_AXP_s116
                                 + 0.635978 w_k1_DIS_s116
                                 + 0.702141 w_k1_FDX_s116
                                 + 0.709894 w_k1_IBM_s116
                                 + 0.727628 w_k1_INTC_s116
                                 + 0.835837 w_k1_KO_s116
                                 + 0.638273 w_k1_MAT_s116
                                 + 0.749808 w_k1_MSFT_s116
                                 + 0.894587 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.861776 w_k1_ADM_s116
                                 + 0.858417 w_k1_AXP_s116
                                 + 0.770824 w_k1_DIS_s116
                                 + 0.826937 w_k1_FDX_s116
                                 + 0.826704 w_k1_IBM_s116
                                 + 0.87666 w_k1_INTC_s116
                                 + 0.96542 w_k1_KO_s116
                                 + 0.969411 w_k1_MAT_s116
                                 + 0.835301 w_k1_MSFT_s116
                                 + 0.933605 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.925065 w_k1_ADM_s116
                                 + 1.02031 w_k1_AXP_s116
                                 + 1.06168 w_k1_DIS_s116
                                 + 0.980647 w_k1_FDX_s116 + 1.067 w_k1_IBM_s116
                                 + 1.09527 w_k1_INTC_s116
                                 + 1.05077 w_k1_KO_s116
                                 + 0.908624 w_k1_MAT_s116
                                 + 0.977448 w_k1_MSFT_s116
                                 + 1.01174 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.20749 w_k1_ADM_s116
                                 + 1.01082 w_k1_AXP_s116
                                 + 0.975995 w_k1_DIS_s116
                                 + 1.08038 w_k1_FDX_s116
                                 + 0.656513 w_k1_IBM_s116
                                 + 0.910165 w_k1_INTC_s116
                                 + 0.848809 w_k1_KO_s116
                                 + 0.628378 w_k1_MAT_s116
                                 + 0.938963 w_k1_MSFT_s116
                                 + 0.932693 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.997154 w_k1_ADM_s116
                                 + 1.13759 w_k1_AXP_s116
                                 + 0.981862 w_k1_DIS_s116
                                 + 1.08583 w_k1_FDX_s116
                                 + 1.03745 w_k1_IBM_s116
                                 + 1.02845 w_k1_INTC_s116
                                 + 1.05998 w_k1_KO_s116 + 1.24567 w_k1_MAT_s116
                                 + 0.987806 w_k1_MSFT_s116
                                 + 1.0961 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.642816 w_k1_ADM_s116
                                 + 0.649682 w_k1_AXP_s116
                                 + 0.66797 w_k1_DIS_s116
                                 + 0.74348 w_k1_FDX_s116
                                 + 0.966518 w_k1_IBM_s116
                                 + 0.925858 w_k1_INTC_s116
                                 + 0.812808 w_k1_KO_s116
                                 + 0.66745 w_k1_MAT_s116
                                 + 0.718032 w_k1_MSFT_s116
                                 + 0.866879 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.32114 w_k1_ADM_s117
                                 + 1.17221 w_k1_AXP_s117
                                 + 1.49088 w_k1_DIS_s117
                                 + 1.18124 w_k1_FDX_s117
                                 + 1.78542 w_k1_IBM_s117
                                 + 1.09517 w_k1_INTC_s117
                                 + 1.59333 w_k1_KO_s117 + 1.39388 w_k1_MAT_s117
                                 + 1.14792 w_k1_MSFT_s117
                                 + 1.22875 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.10171 w_k1_ADM_s117
                                 + 1.15304 w_k1_AXP_s117
                                 + 1.34526 w_k1_DIS_s117
                                 + 1.21857 w_k1_FDX_s117
                                 + 1.13992 w_k1_IBM_s117
                                 + 1.38972 w_k1_INTC_s117
                                 + 1.33985 w_k1_KO_s117
                                 + 0.984528 w_k1_MAT_s117
                                 + 1.14725 w_k1_MSFT_s117 + 1.0963 w_k1_PG_s117
                                 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.798339 w_k1_ADM_s117
                                 + 0.758346 w_k1_AXP_s117
                                 + 0.752372 w_k1_DIS_s117
                                 + 0.889107 w_k1_FDX_s117
                                 + 0.588009 w_k1_IBM_s117
                                 + 0.826354 w_k1_INTC_s117
                                 + 0.713867 w_k1_KO_s117
                                 + 0.530767 w_k1_MAT_s117
                                 + 0.960476 w_k1_MSFT_s117
                                 + 0.829168 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.856149 w_k1_ADM_s117
                                 + 0.568404 w_k1_AXP_s117
                                 + 0.720503 w_k1_DIS_s117
                                 + 0.738894 w_k1_FDX_s117
                                 + 0.724716 w_k1_IBM_s117
                                 + 0.865816 w_k1_INTC_s117
                                 + 0.804434 w_k1_KO_s117
                                 + 0.499714 w_k1_MAT_s117
                                 + 0.745948 w_k1_MSFT_s117
                                 + 0.745571 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.0781 w_k1_ADM_s117
                                 + 1.0965 w_k1_AXP_s117
                                 + 0.834342 w_k1_DIS_s117
                                 + 1.00638 w_k1_FDX_s117
                                 + 0.994461 w_k1_IBM_s117
                                 + 1.10751 w_k1_INTC_s117
                                 + 0.954686 w_k1_KO_s117
                                 + 0.898419 w_k1_MAT_s117
                                 + 0.955131 w_k1_MSFT_s117
                                 + 0.975826 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.933375 w_k1_ADM_s117
                                 + 0.953068 w_k1_AXP_s117
                                 + 1.47541 w_k1_DIS_s117
                                 + 1.06393 w_k1_FDX_s117
                                 + 1.44715 w_k1_IBM_s117
                                 + 1.28329 w_k1_INTC_s117
                                 + 1.43909 w_k1_KO_s117 + 1.06337 w_k1_MAT_s117
                                 + 1.23178 w_k1_MSFT_s117
                                 + 1.07995 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.859539 w_k1_ADM_s117
                                 + 0.788753 w_k1_AXP_s117
                                 + 0.813148 w_k1_DIS_s117
                                 + 0.790036 w_k1_FDX_s117
                                 + 1.03677 w_k1_IBM_s117
                                 + 0.847393 w_k1_INTC_s117
                                 + 0.841948 w_k1_KO_s117
                                 + 0.762536 w_k1_MAT_s117
                                 + 0.769784 w_k1_MSFT_s117
                                 + 0.878312 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.816297 w_k1_ADM_s117
                                 + 0.719381 w_k1_AXP_s117
                                 + 1.11768 w_k1_DIS_s117
                                 + 0.922427 w_k1_FDX_s117
                                 + 0.89372 w_k1_IBM_s117
                                 + 1.05724 w_k1_INTC_s117
                                 + 1.00737 w_k1_KO_s117
                                 + 0.598282 w_k1_MAT_s117
                                 + 1.01661 w_k1_MSFT_s117
                                 + 0.926944 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0516935631216584 w_k1_ADM_s000
                                 + 0.0267473210766001 w_k1_ADM_s001
                                 + 0.024209195999257 w_k1_ADM_s002
                                 + 0.06593101161938 w_k1_ADM_s003
                                 + 0.0813010214873456 w_k1_ADM_s004
                                 + 0.0925724890994033 w_k1_ADM_s005
                                 + 0.0236489436713043 w_k1_ADM_s006
                                 + 0.0260749421958728 w_k1_ADM_s007
                                 + 0.0251671544612133 w_k1_ADM_s010
                                 + 0.0338833835894123 w_k1_ADM_s011
                                 + 0.0557382758808427 w_k1_ADM_s012
                                 + 0.0351296898000057 w_k1_ADM_s013
                                 + 0.0317003678999145 w_k1_ADM_s014
                                 + 0.0399294800087777 w_k1_ADM_s015
                                 + 0.0216773962552179 w_k1_ADM_s016
                                 + 0.0273095020906453 w_k1_ADM_s017
                                 + 0.0113165866940159 w_k1_ADM_s100
                                 + 0.014202856114603 w_k1_ADM_s101
                                 + 0.0268776025202232 w_k1_ADM_s102
                                 + 0.0113918456250945 w_k1_ADM_s103
                                 + 0.00588133081604299 w_k1_ADM_s104
                                 + 0.0155187860172426 w_k1_ADM_s105
                                 + 0.00851891095576351 w_k1_ADM_s106
                                 + 0.0193262605762458 w_k1_ADM_s107
                                 + 0.025589136718485 w_k1_ADM_s110
                                 + 0.0279312767308805 w_k1_ADM_s111
                                 + 0.0163657827458553 w_k1_ADM_s112
                                 + 0.0234181211119127 w_k1_ADM_s113
                                 + 0.0536722821626315 w_k1_ADM_s114
                                 + 0.0191785092155331 w_k1_ADM_s115
                                 + 0.0404699664483164 w_k1_ADM_s116
                                 + 0.0322555213375623 w_k1_ADM_s117
                                 + 0.0457136634509028 w_k1_AXP_s000
                                 + 0.0258811353544554 w_k1_AXP_s001
                                 + 0.0243382848848305 w_k1_AXP_s002
                                 + 0.0741860944877882 w_k1_AXP_s003
                                 + 0.0832908055141399 w_k1_AXP_s004
                                 + 0.0952196707150196 w_k1_AXP_s005
                                 + 0.0258466261309456 w_k1_AXP_s006
                                 + 0.0296460788196626 w_k1_AXP_s007
                                 + 0.0275969157891627 w_k1_AXP_s010
                                 + 0.0304726303800914 w_k1_AXP_s011
                                 + 0.0597909354163861 w_k1_AXP_s012
                                 + 0.0343437019009726 w_k1_AXP_s013
                                 + 0.030427662721601 w_k1_AXP_s014
                                 + 0.040764531402678 w_k1_AXP_s015
                                 + 0.0218506224151526 w_k1_AXP_s016
                                 + 0.0250482874116028 w_k1_AXP_s017
                                 + 0.0109222122639419 w_k1_AXP_s100
                                 + 0.015328961789665 w_k1_AXP_s101
                                 + 0.0283091775560141 w_k1_AXP_s102
                                 + 0.0116465096830037 w_k1_AXP_s103
                                 + 0.00550964032199413 w_k1_AXP_s104
                                 + 0.0181968011781186 w_k1_AXP_s105
                                 + 0.00896401394609543 w_k1_AXP_s106
                                 + 0.0179117118780411 w_k1_AXP_s107
                                 + 0.0268706205316812 w_k1_AXP_s110
                                 + 0.0251234065140735 w_k1_AXP_s111
                                 + 0.0167619869382362 w_k1_AXP_s112
                                 + 0.0221787411247062 w_k1_AXP_s113
                                 + 0.0636890982805115 w_k1_AXP_s114
                                 + 0.0181871747787793 w_k1_AXP_s115
                                 + 0.038819477074586 w_k1_AXP_s116
                                 + 0.0309152013165354 w_k1_AXP_s117
                                 + 0.0447071782163261 w_k1_DIS_s000
                                 + 0.0286974579723858 w_k1_DIS_s001
                                 + 0.023748907139124 w_k1_DIS_s002
                                 + 0.0698464392412022 w_k1_DIS_s003
                                 + 0.0864365073152749 w_k1_DIS_s004
                                 + 0.0898267737770771 w_k1_DIS_s005
                                 + 0.0231214103160032 w_k1_DIS_s006
                                 + 0.0273726919658347 w_k1_DIS_s007
                                 + 0.0261508172969592 w_k1_DIS_s010
                                 + 0.032214053504298 w_k1_DIS_s011
                                 + 0.056356939092293 w_k1_DIS_s012
                                 + 0.036267929406576 w_k1_DIS_s013
                                 + 0.0301345894853617 w_k1_DIS_s014
                                 + 0.0445518000996219 w_k1_DIS_s015
                                 + 0.0204378161423382 w_k1_DIS_s016
                                 + 0.0258752462064416 w_k1_DIS_s017
                                 + 0.0104547487637977 w_k1_DIS_s100
                                 + 0.0154711400103529 w_k1_DIS_s101
                                 + 0.0313003245070907 w_k1_DIS_s102
                                 + 0.0102514212213712 w_k1_DIS_s103
                                 + 0.00544012966455959 w_k1_DIS_s104
                                 + 0.0158013131070109 w_k1_DIS_s105
                                 + 0.00855552491428449 w_k1_DIS_s106
                                 + 0.0172129391879325 w_k1_DIS_s107
                                 + 0.0253139320127664 w_k1_DIS_s110
                                 + 0.0261383650673499 w_k1_DIS_s111
                                 + 0.0168699821315388 w_k1_DIS_s112
                                 + 0.0242107508393205 w_k1_DIS_s113
                                 + 0.0623277776685357 w_k1_DIS_s114
                                 + 0.0180371631490421 w_k1_DIS_s115
                                 + 0.0393070917728975 w_k1_DIS_s116
                                 + 0.0333539437804533 w_k1_DIS_s117
                                 + 0.0466967387712894 w_k1_FDX_s000
                                 + 0.028290948764298 w_k1_FDX_s001
                                 + 0.0241437802552902 w_k1_FDX_s002
                                 + 0.0701554536063646 w_k1_FDX_s003
                                 + 0.0855577429608787 w_k1_FDX_s004
                                 + 0.0924221792816566 w_k1_FDX_s005
                                 + 0.0239985951263606 w_k1_FDX_s006
                                 + 0.0271831101711785 w_k1_FDX_s007
                                 + 0.0242960299528123 w_k1_FDX_s010
                                 + 0.0292845721183365 w_k1_FDX_s011
                                 + 0.0574647311578242 w_k1_FDX_s012
                                 + 0.0338697436061571 w_k1_FDX_s013
                                 + 0.0292686636209119 w_k1_FDX_s014
                                 + 0.0421983520782759 w_k1_FDX_s015
                                 + 0.0207903661720445 w_k1_FDX_s016
                                 + 0.0258792212297428 w_k1_FDX_s017
                                 + 0.0103578692550427 w_k1_FDX_s100
                                 + 0.014850910446128 w_k1_FDX_s101
                                 + 0.0278194960706276 w_k1_FDX_s102
                                 + 0.0104587656990535 w_k1_FDX_s103
                                 + 0.00593801954978212 w_k1_FDX_s104
                                 + 0.0158785816870836 w_k1_FDX_s105
                                 + 0.00911639197610535 w_k1_FDX_s106
                                 + 0.0176718406015966 w_k1_FDX_s107
                                 + 0.0256094556212445 w_k1_FDX_s110
                                 + 0.0261896569204706 w_k1_FDX_s111
                                 + 0.0173389421611562 w_k1_FDX_s112
                                 + 0.0216601153248973 w_k1_FDX_s113
                                 + 0.0597586876282948 w_k1_FDX_s114
                                 + 0.0179223776117749 w_k1_FDX_s115
                                 + 0.0386759588312291 w_k1_FDX_s116
                                 + 0.0320293594297678 w_k1_FDX_s117
                                 + 0.0482947155470458 w_k1_IBM_s000
                                 + 0.0290907830434994 w_k1_IBM_s001
                                 + 0.0239658310235294 w_k1_IBM_s002
                                 + 0.0703646510362978 w_k1_IBM_s003
                                 + 0.0885706558026861 w_k1_IBM_s004
                                 + 0.0955620636240277 w_k1_IBM_s005
                                 + 0.0236625142184749 w_k1_IBM_s006
                                 + 0.0272392007761109 w_k1_IBM_s007
                                 + 0.0250809424014895 w_k1_IBM_s010
                                 + 0.0283329851797271 w_k1_IBM_s011
                                 + 0.0580910505892374 w_k1_IBM_s012
                                 + 0.0368384584394213 w_k1_IBM_s013
                                 + 0.0293617999287874 w_k1_IBM_s014
                                 + 0.0413268944546022 w_k1_IBM_s015
                                 + 0.0201505793449333 w_k1_IBM_s016
                                 + 0.0252534452349466 w_k1_IBM_s017
                                 + 0.0101516921176353 w_k1_IBM_s100
                                 + 0.0157579743058174 w_k1_IBM_s101
                                 + 0.0292575342866887 w_k1_IBM_s102
                                 + 0.0109507700144295 w_k1_IBM_s103
                                 + 0.00593208841725909 w_k1_IBM_s104
                                 + 0.0155693348493846 w_k1_IBM_s105
                                 + 0.00881546147734111 w_k1_IBM_s106
                                 + 0.0183010590698235 w_k1_IBM_s107
                                 + 0.0255203891335372 w_k1_IBM_s110
                                 + 0.0272444318490987 w_k1_IBM_s111
                                 + 0.0172591195983017 w_k1_IBM_s112
                                 + 0.0228595396694143 w_k1_IBM_s113
                                 + 0.06089379117056 w_k1_IBM_s114
                                 + 0.0176740186514883 w_k1_IBM_s115
                                 + 0.0389359412379533 w_k1_IBM_s116
                                 + 0.0347543302978192 w_k1_IBM_s117
                                 + 0.0459827739323227 w_k1_INTC_s000
                                 + 0.0282927172646883 w_k1_INTC_s001
                                 + 0.0222811747308837 w_k1_INTC_s002
                                 + 0.0670214115530647 w_k1_INTC_s003
                                 + 0.0822427398000763 w_k1_INTC_s004
                                 + 0.0882622318800084 w_k1_INTC_s005
                                 + 0.024655415254841 w_k1_INTC_s006
                                 + 0.0235286845347567 w_k1_INTC_s007
                                 + 0.0287912411319653 w_k1_INTC_s010
                                 + 0.0259605878074877 w_k1_INTC_s011
                                 + 0.0639483265202005 w_k1_INTC_s012
                                 + 0.0343739082223062 w_k1_INTC_s013
                                 + 0.0314255249808435 w_k1_INTC_s014
                                 + 0.0431382184673172 w_k1_INTC_s015
                                 + 0.0204913276774367 w_k1_INTC_s016
                                 + 0.0228809006567402 w_k1_INTC_s017
                                 + 0.00936778770844431 w_k1_INTC_s100
                                 + 0.0148910822289651 w_k1_INTC_s101
                                 + 0.0265198761591312 w_k1_INTC_s102
                                 + 0.010698811897362 w_k1_INTC_s103
                                 + 0.00546407990531824 w_k1_INTC_s104
                                 + 0.0190291646130644 w_k1_INTC_s105
                                 + 0.00914660016803254 w_k1_INTC_s106
                                 + 0.0177225471072957 w_k1_INTC_s107
                                 + 0.0271633396596397 w_k1_INTC_s110
                                 + 0.0260645975463744 w_k1_INTC_s111
                                 + 0.0153804610174929 w_k1_INTC_s112
                                 + 0.0229890747919162 w_k1_INTC_s113
                                 + 0.0489871969982994 w_k1_INTC_s114
                                 + 0.0178232896752858 w_k1_INTC_s115
                                 + 0.0371042204467662 w_k1_INTC_s116
                                 + 0.0351018450271528 w_k1_INTC_s117
                                 + 0.0450574053357943 w_k1_KO_s000
                                 + 0.0280070089757013 w_k1_KO_s001
                                 + 0.0229374544885786 w_k1_KO_s002
                                 + 0.0724163483765716 w_k1_KO_s003
                                 + 0.0807441187631378 w_k1_KO_s004
                                 + 0.0979007402307487 w_k1_KO_s005
                                 + 0.0223949660756593 w_k1_KO_s006
                                 + 0.0246466001902114 w_k1_KO_s007
                                 + 0.0240810788556176 w_k1_KO_s010
                                 + 0.0275321657475968 w_k1_KO_s011
                                 + 0.0630621018569208 w_k1_KO_s012
                                 + 0.0365497263033351 w_k1_KO_s013
                                 + 0.0289054719706009 w_k1_KO_s014
                                 + 0.0423815845281422 w_k1_KO_s015
                                 + 0.0208269950804732 w_k1_KO_s016
                                 + 0.0251553509296904 w_k1_KO_s017
                                 + 0.00976492779913352 w_k1_KO_s100
                                 + 0.0155180604798164 w_k1_KO_s101
                                 + 0.0265690328249826 w_k1_KO_s102
                                 + 0.0114468417560762 w_k1_KO_s103
                                 + 0.00528662306699044 w_k1_KO_s104
                                 + 0.0170563257931788 w_k1_KO_s105
                                 + 0.00928788407599576 w_k1_KO_s106
                                 + 0.0176288920938992 w_k1_KO_s107
                                 + 0.0237062436815805 w_k1_KO_s110
                                 + 0.0296365900201609 w_k1_KO_s111
                                 + 0.0184567746922344 w_k1_KO_s112
                                 + 0.0231485576924631 w_k1_KO_s113
                                 + 0.0593522956703108 w_k1_KO_s114
                                 + 0.0159245088651384 w_k1_KO_s115
                                 + 0.040747150121211 w_k1_KO_s116
                                 + 0.0345201163656642 w_k1_KO_s117
                                 + 0.042291491906957 w_k1_MAT_s000
                                 + 0.0295317947387726 w_k1_MAT_s001
                                 + 0.0252875187726012 w_k1_MAT_s002
                                 + 0.0795634640887871 w_k1_MAT_s003
                                 + 0.0948415009257846 w_k1_MAT_s004
                                 + 0.100489263592795 w_k1_MAT_s005
                                 + 0.0260279900524685 w_k1_MAT_s006
                                 + 0.0309423485203618 w_k1_MAT_s007
                                 + 0.0236824920473536 w_k1_MAT_s010
                                 + 0.0266590577223876 w_k1_MAT_s011
                                 + 0.0565630908062673 w_k1_MAT_s012
                                 + 0.0459529303612636 w_k1_MAT_s013
                                 + 0.0286973286728002 w_k1_MAT_s014
                                 + 0.0485310005531221 w_k1_MAT_s015
                                 + 0.0170025585193313 w_k1_MAT_s016
                                 + 0.0234999143932393 w_k1_MAT_s017
                                 + 0.0092074832510775 w_k1_MAT_s100
                                 + 0.0175947488760178 w_k1_MAT_s101
                                 + 0.028163024978869 w_k1_MAT_s102
                                 + 0.0136746358945162 w_k1_MAT_s103
                                 + 0.00506284616617498 w_k1_MAT_s104
                                 + 0.018836592907316 w_k1_MAT_s105
                                 + 0.00884115660575764 w_k1_MAT_s106
                                 + 0.0141735003225064 w_k1_MAT_s107
                                 + 0.0262345139616805 w_k1_MAT_s110
                                 + 0.0269123796398076 w_k1_MAT_s111
                                 + 0.0216927323660732 w_k1_MAT_s112
                                 + 0.0221105814118035 w_k1_MAT_s113
                                 + 0.0706677570114707 w_k1_MAT_s114
                                 + 0.0168842072409218 w_k1_MAT_s115
                                 + 0.0402495967439553 w_k1_MAT_s116
                                 + 0.0279501859350932 w_k1_MAT_s117
                                 + 0.0461886405570334 w_k1_MSFT_s000
                                 + 0.0293243062263435 w_k1_MSFT_s001
                                 + 0.0231686273286499 w_k1_MSFT_s002
                                 + 0.0646411320492458 w_k1_MSFT_s003
                                 + 0.0778262818178629 w_k1_MSFT_s004
                                 + 0.0887632781480608 w_k1_MSFT_s005
                                 + 0.0244700525652968 w_k1_MSFT_s006
                                 + 0.0260425670898655 w_k1_MSFT_s007
                                 + 0.0260598638023979 w_k1_MSFT_s010
                                 + 0.0276669738602565 w_k1_MSFT_s011
                                 + 0.056196505639842 w_k1_MSFT_s012
                                 + 0.0334061499015971 w_k1_MSFT_s013
                                 + 0.0302876131862143 w_k1_MSFT_s014
                                 + 0.0426115596730757 w_k1_MSFT_s015
                                 + 0.0197119524409905 w_k1_MSFT_s016
                                 + 0.0254439906861015 w_k1_MSFT_s017
                                 + 0.0110101560283823 w_k1_MSFT_s100
                                 + 0.0141314505091063 w_k1_MSFT_s101
                                 + 0.0272511779881777 w_k1_MSFT_s102
                                 + 0.0118424226760579 w_k1_MSFT_s103
                                 + 0.00549980489210745 w_k1_MSFT_s104
                                 + 0.0165902344856584 w_k1_MSFT_s105
                                 + 0.0085773198107877 w_k1_MSFT_s106
                                 + 0.0175200280519647 w_k1_MSFT_s107
                                 + 0.0232187659674143 w_k1_MSFT_s110
                                 + 0.0305342213451558 w_k1_MSFT_s111
                                 + 0.0156438024421883 w_k1_MSFT_s112
                                 + 0.0249570812611801 w_k1_MSFT_s113
                                 + 0.0583171222293542 w_k1_MSFT_s114
                                 + 0.0191330651749582 w_k1_MSFT_s115
                                 + 0.0365132043431873 w_k1_MSFT_s116
                                 + 0.0319077532916651 w_k1_MSFT_s117
                                 + 0.0492016188936809 w_k1_PG_s000
                                 + 0.0252863711152829 w_k1_PG_s001
                                 + 0.0248838317511596 w_k1_PG_s002
                                 + 0.0646720767261514 w_k1_PG_s003
                                 + 0.0778017667577617 w_k1_PG_s004
                                 + 0.095210775367419 w_k1_PG_s005
                                 + 0.025485903697948 w_k1_PG_s006
                                 + 0.0277707170896188 w_k1_PG_s007
                                 + 0.0254249361997246 w_k1_PG_s010
                                 + 0.02694720620951 w_k1_PG_s011
                                 + 0.0569269852004582 w_k1_PG_s012
                                 + 0.03501206332748 w_k1_PG_s013
                                 + 0.0306496406264577 w_k1_PG_s014
                                 + 0.0403896841743122 w_k1_PG_s015
                                 + 0.0217507509874037 w_k1_PG_s016
                                 + 0.0265868936178536 w_k1_PG_s017
                                 + 0.0100575703281764 w_k1_PG_s100
                                 + 0.0162952646370604 w_k1_PG_s101
                                 + 0.0277108648911061 w_k1_PG_s102
                                 + 0.0108652025539822 w_k1_PG_s103
                                 + 0.00541690708835466 w_k1_PG_s104
                                 + 0.0155945405834383 w_k1_PG_s105
                                 + 0.00929014786738016 w_k1_PG_s106
                                 + 0.0184156176588453 w_k1_PG_s107
                                 + 0.0240083849892997 w_k1_PG_s110
                                 + 0.0270617768675568 w_k1_PG_s111
                                 + 0.0170529695249527 w_k1_PG_s112
                                 + 0.0224043087756644 w_k1_PG_s113
                                 + 0.0579842310110319 w_k1_PG_s114
                                 + 0.0170777517745468 w_k1_PG_s115
                                 + 0.0414495256097724 w_k1_PG_s116
                                 + 0.031734976950838 w_k1_PG_s117 >= 1.1
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
