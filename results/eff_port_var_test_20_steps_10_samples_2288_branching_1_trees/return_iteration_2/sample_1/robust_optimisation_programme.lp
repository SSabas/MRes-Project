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
 w_k1_ADM_s0_return_constraint:  1.14823 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.944599 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.925036 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.27605 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.00941 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.881346 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.83546 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.705115 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.955191 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.20417 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.11709 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.05577 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.26875 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.86228 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.08333 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.806492 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.870087 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.12436 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.65395 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.1016 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.10325 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.88037 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.935087 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.747336 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.17404 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.991627 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.44009 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.40667 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.03091 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.775846 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.760238 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.11893 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.989946 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.905066 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.777991 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.932034 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.28666 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.28837 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.24519 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.935331 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.820285 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.31475 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.22222 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.747977 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.700195 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.763825 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.30967 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.49985 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.18614 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.10551 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.01598 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.920991 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.48928 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.14139 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.8559 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.3883 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.674405 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.06087 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.06007 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.890675 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.777963 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.671 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.14343 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.884813 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.23595 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.57598 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.906866 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.921422 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.81034 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.40457 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.0401 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.854061 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.746587 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.1677 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.29146 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.42168 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.30438 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.926564 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.79468 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.35279 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.28068 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.74717 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.562728 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.750312 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.19322 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.80576 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.66692 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.16193 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.07293 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.888813 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.93826 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.09336 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.796767 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.17104 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.627524 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.38261 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.11128 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.952132 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.76153 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.69194 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.11129 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.850567 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.45767 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.73468 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.837828 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.779556 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.613657 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.28102 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.926506 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.684136 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.694041 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.10756 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.14468 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.63497 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.21597 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.91307 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.882157 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.19501 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.15046 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.847571 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.732178 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.758674 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.975853 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.43494 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.3261 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.05045 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.1384 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.957488 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.37557 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.968799 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.974514 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.20048 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.713836 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.31282 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.1096 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.896705 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.79743 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.750508 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.03747 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.9685 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.30703 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.39012 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.01976 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.744897 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.677263 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.15023 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.02559 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.890095 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.749329 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.03336 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.2286 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.43852 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.05224 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.02632 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.833086 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.25767 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.21382 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.780895 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.663602 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.0279 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.76034 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.7115 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.0193 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.22044 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.704212 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.860743 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.52205 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.61237 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.615067 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.12351 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.848843 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.804431 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.969501 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.1036 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.08206 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.80193 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.24866 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.831033 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.883542 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.65407 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.575547 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.62654 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.1145 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.80818 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.943458 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.601179 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.770911 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.28659 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.761975 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.07967 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.16648 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.929886 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.936756 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.13701 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.13888 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.846128 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.676471 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.885575 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.982366 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.4106 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.22625 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.06259 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.11003 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.921523 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.18592 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.00804 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.886957 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.08333 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.832781 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.3775 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.938304 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.983776 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.915703 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.79211 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.05731 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.911352 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.17577 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.28247 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.986635 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.980355 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.02819 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.05853 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.02359 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.901057 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.829562 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.03025 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.13182 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.3136 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.1465 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.981452 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.90343 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.19649 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.0863 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.783303 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.647269 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.84698 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.19155 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.493 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.24041 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.17774 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.809597 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.836267 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.6206 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.19685 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.79833 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.24319 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.783798 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.09654 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.06865 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.11648 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.906407 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.778356 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.35515 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.876869 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.21229 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.54684 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.799859 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.15945 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.87329 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.41074 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.992822 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.719488 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.787375 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.13046 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.943707 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.41662 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.11822 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.97046 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.781058 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.54108 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.29459 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.756468 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.660592 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.794014 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 2.01452 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.71868 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.901468 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.10467 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.655438 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.896337 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.69741 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.19785 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.621225 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.05971 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.787965 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.598573 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.09361 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.0426 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.920829 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.70738 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.51299 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.942989 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.0392 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.54777 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.594765 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.47045 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.14928 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.76506 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.01251 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.62599 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.733901 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.33769 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.806963 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.29776 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.16778 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.987758 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.915301 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.2193 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.07954 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.838418 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.694487 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.809006 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.918713 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.31797 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.33763 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.07218 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.08481 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.972376 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.32811 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.916429 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.912514 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.08118 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.745227 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.20378 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.00992 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.966809 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.816675 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.840878 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.03114 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.96296 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.24915 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.1914 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.950507 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.796421 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.730932 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.98158 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.934206 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.820205 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.735343 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.920072 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.11975 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.40822 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.12116 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.936377 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.923515 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.11262 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.18431 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.894911 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.742095 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.816703 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.00212 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.23695 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.2401 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.01694 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.962127 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.925668 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.33376 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.976325 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.915186 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.23899 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.847799 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.09714 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.04542 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.00526 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.799433 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.785894 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.04871 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.888045 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.08922 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.20196 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.853006 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.957721 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.851378 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.15286 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.04814 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.872943 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.870274 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.01811 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.04606 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.16413 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.101000460342253 z_k1_s0000
                                 - 0.149000695166375 z_k1_s0001
                                 - 0.489001885768182 z_k1_s0002
                                 - 0.204000998002519 z_k1_s0003
                                 - 0.210002040360659 z_k1_s0004
                                 - 0.29200075317666 z_k1_s0005
                                 - 0.188000987261487 z_k1_s0006
                                 - 0.393001416119938 z_k1_s0007
                                 - 0.270001906329853 z_k1_s0010
                                 - 0.105000511905126 z_k1_s0011
                                 - 0.164000691379951 z_k1_s0012
                                 - 0.510001306391559 z_k1_s0013
                                 - 0.315000503211206 z_k1_s0014
                                 - 0.42800188995534 z_k1_s0015
                                 - 0.143000588998926 z_k1_s0016
                                 - 0.130000535453569 z_k1_s0017
                                 - 0.162000110101736 z_k1_s0020
                                 - 0.17899957673863 z_k1_s0021
                                 - 0.343000914533711 z_k1_s0022
                                 - 0.118000127864256 z_k1_s0023
                                 - 0.144000117368221 z_k1_s0024
                                 - 0.46200051549396 z_k1_s0025
                                 - 0.152000114138672 z_k1_s0026
                                 - 0.194999921279741 z_k1_s0027
                                 - 0.136999992915514 z_k1_s0030
                                 - 0.0989999902265391 z_k1_s0031
                                 - 0.152000010755898 z_k1_s0032
                                 - 0.127000034155256 z_k1_s0033
                                 - 0.258000051814615 z_k1_s0034
                                 - 0.181999727636674 z_k1_s0035
                                 - 0.43499940157117 z_k1_s0036
                                 - 0.204000115109229 z_k1_s0037
                                 - 0.0920000722313347 z_k1_s0040
                                 - 0.210999534878331 z_k1_s0041
                                 - 0.222999263951958 z_k1_s0042
                                 - 0.193999381357311 z_k1_s0043
                                 - 0.181000212194263 z_k1_s0044
                                 - 0.130999997720695 z_k1_s0045
                                 - 0.166000550852228 z_k1_s0046
                                 - 0.414000325041006 z_k1_s0047
                                 - 0.364999806472561 z_k1_s0050
                                 - 0.401999858450329 z_k1_s0051
                                 - 0.963000380242968 z_k1_s0052
                                 - 0.408999914597476 z_k1_s0053
                                 - 0.296999873043153 z_k1_s0054
                                 - 0.998998832999682 z_k1_s0055
                                 - 0.501999130552387 z_k1_s0056
                                 - 0.347000151694196 z_k1_s0057
                                 - 0.220000479663454 z_k1_s0060
                                 - 0.947002219512589 z_k1_s0061
                                 - 0.224000571861164 z_k1_s0062
                                 - 0.368000830171661 z_k1_s0063
                                 - 0.447000259820921 z_k1_s0064
                                 - 0.979002957094264 z_k1_s0065
                                 - 0.328000673396327 z_k1_s0066
                                 - 0.313000805906017 z_k1_s0067
                                 - 0.243000888172997 z_k1_s0070
                                 - 0.313000879153504 z_k1_s0071
                                 - 0.70700391263846 z_k1_s0072
                                 - 0.365001292374918 z_k1_s0073
                                 - 0.207000719050928 z_k1_s0074
                                 - 0.162000507648341 z_k1_s0075
                                 - 0.246000902266503 z_k1_s0076
                                 - 0.291000606867148 z_k1_s0077
                                 - 0.144000958130973 z_k1_s0100
                                 - 0.0810002081978773 z_k1_s0101
                                 - 0.220000875809795 z_k1_s0102
                                 - 0.0790002242905733 z_k1_s0103
                                 - 0.238000334074577 z_k1_s0104
                                 - 0.303000803314341 z_k1_s0105
                                 - 0.154000216165903 z_k1_s0106
                                 - 0.10400022158235 z_k1_s0107
                                 - 0.115000179063926 z_k1_s0110
                                 - 0.247999652432438 z_k1_s0111
                                 - 0.243000175126086 z_k1_s0112
                                 - 0.14700021482952 z_k1_s0113
                                 - 0.0840001857598177 z_k1_s0114
                                 - 0.324999834962289 z_k1_s0115
                                 - 0.194000593516962 z_k1_s0116
                                 - 0.114000136602441 z_k1_s0117
                                 - 0.285999705832626 z_k1_s0120
                                 - 0.402999651127704 z_k1_s0121
                                 - 0.256999603378443 z_k1_s0122
                                 - 0.311998891454815 z_k1_s0123
                                 - 0.404998144485166 z_k1_s0124
                                 - 0.548999410077346 z_k1_s0125
                                 - 0.182999899625655 z_k1_s0126
                                 - 0.49299827612568 z_k1_s0127
                                 - 0.0699998926451642 z_k1_s0130
                                 - 0.0439999116284119 z_k1_s0131
                                 - 0.128999990183025 z_k1_s0132
                                 - 0.107000034368819 z_k1_s0133
                                 - 0.0719999052464374 z_k1_s0134
                                 - 0.195999772526379 z_k1_s0135
                                 - 0.105000204567344 z_k1_s0136
                                 - 0.191000198022694 z_k1_s0137
                                 - 0.107999965452905 z_k1_s0140
                                 - 0.101999993949959 z_k1_s0141
                                 - 0.266999748480841 z_k1_s0142
                                 - 0.270000511632094 z_k1_s0143
                                 - 0.157000071593541 z_k1_s0144
                                 - 0.149999885573498 z_k1_s0145
                                 - 0.0570000282779091 z_k1_s0146
                                 - 0.0849999351583158 z_k1_s0147
                                 - 0.231999873763909 z_k1_s0150
                                 - 0.165999486227673 z_k1_s0151
                                 - 0.0860000537745413 z_k1_s0152
                                 - 0.136999962507132 z_k1_s0153
                                 - 0.352999305080488 z_k1_s0154
                                 - 0.197000342721889 z_k1_s0155
                                 - 0.206000046704143 z_k1_s0156
                                 - 0.15800060213224 z_k1_s0157
                                 - 0.0869999195380809 z_k1_s0160
                                 - 0.107000349529022 z_k1_s0161
                                 - 0.0589999589499456 z_k1_s0162
                                 - 0.19599983101681 z_k1_s0163
                                 - 0.167000036003889 z_k1_s0164
                                 - 0.191000124393564 z_k1_s0165
                                 - 0.107000349529022 z_k1_s0166
                                 - 0.154999457309734 z_k1_s0167
                                 - 0.186999914018192 z_k1_s0170
                                 - 0.167000059434966 z_k1_s0171
                                 - 0.114000061654409 z_k1_s0172
                                 - 0.529999977805577 z_k1_s0173
                                 - 0.291000359030995 z_k1_s0174
                                 - 0.160000079265837 z_k1_s0175
                                 - 0.406000920809578 z_k1_s0176
                                 - 0.215999092218855 z_k1_s0177
                                 - 0.459998656541566 z_k1_s1000
                                 - 0.931996604864341 z_k1_s1001
                                 - 0.44199868667261 z_k1_s1002
                                 - 0.4339986427199 z_k1_s1003
                                 - 0.851996165337236 z_k1_s1004
                                 - 0.320998860548043 z_k1_s1005
                                 - 0.436998723714488 z_k1_s1006
                                 - 1.28299524316112 z_k1_s1007
                                 - 0.583000230114047 z_k1_s1010
                                 - 0.245000885781318 z_k1_s1011
                                 - 0.268000118110761 z_k1_s1012
                                 - 0.339000873566369 z_k1_s1013
                                 - 0.183000236211644 z_k1_s1014
                                 - 0.298000471063876 z_k1_s1015
                                 - 0.292000879673844 z_k1_s1016
                                 - 0.188000215170398 z_k1_s1017
                                 - 0.481999687035696 z_k1_s1020
                                 - 0.505999221228055 z_k1_s1021
                                 - 1.06599655543993 z_k1_s1022
                                 - 0.494999481331501 z_k1_s1023
                                 - 0.501999671728881 z_k1_s1024
                                 - 1.05399958484039 z_k1_s1025
                                 - 1.01399849685536 z_k1_s1026
                                 - 0.47399946943869 z_k1_s1027
                                 - 1.42099980548547 z_k1_s1030
                                 - 0.472001176341191 z_k1_s1031
                                 - 1.10399883170509 z_k1_s1032
                                 - 1.00200155812912 z_k1_s1033
                                 - 0.778001830087583 z_k1_s1034
                                 - 1.99300199322257 z_k1_s1035
                                 - 0.474000650527966 z_k1_s1036
                                 - 0.786001332838045 z_k1_s1037
                                 - 0.332000212595517 z_k1_s1040
                                 - 0.314000252906663 z_k1_s1041
                                 - 0.271000330083268 z_k1_s1042
                                 - 0.304000198812767 z_k1_s1043
                                 - 0.923000896881864 z_k1_s1044
                                 - 0.336000509593392 z_k1_s1045
                                 - 0.197000342828914 z_k1_s1046
                                 - 0.76500258928124 z_k1_s1047
                                 - 0.300000826533464 z_k1_s1050
                                 - 1.01000468762622 z_k1_s1051
                                 - 1.1950046118514 z_k1_s1052
                                 - 0.535002898338163 z_k1_s1053
                                 - 0.747004370239689 z_k1_s1054
                                 - 0.630001158988188 z_k1_s1055
                                 - 0.304001250004735 z_k1_s1056
                                 - 0.522001784207479 z_k1_s1057
                                 - 0.504001067496247 z_k1_s1060
                                 - 0.671001229862267 z_k1_s1061
                                 - 1.56500688077647 z_k1_s1062
                                 - 1.0960045479758 z_k1_s1063
                                 - 0.696001682904772 z_k1_s1064
                                 - 0.584001202149547 z_k1_s1065
                                 - 0.97000501170325 z_k1_s1066
                                 - 1.22000223611321 z_k1_s1067
                                 - 0.203999999844655 z_k1_s1070
                                 - 0.199000085326438 z_k1_s1071
                                 - 0.322001004975793 z_k1_s1072
                                 - 0.474000266330336 z_k1_s1073
                                 - 0.214000061381183 z_k1_s1074
                                 - 0.323000755379342 z_k1_s1075
                                 - 0.48400009536677 z_k1_s1076
                                 - 0.105000064883305 z_k1_s1077
                                 - 0.148999716179243 z_k1_s1100
                                 - 0.166999787685313 z_k1_s1101
                                 - 0.348999224459589 z_k1_s1102
                                 - 0.282000791862218 z_k1_s1103
                                 - 0.508000430795737 z_k1_s1104
                                 - 0.340999411612142 z_k1_s1105
                                 - 0.412000213629803 z_k1_s1106
                                 - 0.254998960505516 z_k1_s1107
                                 - 0.267000837394193 z_k1_s1110
                                 - 0.179999904825839 z_k1_s1111
                                 - 0.370000551063613 z_k1_s1112
                                 - 0.247000297769551 z_k1_s1113
                                 - 0.566999499771557 z_k1_s1114
                                 - 0.161999890763274 z_k1_s1115
                                 - 0.27399937566417 z_k1_s1116
                                 - 0.291000541744537 z_k1_s1117
                                 - 0.810998731369026 z_k1_s1120
                                 - 0.511000637832009 z_k1_s1121
                                 - 0.665001831536341 z_k1_s1122
                                 - 0.277000370278446 z_k1_s1123
                                 - 0.942003246216388 z_k1_s1124
                                 - 0.905000097485635 z_k1_s1125
                                 - 0.842001792900847 z_k1_s1126
                                 - 0.269000254013202 z_k1_s1127
                                 - 0.584001286234978 z_k1_s1130
                                 - 0.322000377240265 z_k1_s1131
                                 - 0.446001840789698 z_k1_s1132
                                 - 0.211000024024972 z_k1_s1133
                                 - 0.467999027597733 z_k1_s1134
                                 - 0.613000806393491 z_k1_s1135
                                 - 0.699999366869031 z_k1_s1136
                                 - 0.834000463259176 z_k1_s1137
                                 - 0.439999199752516 z_k1_s1140
                                 - 0.348000699075899 z_k1_s1141
                                 - 0.391000287870168 z_k1_s1142
                                 - 0.238999765119953 z_k1_s1143
                                 - 0.535000025970991 z_k1_s1144
                                 - 0.330000065813842 z_k1_s1145
                                 - 0.236999901957333 z_k1_s1146
                                 - 0.144000004500605 z_k1_s1147
                                 - 1.04499922506317 z_k1_s1150
                                 - 0.285999760101542 z_k1_s1151
                                 - 0.417999161626078 z_k1_s1152
                                 - 0.621999821907021 z_k1_s1153
                                 - 1.1919998795771 z_k1_s1154
                                 - 0.870997371966696 z_k1_s1155
                                 - 0.520998958833455 z_k1_s1156
                                 - 0.328999300944031 z_k1_s1157
                                 - 0.216998861926864 z_k1_s1160
                                 - 0.263998719411919 z_k1_s1161
                                 - 0.743998095625608 z_k1_s1162
                                 - 0.173999134307957 z_k1_s1163
                                 - 0.39599919141267 z_k1_s1164
                                 - 0.765996108602073 z_k1_s1165
                                 - 0.358997910701171 z_k1_s1166
                                 - 0.257998683511953 z_k1_s1167
                                 - 0.439002481532203 z_k1_s1170
                                 - 0.650000890733745 z_k1_s1171
                                 - 0.207000670711802 z_k1_s1172
                                 - 0.166000490118977 z_k1_s1173
                                 - 0.570001195798579 z_k1_s1174
                                 - 0.408001139677764 z_k1_s1175
                                 - 0.351000121595099 z_k1_s1176
                                 - 0.204000569838882 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.710096 w_k1_ADM_s000
                                 + 0.701488 w_k1_AXP_s000
                                 + 0.89217 w_k1_DIS_s000
                                 + 0.852756 w_k1_FDX_s000
                                 + 0.793993 w_k1_IBM_s000
                                 + 1.03217 w_k1_INTC_s000
                                 + 0.973282 w_k1_KO_s000
                                 + 0.678393 w_k1_MAT_s000
                                 + 0.988109 w_k1_MSFT_s000
                                 + 0.920159 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.836822 w_k1_ADM_s000
                                 + 0.954288 w_k1_AXP_s000
                                 + 0.954797 w_k1_DIS_s000
                                 + 0.809602 w_k1_FDX_s000
                                 + 1.68206 w_k1_IBM_s000
                                 + 0.818355 w_k1_INTC_s000
                                 + 1.27154 w_k1_KO_s000 + 1.31381 w_k1_MAT_s000
                                 + 0.878319 w_k1_MSFT_s000
                                 + 1.07707 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.32943 w_k1_ADM_s000
                                 + 1.43036 w_k1_AXP_s000
                                 + 1.36178 w_k1_DIS_s000 + 1.1983 w_k1_FDX_s000
                                 + 1.24111 w_k1_IBM_s000
                                 + 1.12972 w_k1_INTC_s000
                                 + 1.09793 w_k1_KO_s000 + 1.1816 w_k1_MAT_s000
                                 + 1.10331 w_k1_MSFT_s000
                                 + 1.15863 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.876812 w_k1_ADM_s000
                                 + 0.771795 w_k1_AXP_s000
                                 + 0.709313 w_k1_DIS_s000
                                 + 0.830781 w_k1_FDX_s000
                                 + 0.579735 w_k1_IBM_s000
                                 + 1.08654 w_k1_INTC_s000
                                 + 0.62551 w_k1_KO_s000
                                 + 0.571375 w_k1_MAT_s000
                                 + 0.91544 w_k1_MSFT_s000
                                 + 0.832499 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.718318 w_k1_ADM_s000
                                 + 0.603072 w_k1_AXP_s000
                                 + 0.694104 w_k1_DIS_s000
                                 + 0.660366 w_k1_FDX_s000
                                 + 0.880311 w_k1_IBM_s000
                                 + 0.785425 w_k1_INTC_s000
                                 + 0.791313 w_k1_KO_s000
                                 + 0.739498 w_k1_MAT_s000
                                 + 0.777257 w_k1_MSFT_s000
                                 + 0.798486 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.933452 w_k1_ADM_s000
                                 + 0.912005 w_k1_AXP_s000
                                 + 1.38778 w_k1_DIS_s000
                                 + 1.26559 w_k1_FDX_s000
                                 + 0.824719 w_k1_IBM_s000
                                 + 1.24216 w_k1_INTC_s000
                                 + 1.02951 w_k1_KO_s000
                                 + 0.839528 w_k1_MAT_s000
                                 + 1.36068 w_k1_MSFT_s000
                                 + 1.06593 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.736032 w_k1_ADM_s000
                                 + 0.804746 w_k1_AXP_s000
                                 + 0.956055 w_k1_DIS_s000
                                 + 0.876738 w_k1_FDX_s000
                                 + 1.13175 w_k1_IBM_s000
                                 + 1.00032 w_k1_INTC_s000
                                 + 0.894365 w_k1_KO_s000
                                 + 0.964231 w_k1_MAT_s000
                                 + 0.899592 w_k1_MSFT_s000
                                 + 0.967598 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.908205 w_k1_ADM_s000
                                 + 0.840938 w_k1_AXP_s000
                                 + 1.03357 w_k1_DIS_s000
                                 + 0.965233 w_k1_FDX_s000
                                 + 0.861459 w_k1_IBM_s000
                                 + 1.07409 w_k1_INTC_s000
                                 + 0.852228 w_k1_KO_s000
                                 + 0.696759 w_k1_MAT_s000
                                 + 1.04509 w_k1_MSFT_s000
                                 + 0.988841 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.09985 w_k1_ADM_s001
                                 + 1.03837 w_k1_AXP_s001
                                 + 0.903597 w_k1_DIS_s001
                                 + 0.898601 w_k1_FDX_s001
                                 + 1.31593 w_k1_IBM_s001
                                 + 0.843106 w_k1_INTC_s001
                                 + 0.975813 w_k1_KO_s001
                                 + 1.82888 w_k1_MAT_s001
                                 + 0.836971 w_k1_MSFT_s001
                                 + 1.04797 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.985946 w_k1_ADM_s001
                                 + 0.655636 w_k1_AXP_s001
                                 + 0.606701 w_k1_DIS_s001
                                 + 0.660416 w_k1_FDX_s001
                                 + 1.10404 w_k1_IBM_s001
                                 + 0.577831 w_k1_INTC_s001
                                 + 0.812573 w_k1_KO_s001
                                 + 1.10192 w_k1_MAT_s001
                                 + 0.650789 w_k1_MSFT_s001
                                 + 0.839899 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.59564 w_k1_ADM_s001
                                 + 1.47605 w_k1_AXP_s001
                                 + 1.67035 w_k1_DIS_s001
                                 + 1.44916 w_k1_FDX_s001
                                 + 1.29516 w_k1_IBM_s001
                                 + 1.12117 w_k1_INTC_s001
                                 + 1.38377 w_k1_KO_s001 + 1.37623 w_k1_MAT_s001
                                 + 1.32418 w_k1_MSFT_s001
                                 + 1.19936 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.40996 w_k1_ADM_s001
                                 + 1.21649 w_k1_AXP_s001
                                 + 1.26768 w_k1_DIS_s001
                                 + 1.16015 w_k1_FDX_s001
                                 + 1.09804 w_k1_IBM_s001
                                 + 0.871315 w_k1_INTC_s001
                                 + 0.923027 w_k1_KO_s001
                                 + 1.34644 w_k1_MAT_s001
                                 + 0.961963 w_k1_MSFT_s001
                                 + 1.12388 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.906521 w_k1_ADM_s001
                                 + 0.898467 w_k1_AXP_s001
                                 + 0.862743 w_k1_DIS_s001
                                 + 0.856449 w_k1_FDX_s001
                                 + 0.938144 w_k1_IBM_s001
                                 + 0.899791 w_k1_INTC_s001
                                 + 0.932111 w_k1_KO_s001
                                 + 1.02658 w_k1_MAT_s001
                                 + 0.910717 w_k1_MSFT_s001
                                 + 0.945182 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.952851 w_k1_ADM_s001
                                 + 0.90034 w_k1_AXP_s001
                                 + 0.723045 w_k1_DIS_s001
                                 + 0.910338 w_k1_FDX_s001
                                 + 0.69439 w_k1_IBM_s001
                                 + 0.792354 w_k1_INTC_s001
                                 + 0.724166 w_k1_KO_s001
                                 + 0.799272 w_k1_MAT_s001
                                 + 0.839446 w_k1_MSFT_s001
                                 + 0.903733 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.14393 w_k1_ADM_s001
                                 + 0.975311 w_k1_AXP_s001
                                 + 1.20461 w_k1_DIS_s001
                                 + 0.927671 w_k1_FDX_s001
                                 + 1.43912 w_k1_IBM_s001
                                 + 0.950625 w_k1_INTC_s001
                                 + 1.33314 w_k1_KO_s001 + 1.25522 w_k1_MAT_s001
                                 + 1.01147 w_k1_MSFT_s001
                                 + 1.15856 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.39208 w_k1_ADM_s001
                                 + 1.15715 w_k1_AXP_s001
                                 + 1.19111 w_k1_DIS_s001
                                 + 1.24665 w_k1_FDX_s001
                                 + 0.707364 w_k1_IBM_s001
                                 + 1.03851 w_k1_INTC_s001
                                 + 0.805766 w_k1_KO_s001
                                 + 0.814349 w_k1_MAT_s001
                                 + 1.2812 w_k1_MSFT_s001 + 1.04495 w_k1_PG_s001
                                 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.47165 w_k1_ADM_s002
                                 + 1.36414 w_k1_AXP_s002
                                 + 1.21968 w_k1_DIS_s002
                                 + 1.32029 w_k1_FDX_s002
                                 + 0.956059 w_k1_IBM_s002
                                 + 1.29657 w_k1_INTC_s002
                                 + 1.09395 w_k1_KO_s002
                                 + 0.901208 w_k1_MAT_s002
                                 + 1.26502 w_k1_MSFT_s002
                                 + 1.18994 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.2555 w_k1_ADM_s002
                                 + 1.37151 w_k1_AXP_s002
                                 + 1.36081 w_k1_DIS_s002
                                 + 1.33249 w_k1_FDX_s002
                                 + 1.23775 w_k1_IBM_s002
                                 + 1.52913 w_k1_INTC_s002 + 1.2936 w_k1_KO_s002
                                 + 1.39696 w_k1_MAT_s002
                                 + 1.36201 w_k1_MSFT_s002
                                 + 1.36789 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.816678 w_k1_ADM_s002
                                 + 0.966709 w_k1_AXP_s002
                                 + 0.756793 w_k1_DIS_s002
                                 + 0.883178 w_k1_FDX_s002
                                 + 1.10043 w_k1_IBM_s002
                                 + 0.981023 w_k1_INTC_s002
                                 + 0.882906 w_k1_KO_s002
                                 + 1.05142 w_k1_MAT_s002
                                 + 0.966688 w_k1_MSFT_s002
                                 + 0.988677 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.13069 w_k1_ADM_s002
                                 + 1.05951 w_k1_AXP_s002
                                 + 1.35117 w_k1_DIS_s002
                                 + 1.22798 w_k1_FDX_s002
                                 + 0.822576 w_k1_IBM_s002
                                 + 1.27204 w_k1_INTC_s002
                                 + 1.18928 w_k1_KO_s002
                                 + 0.760702 w_k1_MAT_s002
                                 + 1.286 w_k1_MSFT_s002 + 1.14506 w_k1_PG_s002
                                 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.830067 w_k1_ADM_s002
                                 + 0.880515 w_k1_AXP_s002
                                 + 1.02093 w_k1_DIS_s002
                                 + 0.807921 w_k1_FDX_s002
                                 + 1.43432 w_k1_IBM_s002
                                 + 0.849103 w_k1_INTC_s002
                                 + 1.09662 w_k1_KO_s002 + 1.20984 w_k1_MAT_s002
                                 + 0.937445 w_k1_MSFT_s002
                                 + 1.00152 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.820758 w_k1_ADM_s002
                                 + 0.709705 w_k1_AXP_s002
                                 + 0.807233 w_k1_DIS_s002
                                 + 0.897656 w_k1_FDX_s002
                                 + 0.811644 w_k1_IBM_s002
                                 + 0.862716 w_k1_INTC_s002
                                 + 0.913432 w_k1_KO_s002
                                 + 0.545167 w_k1_MAT_s002
                                 + 0.961727 w_k1_MSFT_s002
                                 + 0.999287 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.01726 w_k1_ADM_s002
                                 + 1.31148 w_k1_AXP_s002
                                 + 1.42545 w_k1_DIS_s002
                                 + 1.16467 w_k1_FDX_s002
                                 + 1.54473 w_k1_IBM_s002
                                 + 1.07795 w_k1_INTC_s002
                                 + 1.31947 w_k1_KO_s002 + 1.67185 w_k1_MAT_s002
                                 + 1.29143 w_k1_MSFT_s002
                                 + 1.16535 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.10539 w_k1_ADM_s002
                                 + 0.794454 w_k1_AXP_s002
                                 + 0.768505 w_k1_DIS_s002
                                 + 0.937401 w_k1_FDX_s002
                                 + 0.657817 w_k1_IBM_s002
                                 + 1.08683 w_k1_INTC_s002
                                 + 0.756341 w_k1_KO_s002
                                 + 0.480554 w_k1_MAT_s002
                                 + 1.08262 w_k1_MSFT_s002
                                 + 0.915152 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.722158 w_k1_ADM_s003
                                 + 1.00277 w_k1_AXP_s003
                                 + 1.34556 w_k1_DIS_s003
                                 + 1.01864 w_k1_FDX_s003
                                 + 1.00636 w_k1_IBM_s003
                                 + 1.02886 w_k1_INTC_s003 + 1.1918 w_k1_KO_s003
                                 + 1.24465 w_k1_MAT_s003
                                 + 1.18697 w_k1_MSFT_s003
                                 + 1.11518 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.25568 w_k1_ADM_s003
                                 + 1.36947 w_k1_AXP_s003
                                 + 1.17115 w_k1_DIS_s003
                                 + 1.02775 w_k1_FDX_s003
                                 + 1.68838 w_k1_IBM_s003
                                 + 0.87516 w_k1_INTC_s003
                                 + 1.26482 w_k1_KO_s003 + 1.71804 w_k1_MAT_s003
                                 + 1.04591 w_k1_MSFT_s003
                                 + 1.09501 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.40866 w_k1_ADM_s003
                                 + 1.33391 w_k1_AXP_s003
                                 + 1.38829 w_k1_DIS_s003
                                 + 1.37181 w_k1_FDX_s003
                                 + 0.714924 w_k1_IBM_s003
                                 + 1.30085 w_k1_INTC_s003
                                 + 0.985101 w_k1_KO_s003
                                 + 0.888666 w_k1_MAT_s003
                                 + 1.311 w_k1_MSFT_s003 + 1.15772 w_k1_PG_s003
                                 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.54673 w_k1_ADM_s003
                                 + 1.51106 w_k1_AXP_s003
                                 + 1.61573 w_k1_DIS_s003
                                 + 1.32389 w_k1_FDX_s003
                                 + 1.26968 w_k1_IBM_s003
                                 + 1.13966 w_k1_INTC_s003 + 1.4099 w_k1_KO_s003
                                 + 1.59877 w_k1_MAT_s003
                                 + 1.36917 w_k1_MSFT_s003
                                 + 1.44193 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.892953 w_k1_ADM_s003
                                 + 1.28039 w_k1_AXP_s003
                                 + 0.916152 w_k1_DIS_s003
                                 + 0.969867 w_k1_FDX_s003
                                 + 1.24286 w_k1_IBM_s003
                                 + 0.933007 w_k1_INTC_s003
                                 + 1.0481 w_k1_KO_s003 + 1.45315 w_k1_MAT_s003
                                 + 0.947194 w_k1_MSFT_s003
                                 + 1.05977 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.988394 w_k1_ADM_s003
                                 + 0.730395 w_k1_AXP_s003
                                 + 0.869799 w_k1_DIS_s003
                                 + 0.730842 w_k1_FDX_s003
                                 + 1.01267 w_k1_IBM_s003
                                 + 0.665458 w_k1_INTC_s003
                                 + 0.953934 w_k1_KO_s003
                                 + 0.80875 w_k1_MAT_s003
                                 + 0.773961 w_k1_MSFT_s003
                                 + 0.90395 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.848103 w_k1_ADM_s003
                                 + 0.885181 w_k1_AXP_s003
                                 + 0.728103 w_k1_DIS_s003
                                 + 0.834159 w_k1_FDX_s003
                                 + 0.760795 w_k1_IBM_s003
                                 + 0.820581 w_k1_INTC_s003
                                 + 0.726864 w_k1_KO_s003
                                 + 0.791648 w_k1_MAT_s003
                                 + 0.818365 w_k1_MSFT_s003
                                 + 0.811848 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.11325 w_k1_ADM_s003
                                 + 1.15525 w_k1_AXP_s003
                                 + 0.877416 w_k1_DIS_s003
                                 + 1.09112 w_k1_FDX_s003
                                 + 0.665164 w_k1_IBM_s003
                                 + 0.938604 w_k1_INTC_s003
                                 + 0.848992 w_k1_KO_s003
                                 + 1.00349 w_k1_MAT_s003
                                 + 1.01695 w_k1_MSFT_s003
                                 + 1.01281 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.1827 w_k1_ADM_s004
                                 + 0.994547 w_k1_AXP_s004
                                 + 1.57624 w_k1_DIS_s004
                                 + 1.19487 w_k1_FDX_s004
                                 + 0.879039 w_k1_IBM_s004
                                 + 1.1308 w_k1_INTC_s004 + 1.02042 w_k1_KO_s004
                                 + 1.16486 w_k1_MAT_s004
                                 + 1.39152 w_k1_MSFT_s004
                                 + 0.989648 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.0614 w_k1_ADM_s004
                                 + 0.954362 w_k1_AXP_s004
                                 + 0.793103 w_k1_DIS_s004
                                 + 0.949459 w_k1_FDX_s004
                                 + 0.747755 w_k1_IBM_s004
                                 + 0.813775 w_k1_INTC_s004
                                 + 0.794248 w_k1_KO_s004
                                 + 1.01068 w_k1_MAT_s004
                                 + 0.932186 w_k1_MSFT_s004
                                 + 0.860968 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.789155 w_k1_ADM_s004
                                 + 0.688341 w_k1_AXP_s004
                                 + 0.640635 w_k1_DIS_s004
                                 + 0.714709 w_k1_FDX_s004
                                 + 0.89312 w_k1_IBM_s004
                                 + 0.703209 w_k1_INTC_s004
                                 + 0.769403 w_k1_KO_s004
                                 + 1.56207 w_k1_MAT_s004
                                 + 0.785233 w_k1_MSFT_s004
                                 + 0.838912 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.15616 w_k1_ADM_s004
                                 + 1.25885 w_k1_AXP_s004
                                 + 0.907004 w_k1_DIS_s004
                                 + 1.15524 w_k1_FDX_s004
                                 + 1.01648 w_k1_IBM_s004
                                 + 1.1375 w_k1_INTC_s004
                                 + 0.877826 w_k1_KO_s004
                                 + 1.38996 w_k1_MAT_s004
                                 + 1.03146 w_k1_MSFT_s004
                                 + 0.894333 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.97735 w_k1_ADM_s004
                                 + 1.01663 w_k1_AXP_s004
                                 + 0.981899 w_k1_DIS_s004
                                 + 0.868058 w_k1_FDX_s004
                                 + 0.927901 w_k1_IBM_s004
                                 + 0.942618 w_k1_INTC_s004
                                 + 0.875361 w_k1_KO_s004 + 1.0366 w_k1_MAT_s004
                                 + 0.9204 w_k1_MSFT_s004
                                 + 0.887261 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.04325 w_k1_ADM_s004
                                 + 0.752217 w_k1_AXP_s004
                                 + 1.0389 w_k1_DIS_s004
                                 + 0.973819 w_k1_FDX_s004
                                 + 1.00958 w_k1_IBM_s004 + 1.032 w_k1_INTC_s004
                                 + 1.04434 w_k1_KO_s004
                                 + 0.833138 w_k1_MAT_s004
                                 + 1.10189 w_k1_MSFT_s004
                                 + 0.866429 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.45072 w_k1_ADM_s004
                                 + 1.64016 w_k1_AXP_s004
                                 + 1.54903 w_k1_DIS_s004
                                 + 1.34425 w_k1_FDX_s004
                                 + 1.28397 w_k1_IBM_s004
                                 + 1.31346 w_k1_INTC_s004
                                 + 1.24651 w_k1_KO_s004 + 1.3181 w_k1_MAT_s004
                                 + 1.26334 w_k1_MSFT_s004 + 1.238 w_k1_PG_s004
                                 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.999749 w_k1_ADM_s004
                                 + 1.08041 w_k1_AXP_s004
                                 + 1.14707 w_k1_DIS_s004
                                 + 1.01251 w_k1_FDX_s004
                                 + 1.36228 w_k1_IBM_s004
                                 + 1.00925 w_k1_INTC_s004
                                 + 0.965789 w_k1_KO_s004
                                 + 1.68021 w_k1_MAT_s004
                                 + 0.937385 w_k1_MSFT_s004
                                 + 0.958225 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.01556 w_k1_ADM_s005
                                 + 1.5488 w_k1_AXP_s005 + 1.19092 w_k1_DIS_s005
                                 + 1.1583 w_k1_FDX_s005 + 1.19795 w_k1_IBM_s005
                                 + 1.10018 w_k1_INTC_s005
                                 + 0.99528 w_k1_KO_s005 + 1.84005 w_k1_MAT_s005
                                 + 1.15199 w_k1_MSFT_s005
                                 + 1.19074 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.859194 w_k1_ADM_s005
                                 + 0.959399 w_k1_AXP_s005
                                 + 0.863654 w_k1_DIS_s005
                                 + 0.870092 w_k1_FDX_s005
                                 + 1.23032 w_k1_IBM_s005
                                 + 0.9289 w_k1_INTC_s005
                                 + 0.867898 w_k1_KO_s005 + 1.4696 w_k1_MAT_s005
                                 + 0.768481 w_k1_MSFT_s005
                                 + 1.05505 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.936745 w_k1_ADM_s005
                                 + 0.924003 w_k1_AXP_s005
                                 + 0.715051 w_k1_DIS_s005
                                 + 0.897106 w_k1_FDX_s005 + 0.74 w_k1_IBM_s005
                                 + 0.83544 w_k1_INTC_s005
                                 + 0.672178 w_k1_KO_s005
                                 + 0.87566 w_k1_MAT_s005
                                 + 0.80578 w_k1_MSFT_s005
                                 + 0.815279 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.00792 w_k1_ADM_s005
                                 + 1.26533 w_k1_AXP_s005 + 1.2775 w_k1_DIS_s005
                                 + 1.21816 w_k1_FDX_s005
                                 + 0.859798 w_k1_IBM_s005
                                 + 1.06032 w_k1_INTC_s005
                                 + 0.910494 w_k1_KO_s005
                                 + 1.14034 w_k1_MAT_s005
                                 + 1.13626 w_k1_MSFT_s005
                                 + 1.05363 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.17704 w_k1_ADM_s005
                                 + 1.11352 w_k1_AXP_s005
                                 + 1.41882 w_k1_DIS_s005
                                 + 1.13113 w_k1_FDX_s005
                                 + 1.47451 w_k1_IBM_s005
                                 + 0.89212 w_k1_INTC_s005
                                 + 1.28264 w_k1_KO_s005 + 1.37619 w_k1_MAT_s005
                                 + 1.15951 w_k1_MSFT_s005
                                 + 1.15004 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.935601 w_k1_ADM_s005
                                 + 1.15267 w_k1_AXP_s005
                                 + 1.01821 w_k1_DIS_s005
                                 + 1.03115 w_k1_FDX_s005
                                 + 1.06209 w_k1_IBM_s005
                                 + 1.04232 w_k1_INTC_s005
                                 + 0.809662 w_k1_KO_s005
                                 + 0.802599 w_k1_MAT_s005
                                 + 0.940287 w_k1_MSFT_s005
                                 + 1.04873 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.769435 w_k1_ADM_s005
                                 + 0.790161 w_k1_AXP_s005
                                 + 0.730068 w_k1_DIS_s005
                                 + 0.729872 w_k1_FDX_s005
                                 + 1.07129 w_k1_IBM_s005
                                 + 0.907133 w_k1_INTC_s005
                                 + 0.886507 w_k1_KO_s005
                                 + 1.32585 w_k1_MAT_s005
                                 + 0.847919 w_k1_MSFT_s005
                                 + 0.891927 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.33157 w_k1_ADM_s005
                                 + 1.54785 w_k1_AXP_s005
                                 + 1.55754 w_k1_DIS_s005 + 1.4833 w_k1_FDX_s005
                                 + 0.954476 w_k1_IBM_s005
                                 + 1.26169 w_k1_INTC_s005
                                 + 1.11493 w_k1_KO_s005 + 1.60761 w_k1_MAT_s005
                                 + 1.34428 w_k1_MSFT_s005
                                 + 1.18004 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.782438 w_k1_ADM_s006
                                 + 0.668226 w_k1_AXP_s006
                                 + 0.778606 w_k1_DIS_s006
                                 + 0.756064 w_k1_FDX_s006
                                 + 1.08703 w_k1_IBM_s006
                                 + 0.825948 w_k1_INTC_s006
                                 + 0.861753 w_k1_KO_s006
                                 + 0.79535 w_k1_MAT_s006
                                 + 0.770303 w_k1_MSFT_s006
                                 + 0.79586 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.11514 w_k1_ADM_s006
                                 + 1.20837 w_k1_AXP_s006
                                 + 1.29805 w_k1_DIS_s006
                                 + 1.10721 w_k1_FDX_s006
                                 + 1.39304 w_k1_IBM_s006
                                 + 1.10157 w_k1_INTC_s006
                                 + 1.48252 w_k1_KO_s006 + 1.52365 w_k1_MAT_s006
                                 + 1.06215 w_k1_MSFT_s006
                                 + 1.18179 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.717927 w_k1_ADM_s006
                                 + 0.747667 w_k1_AXP_s006
                                 + 0.461897 w_k1_DIS_s006
                                 + 0.693512 w_k1_FDX_s006
                                 + 0.68252 w_k1_IBM_s006
                                 + 0.675133 w_k1_INTC_s006
                                 + 0.663423 w_k1_KO_s006
                                 + 0.737439 w_k1_MAT_s006
                                 + 0.680441 w_k1_MSFT_s006
                                 + 0.782571 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.865357 w_k1_ADM_s006
                                 + 1.09082 w_k1_AXP_s006
                                 + 0.808754 w_k1_DIS_s006
                                 + 0.948936 w_k1_FDX_s006
                                 + 0.982211 w_k1_IBM_s006
                                 + 0.876635 w_k1_INTC_s006
                                 + 0.967471 w_k1_KO_s006
                                 + 0.890166 w_k1_MAT_s006
                                 + 0.902748 w_k1_MSFT_s006
                                 + 0.966115 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.869288 w_k1_ADM_s006
                                 + 0.987638 w_k1_AXP_s006
                                 + 1.03345 w_k1_DIS_s006 + 1.0142 w_k1_FDX_s006
                                 + 0.962444 w_k1_IBM_s006
                                 + 0.909207 w_k1_INTC_s006
                                 + 1.00138 w_k1_KO_s006 + 1.11856 w_k1_MAT_s006
                                 + 1.03339 w_k1_MSFT_s006
                                 + 0.893754 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.965411 w_k1_ADM_s006
                                 + 1.13172 w_k1_AXP_s006
                                 + 1.11368 w_k1_DIS_s006
                                 + 1.20713 w_k1_FDX_s006
                                 + 0.782563 w_k1_IBM_s006
                                 + 1.1288 w_k1_INTC_s006
                                 + 0.969963 w_k1_KO_s006
                                 + 1.07046 w_k1_MAT_s006
                                 + 1.1002 w_k1_MSFT_s006 + 1.08842 w_k1_PG_s006
                                 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.823842 w_k1_ADM_s006
                                 + 1.08424 w_k1_AXP_s006
                                 + 0.755718 w_k1_DIS_s006
                                 + 0.827627 w_k1_FDX_s006
                                 + 1.1286 w_k1_IBM_s006
                                 + 0.843496 w_k1_INTC_s006
                                 + 0.89301 w_k1_KO_s006 + 1.4211 w_k1_MAT_s006
                                 + 0.900249 w_k1_MSFT_s006
                                 + 0.986612 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.66245 w_k1_ADM_s006
                                 + 0.670115 w_k1_AXP_s006
                                 + 0.597993 w_k1_DIS_s006
                                 + 0.793739 w_k1_FDX_s006
                                 + 0.623326 w_k1_IBM_s006
                                 + 0.774261 w_k1_INTC_s006
                                 + 0.781942 w_k1_KO_s006
                                 + 0.656511 w_k1_MAT_s006
                                 + 0.807855 w_k1_MSFT_s006
                                 + 0.738018 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.06352 w_k1_ADM_s007
                                 + 0.88535 w_k1_AXP_s007
                                 + 1.14869 w_k1_DIS_s007 + 0.9216 w_k1_FDX_s007
                                 + 1.23654 w_k1_IBM_s007
                                 + 0.833592 w_k1_INTC_s007
                                 + 1.23669 w_k1_KO_s007 + 1.09993 w_k1_MAT_s007
                                 + 0.964384 w_k1_MSFT_s007
                                 + 1.08259 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.841682 w_k1_ADM_s007
                                 + 0.609279 w_k1_AXP_s007
                                 + 0.647622 w_k1_DIS_s007
                                 + 0.748048 w_k1_FDX_s007
                                 + 0.747142 w_k1_IBM_s007
                                 + 0.8595 w_k1_INTC_s007
                                 + 0.805972 w_k1_KO_s007
                                 + 0.627979 w_k1_MAT_s007
                                 + 0.796977 w_k1_MSFT_s007
                                 + 0.784133 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.13208 w_k1_ADM_s007
                                 + 0.999846 w_k1_AXP_s007
                                 + 0.992658 w_k1_DIS_s007
                                 + 1.07595 w_k1_FDX_s007
                                 + 0.947944 w_k1_IBM_s007
                                 + 0.962014 w_k1_INTC_s007
                                 + 0.897736 w_k1_KO_s007
                                 + 0.88761 w_k1_MAT_s007
                                 + 0.978859 w_k1_MSFT_s007
                                 + 1.01793 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.17863 w_k1_ADM_s007
                                 + 1.37146 w_k1_AXP_s007 + 1.4939 w_k1_DIS_s007
                                 + 1.17854 w_k1_FDX_s007
                                 + 1.27446 w_k1_IBM_s007
                                 + 1.25373 w_k1_INTC_s007 + 1.1995 w_k1_KO_s007
                                 + 1.44387 w_k1_MAT_s007
                                 + 1.15124 w_k1_MSFT_s007
                                 + 1.07049 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.963737 w_k1_ADM_s007
                                 + 1.00148 w_k1_AXP_s007
                                 + 1.13497 w_k1_DIS_s007
                                 + 0.917709 w_k1_FDX_s007
                                 + 1.53025 w_k1_IBM_s007
                                 + 0.960959 w_k1_INTC_s007
                                 + 1.25606 w_k1_KO_s007 + 1.63427 w_k1_MAT_s007
                                 + 1.03693 w_k1_MSFT_s007
                                 + 1.00049 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.57849 w_k1_ADM_s007
                                 + 1.53569 w_k1_AXP_s007
                                 + 1.68751 w_k1_DIS_s007
                                 + 1.34798 w_k1_FDX_s007
                                 + 1.33592 w_k1_IBM_s007
                                 + 1.50183 w_k1_INTC_s007
                                 + 1.49955 w_k1_KO_s007 + 1.41009 w_k1_MAT_s007
                                 + 1.39239 w_k1_MSFT_s007
                                 + 1.25885 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.911589 w_k1_ADM_s007
                                 + 1.03819 w_k1_AXP_s007
                                 + 0.693532 w_k1_DIS_s007
                                 + 0.875026 w_k1_FDX_s007
                                 + 0.790502 w_k1_IBM_s007
                                 + 0.97632 w_k1_INTC_s007
                                 + 0.789662 w_k1_KO_s007
                                 + 0.960077 w_k1_MAT_s007
                                 + 0.768327 w_k1_MSFT_s007
                                 + 0.879261 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.794465 w_k1_ADM_s007
                                 + 0.858531 w_k1_AXP_s007
                                 + 0.904272 w_k1_DIS_s007
                                 + 0.825262 w_k1_FDX_s007
                                 + 1.00344 w_k1_IBM_s007
                                 + 0.927332 w_k1_INTC_s007
                                 + 0.910217 w_k1_KO_s007
                                 + 0.911932 w_k1_MAT_s007
                                 + 0.926316 w_k1_MSFT_s007
                                 + 0.973098 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.29535 w_k1_ADM_s010
                                 + 0.938964 w_k1_AXP_s010
                                 + 1.14754 w_k1_DIS_s010
                                 + 1.03754 w_k1_FDX_s010
                                 + 1.44565 w_k1_IBM_s010
                                 + 1.05227 w_k1_INTC_s010
                                 + 1.14908 w_k1_KO_s010
                                 + 0.929264 w_k1_MAT_s010
                                 + 1.0683 w_k1_MSFT_s010 + 1.15535 w_k1_PG_s010
                                 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.61417 w_k1_ADM_s010
                                 + 1.86127 w_k1_AXP_s010
                                 + 1.99201 w_k1_DIS_s010
                                 + 1.63233 w_k1_FDX_s010
                                 + 1.46447 w_k1_IBM_s010
                                 + 1.5482 w_k1_INTC_s010 + 1.36238 w_k1_KO_s010
                                 + 1.0729 w_k1_MAT_s010
                                 + 1.53582 w_k1_MSFT_s010
                                 + 1.38096 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.919559 w_k1_ADM_s010
                                 + 0.924463 w_k1_AXP_s010
                                 + 1.14405 w_k1_DIS_s010
                                 + 1.07252 w_k1_FDX_s010
                                 + 0.898437 w_k1_IBM_s010
                                 + 1.45906 w_k1_INTC_s010
                                 + 0.997463 w_k1_KO_s010
                                 + 0.704027 w_k1_MAT_s010
                                 + 1.21003 w_k1_MSFT_s010
                                 + 0.984907 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.00555 w_k1_ADM_s010
                                 + 1.33409 w_k1_AXP_s010
                                 + 0.845113 w_k1_DIS_s010
                                 + 1.02109 w_k1_FDX_s010 + 1.5315 w_k1_IBM_s010
                                 + 1.01088 w_k1_INTC_s010
                                 + 1.03521 w_k1_KO_s010 + 1.36311 w_k1_MAT_s010
                                 + 0.927212 w_k1_MSFT_s010
                                 + 1.04104 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.37032 w_k1_ADM_s010
                                 + 1.19083 w_k1_AXP_s010
                                 + 1.13116 w_k1_DIS_s010
                                 + 1.25985 w_k1_FDX_s010
                                 + 0.80056 w_k1_IBM_s010
                                 + 1.22922 w_k1_INTC_s010
                                 + 1.07726 w_k1_KO_s010 + 1.0271 w_k1_MAT_s010
                                 + 1.29636 w_k1_MSFT_s010
                                 + 1.05211 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.838374 w_k1_ADM_s010
                                 + 0.729719 w_k1_AXP_s010
                                 + 0.61765 w_k1_DIS_s010
                                 + 0.775713 w_k1_FDX_s010
                                 + 0.808784 w_k1_IBM_s010
                                 + 0.972564 w_k1_INTC_s010
                                 + 0.803314 w_k1_KO_s010
                                 + 0.586307 w_k1_MAT_s010
                                 + 0.779606 w_k1_MSFT_s010
                                 + 0.827769 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.12056 w_k1_ADM_s010
                                 + 1.08766 w_k1_AXP_s010
                                 + 0.792227 w_k1_DIS_s010
                                 + 0.986166 w_k1_FDX_s010
                                 + 1.14575 w_k1_IBM_s010
                                 + 0.982944 w_k1_INTC_s010
                                 + 1.05542 w_k1_KO_s010 + 1.06776 w_k1_MAT_s010
                                 + 0.961979 w_k1_MSFT_s010
                                 + 0.960043 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.10329 w_k1_ADM_s010
                                 + 0.973036 w_k1_AXP_s010
                                 + 0.77863 w_k1_DIS_s010
                                 + 0.762604 w_k1_FDX_s010
                                 + 1.49594 w_k1_IBM_s010
                                 + 1.04914 w_k1_INTC_s010
                                 + 0.96505 w_k1_KO_s010
                                 + 0.930343 w_k1_MAT_s010
                                 + 0.836355 w_k1_MSFT_s010
                                 + 0.928894 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.26141 w_k1_ADM_s011
                                 + 1.26281 w_k1_AXP_s011 + 1.2811 w_k1_DIS_s011
                                 + 1.29308 w_k1_FDX_s011
                                 + 0.976831 w_k1_IBM_s011
                                 + 1.18421 w_k1_INTC_s011
                                 + 1.13687 w_k1_KO_s011 + 1.46915 w_k1_MAT_s011
                                 + 1.16515 w_k1_MSFT_s011
                                 + 1.34765 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.958953 w_k1_ADM_s011
                                 + 0.764784 w_k1_AXP_s011
                                 + 0.659122 w_k1_DIS_s011
                                 + 0.74781 w_k1_FDX_s011
                                 + 0.822577 w_k1_IBM_s011
                                 + 0.890824 w_k1_INTC_s011
                                 + 0.759835 w_k1_KO_s011
                                 + 0.94322 w_k1_MAT_s011
                                 + 0.81037 w_k1_MSFT_s011
                                 + 0.812247 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.993874 w_k1_ADM_s011
                                 + 0.64293 w_k1_AXP_s011
                                 + 0.72267 w_k1_DIS_s011
                                 + 0.699452 w_k1_FDX_s011
                                 + 1.20001 w_k1_IBM_s011
                                 + 0.827275 w_k1_INTC_s011
                                 + 1.08674 w_k1_KO_s011 + 1.219 w_k1_MAT_s011
                                 + 0.860796 w_k1_MSFT_s011
                                 + 0.960795 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.67151 w_k1_ADM_s011
                                 + 1.24263 w_k1_AXP_s011 + 1.4422 w_k1_DIS_s011
                                 + 1.30137 w_k1_FDX_s011
                                 + 1.27706 w_k1_IBM_s011
                                 + 1.17681 w_k1_INTC_s011
                                 + 1.46297 w_k1_KO_s011 + 1.45679 w_k1_MAT_s011
                                 + 1.33052 w_k1_MSFT_s011 + 1.393 w_k1_PG_s011
                                 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.18367 w_k1_ADM_s011
                                 + 1.04298 w_k1_AXP_s011
                                 + 0.876978 w_k1_DIS_s011
                                 + 1.15065 w_k1_FDX_s011
                                 + 0.738052 w_k1_IBM_s011
                                 + 0.946352 w_k1_INTC_s011
                                 + 0.820887 w_k1_KO_s011
                                 + 1.27557 w_k1_MAT_s011
                                 + 1.03975 w_k1_MSFT_s011
                                 + 1.08755 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.1087 w_k1_ADM_s011
                                 + 0.925711 w_k1_AXP_s011
                                 + 1.05776 w_k1_DIS_s011
                                 + 0.979217 w_k1_FDX_s011
                                 + 1.04496 w_k1_IBM_s011
                                 + 1.0419 w_k1_INTC_s011 + 1.17796 w_k1_KO_s011
                                 + 1.32278 w_k1_MAT_s011
                                 + 1.08491 w_k1_MSFT_s011 + 1.0897 w_k1_PG_s011
                                 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.17464 w_k1_ADM_s011
                                 + 0.751779 w_k1_AXP_s011
                                 + 0.915259 w_k1_DIS_s011
                                 + 0.944793 w_k1_FDX_s011
                                 + 0.763914 w_k1_IBM_s011
                                 + 0.755151 w_k1_INTC_s011
                                 + 0.958801 w_k1_KO_s011
                                 + 0.980647 w_k1_MAT_s011
                                 + 1.06136 w_k1_MSFT_s011
                                 + 1.02455 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.32166 w_k1_ADM_s011
                                 + 1.37445 w_k1_AXP_s011
                                 + 1.31742 w_k1_DIS_s011
                                 + 1.05862 w_k1_FDX_s011
                                 + 1.79452 w_k1_IBM_s011
                                 + 1.02312 w_k1_INTC_s011
                                 + 1.44659 w_k1_KO_s011 + 1.75388 w_k1_MAT_s011
                                 + 1.03499 w_k1_MSFT_s011
                                 + 1.24751 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.15805 w_k1_ADM_s012
                                 + 1.52144 w_k1_AXP_s012 + 1.3128 w_k1_DIS_s012
                                 + 1.31395 w_k1_FDX_s012
                                 + 1.15723 w_k1_IBM_s012
                                 + 1.36301 w_k1_INTC_s012
                                 + 1.24751 w_k1_KO_s012 + 1.27396 w_k1_MAT_s012
                                 + 1.27476 w_k1_MSFT_s012
                                 + 1.19909 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.926565 w_k1_ADM_s012
                                 + 0.917475 w_k1_AXP_s012
                                 + 1.2642 w_k1_DIS_s012 + 1.03262 w_k1_FDX_s012
                                 + 1.22687 w_k1_IBM_s012
                                 + 1.04121 w_k1_INTC_s012
                                 + 1.02815 w_k1_KO_s012
                                 + 0.933375 w_k1_MAT_s012
                                 + 1.0429 w_k1_MSFT_s012 + 0.96924 w_k1_PG_s012
                                 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.3977 w_k1_ADM_s012
                                 + 1.08662 w_k1_AXP_s012
                                 + 1.60282 w_k1_DIS_s012
                                 + 1.29396 w_k1_FDX_s012
                                 + 1.16443 w_k1_IBM_s012
                                 + 1.28271 w_k1_INTC_s012
                                 + 1.25084 w_k1_KO_s012 + 1.0499 w_k1_MAT_s012
                                 + 1.38134 w_k1_MSFT_s012
                                 + 1.18866 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.989962 w_k1_ADM_s012
                                 + 1.0798 w_k1_AXP_s012 + 1.03225 w_k1_DIS_s012
                                 + 0.976544 w_k1_FDX_s012
                                 + 0.956059 w_k1_IBM_s012
                                 + 1.22538 w_k1_INTC_s012
                                 + 0.956586 w_k1_KO_s012
                                 + 0.963186 w_k1_MAT_s012
                                 + 0.918048 w_k1_MSFT_s012
                                 + 0.994681 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.07643 w_k1_ADM_s012
                                 + 0.870249 w_k1_AXP_s012
                                 + 0.898378 w_k1_DIS_s012
                                 + 0.947909 w_k1_FDX_s012
                                 + 0.771491 w_k1_IBM_s012
                                 + 1.09299 w_k1_INTC_s012
                                 + 0.861842 w_k1_KO_s012
                                 + 0.598279 w_k1_MAT_s012
                                 + 0.961721 w_k1_MSFT_s012
                                 + 0.851944 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.704707 w_k1_ADM_s012
                                 + 0.693736 w_k1_AXP_s012
                                 + 0.646933 w_k1_DIS_s012
                                 + 0.732191 w_k1_FDX_s012 + 0.895 w_k1_IBM_s012
                                 + 0.90614 w_k1_INTC_s012
                                 + 0.829019 w_k1_KO_s012
                                 + 1.02619 w_k1_MAT_s012
                                 + 0.836045 w_k1_MSFT_s012
                                 + 0.830907 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.902976 w_k1_ADM_s012
                                 + 0.778172 w_k1_AXP_s012
                                 + 1.08019 w_k1_DIS_s012
                                 + 0.790987 w_k1_FDX_s012
                                 + 1.27115 w_k1_IBM_s012
                                 + 0.987437 w_k1_INTC_s012
                                 + 1.13502 w_k1_KO_s012 + 1.04658 w_k1_MAT_s012
                                 + 0.965482 w_k1_MSFT_s012
                                 + 1.08369 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.813009 w_k1_ADM_s012
                                 + 1.15967 w_k1_AXP_s012
                                 + 0.913518 w_k1_DIS_s012
                                 + 0.884166 w_k1_FDX_s012
                                 + 1.45993 w_k1_IBM_s012
                                 + 1.06999 w_k1_INTC_s012
                                 + 1.06984 w_k1_KO_s012 + 1.4892 w_k1_MAT_s012
                                 + 0.844001 w_k1_MSFT_s012
                                 + 0.958235 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.789812 w_k1_ADM_s013
                                 + 0.818196 w_k1_AXP_s013
                                 + 0.598309 w_k1_DIS_s013
                                 + 0.848717 w_k1_FDX_s013
                                 + 0.74568 w_k1_IBM_s013
                                 + 0.999333 w_k1_INTC_s013
                                 + 0.69512 w_k1_KO_s013
                                 + 0.737411 w_k1_MAT_s013
                                 + 0.819526 w_k1_MSFT_s013
                                 + 0.729893 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.856175 w_k1_ADM_s013
                                 + 0.603168 w_k1_AXP_s013
                                 + 0.551957 w_k1_DIS_s013
                                 + 0.661436 w_k1_FDX_s013
                                 + 0.744598 w_k1_IBM_s013
                                 + 0.645765 w_k1_INTC_s013
                                 + 0.646749 w_k1_KO_s013
                                 + 0.598849 w_k1_MAT_s013
                                 + 0.721131 w_k1_MSFT_s013
                                 + 0.729824 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.676982 w_k1_ADM_s013
                                 + 0.733849 w_k1_AXP_s013
                                 + 1.00145 w_k1_DIS_s013
                                 + 0.908227 w_k1_FDX_s013
                                 + 1.0543 w_k1_IBM_s013
                                 + 0.884843 w_k1_INTC_s013
                                 + 0.921276 w_k1_KO_s013
                                 + 0.961081 w_k1_MAT_s013
                                 + 0.941255 w_k1_MSFT_s013
                                 + 0.829602 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.21849 w_k1_ADM_s013
                                 + 0.899746 w_k1_AXP_s013
                                 + 1.08222 w_k1_DIS_s013 + 1.0938 w_k1_FDX_s013
                                 + 1.02193 w_k1_IBM_s013
                                 + 1.10793 w_k1_INTC_s013
                                 + 1.09279 w_k1_KO_s013
                                 + 0.925006 w_k1_MAT_s013
                                 + 1.0664 w_k1_MSFT_s013
                                 + 0.991792 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.935257 w_k1_ADM_s013
                                 + 0.804469 w_k1_AXP_s013
                                 + 0.864802 w_k1_DIS_s013
                                 + 0.958193 w_k1_FDX_s013
                                 + 0.811688 w_k1_IBM_s013
                                 + 0.94572 w_k1_INTC_s013
                                 + 0.866589 w_k1_KO_s013
                                 + 0.848684 w_k1_MAT_s013
                                 + 0.942754 w_k1_MSFT_s013
                                 + 0.818982 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.892485 w_k1_ADM_s013
                                 + 1.22995 w_k1_AXP_s013
                                 + 1.06221 w_k1_DIS_s013
                                 + 1.01592 w_k1_FDX_s013
                                 + 1.12374 w_k1_IBM_s013
                                 + 1.22019 w_k1_INTC_s013
                                 + 0.990873 w_k1_KO_s013
                                 + 1.21038 w_k1_MAT_s013
                                 + 1.10121 w_k1_MSFT_s013
                                 + 0.880226 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.14023 w_k1_ADM_s013
                                 + 1.14587 w_k1_AXP_s013
                                 + 1.02355 w_k1_DIS_s013
                                 + 1.17752 w_k1_FDX_s013
                                 + 0.656737 w_k1_IBM_s013
                                 + 0.874628 w_k1_INTC_s013
                                 + 0.822011 w_k1_KO_s013
                                 + 0.941812 w_k1_MAT_s013
                                 + 1.00902 w_k1_MSFT_s013
                                 + 0.966902 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.45527 w_k1_ADM_s013
                                 + 1.25634 w_k1_AXP_s013
                                 + 1.53683 w_k1_DIS_s013
                                 + 1.31007 w_k1_FDX_s013
                                 + 1.28095 w_k1_IBM_s013
                                 + 1.13892 w_k1_INTC_s013
                                 + 1.30754 w_k1_KO_s013 + 1.34172 w_k1_MAT_s013
                                 + 1.19879 w_k1_MSFT_s013
                                 + 1.03217 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.23922 w_k1_ADM_s014
                                 + 1.23463 w_k1_AXP_s014 + 1.0895 w_k1_DIS_s014
                                 + 1.00671 w_k1_FDX_s014
                                 + 1.72493 w_k1_IBM_s014
                                 + 0.926409 w_k1_INTC_s014
                                 + 1.35914 w_k1_KO_s014 + 1.19762 w_k1_MAT_s014
                                 + 0.986661 w_k1_MSFT_s014
                                 + 1.10216 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.0045 w_k1_ADM_s014
                                 + 0.608539 w_k1_AXP_s014
                                 + 0.624973 w_k1_DIS_s014
                                 + 0.697941 w_k1_FDX_s014
                                 + 0.630687 w_k1_IBM_s014
                                 + 0.710706 w_k1_INTC_s014
                                 + 0.701595 w_k1_KO_s014
                                 + 0.434635 w_k1_MAT_s014
                                 + 0.771188 w_k1_MSFT_s014
                                 + 0.772442 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.21815 w_k1_ADM_s014
                                 + 1.22687 w_k1_AXP_s014
                                 + 1.30648 w_k1_DIS_s014
                                 + 1.19068 w_k1_FDX_s014 + 1.0174 w_k1_IBM_s014
                                 + 1.17666 w_k1_INTC_s014
                                 + 1.08653 w_k1_KO_s014
                                 + 0.777594 w_k1_MAT_s014
                                 + 1.28716 w_k1_MSFT_s014
                                 + 1.09041 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.851366 w_k1_ADM_s014
                                 + 0.701113 w_k1_AXP_s014
                                 + 0.879645 w_k1_DIS_s014
                                 + 0.924601 w_k1_FDX_s014
                                 + 0.829127 w_k1_IBM_s014
                                 + 0.800353 w_k1_INTC_s014
                                 + 0.905043 w_k1_KO_s014
                                 + 0.743224 w_k1_MAT_s014
                                 + 0.904907 w_k1_MSFT_s014
                                 + 0.898038 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.61901 w_k1_ADM_s014
                                 + 1.27629 w_k1_AXP_s014
                                 + 1.55814 w_k1_DIS_s014
                                 + 1.33043 w_k1_FDX_s014
                                 + 1.36917 w_k1_IBM_s014
                                 + 1.06166 w_k1_INTC_s014
                                 + 1.35533 w_k1_KO_s014 + 1.34125 w_k1_MAT_s014
                                 + 1.36681 w_k1_MSFT_s014
                                 + 1.29397 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.952037 w_k1_ADM_s014
                                 + 0.785306 w_k1_AXP_s014
                                 + 0.846762 w_k1_DIS_s014
                                 + 0.829932 w_k1_FDX_s014
                                 + 1.16081 w_k1_IBM_s014
                                 + 0.813753 w_k1_INTC_s014
                                 + 1.01553 w_k1_KO_s014 + 0.9065 w_k1_MAT_s014
                                 + 0.866412 w_k1_MSFT_s014
                                 + 0.977482 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.10405 w_k1_ADM_s014
                                 + 1.2707 w_k1_AXP_s014
                                 + 0.892666 w_k1_DIS_s014
                                 + 1.01911 w_k1_FDX_s014
                                 + 1.17024 w_k1_IBM_s014
                                 + 0.851376 w_k1_INTC_s014
                                 + 0.959391 w_k1_KO_s014
                                 + 1.25337 w_k1_MAT_s014
                                 + 0.832836 w_k1_MSFT_s014
                                 + 0.940581 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.22207 w_k1_ADM_s014
                                 + 1.19825 w_k1_AXP_s014
                                 + 0.800478 w_k1_DIS_s014
                                 + 0.959921 w_k1_FDX_s014
                                 + 1.03805 w_k1_IBM_s014
                                 + 1.00589 w_k1_INTC_s014
                                 + 0.811827 w_k1_KO_s014
                                 + 0.527785 w_k1_MAT_s014
                                 + 0.785657 w_k1_MSFT_s014
                                 + 0.870176 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.39215 w_k1_ADM_s015
                                 + 1.42471 w_k1_AXP_s015
                                 + 1.59295 w_k1_DIS_s015
                                 + 1.31264 w_k1_FDX_s015
                                 + 1.28195 w_k1_IBM_s015
                                 + 0.913755 w_k1_INTC_s015 + 1.346 w_k1_KO_s015
                                 + 1.4664 w_k1_MAT_s015 + 1.1566 w_k1_MSFT_s015
                                 + 1.2971 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.12186 w_k1_ADM_s015
                                 + 1.03591 w_k1_AXP_s015
                                 + 1.02415 w_k1_DIS_s015
                                 + 1.14464 w_k1_FDX_s015
                                 + 0.774191 w_k1_IBM_s015
                                 + 1.06653 w_k1_INTC_s015
                                 + 0.887427 w_k1_KO_s015
                                 + 0.725194 w_k1_MAT_s015
                                 + 1.06944 w_k1_MSFT_s015
                                 + 0.943401 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.12636 w_k1_ADM_s015
                                 + 1.14116 w_k1_AXP_s015
                                 + 1.00127 w_k1_DIS_s015
                                 + 0.905206 w_k1_FDX_s015
                                 + 1.2736 w_k1_IBM_s015
                                 + 0.85114 w_k1_INTC_s015
                                 + 1.20339 w_k1_KO_s015 + 1.55518 w_k1_MAT_s015
                                 + 1.05144 w_k1_MSFT_s015
                                 + 0.927941 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.10322 w_k1_ADM_s015
                                 + 1.46423 w_k1_AXP_s015
                                 + 1.22255 w_k1_DIS_s015
                                 + 1.25134 w_k1_FDX_s015
                                 + 1.08699 w_k1_IBM_s015
                                 + 1.03021 w_k1_INTC_s015 + 1.0713 w_k1_KO_s015
                                 + 1.32151 w_k1_MAT_s015
                                 + 1.12247 w_k1_MSFT_s015
                                 + 1.06882 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.955351 w_k1_ADM_s015
                                 + 0.892971 w_k1_AXP_s015
                                 + 0.743525 w_k1_DIS_s015
                                 + 0.839845 w_k1_FDX_s015
                                 + 0.766021 w_k1_IBM_s015
                                 + 0.663271 w_k1_INTC_s015
                                 + 0.783852 w_k1_KO_s015
                                 + 0.701682 w_k1_MAT_s015
                                 + 0.808569 w_k1_MSFT_s015
                                 + 0.887714 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.1265 w_k1_ADM_s015
                                 + 1.10997 w_k1_AXP_s015
                                 + 1.09636 w_k1_DIS_s015
                                 + 1.01717 w_k1_FDX_s015
                                 + 1.00313 w_k1_IBM_s015
                                 + 0.817866 w_k1_INTC_s015
                                 + 0.984178 w_k1_KO_s015
                                 + 0.819939 w_k1_MAT_s015
                                 + 0.936498 w_k1_MSFT_s015
                                 + 1.00426 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.842488 w_k1_ADM_s015
                                 + 1.01687 w_k1_AXP_s015
                                 + 1.23511 w_k1_DIS_s015
                                 + 0.93026 w_k1_FDX_s015 + 1.3547 w_k1_IBM_s015
                                 + 0.89473 w_k1_INTC_s015
                                 + 1.20998 w_k1_KO_s015 + 1.58349 w_k1_MAT_s015
                                 + 1.0095 w_k1_MSFT_s015 + 1.10559 w_k1_PG_s015
                                 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.628074 w_k1_ADM_s015
                                 + 0.788504 w_k1_AXP_s015
                                 + 0.760957 w_k1_DIS_s015
                                 + 0.681447 w_k1_FDX_s015
                                 + 1.02733 w_k1_IBM_s015
                                 + 0.683294 w_k1_INTC_s015
                                 + 0.807165 w_k1_KO_s015
                                 + 1.29342 w_k1_MAT_s015
                                 + 0.769318 w_k1_MSFT_s015
                                 + 0.91613 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.82046 w_k1_ADM_s016
                                 + 1.16866 w_k1_AXP_s016
                                 + 0.795457 w_k1_DIS_s016
                                 + 1.13758 w_k1_FDX_s016
                                 + 0.718091 w_k1_IBM_s016
                                 + 1.28212 w_k1_INTC_s016
                                 + 0.758789 w_k1_KO_s016
                                 + 0.682336 w_k1_MAT_s016
                                 + 1.05186 w_k1_MSFT_s016
                                 + 0.957776 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.907674 w_k1_ADM_s016
                                 + 1.20437 w_k1_AXP_s016
                                 + 0.821048 w_k1_DIS_s016
                                 + 0.841029 w_k1_FDX_s016
                                 + 1.16717 w_k1_IBM_s016
                                 + 0.966788 w_k1_INTC_s016
                                 + 0.995798 w_k1_KO_s016
                                 + 1.17044 w_k1_MAT_s016
                                 + 0.858827 w_k1_MSFT_s016
                                 + 0.979716 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.828689 w_k1_ADM_s016
                                 + 0.744552 w_k1_AXP_s016
                                 + 0.553866 w_k1_DIS_s016
                                 + 0.800194 w_k1_FDX_s016
                                 + 0.76998 w_k1_IBM_s016
                                 + 0.752934 w_k1_INTC_s016
                                 + 0.625924 w_k1_KO_s016
                                 + 0.649864 w_k1_MAT_s016
                                 + 0.753643 w_k1_MSFT_s016
                                 + 0.960323 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.04643 w_k1_ADM_s016
                                 + 1.0168 w_k1_AXP_s016 + 1.09543 w_k1_DIS_s016
                                 + 0.980848 w_k1_FDX_s016
                                 + 1.05049 w_k1_IBM_s016
                                 + 1.04634 w_k1_INTC_s016
                                 + 1.10376 w_k1_KO_s016
                                 + 0.729966 w_k1_MAT_s016
                                 + 0.942119 w_k1_MSFT_s016
                                 + 1.10036 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.74384 w_k1_ADM_s016
                                 + 0.75063 w_k1_AXP_s016
                                 + 0.778862 w_k1_DIS_s016
                                 + 0.858024 w_k1_FDX_s016
                                 + 0.7286 w_k1_IBM_s016
                                 + 1.17296 w_k1_INTC_s016
                                 + 0.934656 w_k1_KO_s016
                                 + 0.583473 w_k1_MAT_s016
                                 + 1.03878 w_k1_MSFT_s016
                                 + 0.933887 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.727041 w_k1_ADM_s016
                                 + 1.37164 w_k1_AXP_s016
                                 + 1.32229 w_k1_DIS_s016
                                 + 1.15421 w_k1_FDX_s016 + 1.5815 w_k1_IBM_s016
                                 + 1.37859 w_k1_INTC_s016 + 1.3006 w_k1_KO_s016
                                 + 1.2523 w_k1_MAT_s016
                                 + 1.10803 w_k1_MSFT_s016
                                 + 1.14172 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.908098 w_k1_ADM_s016
                                 + 0.833436 w_k1_AXP_s016
                                 + 0.791241 w_k1_DIS_s016
                                 + 0.978394 w_k1_FDX_s016
                                 + 0.946232 w_k1_IBM_s016
                                 + 1.04506 w_k1_INTC_s016
                                 + 0.86854 w_k1_KO_s016
                                 + 0.540187 w_k1_MAT_s016
                                 + 0.906861 w_k1_MSFT_s016
                                 + 1.00908 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.878615 w_k1_ADM_s016
                                 + 1.04467 w_k1_AXP_s016
                                 + 1.13047 w_k1_DIS_s016
                                 + 1.15177 w_k1_FDX_s016
                                 + 0.865151 w_k1_IBM_s016
                                 + 1.25017 w_k1_INTC_s016
                                 + 0.890465 w_k1_KO_s016
                                 + 0.711131 w_k1_MAT_s016
                                 + 1.01395 w_k1_MSFT_s016
                                 + 0.976735 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.18096 w_k1_ADM_s017
                                 + 1.48274 w_k1_AXP_s017
                                 + 1.28669 w_k1_DIS_s017 + 1.2039 w_k1_FDX_s017
                                 + 1.47658 w_k1_IBM_s017
                                 + 1.0726 w_k1_INTC_s017 + 1.2769 w_k1_KO_s017
                                 + 1.46948 w_k1_MAT_s017
                                 + 1.04655 w_k1_MSFT_s017
                                 + 1.17278 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.39059 w_k1_ADM_s017
                                 + 1.23955 w_k1_AXP_s017
                                 + 1.25772 w_k1_DIS_s017
                                 + 1.23867 w_k1_FDX_s017
                                 + 0.926374 w_k1_IBM_s017
                                 + 0.961883 w_k1_INTC_s017
                                 + 1.09136 w_k1_KO_s017
                                 + 0.894882 w_k1_MAT_s017
                                 + 0.979588 w_k1_MSFT_s017
                                 + 1.12529 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.512 w_k1_ADM_s017
                                 + 1.23448 w_k1_AXP_s017
                                 + 1.63949 w_k1_DIS_s017
                                 + 1.35163 w_k1_FDX_s017
                                 + 0.979823 w_k1_IBM_s017
                                 + 1.04318 w_k1_INTC_s017
                                 + 1.05754 w_k1_KO_s017
                                 + 0.850336 w_k1_MAT_s017
                                 + 1.3029 w_k1_MSFT_s017 + 1.19082 w_k1_PG_s017
                                 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.820602 w_k1_ADM_s017
                                 + 0.821276 w_k1_AXP_s017
                                 + 0.76434 w_k1_DIS_s017
                                 + 0.82823 w_k1_FDX_s017
                                 + 1.08275 w_k1_IBM_s017
                                 + 0.84863 w_k1_INTC_s017
                                 + 0.862528 w_k1_KO_s017
                                 + 1.02105 w_k1_MAT_s017
                                 + 0.73384 w_k1_MSFT_s017
                                 + 0.919394 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.850476 w_k1_ADM_s017
                                 + 0.869002 w_k1_AXP_s017
                                 + 0.952541 w_k1_DIS_s017
                                 + 0.929011 w_k1_FDX_s017
                                 + 0.957818 w_k1_IBM_s017
                                 + 0.953909 w_k1_INTC_s017
                                 + 0.836494 w_k1_KO_s017
                                 + 0.742066 w_k1_MAT_s017
                                 + 1.0049 w_k1_MSFT_s017 + 0.94791 w_k1_PG_s017
                                 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.02884 w_k1_ADM_s017
                                 + 0.904822 w_k1_AXP_s017
                                 + 1.40765 w_k1_DIS_s017
                                 + 1.07876 w_k1_FDX_s017
                                 + 1.22051 w_k1_IBM_s017
                                 + 1.10842 w_k1_INTC_s017
                                 + 1.21546 w_k1_KO_s017
                                 + 0.683343 w_k1_MAT_s017
                                 + 1.12427 w_k1_MSFT_s017
                                 + 1.06694 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.901349 w_k1_ADM_s017
                                 + 0.852799 w_k1_AXP_s017
                                 + 0.798359 w_k1_DIS_s017
                                 + 0.995039 w_k1_FDX_s017
                                 + 0.695875 w_k1_IBM_s017
                                 + 1.07662 w_k1_INTC_s017
                                 + 0.855979 w_k1_KO_s017
                                 + 0.596153 w_k1_MAT_s017
                                 + 0.902246 w_k1_MSFT_s017
                                 + 0.8419 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.847729 w_k1_ADM_s017
                                 + 1.25943 w_k1_AXP_s017
                                 + 0.897085 w_k1_DIS_s017
                                 + 1.04741 w_k1_FDX_s017
                                 + 0.781763 w_k1_IBM_s017
                                 + 0.806998 w_k1_INTC_s017
                                 + 0.751178 w_k1_KO_s017
                                 + 1.03484 w_k1_MAT_s017
                                 + 0.931883 w_k1_MSFT_s017
                                 + 0.880779 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.25266 w_k1_ADM_s100
                                 + 0.974027 w_k1_AXP_s100
                                 + 1.03925 w_k1_DIS_s100
                                 + 1.11215 w_k1_FDX_s100
                                 + 0.843406 w_k1_IBM_s100
                                 + 1.03782 w_k1_INTC_s100
                                 + 1.13501 w_k1_KO_s100
                                 + 0.750051 w_k1_MAT_s100
                                 + 1.09151 w_k1_MSFT_s100
                                 + 1.15914 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.83217 w_k1_ADM_s100
                                 + 0.941956 w_k1_AXP_s100
                                 + 0.737801 w_k1_DIS_s100
                                 + 0.851628 w_k1_FDX_s100
                                 + 0.922945 w_k1_IBM_s100
                                 + 0.888886 w_k1_INTC_s100
                                 + 0.893102 w_k1_KO_s100
                                 + 1.07356 w_k1_MAT_s100
                                 + 0.888066 w_k1_MSFT_s100
                                 + 0.911129 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.852603 w_k1_ADM_s100
                                 + 0.686805 w_k1_AXP_s100
                                 + 0.509434 w_k1_DIS_s100
                                 + 0.746313 w_k1_FDX_s100
                                 + 0.643203 w_k1_IBM_s100
                                 + 0.592727 w_k1_INTC_s100
                                 + 0.697497 w_k1_KO_s100
                                 + 0.947448 w_k1_MAT_s100
                                 + 0.737758 w_k1_MSFT_s100
                                 + 0.880151 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.989569 w_k1_ADM_s100
                                 + 0.725129 w_k1_AXP_s100
                                 + 0.87955 w_k1_DIS_s100
                                 + 0.921323 w_k1_FDX_s100
                                 + 0.700504 w_k1_IBM_s100
                                 + 0.813166 w_k1_INTC_s100
                                 + 0.904789 w_k1_KO_s100
                                 + 0.656565 w_k1_MAT_s100
                                 + 0.897494 w_k1_MSFT_s100
                                 + 0.972836 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.889182 w_k1_ADM_s100
                                 + 1.04036 w_k1_AXP_s100
                                 + 0.988445 w_k1_DIS_s100
                                 + 0.914957 w_k1_FDX_s100
                                 + 1.24504 w_k1_IBM_s100
                                 + 0.984046 w_k1_INTC_s100
                                 + 1.33602 w_k1_KO_s100 + 1.28818 w_k1_MAT_s100
                                 + 0.999162 w_k1_MSFT_s100
                                 + 1.08188 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.858246 w_k1_ADM_s100
                                 + 1.10017 w_k1_AXP_s100
                                 + 1.10302 w_k1_DIS_s100
                                 + 0.99514 w_k1_FDX_s100
                                 + 1.56102 w_k1_IBM_s100
                                 + 1.02941 w_k1_INTC_s100
                                 + 1.42046 w_k1_KO_s100 + 1.36753 w_k1_MAT_s100
                                 + 0.95232 w_k1_MSFT_s100 + 1.1746 w_k1_PG_s100
                                 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.19879 w_k1_ADM_s100
                                 + 1.52637 w_k1_AXP_s100
                                 + 1.80859 w_k1_DIS_s100
                                 + 1.42692 w_k1_FDX_s100
                                 + 1.46699 w_k1_IBM_s100
                                 + 1.30216 w_k1_INTC_s100
                                 + 1.65833 w_k1_KO_s100 + 1.65795 w_k1_MAT_s100
                                 + 1.35392 w_k1_MSFT_s100
                                 + 1.54246 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.01719 w_k1_ADM_s100
                                 + 1.0783 w_k1_AXP_s100 + 1.28304 w_k1_DIS_s100
                                 + 1.14683 w_k1_FDX_s100
                                 + 1.02058 w_k1_IBM_s100
                                 + 1.00461 w_k1_INTC_s100
                                 + 1.30143 w_k1_KO_s100
                                 + 0.842086 w_k1_MAT_s100
                                 + 1.22529 w_k1_MSFT_s100
                                 + 1.13789 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.936423 w_k1_ADM_s101
                                 + 0.90025 w_k1_AXP_s101
                                 + 0.883598 w_k1_DIS_s101
                                 + 0.936651 w_k1_FDX_s101
                                 + 0.83367 w_k1_IBM_s101
                                 + 0.968603 w_k1_INTC_s101
                                 + 1.09983 w_k1_KO_s101
                                 + 0.988696 w_k1_MAT_s101
                                 + 0.930747 w_k1_MSFT_s101
                                 + 0.986109 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.0116 w_k1_ADM_s101
                                 + 0.755215 w_k1_AXP_s101
                                 + 1.00953 w_k1_DIS_s101
                                 + 0.864509 w_k1_FDX_s101
                                 + 1.09702 w_k1_IBM_s101
                                 + 0.917496 w_k1_INTC_s101
                                 + 1.18223 w_k1_KO_s101
                                 + 0.844596 w_k1_MAT_s101
                                 + 0.93586 w_k1_MSFT_s101
                                 + 1.09391 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.10395 w_k1_ADM_s101
                                 + 1.17213 w_k1_AXP_s101
                                 + 1.13244 w_k1_DIS_s101
                                 + 1.04032 w_k1_FDX_s101
                                 + 1.45772 w_k1_IBM_s101
                                 + 1.11365 w_k1_INTC_s101 + 1.3997 w_k1_KO_s101
                                 + 2.047 w_k1_MAT_s101 + 1.15619 w_k1_MSFT_s101
                                 + 1.15487 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.906173 w_k1_ADM_s101
                                 + 0.587219 w_k1_AXP_s101
                                 + 0.666473 w_k1_DIS_s101
                                 + 0.752206 w_k1_FDX_s101
                                 + 0.667364 w_k1_IBM_s101
                                 + 0.812907 w_k1_INTC_s101
                                 + 0.79603 w_k1_KO_s101
                                 + 0.471553 w_k1_MAT_s101
                                 + 0.849474 w_k1_MSFT_s101
                                 + 0.839846 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.760429 w_k1_ADM_s101
                                 + 0.970547 w_k1_AXP_s101
                                 + 0.890347 w_k1_DIS_s101
                                 + 0.875558 w_k1_FDX_s101
                                 + 1.1744 w_k1_IBM_s101
                                 + 0.956682 w_k1_INTC_s101
                                 + 1.11212 w_k1_KO_s101 + 1.56403 w_k1_MAT_s101
                                 + 0.854358 w_k1_MSFT_s101
                                 + 0.973127 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.31114 w_k1_ADM_s101
                                 + 1.14702 w_k1_AXP_s101
                                 + 1.32796 w_k1_DIS_s101
                                 + 1.21661 w_k1_FDX_s101
                                 + 1.14203 w_k1_IBM_s101
                                 + 1.46005 w_k1_INTC_s101
                                 + 1.52628 w_k1_KO_s101 + 1.09895 w_k1_MAT_s101
                                 + 1.31676 w_k1_MSFT_s101
                                 + 1.17828 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.37854 w_k1_ADM_s101
                                 + 1.08234 w_k1_AXP_s101
                                 + 1.25181 w_k1_DIS_s101
                                 + 1.24897 w_k1_FDX_s101
                                 + 0.837912 w_k1_IBM_s101
                                 + 1.35943 w_k1_INTC_s101 + 1.2296 w_k1_KO_s101
                                 + 0.923327 w_k1_MAT_s101
                                 + 1.29719 w_k1_MSFT_s101
                                 + 1.11183 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.48781 w_k1_ADM_s101
                                 + 1.53813 w_k1_AXP_s101
                                 + 2.17693 w_k1_DIS_s101
                                 + 1.43585 w_k1_FDX_s101
                                 + 1.39345 w_k1_IBM_s101
                                 + 1.60292 w_k1_INTC_s101 + 1.6449 w_k1_KO_s101
                                 + 1.54585 w_k1_MAT_s101
                                 + 1.61033 w_k1_MSFT_s101
                                 + 1.46052 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.895092 w_k1_ADM_s102
                                 + 1.19757 w_k1_AXP_s102
                                 + 0.734461 w_k1_DIS_s102
                                 + 0.893133 w_k1_FDX_s102
                                 + 1.11978 w_k1_IBM_s102
                                 + 1.05496 w_k1_INTC_s102
                                 + 0.827687 w_k1_KO_s102
                                 + 1.21833 w_k1_MAT_s102
                                 + 0.86541 w_k1_MSFT_s102
                                 + 0.942795 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.850958 w_k1_ADM_s102
                                 + 0.78464 w_k1_AXP_s102
                                 + 0.702586 w_k1_DIS_s102
                                 + 0.85882 w_k1_FDX_s102
                                 + 0.827553 w_k1_IBM_s102
                                 + 0.798102 w_k1_INTC_s102
                                 + 0.665987 w_k1_KO_s102
                                 + 0.63409 w_k1_MAT_s102
                                 + 0.88136 w_k1_MSFT_s102
                                 + 0.851396 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.94802 w_k1_ADM_s102
                                 + 1.42842 w_k1_AXP_s102
                                 + 1.33799 w_k1_DIS_s102
                                 + 1.30235 w_k1_FDX_s102
                                 + 1.01689 w_k1_IBM_s102
                                 + 1.22014 w_k1_INTC_s102
                                 + 1.03471 w_k1_KO_s102
                                 + 0.921099 w_k1_MAT_s102
                                 + 1.19011 w_k1_MSFT_s102
                                 + 1.04983 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.41485 w_k1_ADM_s102
                                 + 1.40038 w_k1_AXP_s102
                                 + 1.54822 w_k1_DIS_s102
                                 + 1.27703 w_k1_FDX_s102
                                 + 1.57324 w_k1_IBM_s102
                                 + 0.923617 w_k1_INTC_s102
                                 + 1.30962 w_k1_KO_s102 + 1.63 w_k1_MAT_s102
                                 + 1.12002 w_k1_MSFT_s102
                                 + 1.29196 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.739199 w_k1_ADM_s102
                                 + 0.761525 w_k1_AXP_s102
                                 + 0.655548 w_k1_DIS_s102
                                 + 0.838809 w_k1_FDX_s102
                                 + 0.654426 w_k1_IBM_s102
                                 + 1.10234 w_k1_INTC_s102
                                 + 0.734142 w_k1_KO_s102
                                 + 0.694995 w_k1_MAT_s102
                                 + 0.870321 w_k1_MSFT_s102
                                 + 0.924729 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.00037 w_k1_ADM_s102
                                 + 0.97939 w_k1_AXP_s102
                                 + 1.03814 w_k1_DIS_s102
                                 + 0.926201 w_k1_FDX_s102
                                 + 1.12414 w_k1_IBM_s102
                                 + 1.01404 w_k1_INTC_s102
                                 + 1.03928 w_k1_KO_s102
                                 + 0.954484 w_k1_MAT_s102
                                 + 1.1143 w_k1_MSFT_s102 + 1.0161 w_k1_PG_s102
                                 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.243 w_k1_ADM_s102
                                 + 1.08799 w_k1_AXP_s102
                                 + 0.911495 w_k1_DIS_s102
                                 + 1.07412 w_k1_FDX_s102
                                 + 0.819061 w_k1_IBM_s102
                                 + 0.80204 w_k1_INTC_s102
                                 + 0.78219 w_k1_KO_s102
                                 + 0.969425 w_k1_MAT_s102
                                 + 0.992306 w_k1_MSFT_s102
                                 + 0.89675 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.818712 w_k1_ADM_s102
                                 + 0.826261 w_k1_AXP_s102
                                 + 0.802025 w_k1_DIS_s102
                                 + 0.847483 w_k1_FDX_s102
                                 + 1.34999 w_k1_IBM_s102
                                 + 0.863706 w_k1_INTC_s102
                                 + 0.920597 w_k1_KO_s102
                                 + 0.91048 w_k1_MAT_s102
                                 + 0.765719 w_k1_MSFT_s102
                                 + 0.957125 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.825603 w_k1_ADM_s103
                                 + 0.675621 w_k1_AXP_s103
                                 + 0.601749 w_k1_DIS_s103
                                 + 0.747374 w_k1_FDX_s103
                                 + 0.82714 w_k1_IBM_s103
                                 + 0.777019 w_k1_INTC_s103
                                 + 0.855345 w_k1_KO_s103
                                 + 0.620536 w_k1_MAT_s103
                                 + 0.757131 w_k1_MSFT_s103
                                 + 0.805025 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.14992 w_k1_ADM_s103
                                 + 1.85369 w_k1_AXP_s103
                                 + 1.59146 w_k1_DIS_s103
                                 + 1.29171 w_k1_FDX_s103
                                 + 1.60606 w_k1_IBM_s103
                                 + 1.33046 w_k1_INTC_s103
                                 + 1.42952 w_k1_KO_s103 + 1.89443 w_k1_MAT_s103
                                 + 1.12569 w_k1_MSFT_s103 + 1.1951 w_k1_PG_s103
                                 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.982263 w_k1_ADM_s103
                                 + 0.911373 w_k1_AXP_s103
                                 + 1.09493 w_k1_DIS_s103
                                 + 0.98663 w_k1_FDX_s103
                                 + 1.20814 w_k1_IBM_s103
                                 + 1.05544 w_k1_INTC_s103
                                 + 1.12283 w_k1_KO_s103
                                 + 0.763681 w_k1_MAT_s103
                                 + 0.961343 w_k1_MSFT_s103
                                 + 1.00269 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.08352 w_k1_ADM_s103
                                 + 1.36368 w_k1_AXP_s103
                                 + 1.48273 w_k1_DIS_s103
                                 + 1.32458 w_k1_FDX_s103
                                 + 1.12889 w_k1_IBM_s103
                                 + 1.4401 w_k1_INTC_s103 + 1.32578 w_k1_KO_s103
                                 + 1.22059 w_k1_MAT_s103
                                 + 1.20638 w_k1_MSFT_s103
                                 + 1.15524 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.2197 w_k1_ADM_s103
                                 + 1.32407 w_k1_AXP_s103 + 1.2135 w_k1_DIS_s103
                                 + 1.08173 w_k1_FDX_s103 + 1.4951 w_k1_IBM_s103
                                 + 1.10393 w_k1_INTC_s103
                                 + 1.42986 w_k1_KO_s103 + 1.5743 w_k1_MAT_s103
                                 + 0.949489 w_k1_MSFT_s103
                                 + 1.09466 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.931587 w_k1_ADM_s103
                                 + 1.09449 w_k1_AXP_s103
                                 + 0.77039 w_k1_DIS_s103
                                 + 0.901531 w_k1_FDX_s103
                                 + 0.979027 w_k1_IBM_s103
                                 + 1.11194 w_k1_INTC_s103
                                 + 1.06252 w_k1_KO_s103 + 1.05813 w_k1_MAT_s103
                                 + 0.882105 w_k1_MSFT_s103
                                 + 1.03676 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.16676 w_k1_ADM_s103
                                 + 1.43201 w_k1_AXP_s103
                                 + 1.00525 w_k1_DIS_s103
                                 + 1.07574 w_k1_FDX_s103
                                 + 1.08363 w_k1_IBM_s103
                                 + 1.01722 w_k1_INTC_s103
                                 + 0.954121 w_k1_KO_s103
                                 + 1.35475 w_k1_MAT_s103
                                 + 0.971932 w_k1_MSFT_s103
                                 + 1.05665 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.16931 w_k1_ADM_s103
                                 + 0.855244 w_k1_AXP_s103
                                 + 1.06445 w_k1_DIS_s103
                                 + 1.07239 w_k1_FDX_s103
                                 + 0.783619 w_k1_IBM_s103
                                 + 0.998124 w_k1_INTC_s103
                                 + 1.04602 w_k1_KO_s103
                                 + 0.644126 w_k1_MAT_s103
                                 + 1.11287 w_k1_MSFT_s103
                                 + 0.980807 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.839912 w_k1_ADM_s104
                                 + 1.27407 w_k1_AXP_s104
                                 + 1.02535 w_k1_DIS_s104
                                 + 1.04879 w_k1_FDX_s104
                                 + 1.43702 w_k1_IBM_s104
                                 + 1.05572 w_k1_INTC_s104
                                 + 1.36069 w_k1_KO_s104 + 1.28258 w_k1_MAT_s104
                                 + 1.10908 w_k1_MSFT_s104
                                 + 1.21908 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.02715 w_k1_ADM_s104
                                 + 1.40333 w_k1_AXP_s104
                                 + 1.51743 w_k1_DIS_s104
                                 + 1.35553 w_k1_FDX_s104
                                 + 1.44929 w_k1_IBM_s104
                                 + 1.52651 w_k1_INTC_s104
                                 + 1.43355 w_k1_KO_s104 + 1.24985 w_k1_MAT_s104
                                 + 1.44573 w_k1_MSFT_s104
                                 + 1.18413 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.17366 w_k1_ADM_s104
                                 + 1.33999 w_k1_AXP_s104
                                 + 0.855277 w_k1_DIS_s104
                                 + 1.12855 w_k1_FDX_s104
                                 + 1.05249 w_k1_IBM_s104
                                 + 1.32712 w_k1_INTC_s104
                                 + 1.07306 w_k1_KO_s104 + 1.10424 w_k1_MAT_s104
                                 + 0.976219 w_k1_MSFT_s104
                                 + 1.1149 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.1558 w_k1_ADM_s104
                                 + 1.1821 w_k1_AXP_s104 + 1.2578 w_k1_DIS_s104
                                 + 1.2647 w_k1_FDX_s104
                                 + 0.891246 w_k1_IBM_s104
                                 + 1.04913 w_k1_INTC_s104
                                 + 1.12627 w_k1_KO_s104
                                 + 0.761695 w_k1_MAT_s104
                                 + 1.21225 w_k1_MSFT_s104
                                 + 1.09879 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.85425 w_k1_ADM_s104
                                 + 0.935605 w_k1_AXP_s104
                                 + 0.738651 w_k1_DIS_s104
                                 + 0.866381 w_k1_FDX_s104
                                 + 0.832437 w_k1_IBM_s104
                                 + 0.926754 w_k1_INTC_s104
                                 + 0.813532 w_k1_KO_s104
                                 + 0.700774 w_k1_MAT_s104
                                 + 0.890554 w_k1_MSFT_s104
                                 + 0.913364 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.766562 w_k1_ADM_s104
                                 + 0.669986 w_k1_AXP_s104
                                 + 0.693126 w_k1_DIS_s104
                                 + 0.748379 w_k1_FDX_s104
                                 + 0.894086 w_k1_IBM_s104
                                 + 0.864285 w_k1_INTC_s104
                                 + 0.84809 w_k1_KO_s104
                                 + 0.607538 w_k1_MAT_s104
                                 + 0.841133 w_k1_MSFT_s104
                                 + 0.783745 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.13318 w_k1_ADM_s104
                                 + 1.65172 w_k1_AXP_s104
                                 + 1.30686 w_k1_DIS_s104
                                 + 1.52816 w_k1_FDX_s104
                                 + 0.909673 w_k1_IBM_s104
                                 + 1.45695 w_k1_INTC_s104
                                 + 1.08525 w_k1_KO_s104 + 1.38731 w_k1_MAT_s104
                                 + 1.28955 w_k1_MSFT_s104
                                 + 1.15602 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.815322 w_k1_ADM_s104
                                 + 0.853458 w_k1_AXP_s104
                                 + 1.14372 w_k1_DIS_s104
                                 + 1.00432 w_k1_FDX_s104
                                 + 1.00937 w_k1_IBM_s104
                                 + 1.18701 w_k1_INTC_s104
                                 + 1.05724 w_k1_KO_s104
                                 + 0.613958 w_k1_MAT_s104
                                 + 1.08018 w_k1_MSFT_s104
                                 + 0.93025 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.784746 w_k1_ADM_s105
                                 + 0.922239 w_k1_AXP_s105
                                 + 0.633871 w_k1_DIS_s105
                                 + 0.736458 w_k1_FDX_s105
                                 + 1.45233 w_k1_IBM_s105
                                 + 0.910158 w_k1_INTC_s105
                                 + 1.04853 w_k1_KO_s105 + 1.76854 w_k1_MAT_s105
                                 + 0.836865 w_k1_MSFT_s105
                                 + 1.02206 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.51577 w_k1_ADM_s105
                                 + 1.18382 w_k1_AXP_s105
                                 + 1.33106 w_k1_DIS_s105
                                 + 1.27771 w_k1_FDX_s105
                                 + 0.88423 w_k1_IBM_s105
                                 + 1.3828 w_k1_INTC_s105 + 1.21519 w_k1_KO_s105
                                 + 0.970079 w_k1_MAT_s105
                                 + 1.54411 w_k1_MSFT_s105
                                 + 1.14519 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.28978 w_k1_ADM_s105
                                 + 1.08795 w_k1_AXP_s105
                                 + 0.974203 w_k1_DIS_s105
                                 + 0.962001 w_k1_FDX_s105
                                 + 1.06771 w_k1_IBM_s105
                                 + 1.13791 w_k1_INTC_s105
                                 + 1.06763 w_k1_KO_s105 + 1.1991 w_k1_MAT_s105
                                 + 1.17533 w_k1_MSFT_s105
                                 + 1.10696 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.04477 w_k1_ADM_s105
                                 + 1.37333 w_k1_AXP_s105
                                 + 1.59121 w_k1_DIS_s105
                                 + 1.22078 w_k1_FDX_s105
                                 + 1.55337 w_k1_IBM_s105
                                 + 1.47881 w_k1_INTC_s105
                                 + 1.60027 w_k1_KO_s105 + 1.44016 w_k1_MAT_s105
                                 + 1.45387 w_k1_MSFT_s105
                                 + 1.03653 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.692757 w_k1_ADM_s105
                                 + 0.637244 w_k1_AXP_s105
                                 + 0.764582 w_k1_DIS_s105
                                 + 0.777125 w_k1_FDX_s105
                                 + 1.18774 w_k1_IBM_s105
                                 + 0.790656 w_k1_INTC_s105
                                 + 0.991962 w_k1_KO_s105 + 1.3183 w_k1_MAT_s105
                                 + 0.957467 w_k1_MSFT_s105
                                 + 0.842085 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.803129 w_k1_ADM_s105
                                 + 0.806338 w_k1_AXP_s105
                                 + 0.860437 w_k1_DIS_s105
                                 + 0.971432 w_k1_FDX_s105
                                 + 0.631671 w_k1_IBM_s105
                                 + 1.00639 w_k1_INTC_s105
                                 + 0.760426 w_k1_KO_s105
                                 + 0.698247 w_k1_MAT_s105
                                 + 0.998558 w_k1_MSFT_s105
                                 + 0.966762 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.695428 w_k1_ADM_s105
                                 + 0.595901 w_k1_AXP_s105
                                 + 0.528243 w_k1_DIS_s105
                                 + 0.73006 w_k1_FDX_s105
                                 + 0.657128 w_k1_IBM_s105
                                 + 0.745465 w_k1_INTC_s105
                                 + 0.701932 w_k1_KO_s105
                                 + 0.633936 w_k1_MAT_s105
                                 + 0.88809 w_k1_MSFT_s105
                                 + 0.762661 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.845272 w_k1_ADM_s105
                                 + 0.748668 w_k1_AXP_s105
                                 + 0.777387 w_k1_DIS_s105
                                 + 0.756848 w_k1_FDX_s105
                                 + 0.791332 w_k1_IBM_s105
                                 + 0.903986 w_k1_INTC_s105
                                 + 0.913296 w_k1_KO_s105
                                 + 0.732444 w_k1_MAT_s105
                                 + 0.934604 w_k1_MSFT_s105
                                 + 0.902761 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.08169 w_k1_ADM_s106
                                 + 1.49067 w_k1_AXP_s106
                                 + 1.07748 w_k1_DIS_s106
                                 + 1.01511 w_k1_FDX_s106
                                 + 1.25264 w_k1_IBM_s106
                                 + 0.891412 w_k1_INTC_s106
                                 + 1.16009 w_k1_KO_s106 + 1.77381 w_k1_MAT_s106
                                 + 0.998786 w_k1_MSFT_s106
                                 + 1.10295 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.09466 w_k1_ADM_s106
                                 + 1.18226 w_k1_AXP_s106
                                 + 1.31237 w_k1_DIS_s106
                                 + 1.07779 w_k1_FDX_s106
                                 + 1.42066 w_k1_IBM_s106
                                 + 1.09652 w_k1_INTC_s106
                                 + 1.35232 w_k1_KO_s106 + 1.47821 w_k1_MAT_s106
                                 + 1.14396 w_k1_MSFT_s106
                                 + 1.17868 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.946655 w_k1_ADM_s106
                                 + 0.708597 w_k1_AXP_s106
                                 + 0.705991 w_k1_DIS_s106
                                 + 0.780545 w_k1_FDX_s106
                                 + 0.965254 w_k1_IBM_s106
                                 + 0.743 w_k1_INTC_s106 + 0.874854 w_k1_KO_s106
                                 + 1.00722 w_k1_MAT_s106
                                 + 0.81596 w_k1_MSFT_s106
                                 + 0.908964 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.36102 w_k1_ADM_s106
                                 + 1.05693 w_k1_AXP_s106
                                 + 0.982317 w_k1_DIS_s106
                                 + 1.12215 w_k1_FDX_s106
                                 + 0.86684 w_k1_IBM_s106
                                 + 0.94992 w_k1_INTC_s106
                                 + 0.935881 w_k1_KO_s106
                                 + 0.836429 w_k1_MAT_s106
                                 + 1.02793 w_k1_MSFT_s106
                                 + 1.05078 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.840908 w_k1_ADM_s106
                                 + 0.631228 w_k1_AXP_s106
                                 + 0.530714 w_k1_DIS_s106
                                 + 0.682079 w_k1_FDX_s106
                                 + 0.666574 w_k1_IBM_s106
                                 + 0.635779 w_k1_INTC_s106
                                 + 0.679213 w_k1_KO_s106
                                 + 0.522477 w_k1_MAT_s106
                                 + 0.625844 w_k1_MSFT_s106
                                 + 0.794395 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.15934 w_k1_ADM_s106
                                 + 1.40597 w_k1_AXP_s106
                                 + 1.81585 w_k1_DIS_s106
                                 + 1.35946 w_k1_FDX_s106
                                 + 1.65593 w_k1_IBM_s106
                                 + 1.24999 w_k1_INTC_s106
                                 + 1.52218 w_k1_KO_s106 + 1.86516 w_k1_MAT_s106
                                 + 1.37179 w_k1_MSFT_s106
                                 + 1.38908 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.37534 w_k1_ADM_s106
                                 + 1.31834 w_k1_AXP_s106
                                 + 1.57618 w_k1_DIS_s106
                                 + 1.48355 w_k1_FDX_s106
                                 + 1.03776 w_k1_IBM_s106
                                 + 1.32147 w_k1_INTC_s106
                                 + 1.34408 w_k1_KO_s106 + 1.11564 w_k1_MAT_s106
                                 + 1.40696 w_k1_MSFT_s106
                                 + 1.20239 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.971126 w_k1_ADM_s106
                                 + 0.833356 w_k1_AXP_s106
                                 + 0.973385 w_k1_DIS_s106
                                 + 0.894713 w_k1_FDX_s106
                                 + 1.11013 w_k1_IBM_s106 + 0.932 w_k1_INTC_s106
                                 + 1.08853 w_k1_KO_s106
                                 + 0.951489 w_k1_MAT_s106
                                 + 0.971362 w_k1_MSFT_s106
                                 + 0.982976 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.43107 w_k1_ADM_s107
                                 + 1.46131 w_k1_AXP_s107
                                 + 1.42989 w_k1_DIS_s107
                                 + 1.24478 w_k1_FDX_s107
                                 + 1.53702 w_k1_IBM_s107
                                 + 1.14051 w_k1_INTC_s107
                                 + 1.30403 w_k1_KO_s107 + 1.79653 w_k1_MAT_s107
                                 + 1.20068 w_k1_MSFT_s107
                                 + 1.08307 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.863269 w_k1_ADM_s107
                                 + 0.565852 w_k1_AXP_s107
                                 + 0.529763 w_k1_DIS_s107
                                 + 0.730166 w_k1_FDX_s107
                                 + 0.615939 w_k1_IBM_s107
                                 + 0.65414 w_k1_INTC_s107
                                 + 0.602966 w_k1_KO_s107
                                 + 0.487751 w_k1_MAT_s107
                                 + 0.771808 w_k1_MSFT_s107
                                 + 0.697713 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.13665 w_k1_ADM_s107
                                 + 0.937579 w_k1_AXP_s107
                                 + 1.0058 w_k1_DIS_s107 + 0.89859 w_k1_FDX_s107
                                 + 1.283 w_k1_IBM_s107
                                 + 0.923606 w_k1_INTC_s107
                                 + 1.20175 w_k1_KO_s107 + 1.4754 w_k1_MAT_s107
                                 + 1.06006 w_k1_MSFT_s107
                                 + 1.02762 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.15881 w_k1_ADM_s107
                                 + 0.890887 w_k1_AXP_s107
                                 + 0.866364 w_k1_DIS_s107
                                 + 0.996771 w_k1_FDX_s107
                                 + 0.862684 w_k1_IBM_s107
                                 + 0.717358 w_k1_INTC_s107
                                 + 0.91044 w_k1_KO_s107
                                 + 0.896827 w_k1_MAT_s107
                                 + 0.935516 w_k1_MSFT_s107
                                 + 0.986608 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.28198 w_k1_ADM_s107
                                 + 1.01085 w_k1_AXP_s107
                                 + 0.876238 w_k1_DIS_s107
                                 + 1.06262 w_k1_FDX_s107
                                 + 0.698878 w_k1_IBM_s107
                                 + 0.930716 w_k1_INTC_s107
                                 + 0.816935 w_k1_KO_s107
                                 + 0.713912 w_k1_MAT_s107
                                 + 0.936088 w_k1_MSFT_s107
                                 + 0.915612 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.905354 w_k1_ADM_s107
                                 + 0.632784 w_k1_AXP_s107
                                 + 0.795245 w_k1_DIS_s107
                                 + 0.789157 w_k1_FDX_s107
                                 + 0.94402 w_k1_IBM_s107
                                 + 0.73925 w_k1_INTC_s107
                                 + 0.972763 w_k1_KO_s107
                                 + 1.01374 w_k1_MAT_s107
                                 + 0.898249 w_k1_MSFT_s107
                                 + 0.82116 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.33273 w_k1_ADM_s107
                                 + 1.22389 w_k1_AXP_s107
                                 + 1.36519 w_k1_DIS_s107
                                 + 1.29605 w_k1_FDX_s107
                                 + 1.08894 w_k1_IBM_s107
                                 + 1.1985 w_k1_INTC_s107 + 1.09789 w_k1_KO_s107
                                 + 1.13832 w_k1_MAT_s107
                                 + 1.22954 w_k1_MSFT_s107
                                 + 1.05737 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.93104 w_k1_ADM_s107
                                 + 1.67306 w_k1_AXP_s107
                                 + 1.85796 w_k1_DIS_s107
                                 + 1.40632 w_k1_FDX_s107 + 2.0505 w_k1_IBM_s107
                                 + 0.929172 w_k1_INTC_s107
                                 + 1.61897 w_k1_KO_s107 + 2.21337 w_k1_MAT_s107
                                 + 1.26711 w_k1_MSFT_s107
                                 + 1.27775 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.01908 w_k1_ADM_s110
                                 + 0.672802 w_k1_AXP_s110
                                 + 0.865941 w_k1_DIS_s110
                                 + 0.993258 w_k1_FDX_s110
                                 + 0.734361 w_k1_IBM_s110
                                 + 0.69026 w_k1_INTC_s110
                                 + 0.836797 w_k1_KO_s110
                                 + 0.527146 w_k1_MAT_s110
                                 + 1.03671 w_k1_MSFT_s110
                                 + 0.956858 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.750559 w_k1_ADM_s110
                                 + 0.804112 w_k1_AXP_s110
                                 + 0.585554 w_k1_DIS_s110
                                 + 0.808623 w_k1_FDX_s110
                                 + 0.736933 w_k1_IBM_s110
                                 + 0.558063 w_k1_INTC_s110
                                 + 0.696187 w_k1_KO_s110
                                 + 0.756398 w_k1_MAT_s110
                                 + 0.725366 w_k1_MSFT_s110
                                 + 0.758193 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.14312 w_k1_ADM_s110
                                 + 1.2522 w_k1_AXP_s110 + 1.66474 w_k1_DIS_s110
                                 + 1.27626 w_k1_FDX_s110 + 1.4934 w_k1_IBM_s110
                                 + 1.10781 w_k1_INTC_s110
                                 + 1.46006 w_k1_KO_s110 + 1.32759 w_k1_MAT_s110
                                 + 1.36311 w_k1_MSFT_s110
                                 + 1.24324 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.83556 w_k1_ADM_s110
                                 + 0.668601 w_k1_AXP_s110
                                 + 0.767347 w_k1_DIS_s110
                                 + 0.837476 w_k1_FDX_s110
                                 + 1.02732 w_k1_IBM_s110
                                 + 0.599353 w_k1_INTC_s110
                                 + 0.883823 w_k1_KO_s110
                                 + 1.04322 w_k1_MAT_s110
                                 + 0.889351 w_k1_MSFT_s110
                                 + 0.895279 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.02565 w_k1_ADM_s110
                                 + 1.07556 w_k1_AXP_s110
                                 + 1.35214 w_k1_DIS_s110
                                 + 1.28928 w_k1_FDX_s110
                                 + 1.03627 w_k1_IBM_s110
                                 + 0.93935 w_k1_INTC_s110
                                 + 1.10158 w_k1_KO_s110
                                 + 0.899845 w_k1_MAT_s110
                                 + 1.24885 w_k1_MSFT_s110
                                 + 1.20389 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.790729 w_k1_ADM_s110
                                 + 0.947058 w_k1_AXP_s110
                                 + 1.02989 w_k1_DIS_s110
                                 + 0.862244 w_k1_FDX_s110
                                 + 1.36158 w_k1_IBM_s110
                                 + 0.755766 w_k1_INTC_s110
                                 + 1.10559 w_k1_KO_s110 + 1.16219 w_k1_MAT_s110
                                 + 0.96585 w_k1_MSFT_s110
                                 + 0.922842 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.05151 w_k1_ADM_s110
                                 + 0.918698 w_k1_AXP_s110
                                 + 1.09229 w_k1_DIS_s110
                                 + 1.16439 w_k1_FDX_s110
                                 + 0.833323 w_k1_IBM_s110
                                 + 0.717758 w_k1_INTC_s110
                                 + 0.924528 w_k1_KO_s110
                                 + 0.796918 w_k1_MAT_s110
                                 + 1.03584 w_k1_MSFT_s110
                                 + 0.950335 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.808011 w_k1_ADM_s110
                                 + 0.529208 w_k1_AXP_s110
                                 + 0.592992 w_k1_DIS_s110
                                 + 0.775926 w_k1_FDX_s110
                                 + 0.645117 w_k1_IBM_s110
                                 + 0.644344 w_k1_INTC_s110
                                 + 0.712094 w_k1_KO_s110
                                 + 0.491442 w_k1_MAT_s110
                                 + 0.854142 w_k1_MSFT_s110
                                 + 0.757259 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.718284 w_k1_ADM_s111
                                 + 0.753459 w_k1_AXP_s111
                                 + 0.681416 w_k1_DIS_s111
                                 + 0.775426 w_k1_FDX_s111
                                 + 0.640218 w_k1_IBM_s111
                                 + 0.97727 w_k1_INTC_s111
                                 + 0.703362 w_k1_KO_s111
                                 + 0.638348 w_k1_MAT_s111
                                 + 0.918964 w_k1_MSFT_s111
                                 + 0.781678 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.984263 w_k1_ADM_s111
                                 + 1.204 w_k1_AXP_s111 + 0.969051 w_k1_DIS_s111
                                 + 0.857173 w_k1_FDX_s111
                                 + 1.65632 w_k1_IBM_s111
                                 + 1.2843 w_k1_INTC_s111 + 1.52617 w_k1_KO_s111
                                 + 1.54912 w_k1_MAT_s111
                                 + 1.13764 w_k1_MSFT_s111
                                 + 1.15516 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.967059 w_k1_ADM_s111
                                 + 1.22742 w_k1_AXP_s111
                                 + 1.15789 w_k1_DIS_s111
                                 + 1.04469 w_k1_FDX_s111
                                 + 0.784208 w_k1_IBM_s111
                                 + 1.26211 w_k1_INTC_s111
                                 + 1.10219 w_k1_KO_s111 + 1.08444 w_k1_MAT_s111
                                 + 1.2588 w_k1_MSFT_s111 + 1.10153 w_k1_PG_s111
                                 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.807197 w_k1_ADM_s111
                                 + 0.961149 w_k1_AXP_s111
                                 + 0.752668 w_k1_DIS_s111
                                 + 0.812175 w_k1_FDX_s111
                                 + 0.893756 w_k1_IBM_s111
                                 + 1.09889 w_k1_INTC_s111
                                 + 0.867043 w_k1_KO_s111
                                 + 0.958839 w_k1_MAT_s111
                                 + 0.969944 w_k1_MSFT_s111
                                 + 0.873214 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.776752 w_k1_ADM_s111
                                 + 0.844025 w_k1_AXP_s111
                                 + 0.991322 w_k1_DIS_s111
                                 + 0.825209 w_k1_FDX_s111
                                 + 1.19765 w_k1_IBM_s111
                                 + 1.0753 w_k1_INTC_s111 + 1.08811 w_k1_KO_s111
                                 + 1.17371 w_k1_MAT_s111
                                 + 1.08156 w_k1_MSFT_s111
                                 + 1.01986 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.720429 w_k1_ADM_s111
                                 + 0.650339 w_k1_AXP_s111
                                 + 0.803016 w_k1_DIS_s111
                                 + 0.690006 w_k1_FDX_s111
                                 + 0.88247 w_k1_IBM_s111
                                 + 0.827084 w_k1_INTC_s111
                                 + 0.931882 w_k1_KO_s111
                                 + 0.484052 w_k1_MAT_s111
                                 + 0.85245 w_k1_MSFT_s111
                                 + 0.809749 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.29152 w_k1_ADM_s111
                                 + 1.25433 w_k1_AXP_s111
                                 + 1.73958 w_k1_DIS_s111
                                 + 1.27708 w_k1_FDX_s111
                                 + 1.38559 w_k1_IBM_s111
                                 + 1.25846 w_k1_INTC_s111
                                 + 1.51755 w_k1_KO_s111 + 1.50386 w_k1_MAT_s111
                                 + 1.46133 w_k1_MSFT_s111
                                 + 1.41023 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.11947 w_k1_ADM_s111
                                 + 1.32936 w_k1_AXP_s111
                                 + 1.62332 w_k1_DIS_s111
                                 + 1.33329 w_k1_FDX_s111
                                 + 0.934132 w_k1_IBM_s111
                                 + 1.53128 w_k1_INTC_s111
                                 + 1.14485 w_k1_KO_s111 + 1.0107 w_k1_MAT_s111
                                 + 1.46258 w_k1_MSFT_s111
                                 + 0.996579 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.935827 w_k1_ADM_s112
                                 + 1.03072 w_k1_AXP_s112 + 1.0341 w_k1_DIS_s112
                                 + 0.903834 w_k1_FDX_s112
                                 + 1.31038 w_k1_IBM_s112
                                 + 1.00605 w_k1_INTC_s112 + 1.1237 w_k1_KO_s112
                                 + 1.17957 w_k1_MAT_s112
                                 + 1.04233 w_k1_MSFT_s112
                                 + 0.910814 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.16589 w_k1_ADM_s112
                                 + 1.23123 w_k1_AXP_s112
                                 + 1.34866 w_k1_DIS_s112
                                 + 1.17311 w_k1_FDX_s112
                                 + 1.27191 w_k1_IBM_s112
                                 + 1.28838 w_k1_INTC_s112
                                 + 1.35224 w_k1_KO_s112 + 1.37672 w_k1_MAT_s112
                                 + 1.22967 w_k1_MSFT_s112
                                 + 1.21261 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.04649 w_k1_ADM_s112
                                 + 1.10011 w_k1_AXP_s112
                                 + 1.31049 w_k1_DIS_s112
                                 + 1.22275 w_k1_FDX_s112
                                 + 0.754219 w_k1_IBM_s112
                                 + 1.06746 w_k1_INTC_s112
                                 + 0.949306 w_k1_KO_s112
                                 + 0.854817 w_k1_MAT_s112
                                 + 1.26631 w_k1_MSFT_s112
                                 + 1.06179 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.16921 w_k1_ADM_s112
                                 + 0.935476 w_k1_AXP_s112
                                 + 1.62012 w_k1_DIS_s112
                                 + 1.13948 w_k1_FDX_s112
                                 + 1.29097 w_k1_IBM_s112
                                 + 1.20914 w_k1_INTC_s112
                                 + 1.34106 w_k1_KO_s112 + 0.8774 w_k1_MAT_s112
                                 + 1.36694 w_k1_MSFT_s112
                                 + 1.02398 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.758618 w_k1_ADM_s112
                                 + 0.764627 w_k1_AXP_s112
                                 + 0.781194 w_k1_DIS_s112
                                 + 0.895057 w_k1_FDX_s112
                                 + 0.707831 w_k1_IBM_s112
                                 + 0.878281 w_k1_INTC_s112
                                 + 0.826679 w_k1_KO_s112
                                 + 0.695094 w_k1_MAT_s112
                                 + 0.978246 w_k1_MSFT_s112
                                 + 0.851213 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.894598 w_k1_ADM_s112
                                 + 1.1376 w_k1_AXP_s112 + 1.10863 w_k1_DIS_s112
                                 + 1.03867 w_k1_FDX_s112
                                 + 1.01338 w_k1_IBM_s112
                                 + 1.09961 w_k1_INTC_s112
                                 + 0.900546 w_k1_KO_s112
                                 + 1.12671 w_k1_MAT_s112
                                 + 1.1118 w_k1_MSFT_s112
                                 + 0.963565 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.685872 w_k1_ADM_s112
                                 + 0.79293 w_k1_AXP_s112
                                 + 0.89888 w_k1_DIS_s112
                                 + 0.844848 w_k1_FDX_s112
                                 + 0.927179 w_k1_IBM_s112
                                 + 0.851661 w_k1_INTC_s112
                                 + 0.924574 w_k1_KO_s112
                                 + 0.921944 w_k1_MAT_s112
                                 + 0.957505 w_k1_MSFT_s112
                                 + 0.886099 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.3282 w_k1_ADM_s112
                                 + 1.45895 w_k1_AXP_s112
                                 + 1.95245 w_k1_DIS_s112
                                 + 1.42711 w_k1_FDX_s112
                                 + 1.26228 w_k1_IBM_s112
                                 + 1.42988 w_k1_INTC_s112
                                 + 1.29359 w_k1_KO_s112 + 1.10709 w_k1_MAT_s112
                                 + 1.4846 w_k1_MSFT_s112 + 1.22469 w_k1_PG_s112
                                 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.903649 w_k1_ADM_s113
                                 + 0.731879 w_k1_AXP_s113
                                 + 1.15243 w_k1_DIS_s113
                                 + 0.96409 w_k1_FDX_s113
                                 + 0.882055 w_k1_IBM_s113
                                 + 0.982926 w_k1_INTC_s113
                                 + 1.04917 w_k1_KO_s113
                                 + 0.678202 w_k1_MAT_s113
                                 + 1.03309 w_k1_MSFT_s113
                                 + 0.957882 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.830633 w_k1_ADM_s113
                                 + 1.05773 w_k1_AXP_s113
                                 + 1.37727 w_k1_DIS_s113 + 1.0154 w_k1_FDX_s113
                                 + 1.72033 w_k1_IBM_s113
                                 + 0.994648 w_k1_INTC_s113
                                 + 1.43611 w_k1_KO_s113 + 1.43322 w_k1_MAT_s113
                                 + 1.06935 w_k1_MSFT_s113
                                 + 1.04684 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.23796 w_k1_ADM_s113
                                 + 1.33865 w_k1_AXP_s113
                                 + 1.86822 w_k1_DIS_s113 + 1.316 w_k1_FDX_s113
                                 + 1.53495 w_k1_IBM_s113
                                 + 1.38267 w_k1_INTC_s113 + 1.3494 w_k1_KO_s113
                                 + 1.64785 w_k1_MAT_s113
                                 + 1.36516 w_k1_MSFT_s113
                                 + 1.18685 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.00637 w_k1_ADM_s113
                                 + 0.942932 w_k1_AXP_s113
                                 + 0.969831 w_k1_DIS_s113
                                 + 0.840571 w_k1_FDX_s113
                                 + 1.56543 w_k1_IBM_s113
                                 + 0.916821 w_k1_INTC_s113
                                 + 1.18255 w_k1_KO_s113 + 1.35948 w_k1_MAT_s113
                                 + 1.00519 w_k1_MSFT_s113
                                 + 0.969277 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.988286 w_k1_ADM_s113
                                 + 1.05151 w_k1_AXP_s113
                                 + 0.912305 w_k1_DIS_s113
                                 + 1.00192 w_k1_FDX_s113
                                 + 1.12596 w_k1_IBM_s113
                                 + 0.95455 w_k1_INTC_s113
                                 + 1.07184 w_k1_KO_s113 + 1.42177 w_k1_MAT_s113
                                 + 1.05164 w_k1_MSFT_s113
                                 + 0.970306 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.855758 w_k1_ADM_s113
                                 + 0.871745 w_k1_AXP_s113
                                 + 0.803147 w_k1_DIS_s113
                                 + 0.847764 w_k1_FDX_s113
                                 + 1.03303 w_k1_IBM_s113
                                 + 0.835974 w_k1_INTC_s113
                                 + 0.883426 w_k1_KO_s113
                                 + 0.800198 w_k1_MAT_s113
                                 + 0.948064 w_k1_MSFT_s113
                                 + 0.924958 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.819592 w_k1_ADM_s113
                                 + 0.734683 w_k1_AXP_s113
                                 + 0.70379 w_k1_DIS_s113
                                 + 0.859884 w_k1_FDX_s113
                                 + 0.683512 w_k1_IBM_s113
                                 + 0.909261 w_k1_INTC_s113
                                 + 0.717481 w_k1_KO_s113
                                 + 0.540326 w_k1_MAT_s113
                                 + 0.895597 w_k1_MSFT_s113
                                 + 0.724629 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.08739 w_k1_ADM_s113
                                 + 1.06055 w_k1_AXP_s113 + 1.3653 w_k1_DIS_s113
                                 + 1.22852 w_k1_FDX_s113
                                 + 1.02197 w_k1_IBM_s113
                                 + 1.10848 w_k1_INTC_s113
                                 + 0.991659 w_k1_KO_s113
                                 + 0.868743 w_k1_MAT_s113
                                 + 1.27816 w_k1_MSFT_s113
                                 + 1.07968 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.0158 w_k1_ADM_s114
                                 + 1.31541 w_k1_AXP_s114
                                 + 0.970794 w_k1_DIS_s114
                                 + 1.14028 w_k1_FDX_s114
                                 + 0.934151 w_k1_IBM_s114
                                 + 1.09961 w_k1_INTC_s114 + 1.0406 w_k1_KO_s114
                                 + 1.10948 w_k1_MAT_s114
                                 + 1.1213 w_k1_MSFT_s114
                                 + 0.983055 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.24379 w_k1_ADM_s114
                                 + 1.61196 w_k1_AXP_s114
                                 + 1.26153 w_k1_DIS_s114
                                 + 1.36445 w_k1_FDX_s114
                                 + 1.24487 w_k1_IBM_s114
                                 + 1.24935 w_k1_INTC_s114
                                 + 1.21183 w_k1_KO_s114 + 1.95545 w_k1_MAT_s114
                                 + 1.37623 w_k1_MSFT_s114
                                 + 1.15977 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.793427 w_k1_ADM_s114
                                 + 0.553226 w_k1_AXP_s114
                                 + 0.565998 w_k1_DIS_s114
                                 + 0.764282 w_k1_FDX_s114
                                 + 0.662672 w_k1_IBM_s114
                                 + 0.730341 w_k1_INTC_s114
                                 + 0.655535 w_k1_KO_s114
                                 + 0.614834 w_k1_MAT_s114
                                 + 0.889766 w_k1_MSFT_s114
                                 + 0.807895 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.841725 w_k1_ADM_s114
                                 + 0.941357 w_k1_AXP_s114
                                 + 0.723556 w_k1_DIS_s114
                                 + 0.821241 w_k1_FDX_s114
                                 + 1.11887 w_k1_IBM_s114
                                 + 0.923338 w_k1_INTC_s114
                                 + 0.917007 w_k1_KO_s114
                                 + 1.28438 w_k1_MAT_s114
                                 + 0.853998 w_k1_MSFT_s114
                                 + 0.930562 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.880299 w_k1_ADM_s114
                                 + 0.705974 w_k1_AXP_s114
                                 + 0.868381 w_k1_DIS_s114
                                 + 0.931735 w_k1_FDX_s114
                                 + 0.809814 w_k1_IBM_s114
                                 + 0.896562 w_k1_INTC_s114
                                 + 0.801521 w_k1_KO_s114
                                 + 0.580696 w_k1_MAT_s114
                                 + 0.953426 w_k1_MSFT_s114
                                 + 0.919701 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.13622 w_k1_ADM_s114
                                 + 1.01927 w_k1_AXP_s114
                                 + 1.19206 w_k1_DIS_s114
                                 + 1.17266 w_k1_FDX_s114
                                 + 1.16767 w_k1_IBM_s114
                                 + 1.17882 w_k1_INTC_s114
                                 + 1.12037 w_k1_KO_s114 + 1.14192 w_k1_MAT_s114
                                 + 1.14999 w_k1_MSFT_s114
                                 + 1.06539 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.08277 w_k1_ADM_s114
                                 + 1.20011 w_k1_AXP_s114
                                 + 1.46238 w_k1_DIS_s114 + 1.1122 w_k1_FDX_s114
                                 + 1.60351 w_k1_IBM_s114
                                 + 1.09586 w_k1_INTC_s114
                                 + 1.55637 w_k1_KO_s114 + 1.64445 w_k1_MAT_s114
                                 + 1.30191 w_k1_MSFT_s114 + 1.1357 w_k1_PG_s114
                                 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.01315 w_k1_ADM_s114
                                 + 1.02136 w_k1_AXP_s114
                                 + 0.788495 w_k1_DIS_s114
                                 + 0.883234 w_k1_FDX_s114
                                 + 1.59224 w_k1_IBM_s114
                                 + 0.940931 w_k1_INTC_s114
                                 + 1.09344 w_k1_KO_s114 + 1.60911 w_k1_MAT_s114
                                 + 0.967185 w_k1_MSFT_s114
                                 + 1.06498 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.00542 w_k1_ADM_s115
                                 + 1.05666 w_k1_AXP_s115
                                 + 0.827229 w_k1_DIS_s115
                                 + 0.911907 w_k1_FDX_s115
                                 + 1.08053 w_k1_IBM_s115
                                 + 1.15284 w_k1_INTC_s115
                                 + 0.940133 w_k1_KO_s115
                                 + 1.26433 w_k1_MAT_s115
                                 + 1.03917 w_k1_MSFT_s115
                                 + 0.980369 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.11604 w_k1_ADM_s115
                                 + 1.06227 w_k1_AXP_s115
                                 + 0.957075 w_k1_DIS_s115
                                 + 0.902764 w_k1_FDX_s115
                                 + 1.6317 w_k1_IBM_s115
                                 + 0.901113 w_k1_INTC_s115
                                 + 1.24012 w_k1_KO_s115 + 1.38982 w_k1_MAT_s115
                                 + 0.879208 w_k1_MSFT_s115
                                 + 1.04931 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.951069 w_k1_ADM_s115
                                 + 0.847176 w_k1_AXP_s115
                                 + 1.10562 w_k1_DIS_s115
                                 + 0.966208 w_k1_FDX_s115
                                 + 1.33074 w_k1_IBM_s115
                                 + 0.989349 w_k1_INTC_s115
                                 + 1.21414 w_k1_KO_s115 + 1.22301 w_k1_MAT_s115
                                 + 1.03918 w_k1_MSFT_s115
                                 + 1.14236 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.760119 w_k1_ADM_s115
                                 + 0.764887 w_k1_AXP_s115
                                 + 0.678142 w_k1_DIS_s115
                                 + 0.734657 w_k1_FDX_s115
                                 + 1.06392 w_k1_IBM_s115
                                 + 1.02306 w_k1_INTC_s115
                                 + 0.916981 w_k1_KO_s115
                                 + 1.04945 w_k1_MAT_s115
                                 + 0.923405 w_k1_MSFT_s115
                                 + 0.883814 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.20322 w_k1_ADM_s115
                                 + 0.936663 w_k1_AXP_s115
                                 + 0.942803 w_k1_DIS_s115
                                 + 1.06613 w_k1_FDX_s115
                                 + 0.778068 w_k1_IBM_s115
                                 + 1.05617 w_k1_INTC_s115
                                 + 0.914013 w_k1_KO_s115
                                 + 0.497149 w_k1_MAT_s115
                                 + 1.0323 w_k1_MSFT_s115
                                 + 0.996357 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.38243 w_k1_ADM_s115
                                 + 1.44284 w_k1_AXP_s115
                                 + 1.38474 w_k1_DIS_s115 + 1.2141 w_k1_FDX_s115
                                 + 1.30439 w_k1_IBM_s115
                                 + 1.49484 w_k1_INTC_s115
                                 + 1.24746 w_k1_KO_s115 + 1.14335 w_k1_MAT_s115
                                 + 1.32048 w_k1_MSFT_s115
                                 + 1.24502 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.903885 w_k1_ADM_s115
                                 + 0.731063 w_k1_AXP_s115
                                 + 0.6166 w_k1_DIS_s115
                                 + 0.730478 w_k1_FDX_s115
                                 + 0.745195 w_k1_IBM_s115
                                 + 0.861993 w_k1_INTC_s115
                                 + 0.781616 w_k1_KO_s115
                                 + 0.854457 w_k1_MAT_s115
                                 + 0.927438 w_k1_MSFT_s115
                                 + 0.883849 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.64827 w_k1_ADM_s115
                                 + 1.38731 w_k1_AXP_s115
                                 + 1.49062 w_k1_DIS_s115
                                 + 1.36574 w_k1_FDX_s115
                                 + 0.927171 w_k1_IBM_s115
                                 + 1.55652 w_k1_INTC_s115 + 1.1763 w_k1_KO_s115
                                 + 1.12186 w_k1_MAT_s115
                                 + 1.37922 w_k1_MSFT_s115
                                 + 1.29736 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.08342 w_k1_ADM_s116
                                 + 1.31657 w_k1_AXP_s116
                                 + 1.18156 w_k1_DIS_s116
                                 + 1.05068 w_k1_FDX_s116 + 1.6325 w_k1_IBM_s116
                                 + 1.16359 w_k1_INTC_s116 + 1.3561 w_k1_KO_s116
                                 + 1.88471 w_k1_MAT_s116
                                 + 1.12565 w_k1_MSFT_s116
                                 + 1.12704 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.946667 w_k1_ADM_s116
                                 + 0.866528 w_k1_AXP_s116
                                 + 0.782344 w_k1_DIS_s116
                                 + 1.09749 w_k1_FDX_s116
                                 + 0.560848 w_k1_IBM_s116
                                 + 1.07507 w_k1_INTC_s116
                                 + 0.790815 w_k1_KO_s116
                                 + 0.553698 w_k1_MAT_s116
                                 + 0.919779 w_k1_MSFT_s116
                                 + 0.929229 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.0342 w_k1_ADM_s116
                                 + 1.07704 w_k1_AXP_s116
                                 + 1.28011 w_k1_DIS_s116
                                 + 1.15404 w_k1_FDX_s116
                                 + 0.951627 w_k1_IBM_s116
                                 + 1.10315 w_k1_INTC_s116
                                 + 1.18574 w_k1_KO_s116
                                 + 0.948179 w_k1_MAT_s116
                                 + 1.15531 w_k1_MSFT_s116 + 1.1308 w_k1_PG_s116
                                 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.827639 w_k1_ADM_s116
                                 + 0.939064 w_k1_AXP_s116
                                 + 1.32287 w_k1_DIS_s116
                                 + 1.04763 w_k1_FDX_s116
                                 + 0.915495 w_k1_IBM_s116
                                 + 1.05602 w_k1_INTC_s116
                                 + 1.15884 w_k1_KO_s116 + 1.22259 w_k1_MAT_s116
                                 + 1.17217 w_k1_MSFT_s116
                                 + 1.03232 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.925357 w_k1_ADM_s116
                                 + 1.0004 w_k1_AXP_s116
                                 + 0.813611 w_k1_DIS_s116
                                 + 0.935888 w_k1_FDX_s116
                                 + 1.08109 w_k1_IBM_s116
                                 + 0.995338 w_k1_INTC_s116
                                 + 0.957204 w_k1_KO_s116
                                 + 1.42191 w_k1_MAT_s116
                                 + 0.885441 w_k1_MSFT_s116
                                 + 1.0195 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.774277 w_k1_ADM_s116
                                 + 0.682938 w_k1_AXP_s116
                                 + 0.673589 w_k1_DIS_s116
                                 + 0.834669 w_k1_FDX_s116
                                 + 0.665133 w_k1_IBM_s116
                                 + 0.92102 w_k1_INTC_s116
                                 + 0.683625 w_k1_KO_s116
                                 + 0.592245 w_k1_MAT_s116
                                 + 0.779922 w_k1_MSFT_s116
                                 + 0.841705 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.23552 w_k1_ADM_s116
                                 + 1.49107 w_k1_AXP_s116 + 1.3662 w_k1_DIS_s116
                                 + 1.40632 w_k1_FDX_s116
                                 + 1.19408 w_k1_IBM_s116
                                 + 1.44386 w_k1_INTC_s116
                                 + 1.09126 w_k1_KO_s116 + 1.38964 w_k1_MAT_s116
                                 + 1.18655 w_k1_MSFT_s116
                                 + 1.20928 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.873873 w_k1_ADM_s116
                                 + 0.9603 w_k1_AXP_s116
                                 + 0.781028 w_k1_DIS_s116
                                 + 0.820891 w_k1_FDX_s116
                                 + 1.25343 w_k1_IBM_s116
                                 + 0.969494 w_k1_INTC_s116
                                 + 1.09851 w_k1_KO_s116 + 1.63903 w_k1_MAT_s116
                                 + 0.8875 w_k1_MSFT_s116
                                 + 0.974884 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.884059 w_k1_ADM_s117
                                 + 0.71316 w_k1_AXP_s117
                                 + 0.688849 w_k1_DIS_s117
                                 + 0.782224 w_k1_FDX_s117
                                 + 0.745851 w_k1_IBM_s117
                                 + 0.795394 w_k1_INTC_s117
                                 + 0.652568 w_k1_KO_s117
                                 + 0.579716 w_k1_MAT_s117
                                 + 0.749351 w_k1_MSFT_s117
                                 + 0.857653 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.93099 w_k1_ADM_s117
                                 + 1.03621 w_k1_AXP_s117
                                 + 1.01293 w_k1_DIS_s117
                                 + 0.897229 w_k1_FDX_s117
                                 + 1.27801 w_k1_IBM_s117
                                 + 1.17137 w_k1_INTC_s117
                                 + 1.07108 w_k1_KO_s117 + 1.04577 w_k1_MAT_s117
                                 + 0.97391 w_k1_MSFT_s117
                                 + 1.00663 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.17961 w_k1_ADM_s117
                                 + 1.38152 w_k1_AXP_s117
                                 + 1.12228 w_k1_DIS_s117
                                 + 1.08095 w_k1_FDX_s117
                                 + 0.957035 w_k1_IBM_s117
                                 + 1.26013 w_k1_INTC_s117
                                 + 0.807772 w_k1_KO_s117
                                 + 0.74969 w_k1_MAT_s117
                                 + 0.96755 w_k1_MSFT_s117
                                 + 1.05454 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.3405 w_k1_ADM_s117
                                 + 1.20301 w_k1_AXP_s117
                                 + 1.18831 w_k1_DIS_s117
                                 + 1.30202 w_k1_FDX_s117
                                 + 0.690211 w_k1_IBM_s117
                                 + 1.1486 w_k1_INTC_s117 + 0.82046 w_k1_KO_s117
                                 + 0.518204 w_k1_MAT_s117
                                 + 1.06757 w_k1_MSFT_s117
                                 + 1.06764 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.24591 w_k1_ADM_s117
                                 + 1.29706 w_k1_AXP_s117
                                 + 1.62717 w_k1_DIS_s117
                                 + 1.35172 w_k1_FDX_s117
                                 + 1.26597 w_k1_IBM_s117 + 1.219 w_k1_INTC_s117
                                 + 1.28611 w_k1_KO_s117 + 1.31434 w_k1_MAT_s117
                                 + 1.22934 w_k1_MSFT_s117
                                 + 1.21113 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.885301 w_k1_ADM_s117
                                 + 0.981754 w_k1_AXP_s117
                                 + 1.07287 w_k1_DIS_s117
                                 + 1.05334 w_k1_FDX_s117
                                 + 0.942706 w_k1_IBM_s117
                                 + 1.11787 w_k1_INTC_s117
                                 + 0.84295 w_k1_KO_s117
                                 + 0.768791 w_k1_MAT_s117
                                 + 0.977781 w_k1_MSFT_s117
                                 + 1.1348 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.90233 w_k1_ADM_s117
                                 + 0.985953 w_k1_AXP_s117
                                 + 0.857712 w_k1_DIS_s117
                                 + 1.00431 w_k1_FDX_s117
                                 + 0.66482 w_k1_IBM_s117
                                 + 1.07051 w_k1_INTC_s117
                                 + 0.739673 w_k1_KO_s117
                                 + 0.743952 w_k1_MAT_s117
                                 + 0.922829 w_k1_MSFT_s117
                                 + 1.03404 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.992428 w_k1_ADM_s117
                                 + 1.09187 w_k1_AXP_s117
                                 + 0.675332 w_k1_DIS_s117
                                 + 0.875319 w_k1_FDX_s117
                                 + 0.707974 w_k1_IBM_s117
                                 + 0.8088 w_k1_INTC_s117
                                 + 0.663051 w_k1_KO_s117
                                 + 0.755658 w_k1_MAT_s117
                                 + 0.763888 w_k1_MSFT_s117
                                 + 0.878087 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0194408890914039 w_k1_ADM_s000
                                 + 0.0241918434572616 w_k1_ADM_s001
                                 + 0.0174557925969237 w_k1_ADM_s002
                                 + 0.0164009583127371 w_k1_ADM_s003
                                 + 0.0170132173574388 w_k1_ADM_s004
                                 + 0.041629557856825 w_k1_ADM_s005
                                 + 0.0351872524363877 w_k1_ADM_s006
                                 + 0.0266312192013377 w_k1_ADM_s007
                                 + 0.0146649570803928 w_k1_ADM_s010
                                 + 0.0170841236632679 w_k1_ADM_s011
                                 + 0.0276157515609564 w_k1_ADM_s012
                                 + 0.00950614036575047 w_k1_ADM_s013
                                 + 0.0135520658926093 w_k1_ADM_s014
                                 + 0.0158916268980186 w_k1_ADM_s015
                                 + 0.00918931989206963 w_k1_ADM_s016
                                 + 0.0202151568743477 w_k1_ADM_s017
                                 + 0.050201178146537 w_k1_ADM_s100
                                 + 0.0260895240658223 w_k1_ADM_s101
                                 + 0.0564689062505938 w_k1_ADM_s102
                                 + 0.0816375902532661 w_k1_ADM_s103
                                 + 0.0316380102790902 w_k1_ADM_s104
                                 + 0.0554271646111257 w_k1_ADM_s105
                                 + 0.0803408874080029 w_k1_ADM_s106
                                 + 0.0279358217233492 w_k1_ADM_s107
                                 + 0.0234169547482309 w_k1_ADM_s110
                                 + 0.0216290940295396 w_k1_ADM_s111
                                 + 0.0483353789997004 w_k1_ADM_s112
                                 + 0.0402736772655565 w_k1_ADM_s113
                                 + 0.0263961589035515 w_k1_ADM_s114
                                 + 0.0589172403410832 w_k1_ADM_s115
                                 + 0.0302701205764813 w_k1_ADM_s116
                                 + 0.0305049996632617 w_k1_ADM_s117
                                 + 0.0194467123864329 w_k1_AXP_s000
                                 + 0.0216995365060109 w_k1_AXP_s001
                                 + 0.0173203684259249 w_k1_AXP_s002
                                 + 0.0175161272114168 w_k1_AXP_s003
                                 + 0.0169269221718448 w_k1_AXP_s004
                                 + 0.0477432066800638 w_k1_AXP_s005
                                 + 0.0397504926803027 w_k1_AXP_s006
                                 + 0.0257464376391407 w_k1_AXP_s007
                                 + 0.0136797026778289 w_k1_AXP_s010
                                 + 0.0136478801777039 w_k1_AXP_s011
                                 + 0.0286846140313203 w_k1_AXP_s012
                                 + 0.0093402172276275 w_k1_AXP_s013
                                 + 0.0120474036154653 w_k1_AXP_s014
                                 + 0.0166917433153923 w_k1_AXP_s015
                                 + 0.0111220237657025 w_k1_AXP_s016
                                 + 0.0207620863576614 w_k1_AXP_s017
                                 + 0.0523423437475994 w_k1_AXP_s100
                                 + 0.0234770886717187 w_k1_AXP_s101
                                 + 0.0609956322115794 w_k1_AXP_s102
                                 + 0.0877001161238142 w_k1_AXP_s103
                                 + 0.0365310073074569 w_k1_AXP_s104
                                 + 0.0506315561750891 w_k1_AXP_s105
                                 + 0.0736787296398126 w_k1_AXP_s106
                                 + 0.0232363984888079 w_k1_AXP_s107
                                 + 0.0224287910688992 w_k1_AXP_s110
                                 + 0.0242389095174405 w_k1_AXP_s111
                                 + 0.0526568992027442 w_k1_AXP_s112
                                 + 0.0398926964685151 w_k1_AXP_s113
                                 + 0.027265952537318 w_k1_AXP_s114
                                 + 0.0544841539525413 w_k1_AXP_s115
                                 + 0.0318150109794544 w_k1_AXP_s116
                                 + 0.0318098647591926 w_k1_AXP_s117
                                 + 0.0217991836434524 w_k1_DIS_s000
                                 + 0.0213646782202304 w_k1_DIS_s001
                                 + 0.0174667471134017 w_k1_DIS_s002
                                 + 0.0160689118798876 w_k1_DIS_s003
                                 + 0.0167702491743178 w_k1_DIS_s004
                                 + 0.0433850621019935 w_k1_DIS_s005
                                 + 0.0378893399564532 w_k1_DIS_s006
                                 + 0.026709977474719 w_k1_DIS_s007
                                 + 0.0130440192858898 w_k1_DIS_s010
                                 + 0.0144358639221639 w_k1_DIS_s011
                                 + 0.0298596225195626 w_k1_DIS_s012
                                 + 0.0098261861649903 w_k1_DIS_s013
                                 + 0.0125831249602882 w_k1_DIS_s014
                                 + 0.0164628263667626 w_k1_DIS_s015
                                 + 0.0104695212072368 w_k1_DIS_s016
                                 + 0.0206297029758838 w_k1_DIS_s017
                                 + 0.054052795473437 w_k1_DIS_s100
                                 + 0.026253619109726 w_k1_DIS_s101
                                 + 0.0562987676336226 w_k1_DIS_s102
                                 + 0.0809339771418739 w_k1_DIS_s103
                                 + 0.0347810847526955 w_k1_DIS_s104
                                 + 0.0522962403914037 w_k1_DIS_s105
                                 + 0.0775142842257028 w_k1_DIS_s106
                                 + 0.0243186355176485 w_k1_DIS_s107
                                 + 0.0266351343092137 w_k1_DIS_s110
                                 + 0.0261189530431417 w_k1_DIS_s111
                                 + 0.0586933765798386 w_k1_DIS_s112
                                 + 0.0470496698357934 w_k1_DIS_s113
                                 + 0.0257848793619467 w_k1_DIS_s114
                                 + 0.0516371617996566 w_k1_DIS_s115
                                 + 0.031756356187387 w_k1_DIS_s116
                                 + 0.031944307354093 w_k1_DIS_s117
                                 + 0.0201460810825827 w_k1_FDX_s000
                                 + 0.020954401760239 w_k1_FDX_s001
                                 + 0.0179111627778742 w_k1_FDX_s002
                                 + 0.0158663617532355 w_k1_FDX_s003
                                 + 0.0162077363851984 w_k1_FDX_s004
                                 + 0.043818604392257 w_k1_FDX_s005
                                 + 0.0387445575360276 w_k1_FDX_s006
                                 + 0.0251270757861851 w_k1_FDX_s007
                                 + 0.0136431464747152 w_k1_FDX_s010
                                 + 0.0141430308419202 w_k1_FDX_s011
                                 + 0.0279567933408956 w_k1_FDX_s012
                                 + 0.00964684500529078 w_k1_FDX_s013
                                 + 0.0122051845843462 w_k1_FDX_s014
                                 + 0.0153997344372377 w_k1_FDX_s015
                                 + 0.0107537373822857 w_k1_FDX_s016
                                 + 0.0209820552453362 w_k1_FDX_s017
                                 + 0.0522894252292482 w_k1_FDX_s100
                                 + 0.02449095714834 w_k1_FDX_s101
                                 + 0.0577364174920086 w_k1_FDX_s102
                                 + 0.0807931896804217 w_k1_FDX_s103
                                 + 0.0358462334951937 w_k1_FDX_s104
                                 + 0.0512368178402606 w_k1_FDX_s105
                                 + 0.0748551194036623 w_k1_FDX_s106
                                 + 0.0241830588582521 w_k1_FDX_s107
                                 + 0.0259118641904466 w_k1_FDX_s110
                                 + 0.022660550140969 w_k1_FDX_s111
                                 + 0.0533963509108883 w_k1_FDX_s112
                                 + 0.0426937001283605 w_k1_FDX_s113
                                 + 0.0274789584677615 w_k1_FDX_s114
                                 + 0.052301735765175 w_k1_FDX_s115
                                 + 0.0328524143047522 w_k1_FDX_s116
                                 + 0.0309781570321983 w_k1_FDX_s117
                                 + 0.0203300381641631 w_k1_IBM_s000
                                 + 0.021341061145296 w_k1_IBM_s001
                                 + 0.0179554655320147 w_k1_IBM_s002
                                 + 0.0154654128629403 w_k1_IBM_s003
                                 + 0.0171233854814869 w_k1_IBM_s004
                                 + 0.0436406349266272 w_k1_IBM_s005
                                 + 0.0383432573650172 w_k1_IBM_s006
                                 + 0.0268936450785974 w_k1_IBM_s007
                                 + 0.0141306271868304 w_k1_IBM_s010
                                 + 0.0155004891169168 w_k1_IBM_s011
                                 + 0.0317911320906664 w_k1_IBM_s012
                                 + 0.00922624467391407 w_k1_IBM_s013
                                 + 0.0129015175988472 w_k1_IBM_s014
                                 + 0.015937836604769 w_k1_IBM_s015
                                 + 0.0109777412124531 w_k1_IBM_s016
                                 + 0.0204177527220914 w_k1_IBM_s017
                                 + 0.0534878688983083 w_k1_IBM_s100
                                 + 0.0243358768101266 w_k1_IBM_s101
                                 + 0.0580501490139876 w_k1_IBM_s102
                                 + 0.0864232287782785 w_k1_IBM_s103
                                 + 0.0350846247626251 w_k1_IBM_s104
                                 + 0.053337973537718 w_k1_IBM_s105
                                 + 0.0783728145824657 w_k1_IBM_s106
                                 + 0.0245499253334305 w_k1_IBM_s107
                                 + 0.0254194473412587 w_k1_IBM_s110
                                 + 0.0245350228224849 w_k1_IBM_s111
                                 + 0.0527594764281252 w_k1_IBM_s112
                                 + 0.0457494228546428 w_k1_IBM_s113
                                 + 0.0279865190050406 w_k1_IBM_s114
                                 + 0.0557068673038671 w_k1_IBM_s115
                                 + 0.0305927337972111 w_k1_IBM_s116
                                 + 0.0295482886589625 w_k1_IBM_s117
                                 + 0.021381081105379 w_k1_INTC_s000
                                 + 0.0181006763975539 w_k1_INTC_s001
                                 + 0.0186697723022284 w_k1_INTC_s002
                                 + 0.014803173141546 w_k1_INTC_s003
                                 + 0.0159490018110903 w_k1_INTC_s004
                                 + 0.0421260557175359 w_k1_INTC_s005
                                 + 0.0372925665315315 w_k1_INTC_s006
                                 + 0.0256159478635665 w_k1_INTC_s007
                                 + 0.0152551346000997 w_k1_INTC_s010
                                 + 0.0141237433091573 w_k1_INTC_s011
                                 + 0.0316973423340477 w_k1_INTC_s012
                                 + 0.00947678869322213 w_k1_INTC_s013
                                 + 0.0112558038497383 w_k1_INTC_s014
                                 + 0.0129090066344324 w_k1_INTC_s015
                                 + 0.0122928920957933 w_k1_INTC_s016
                                 + 0.0199626178755437 w_k1_INTC_s017
                                 + 0.0494752700056114 w_k1_INTC_s100
                                 + 0.0267199029976443 w_k1_INTC_s101
                                 + 0.0551501995071322 w_k1_INTC_s102
                                 + 0.0868195743643018 w_k1_INTC_s103
                                 + 0.0384928982375653 w_k1_INTC_s104
                                 + 0.0574381291877259 w_k1_INTC_s105
                                 + 0.0698032914428106 w_k1_INTC_s106
                                 + 0.0211585672302159 w_k1_INTC_s107
                                 + 0.0194661726358539 w_k1_INTC_s110
                                 + 0.0276461598246424 w_k1_INTC_s111
                                 + 0.0544329224071256 w_k1_INTC_s112
                                 + 0.0422456586347601 w_k1_INTC_s113
                                 + 0.0268872745146105 w_k1_INTC_s114
                                 + 0.0603447287055278 w_k1_INTC_s115
                                 + 0.0340892455217365 w_k1_INTC_s116
                                 + 0.0325375589405384 w_k1_INTC_s117
                                 + 0.019221201324727 w_k1_KO_s000
                                 + 0.0194542583108566 w_k1_KO_s001
                                 + 0.017799121200688 w_k1_KO_s002
                                 + 0.0155069192318688 w_k1_KO_s003
                                 + 0.0150534604683619 w_k1_KO_s004
                                 + 0.0375357345100908 w_k1_KO_s005
                                 + 0.04033043189349 w_k1_KO_s006
                                 + 0.0258737307596943 w_k1_KO_s007
                                 + 0.0133973958957217 w_k1_KO_s010
                                 + 0.0160102487083645 w_k1_KO_s011
                                 + 0.02930363914387 w_k1_KO_s012
                                 + 0.00905545317304365 w_k1_KO_s013
                                 + 0.0124162191588559 w_k1_KO_s014
                                 + 0.0155721502781287 w_k1_KO_s015
                                 + 0.0106128960256635 w_k1_KO_s016
                                 + 0.0194641260009363 w_k1_KO_s017
                                 + 0.0604410761393733 w_k1_KO_s100
                                 + 0.0290245938824006 w_k1_KO_s101
                                 + 0.0518063400028958 w_k1_KO_s102
                                 + 0.0896268209763161 w_k1_KO_s103
                                 + 0.0359350574731688 w_k1_KO_s104
                                 + 0.055840788822405 w_k1_KO_s105
                                 + 0.0788043905474324 w_k1_KO_s106
                                 + 0.0239492405914968 w_k1_KO_s107
                                 + 0.0249884180754861 w_k1_KO_s110
                                 + 0.026013618149512 w_k1_KO_s111
                                 + 0.0532527664582305 w_k1_KO_s112
                                 + 0.0429893879287504 w_k1_KO_s113
                                 + 0.0267991051762856 w_k1_KO_s114
                                 + 0.0538524456721577 w_k1_KO_s115
                                 + 0.0316475205627584 w_k1_KO_s116
                                 + 0.0275798492671357 w_k1_KO_s117
                                 + 0.0181418485786484 w_k1_MAT_s000
                                 + 0.0247271822954697 w_k1_MAT_s001
                                 + 0.0162036591879146 w_k1_MAT_s002
                                 + 0.0174990760734244 w_k1_MAT_s003
                                 + 0.0214959076268014 w_k1_MAT_s004
                                 + 0.0500599694221204 w_k1_MAT_s005
                                 + 0.043302362430901 w_k1_MAT_s006
                                 + 0.0268668303955656 w_k1_MAT_s007
                                 + 0.011665853661363 w_k1_MAT_s010
                                 + 0.0184047645993527 w_k1_MAT_s011
                                 + 0.0304221548973397 w_k1_MAT_s012
                                 + 0.00954421717880217 w_k1_MAT_s013
                                 + 0.0104481900719104 w_k1_MAT_s014
                                 + 0.0171516050554864 w_k1_MAT_s015
                                 + 0.00870670412844268 w_k1_MAT_s016
                                 + 0.0185317238872968 w_k1_MAT_s017
                                 + 0.053907101763482 w_k1_MAT_s100
                                 + 0.0266572779031128 w_k1_MAT_s101
                                 + 0.0546629677202276 w_k1_MAT_s102
                                 + 0.0832419049343886 w_k1_MAT_s103
                                 + 0.0294300299812075 w_k1_MAT_s104
                                 + 0.0571349429506903 w_k1_MAT_s105
                                 + 0.0807481473241877 w_k1_MAT_s106
                                 + 0.0262729810968018 w_k1_MAT_s107
                                 + 0.0226945499550252 w_k1_MAT_s110
                                 + 0.0253743812184934 w_k1_MAT_s111
                                 + 0.0522016902362099 w_k1_MAT_s112
                                 + 0.0413803158610033 w_k1_MAT_s113
                                 + 0.0302498745471762 w_k1_MAT_s114
                                 + 0.0528540568106411 w_k1_MAT_s115
                                 + 0.0341180647439288 w_k1_MAT_s116
                                 + 0.0265358152581824 w_k1_MAT_s117
                                 + 0.0209733212446682 w_k1_MSFT_s000
                                 + 0.0195944404547318 w_k1_MSFT_s001
                                 + 0.0191877450726557 w_k1_MSFT_s002
                                 + 0.0158799960684112 w_k1_MSFT_s003
                                 + 0.0160865274925131 w_k1_MSFT_s004
                                 + 0.0414594197280816 w_k1_MSFT_s005
                                 + 0.0374712281966611 w_k1_MSFT_s006
                                 + 0.0249484269679644 w_k1_MSFT_s007
                                 + 0.0139757701566498 w_k1_MSFT_s010
                                 + 0.0150355247697085 w_k1_MSFT_s011
                                 + 0.0286756069105063 w_k1_MSFT_s012
                                 + 0.00943254696677702 w_k1_MSFT_s013
                                 + 0.0123202071280389 w_k1_MSFT_s014
                                 + 0.0148948426544974 w_k1_MSFT_s015
                                 + 0.0105183290435629 w_k1_MSFT_s016
                                 + 0.0193667001703728 w_k1_MSFT_s017
                                 + 0.0536604525636081 w_k1_MSFT_s100
                                 + 0.026000095524625 w_k1_MSFT_s101
                                 + 0.0566667708830072 w_k1_MSFT_s102
                                 + 0.0770948110291026 w_k1_MSFT_s103
                                 + 0.0364023984000691 w_k1_MSFT_s104
                                 + 0.0609513812048303 w_k1_MSFT_s105
                                 + 0.0746112429622667 w_k1_MSFT_s106
                                 + 0.0240190590060851 w_k1_MSFT_s107
                                 + 0.0261046972596614 w_k1_MSFT_s110
                                 + 0.0273282826559798 w_k1_MSFT_s111
                                 + 0.0582770053723422 w_k1_MSFT_s112
                                 + 0.0453484983830517 w_k1_MSFT_s113
                                 + 0.0286171108547859 w_k1_MSFT_s114
                                 + 0.0566371196976753 w_k1_MSFT_s115
                                 + 0.0315358419180504 w_k1_MSFT_s116
                                 + 0.0291891766911952 w_k1_MSFT_s117
                                 + 0.0203928488769732 w_k1_PG_s000
                                 + 0.0212707091971989 w_k1_PG_s001
                                 + 0.0187333449127369 w_k1_PG_s002
                                 + 0.0161799036255857 w_k1_PG_s003
                                 + 0.0150959911199093 w_k1_PG_s004
                                 + 0.0432126080488633 w_k1_PG_s005
                                 + 0.0384475910623657 w_k1_PG_s006
                                 + 0.0252942297249231 w_k1_PG_s007
                                 + 0.0132282156537321 w_k1_PG_s010
                                 + 0.0158114795820479 w_k1_PG_s011
                                 + 0.0282129384341187 w_k1_PG_s012
                                 + 0.0082650535544929 w_k1_PG_s013
                                 + 0.0120878585442036 w_k1_PG_s014
                                 + 0.0156746506595146 w_k1_PG_s015
                                 + 0.0109387891329538 w_k1_PG_s016
                                 + 0.0200887788363369 w_k1_PG_s017
                                 + 0.0562637044857571 w_k1_PG_s100
                                 + 0.0256556839516388 w_k1_PG_s101
                                 + 0.055420298284301 w_k1_PG_s102
                                 + 0.0816223046928613 w_k1_PG_s103
                                 + 0.0345847709067546 w_k1_PG_s104
                                 + 0.0528183007796077 w_k1_PG_s105
                                 + 0.0765066017114968 w_k1_PG_s106
                                 + 0.0226544569209323 w_k1_PG_s107
                                 + 0.0246645252169806 w_k1_PG_s110
                                 + 0.0242573472347768 w_k1_PG_s111
                                 + 0.0509745931174973 w_k1_PG_s112
                                 + 0.040591376757386 w_k1_PG_s113
                                 + 0.0264057220352734 w_k1_PG_s114
                                 + 0.0551120644032273 w_k1_PG_s115
                                 + 0.032449292085144 w_k1_PG_s116
                                 + 0.0312176509817207 w_k1_PG_s117 >= 1.05
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
