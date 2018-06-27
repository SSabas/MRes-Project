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
 w_k1_ADM_s0_return_constraint:  1.09279 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.971797 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.941527 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.14973 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.23541 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.836738 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.73072 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.875768 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.966988 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.29534 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.923734 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.23666 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.20184 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.3026 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.889269 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.897156 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.831246 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.3763 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.28463 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.54968 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.01904 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.18263 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.956102 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.960563 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.06338 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.0195 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.28601 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.753697 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.879075 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.07693 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.25718 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.965347 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.12886 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.915254 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.806739 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.18896 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.746848 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.993687 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.12272 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.858002 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.899512 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.51121 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.39139 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.922292 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.736277 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.06978 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.01825 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.46973 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.974052 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.79554 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.38362 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.41203 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.716117 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.802679 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.952593 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.26345 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.27904 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.42512 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.818817 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.858525 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.96938 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.85298 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.31447 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.14771 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.35206 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.621357 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.20865 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.956703 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.4322 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.01304 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.02984 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.838469 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.756498 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.13134 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.63429 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.722813 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.13619 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.873819 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.899691 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.57812 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.55984 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.874164 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.614368 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.912831 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.897137 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.38835 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.991419 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.43747 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.817 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.51142 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.634266 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.804695 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.05316 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.29384 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.39436 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.9081 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.866908 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.13738 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.28711 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.951147 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.06014 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.06727 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.52516 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.617288 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.52404 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.896449 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.52487 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.900485 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.03782 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.772998 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.762567 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.05417 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.498661 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.729577 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.07731 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.941953 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.907522 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.31441 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.33308 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.918606 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.696881 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.84494 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.07056 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.28936 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.990677 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.31626 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.56604 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.27594 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.796738 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.802365 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.900112 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.22741 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.44971 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.45938 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.917015 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.08893 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.06247 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.836821 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.11442 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.92332 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.37888 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.765173 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.15221 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.00389 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.31505 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.795912 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.933092 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.859046 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.724946 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.11683 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.627365 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.854026 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.18055 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.788372 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.952764 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.34938 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.32215 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.985311 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.891992 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.61988 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.694244 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.00603 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.09344 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.33251 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.900744 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.67784 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.619283 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.24377 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.73811 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.14865 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.733392 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.52555 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.961898 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.667462 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.951439 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.18364 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.20814 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.68851 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.988872 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.741313 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.62739 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.780241 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.20443 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.40115 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.50013 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.848467 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.01155 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.792235 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.714759 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.539613 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.10231 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.954151 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.877417 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.32095 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.19683 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.0109 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.757533 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.06471 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.11641 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.17054 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.13654 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.34435 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.5145 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.21379 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.819128 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.719282 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.940491 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.970967 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.30597 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.16737 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.770891 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.880627 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.10296 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.959789 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.19781 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.875943 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.30444 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.880154 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.261 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.954278 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.10386 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.895883 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.917065 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.787505 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.749585 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.971434 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.68324 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.853688 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.09686 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.881297 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.950171 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.4394 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.31408 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.93679 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.745084 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.14864 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.878841 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.14809 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.04166 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.23173 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.15353 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.47216 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.668114 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.976984 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.23667 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.16921 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.03185 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.49123 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.845817 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.827235 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.20912 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.18324 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.21277 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.13946 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.19093 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.774271 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.49639 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.826259 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.14514 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.03596 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.21325 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.843216 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.937539 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.942455 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.641343 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.646983 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.03247 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.828642 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.9553 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.73431 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.19828 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.771837 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.992777 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.81856 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.683775 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.11437 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.26127 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.82108 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.969535 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.9373 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.674441 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.10887 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.89085 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.42166 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.759053 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.51488 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.877815 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.754388 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.25598 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.20785 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.46715 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.56102 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.32678 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.859527 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.8209 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.758156 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.36068 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.57515 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.67197 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.9733 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.02922 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.963989 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.704934 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.66886 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.06693 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.978061 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.914181 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.28918 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.22181 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.915079 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.748175 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.911661 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.04312 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.22906 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.03584 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.27808 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.50154 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.28723 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.81046 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.806848 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.967383 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.18052 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.2818 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.42273 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.87192 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.05792 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.14041 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.955943 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.06634 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.926356 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.36395 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.838699 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.22802 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.04793 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.11952 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.763626 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.03545 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.847929 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.822693 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.06421 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.639447 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.8494 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.05473 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.932138 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.94111 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.2417 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.17561 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.981896 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.764529 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.05047 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.977934 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.16892 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.07452 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.15553 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.2035 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.24744 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.770159 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.927081 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.07015 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.10909 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.16478 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.39018 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.941503 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.941774 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.16682 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.97373 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.08511 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.11878 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.20109 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.872167 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.1628 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.898996 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.09816 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.924079 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.03711 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.859916 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.875318 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.909992 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.671679 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.784595 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.887998385837153 z_k1_s0000
                                 - 1.9169926879437 z_k1_s0001
                                 - 0.770999363999558 z_k1_s0002
                                 - 1.10199412215854 z_k1_s0003
                                 - 1.52500285094089 z_k1_s0004
                                 - 1.02899849810138 z_k1_s0005
                                 - 1.66699531598237 z_k1_s0006
                                 - 1.59099535962705 z_k1_s0007
                                 - 0.528002164409317 z_k1_s0010
                                 - 0.722000472331362 z_k1_s0011
                                 - 0.398000292171596 z_k1_s0012
                                 - 0.358999960100378 z_k1_s0013
                                 - 0.467002010021161 z_k1_s0014
                                 - 0.794999527910142 z_k1_s0015
                                 - 0.745001810281548 z_k1_s0016
                                 - 0.577999882037912 z_k1_s0017
                                 - 0.855999848881468 z_k1_s0020
                                 - 0.645999663432091 z_k1_s0021
                                 - 2.39799654100506 z_k1_s0022
                                 - 2.1689977833309 z_k1_s0023
                                 - 1.22199691344824 z_k1_s0024
                                 - 0.645999663432091 z_k1_s0025
                                 - 1.27600139557996 z_k1_s0026
                                 - 2.54499502469999 z_k1_s0027
                                 - 0.610000498858656 z_k1_s0030
                                 - 0.290000318454179 z_k1_s0031
                                 - 0.790000572161153 z_k1_s0032
                                 - 1.04200247798606 z_k1_s0033
                                 - 0.276000262663946 z_k1_s0034
                                 - 0.39100033454056 z_k1_s0035
                                 - 0.801000100795932 z_k1_s0036
                                 - 0.308000325784429 z_k1_s0037
                                 - 0.48499931111964 z_k1_s0040
                                 - 1.7429957126673 z_k1_s0041
                                 - 1.14899952249491 z_k1_s0042
                                 - 1.11800068472353 z_k1_s0043
                                 - 0.624999429937206 z_k1_s0044
                                 - 1.6799943826213 z_k1_s0045
                                 - 1.21999487261941 z_k1_s0046
                                 - 0.783998552421524 z_k1_s0047
                                 - 0.306000792084176 z_k1_s0050
                                 - 0.284999853352626 z_k1_s0051
                                 - 0.160999964206235 z_k1_s0052
                                 - 0.180000035696436 z_k1_s0053
                                 - 0.160000140085756 z_k1_s0054
                                 - 0.498000667626989 z_k1_s0055
                                 - 0.640999165463112 z_k1_s0056
                                 - 0.207000162950941 z_k1_s0057
                                 - 0.291999748223421 z_k1_s0060
                                 - 0.200000106833909 z_k1_s0061
                                 - 0.156999921518084 z_k1_s0062
                                 - 0.422999557690077 z_k1_s0063
                                 - 0.593999539543546 z_k1_s0064
                                 - 0.399999206867606 z_k1_s0065
                                 - 0.272998911239527 z_k1_s0066
                                 - 0.178000012021161 z_k1_s0067
                                 - 0.209000138913225 z_k1_s0070
                                 - 0.525000604595505 z_k1_s0071
                                 - 0.417000803501677 z_k1_s0072
                                 - 0.261999936535162 z_k1_s0073
                                 - 0.362001428629991 z_k1_s0074
                                 - 0.295999978683253 z_k1_s0075
                                 - 0.595001523341845 z_k1_s0076
                                 - 0.477000692998248 z_k1_s0077
                                 - 0.19099966457536 z_k1_s0100
                                 - 0.184999716562473 z_k1_s0101
                                 - 0.0839999507796248 z_k1_s0102
                                 - 0.100999878882716 z_k1_s0103
                                 - 0.0699998835831085 z_k1_s0104
                                 - 0.307999216325993 z_k1_s0105
                                 - 0.0769998606314325 z_k1_s0106
                                 - 0.114999946079233 z_k1_s0107
                                 - 0.216000005560023 z_k1_s0110
                                 - 0.0779998587857643 z_k1_s0111
                                 - 0.118999797778263 z_k1_s0112
                                 - 0.27500003094907 z_k1_s0113
                                 - 0.118999797778263 z_k1_s0114
                                 - 0.167000028169058 z_k1_s0115
                                 - 0.163999240370716 z_k1_s0116
                                 - 0.150999693572487 z_k1_s0117
                                 - 0.185999348799074 z_k1_s0120
                                 - 0.107999874753987 z_k1_s0121
                                 - 0.142000383490672 z_k1_s0122
                                 - 0.0639999734467478 z_k1_s0123
                                 - 0.245999214218037 z_k1_s0124
                                 - 0.257999889300957 z_k1_s0125
                                 - 0.109999870267952 z_k1_s0126
                                 - 0.0559998743910316 z_k1_s0127
                                 - 0.253001343064879 z_k1_s0130
                                 - 0.229999399147362 z_k1_s0131
                                 - 0.228999401759765 z_k1_s0132
                                 - 0.358001068762588 z_k1_s0133
                                 - 0.0890001682967094 z_k1_s0134
                                 - 0.423000898956408 z_k1_s0135
                                 - 0.307001201995129 z_k1_s0136
                                 - 0.115000100374237 z_k1_s0137
                                 - 0.0829999128586958 z_k1_s0140
                                 - 0.140999514172245 z_k1_s0141
                                 - 0.116000052226485 z_k1_s0142
                                 - 0.0939998885146768 z_k1_s0143
                                 - 0.0879999114477711 z_k1_s0144
                                 - 0.0649999816579787 z_k1_s0145
                                 - 0.180999715284694 z_k1_s0146
                                 - 0.293999619677842 z_k1_s0147
                                 - 0.0509999639208998 z_k1_s0150
                                 - 0.136999503081916 z_k1_s0151
                                 - 0.0709999397722401 z_k1_s0152
                                 - 0.157999588225745 z_k1_s0153
                                 - 0.0679999518945331 z_k1_s0154
                                 - 0.159000037517993 z_k1_s0155
                                 - 0.0910000856234602 z_k1_s0156
                                 - 0.114999818645237 z_k1_s0157
                                 - 0.723999289908232 z_k1_s0160
                                 - 0.398997894369731 z_k1_s0161
                                 - 0.20599946873472 z_k1_s0162
                                 - 0.588999578159861 z_k1_s0163
                                 - 0.459997581293408 z_k1_s0164
                                 - 0.233999735430074 z_k1_s0165
                                 - 0.48699963923916 z_k1_s0166
                                 - 0.426997808465738 z_k1_s0167
                                 - 0.1599997726672 z_k1_s0170
                                 - 0.614998448127633 z_k1_s0171
                                 - 0.24000023000031 z_k1_s0172
                                 - 0.241998728284937 z_k1_s0173
                                 - 0.395999580101198 z_k1_s0174
                                 - 0.218999781625651 z_k1_s0175
                                 - 0.243999510567603 z_k1_s0176
                                 - 0.167999932600413 z_k1_s0177
                                 - 0.100000352664495 z_k1_s1000
                                 - 0.0430000944056388 z_k1_s1001
                                 - 0.0840003153182076 z_k1_s1002
                                 - 0.067000007324835 z_k1_s1003
                                 - 0.0660002136785356 z_k1_s1004
                                 - 0.0550000985653154 z_k1_s1005
                                 - 0.038000076772414 z_k1_s1006
                                 - 0.0240000560194318 z_k1_s1007
                                 - 0.119999711275429 z_k1_s1010
                                 - 0.101999798934167 z_k1_s1011
                                 - 0.100000498563734 z_k1_s1012
                                 - 0.182000197785154 z_k1_s1013
                                 - 0.0600000330379248 z_k1_s1014
                                 - 0.0969997740059805 z_k1_s1015
                                 - 0.0780001227793969 z_k1_s1016
                                 - 0.148000560474116 z_k1_s1017
                                 - 0.0800002214830233 z_k1_s1020
                                 - 0.0260000854570204 z_k1_s1021
                                 - 0.048000132889814 z_k1_s1022
                                 - 0.0250000759509637 z_k1_s1023
                                 - 0.0930003989619117 z_k1_s1024
                                 - 0.0300000695810959 z_k1_s1025
                                 - 0.116000132499854 z_k1_s1026
                                 - 0.121000449530894 z_k1_s1027
                                 - 0.0760001782726034 z_k1_s1030
                                 - 0.0420001307296739 z_k1_s1031
                                 - 0.0710001262808882 z_k1_s1032
                                 - 0.154000377341893 z_k1_s1033
                                 - 0.0770001886709464 z_k1_s1034
                                 - 0.0980004070361503 z_k1_s1035
                                 - 0.0490001423179284 z_k1_s1036
                                 - 0.0450001007245562 z_k1_s1037
                                 - 0.102000064192926 z_k1_s1040
                                 - 0.266999304834398 z_k1_s1041
                                 - 0.225999876631377 z_k1_s1042
                                 - 0.0800000337473923 z_k1_s1043
                                 - 0.202000176927168 z_k1_s1044
                                 - 0.195000205721772 z_k1_s1045
                                 - 0.196000604751126 z_k1_s1046
                                 - 0.14300062119598 z_k1_s1047
                                 - 0.0900005411150354 z_k1_s1050
                                 - 0.215001014528853 z_k1_s1051
                                 - 0.164001017818681 z_k1_s1052
                                 - 0.0650003510717544 z_k1_s1053
                                 - 0.251000801852539 z_k1_s1054
                                 - 0.0760004251547463 z_k1_s1055
                                 - 0.0610003349689071 z_k1_s1056
                                 - 0.270001384943813 z_k1_s1057
                                 - 0.102000278412434 z_k1_s1060
                                 - 0.0849999620097831 z_k1_s1061
                                 - 0.0960003255647793 z_k1_s1062
                                 - 0.0589999863362296 z_k1_s1063
                                 - 0.0579999041947612 z_k1_s1064
                                 - 0.0730000563144748 z_k1_s1065
                                 - 0.0420001019345042 z_k1_s1066
                                 - 0.025000055532432 z_k1_s1067
                                 - 0.416999942616931 z_k1_s1070
                                 - 0.269000539873532 z_k1_s1071
                                 - 0.268000767591768 z_k1_s1072
                                 - 0.339999820898527 z_k1_s1073
                                 - 0.139000128005832 z_k1_s1074
                                 - 0.365000748951096 z_k1_s1075
                                 - 0.177000302724152 z_k1_s1076
                                 - 0.232000137436975 z_k1_s1077
                                 - 1.89199752436487 z_k1_s1100
                                 - 1.51399888356313 z_k1_s1101
                                 - 0.803998507137669 z_k1_s1102
                                 - 1.79099955197831 z_k1_s1103
                                 - 1.07000147874657 z_k1_s1104
                                 - 0.735999017960152 z_k1_s1105
                                 - 0.801998680773336 z_k1_s1106
                                 - 0.37899900338309 z_k1_s1107
                                 - 0.343000346002354 z_k1_s1110
                                 - 0.315000551736615 z_k1_s1111
                                 - 0.420998953957751 z_k1_s1112
                                 - 0.162999801424523 z_k1_s1113
                                 - 0.292999239315048 z_k1_s1114
                                 - 0.103999825472137 z_k1_s1115
                                 - 0.407999950297734 z_k1_s1116
                                 - 0.245999666547475 z_k1_s1117
                                 - 0.36899936267959 z_k1_s1120
                                 - 0.355998918447147 z_k1_s1121
                                 - 0.485000786139245 z_k1_s1122
                                 - 0.14699985159183 z_k1_s1123
                                 - 0.142999932658691 z_k1_s1124
                                 - 0.133999879159214 z_k1_s1125
                                 - 0.60400049749936 z_k1_s1126
                                 - 0.120999906726598 z_k1_s1127
                                 - 0.618003247390725 z_k1_s1130
                                 - 0.288000861453927 z_k1_s1131
                                 - 0.251000642289037 z_k1_s1132
                                 - 0.32100096708721 z_k1_s1133
                                 - 0.636002761079984 z_k1_s1134
                                 - 0.60000040969155 z_k1_s1135
                                 - 0.194000520268091 z_k1_s1136
                                 - 0.415999840851486 z_k1_s1137
                                 - 0.501999373506225 z_k1_s1140
                                 - 0.3869986376679 z_k1_s1141
                                 - 0.255998905485908 z_k1_s1142
                                 - 0.205999734963916 z_k1_s1143
                                 - 0.402000034863092 z_k1_s1144
                                 - 0.200999768031759 z_k1_s1145
                                 - 0.153999879349657 z_k1_s1146
                                 - 0.385998544521554 z_k1_s1147
                                 - 0.487000560643301 z_k1_s1150
                                 - 0.501000176825831 z_k1_s1151
                                 - 0.558000118141299 z_k1_s1152
                                 - 1.21300396897958 z_k1_s1153
                                 - 0.789001681157668 z_k1_s1154
                                 - 1.95399794207758 z_k1_s1155
                                 - 0.55200047063461 z_k1_s1156
                                 - 0.526000243437932 z_k1_s1157
                                 - 0.287999738649121 z_k1_s1160
                                 - 0.422999416172185 z_k1_s1161
                                 - 0.116999933819948 z_k1_s1162
                                 - 0.176999932696838 z_k1_s1163
                                 - 0.217998957859564 z_k1_s1164
                                 - 0.31099987330857 z_k1_s1165
                                 - 0.186999861409667 z_k1_s1166
                                 - 0.411999537248065 z_k1_s1167
                                 - 0.20900000446559 z_k1_s1170
                                 - 0.120999914238008 z_k1_s1171
                                 - 0.345000122108224 z_k1_s1172
                                 - 0.293999478492476 z_k1_s1173
                                 - 0.32200091330069 z_k1_s1174
                                 - 0.107999881833941 z_k1_s1175
                                 - 0.523998760564945 z_k1_s1176
                                 - 0.475000446148893 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.889703 w_k1_ADM_s000
                                 + 0.997002 w_k1_AXP_s000
                                 + 1.03169 w_k1_DIS_s000
                                 + 0.966467 w_k1_FDX_s000
                                 + 1.66413 w_k1_IBM_s000
                                 + 1.14864 w_k1_INTC_s000
                                 + 1.47016 w_k1_KO_s000 + 1.4404 w_k1_MAT_s000
                                 + 1.05454 w_k1_MSFT_s000
                                 + 1.20948 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.670744 w_k1_ADM_s000
                                 + 0.674308 w_k1_AXP_s000
                                 + 0.600977 w_k1_DIS_s000
                                 + 0.738476 w_k1_FDX_s000
                                 + 0.710276 w_k1_IBM_s000
                                 + 0.874552 w_k1_INTC_s000
                                 + 0.711465 w_k1_KO_s000
                                 + 0.574308 w_k1_MAT_s000
                                 + 0.785403 w_k1_MSFT_s000
                                 + 0.970797 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.20021 w_k1_ADM_s000
                                 + 1.60828 w_k1_AXP_s000
                                 + 1.40651 w_k1_DIS_s000
                                 + 1.29022 w_k1_FDX_s000
                                 + 1.47437 w_k1_IBM_s000
                                 + 1.27645 w_k1_INTC_s000
                                 + 1.49542 w_k1_KO_s000 + 1.55334 w_k1_MAT_s000
                                 + 1.22412 w_k1_MSFT_s000
                                 + 1.37651 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.976374 w_k1_ADM_s000
                                 + 0.81553 w_k1_AXP_s000
                                 + 0.801966 w_k1_DIS_s000
                                 + 1.02214 w_k1_FDX_s000
                                 + 0.592475 w_k1_IBM_s000
                                 + 1.04018 w_k1_INTC_s000
                                 + 0.796666 w_k1_KO_s000
                                 + 0.542927 w_k1_MAT_s000
                                 + 0.969329 w_k1_MSFT_s000
                                 + 0.925256 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.15896 w_k1_ADM_s000
                                 + 1.10035 w_k1_AXP_s000
                                 + 1.17242 w_k1_DIS_s000
                                 + 1.12039 w_k1_FDX_s000
                                 + 1.04367 w_k1_IBM_s000
                                 + 1.07938 w_k1_INTC_s000
                                 + 1.13113 w_k1_KO_s000
                                 + 0.921623 w_k1_MAT_s000
                                 + 1.02387 w_k1_MSFT_s000
                                 + 1.13395 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.32209 w_k1_ADM_s000
                                 + 1.54497 w_k1_AXP_s000
                                 + 1.52273 w_k1_DIS_s000
                                 + 1.41714 w_k1_FDX_s000
                                 + 1.02177 w_k1_IBM_s000
                                 + 1.20157 w_k1_INTC_s000
                                 + 1.33354 w_k1_KO_s000 + 1.39922 w_k1_MAT_s000
                                 + 1.37739 w_k1_MSFT_s000
                                 + 1.31071 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.895159 w_k1_ADM_s000
                                 + 0.924058 w_k1_AXP_s000
                                 + 0.910142 w_k1_DIS_s000
                                 + 0.912942 w_k1_FDX_s000
                                 + 1.12513 w_k1_IBM_s000
                                 + 0.916444 w_k1_INTC_s000
                                 + 1.07932 w_k1_KO_s000 + 1.07515 w_k1_MAT_s000
                                 + 0.86941 w_k1_MSFT_s000 + 1.0569 w_k1_PG_s000
                                 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.16663 w_k1_ADM_s000
                                 + 1.0148 w_k1_AXP_s000 + 1.15553 w_k1_DIS_s000
                                 + 1.15664 w_k1_FDX_s000
                                 + 0.79993 w_k1_IBM_s000
                                 + 1.05912 w_k1_INTC_s000
                                 + 0.956101 w_k1_KO_s000
                                 + 0.584455 w_k1_MAT_s000
                                 + 1.15228 w_k1_MSFT_s000
                                 + 1.06043 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.2574 w_k1_ADM_s001
                                 + 1.47366 w_k1_AXP_s001
                                 + 1.76351 w_k1_DIS_s001
                                 + 1.31774 w_k1_FDX_s001
                                 + 1.34899 w_k1_IBM_s001
                                 + 1.05203 w_k1_INTC_s001
                                 + 1.42633 w_k1_KO_s001 + 1.09779 w_k1_MAT_s001
                                 + 1.31744 w_k1_MSFT_s001
                                 + 1.20511 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.20586 w_k1_ADM_s001
                                 + 1.01153 w_k1_AXP_s001
                                 + 0.864698 w_k1_DIS_s001
                                 + 1.0347 w_k1_FDX_s001
                                 + 0.711403 w_k1_IBM_s001
                                 + 0.935101 w_k1_INTC_s001
                                 + 0.875885 w_k1_KO_s001
                                 + 0.920576 w_k1_MAT_s001
                                 + 0.973201 w_k1_MSFT_s001
                                 + 0.88839 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.87204 w_k1_ADM_s001
                                 + 0.650746 w_k1_AXP_s001
                                 + 0.738066 w_k1_DIS_s001
                                 + 0.784442 w_k1_FDX_s001
                                 + 0.850214 w_k1_IBM_s001
                                 + 0.679568 w_k1_INTC_s001
                                 + 0.814298 w_k1_KO_s001
                                 + 0.661346 w_k1_MAT_s001
                                 + 0.816351 w_k1_MSFT_s001
                                 + 0.724928 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.15363 w_k1_ADM_s001
                                 + 0.842163 w_k1_AXP_s001
                                 + 1.38237 w_k1_DIS_s001
                                 + 1.17828 w_k1_FDX_s001
                                 + 1.08437 w_k1_IBM_s001
                                 + 0.747885 w_k1_INTC_s001
                                 + 1.14476 w_k1_KO_s001
                                 + 0.858024 w_k1_MAT_s001
                                 + 1.08267 w_k1_MSFT_s001
                                 + 1.03495 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.993547 w_k1_ADM_s001
                                 + 0.678337 w_k1_AXP_s001
                                 + 0.812013 w_k1_DIS_s001
                                 + 0.913514 w_k1_FDX_s001
                                 + 0.614682 w_k1_IBM_s001
                                 + 0.862163 w_k1_INTC_s001
                                 + 0.811925 w_k1_KO_s001 + 0.4838 w_k1_MAT_s001
                                 + 1.03912 w_k1_MSFT_s001
                                 + 0.808119 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.00073 w_k1_ADM_s001
                                 + 0.868242 w_k1_AXP_s001
                                 + 0.83543 w_k1_DIS_s001
                                 + 0.892551 w_k1_FDX_s001
                                 + 1.00816 w_k1_IBM_s001
                                 + 0.831021 w_k1_INTC_s001
                                 + 0.900168 w_k1_KO_s001
                                 + 1.06013 w_k1_MAT_s001
                                 + 0.91088 w_k1_MSFT_s001
                                 + 0.849036 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.10219 w_k1_ADM_s001
                                 + 1.07288 w_k1_AXP_s001
                                 + 1.24681 w_k1_DIS_s001
                                 + 1.15079 w_k1_FDX_s001
                                 + 0.951599 w_k1_IBM_s001
                                 + 0.889314 w_k1_INTC_s001
                                 + 1.11371 w_k1_KO_s001 + 1.0879 w_k1_MAT_s001
                                 + 1.12009 w_k1_MSFT_s001
                                 + 1.08391 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.23837 w_k1_ADM_s001
                                 + 1.20221 w_k1_AXP_s001
                                 + 1.12774 w_k1_DIS_s001
                                 + 1.01658 w_k1_FDX_s001
                                 + 1.44875 w_k1_IBM_s001
                                 + 0.863038 w_k1_INTC_s001
                                 + 1.26817 w_k1_KO_s001 + 1.38489 w_k1_MAT_s001
                                 + 0.919729 w_k1_MSFT_s001
                                 + 1.14672 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.773695 w_k1_ADM_s002
                                 + 0.534338 w_k1_AXP_s002
                                 + 0.556882 w_k1_DIS_s002
                                 + 0.698021 w_k1_FDX_s002
                                 + 0.671922 w_k1_IBM_s002
                                 + 0.719742 w_k1_INTC_s002
                                 + 0.653589 w_k1_KO_s002
                                 + 0.522604 w_k1_MAT_s002
                                 + 0.724841 w_k1_MSFT_s002
                                 + 0.771546 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.71805 w_k1_ADM_s002
                                 + 1.6014 w_k1_AXP_s002 + 1.42105 w_k1_DIS_s002
                                 + 1.51689 w_k1_FDX_s002
                                 + 1.09986 w_k1_IBM_s002
                                 + 1.14869 w_k1_INTC_s002
                                 + 1.13664 w_k1_KO_s002 + 1.65935 w_k1_MAT_s002
                                 + 1.2298 w_k1_MSFT_s002 + 1.15098 w_k1_PG_s002
                                 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.951281 w_k1_ADM_s002
                                 + 0.932801 w_k1_AXP_s002
                                 + 1.05361 w_k1_DIS_s002
                                 + 0.95179 w_k1_FDX_s002
                                 + 1.07856 w_k1_IBM_s002
                                 + 0.951035 w_k1_INTC_s002
                                 + 1.12592 w_k1_KO_s002 + 1.43777 w_k1_MAT_s002
                                 + 1.09494 w_k1_MSFT_s002
                                 + 1.04778 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.12918 w_k1_ADM_s002
                                 + 1.2347 w_k1_AXP_s002 + 1.15049 w_k1_DIS_s002
                                 + 1.15462 w_k1_FDX_s002
                                 + 1.19711 w_k1_IBM_s002
                                 + 0.81173 w_k1_INTC_s002
                                 + 0.989371 w_k1_KO_s002
                                 + 1.49695 w_k1_MAT_s002
                                 + 0.982643 w_k1_MSFT_s002
                                 + 1.05984 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.22836 w_k1_ADM_s002
                                 + 0.971552 w_k1_AXP_s002
                                 + 1.4012 w_k1_DIS_s002 + 1.2535 w_k1_FDX_s002
                                 + 0.87336 w_k1_IBM_s002
                                 + 1.23762 w_k1_INTC_s002
                                 + 1.07098 w_k1_KO_s002 + 0.94068 w_k1_MAT_s002
                                 + 1.38829 w_k1_MSFT_s002
                                 + 1.17029 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.29985 w_k1_ADM_s002
                                 + 1.40336 w_k1_AXP_s002
                                 + 2.17281 w_k1_DIS_s002
                                 + 1.51736 w_k1_FDX_s002 + 1.4013 w_k1_IBM_s002
                                 + 1.18333 w_k1_INTC_s002
                                 + 1.41245 w_k1_KO_s002 + 2.13616 w_k1_MAT_s002
                                 + 1.31272 w_k1_MSFT_s002
                                 + 1.32588 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.835441 w_k1_ADM_s002
                                 + 0.829974 w_k1_AXP_s002
                                 + 0.688754 w_k1_DIS_s002
                                 + 0.77376 w_k1_FDX_s002
                                 + 1.04578 w_k1_IBM_s002
                                 + 0.75203 w_k1_INTC_s002
                                 + 0.796993 w_k1_KO_s002
                                 + 1.16954 w_k1_MAT_s002
                                 + 0.743503 w_k1_MSFT_s002
                                 + 0.797846 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.02271 w_k1_ADM_s002
                                 + 0.838794 w_k1_AXP_s002
                                 + 0.859588 w_k1_DIS_s002
                                 + 0.915802 w_k1_FDX_s002
                                 + 0.755799 w_k1_IBM_s002
                                 + 0.794984 w_k1_INTC_s002
                                 + 0.793036 w_k1_KO_s002
                                 + 0.675244 w_k1_MAT_s002
                                 + 0.923789 w_k1_MSFT_s002
                                 + 0.860317 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.11999 w_k1_ADM_s003
                                 + 1.34375 w_k1_AXP_s003
                                 + 1.44204 w_k1_DIS_s003
                                 + 1.44115 w_k1_FDX_s003
                                 + 1.16009 w_k1_IBM_s003
                                 + 1.36462 w_k1_INTC_s003
                                 + 1.20679 w_k1_KO_s003 + 1.52072 w_k1_MAT_s003
                                 + 1.32905 w_k1_MSFT_s003
                                 + 1.05619 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.834403 w_k1_ADM_s003
                                 + 0.889678 w_k1_AXP_s003
                                 + 0.65354 w_k1_DIS_s003
                                 + 0.774025 w_k1_FDX_s003
                                 + 0.947374 w_k1_IBM_s003
                                 + 0.74294 w_k1_INTC_s003
                                 + 0.796688 w_k1_KO_s003
                                 + 1.07663 w_k1_MAT_s003
                                 + 0.705092 w_k1_MSFT_s003
                                 + 0.80647 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.680291 w_k1_ADM_s003
                                 + 0.702656 w_k1_AXP_s003
                                 + 0.700146 w_k1_DIS_s003
                                 + 0.832871 w_k1_FDX_s003
                                 + 0.770925 w_k1_IBM_s003
                                 + 1.00691 w_k1_INTC_s003
                                 + 0.716305 w_k1_KO_s003
                                 + 0.854665 w_k1_MAT_s003
                                 + 0.8715 w_k1_MSFT_s003
                                 + 0.813019 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.901688 w_k1_ADM_s003
                                 + 0.982547 w_k1_AXP_s003
                                 + 1.05098 w_k1_DIS_s003
                                 + 1.05194 w_k1_FDX_s003
                                 + 1.16528 w_k1_IBM_s003
                                 + 1.00758 w_k1_INTC_s003
                                 + 0.981613 w_k1_KO_s003
                                 + 1.48112 w_k1_MAT_s003
                                 + 1.09157 w_k1_MSFT_s003
                                 + 0.983943 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.23511 w_k1_ADM_s003
                                 + 1.11642 w_k1_AXP_s003
                                 + 0.960518 w_k1_DIS_s003
                                 + 1.17596 w_k1_FDX_s003
                                 + 0.751074 w_k1_IBM_s003
                                 + 0.979267 w_k1_INTC_s003
                                 + 0.831698 w_k1_KO_s003
                                 + 0.906775 w_k1_MAT_s003
                                 + 0.96092 w_k1_MSFT_s003
                                 + 0.939297 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.29071 w_k1_ADM_s003
                                 + 1.22381 w_k1_AXP_s003
                                 + 0.935405 w_k1_DIS_s003
                                 + 1.06182 w_k1_FDX_s003
                                 + 1.16296 w_k1_IBM_s003
                                 + 0.99224 w_k1_INTC_s003
                                 + 1.01793 w_k1_KO_s003 + 1.38104 w_k1_MAT_s003
                                 + 0.934889 w_k1_MSFT_s003
                                 + 0.970787 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.933203 w_k1_ADM_s003
                                 + 0.756032 w_k1_AXP_s003
                                 + 0.838812 w_k1_DIS_s003
                                 + 0.875888 w_k1_FDX_s003
                                 + 1.20524 w_k1_IBM_s003
                                 + 0.884931 w_k1_INTC_s003
                                 + 0.965237 w_k1_KO_s003
                                 + 0.962278 w_k1_MAT_s003
                                 + 0.907002 w_k1_MSFT_s003
                                 + 0.95457 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.908208 w_k1_ADM_s003
                                 + 0.851797 w_k1_AXP_s003
                                 + 1.29744 w_k1_DIS_s003
                                 + 1.19499 w_k1_FDX_s003
                                 + 0.929762 w_k1_IBM_s003
                                 + 1.08792 w_k1_INTC_s003
                                 + 1.17157 w_k1_KO_s003
                                 + 0.928212 w_k1_MAT_s003
                                 + 1.16094 w_k1_MSFT_s003 + 1.1153 w_k1_PG_s003
                                 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.59815 w_k1_ADM_s004
                                 + 1.54297 w_k1_AXP_s004
                                 + 1.82143 w_k1_DIS_s004
                                 + 1.22845 w_k1_FDX_s004
                                 + 1.59769 w_k1_IBM_s004
                                 + 1.07531 w_k1_INTC_s004
                                 + 1.53227 w_k1_KO_s004 + 1.5282 w_k1_MAT_s004
                                 + 1.26866 w_k1_MSFT_s004
                                 + 1.23263 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.932225 w_k1_ADM_s004
                                 + 0.774055 w_k1_AXP_s004
                                 + 0.750937 w_k1_DIS_s004
                                 + 0.83254 w_k1_FDX_s004
                                 + 0.766601 w_k1_IBM_s004
                                 + 0.732811 w_k1_INTC_s004
                                 + 0.783765 w_k1_KO_s004
                                 + 0.595121 w_k1_MAT_s004
                                 + 0.797737 w_k1_MSFT_s004
                                 + 0.863577 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.33212 w_k1_ADM_s004
                                 + 1.14397 w_k1_AXP_s004
                                 + 1.18946 w_k1_DIS_s004
                                 + 1.22109 w_k1_FDX_s004
                                 + 0.785547 w_k1_IBM_s004
                                 + 0.921804 w_k1_INTC_s004
                                 + 0.875096 w_k1_KO_s004
                                 + 0.891832 w_k1_MAT_s004
                                 + 1.10217 w_k1_MSFT_s004
                                 + 1.05072 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.13303 w_k1_ADM_s004
                                 + 1.24517 w_k1_AXP_s004
                                 + 0.961025 w_k1_DIS_s004
                                 + 0.959121 w_k1_FDX_s004
                                 + 0.935295 w_k1_IBM_s004
                                 + 0.843847 w_k1_INTC_s004
                                 + 0.881651 w_k1_KO_s004
                                 + 1.13162 w_k1_MAT_s004
                                 + 0.839501 w_k1_MSFT_s004
                                 + 0.930254 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.17416 w_k1_ADM_s004
                                 + 1.32962 w_k1_AXP_s004
                                 + 1.86451 w_k1_DIS_s004
                                 + 1.36744 w_k1_FDX_s004
                                 + 1.15127 w_k1_IBM_s004
                                 + 1.14271 w_k1_INTC_s004
                                 + 1.28732 w_k1_KO_s004 + 1.11634 w_k1_MAT_s004
                                 + 1.4099 w_k1_MSFT_s004 + 1.06075 w_k1_PG_s004
                                 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.758293 w_k1_ADM_s004
                                 + 0.985083 w_k1_AXP_s004
                                 + 1.15432 w_k1_DIS_s004
                                 + 0.931058 w_k1_FDX_s004
                                 + 1.28169 w_k1_IBM_s004
                                 + 0.909767 w_k1_INTC_s004
                                 + 1.25442 w_k1_KO_s004 + 1.33295 w_k1_MAT_s004
                                 + 1.01112 w_k1_MSFT_s004
                                 + 0.977405 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.08304 w_k1_ADM_s004
                                 + 1.1251 w_k1_AXP_s004 + 1.06182 w_k1_DIS_s004
                                 + 1.0765 w_k1_FDX_s004 + 1.11953 w_k1_IBM_s004
                                 + 0.947244 w_k1_INTC_s004
                                 + 1.00344 w_k1_KO_s004 + 0.9844 w_k1_MAT_s004
                                 + 1.04198 w_k1_MSFT_s004
                                 + 0.994004 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.93125 w_k1_ADM_s004
                                 + 0.75468 w_k1_AXP_s004
                                 + 1.06564 w_k1_DIS_s004
                                 + 0.920033 w_k1_FDX_s004
                                 + 0.748547 w_k1_IBM_s004
                                 + 0.77687 w_k1_INTC_s004
                                 + 1.02543 w_k1_KO_s004
                                 + 0.655776 w_k1_MAT_s004
                                 + 1.05322 w_k1_MSFT_s004
                                 + 0.924333 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.833502 w_k1_ADM_s005
                                 + 0.722581 w_k1_AXP_s005
                                 + 0.901055 w_k1_DIS_s005
                                 + 0.878183 w_k1_FDX_s005
                                 + 0.815113 w_k1_IBM_s005
                                 + 0.850734 w_k1_INTC_s005
                                 + 0.834488 w_k1_KO_s005
                                 + 0.701993 w_k1_MAT_s005
                                 + 0.885132 w_k1_MSFT_s005
                                 + 0.993565 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.728489 w_k1_ADM_s005
                                 + 0.835588 w_k1_AXP_s005
                                 + 0.615607 w_k1_DIS_s005
                                 + 0.767444 w_k1_FDX_s005
                                 + 0.933154 w_k1_IBM_s005
                                 + 0.924943 w_k1_INTC_s005
                                 + 0.901779 w_k1_KO_s005
                                 + 0.869541 w_k1_MAT_s005
                                 + 0.762274 w_k1_MSFT_s005
                                 + 1.01769 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.00753 w_k1_ADM_s005
                                 + 0.791463 w_k1_AXP_s005
                                 + 1.15486 w_k1_DIS_s005
                                 + 1.06731 w_k1_FDX_s005
                                 + 0.788435 w_k1_IBM_s005
                                 + 0.863825 w_k1_INTC_s005
                                 + 1.09701 w_k1_KO_s005
                                 + 0.946245 w_k1_MAT_s005
                                 + 0.989542 w_k1_MSFT_s005
                                 + 1.13859 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.17298 w_k1_ADM_s005
                                 + 0.79233 w_k1_AXP_s005 + 1.0563 w_k1_DIS_s005
                                 + 1.02086 w_k1_FDX_s005
                                 + 1.03567 w_k1_IBM_s005
                                 + 0.774151 w_k1_INTC_s005
                                 + 1.02439 w_k1_KO_s005
                                 + 0.836341 w_k1_MAT_s005
                                 + 0.898055 w_k1_MSFT_s005
                                 + 1.12251 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.931682 w_k1_ADM_s005
                                 + 0.90707 w_k1_AXP_s005
                                 + 0.682464 w_k1_DIS_s005
                                 + 0.878773 w_k1_FDX_s005
                                 + 0.650049 w_k1_IBM_s005
                                 + 0.888577 w_k1_INTC_s005
                                 + 0.764439 w_k1_KO_s005
                                 + 0.817999 w_k1_MAT_s005
                                 + 0.74739 w_k1_MSFT_s005
                                 + 0.896388 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.028 w_k1_ADM_s005
                                 + 1.27801 w_k1_AXP_s005
                                 + 1.35187 w_k1_DIS_s005
                                 + 1.29982 w_k1_FDX_s005
                                 + 1.05017 w_k1_IBM_s005
                                 + 1.20414 w_k1_INTC_s005
                                 + 1.14358 w_k1_KO_s005
                                 + 0.989336 w_k1_MAT_s005
                                 + 1.21057 w_k1_MSFT_s005
                                 + 1.24212 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.05446 w_k1_ADM_s005
                                 + 1.23683 w_k1_AXP_s005 + 1.3498 w_k1_DIS_s005
                                 + 1.11047 w_k1_FDX_s005
                                 + 1.27693 w_k1_IBM_s005
                                 + 0.950033 w_k1_INTC_s005
                                 + 1.35849 w_k1_KO_s005 + 1.47083 w_k1_MAT_s005
                                 + 1.0957 w_k1_MSFT_s005 + 1.21655 w_k1_PG_s005
                                 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.936611 w_k1_ADM_s005
                                 + 0.965261 w_k1_AXP_s005
                                 + 0.856043 w_k1_DIS_s005
                                 + 0.95359 w_k1_FDX_s005
                                 + 1.48295 w_k1_IBM_s005
                                 + 0.949312 w_k1_INTC_s005
                                 + 1.08938 w_k1_KO_s005 + 1.50896 w_k1_MAT_s005
                                 + 0.854091 w_k1_MSFT_s005
                                 + 1.00104 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.02592 w_k1_ADM_s006
                                 + 0.82518 w_k1_AXP_s006
                                 + 0.935947 w_k1_DIS_s006
                                 + 0.898705 w_k1_FDX_s006
                                 + 0.987637 w_k1_IBM_s006
                                 + 0.731346 w_k1_INTC_s006
                                 + 1.16418 w_k1_KO_s006
                                 + 0.971993 w_k1_MAT_s006
                                 + 0.952039 w_k1_MSFT_s006
                                 + 0.955814 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.50428 w_k1_ADM_s006
                                 + 1.56792 w_k1_AXP_s006
                                 + 1.17908 w_k1_DIS_s006
                                 + 1.12899 w_k1_FDX_s006
                                 + 1.56062 w_k1_IBM_s006
                                 + 0.956196 w_k1_INTC_s006
                                 + 1.23087 w_k1_KO_s006 + 1.69149 w_k1_MAT_s006
                                 + 1.04407 w_k1_MSFT_s006
                                 + 1.19047 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.06078 w_k1_ADM_s006
                                 + 1.08497 w_k1_AXP_s006
                                 + 0.751926 w_k1_DIS_s006
                                 + 0.982944 w_k1_FDX_s006
                                 + 0.788353 w_k1_IBM_s006
                                 + 0.878678 w_k1_INTC_s006
                                 + 0.899178 w_k1_KO_s006
                                 + 1.46068 w_k1_MAT_s006
                                 + 0.967021 w_k1_MSFT_s006
                                 + 0.966039 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.02604 w_k1_ADM_s006
                                 + 0.725223 w_k1_AXP_s006
                                 + 0.633427 w_k1_DIS_s006
                                 + 0.777864 w_k1_FDX_s006
                                 + 0.732787 w_k1_IBM_s006
                                 + 0.748299 w_k1_INTC_s006
                                 + 0.840532 w_k1_KO_s006
                                 + 0.79646 w_k1_MAT_s006
                                 + 0.857757 w_k1_MSFT_s006
                                 + 0.963168 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.25511 w_k1_ADM_s006
                                 + 1.44826 w_k1_AXP_s006
                                 + 1.44306 w_k1_DIS_s006
                                 + 1.29327 w_k1_FDX_s006
                                 + 1.04383 w_k1_IBM_s006
                                 + 1.1659 w_k1_INTC_s006 + 1.37097 w_k1_KO_s006
                                 + 1.26321 w_k1_MAT_s006
                                 + 1.2435 w_k1_MSFT_s006 + 1.24208 w_k1_PG_s006
                                 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.30495 w_k1_ADM_s006
                                 + 1.04964 w_k1_AXP_s006
                                 + 0.936832 w_k1_DIS_s006
                                 + 1.09511 w_k1_FDX_s006
                                 + 0.813815 w_k1_IBM_s006
                                 + 0.8322 w_k1_INTC_s006
                                 + 0.914224 w_k1_KO_s006
                                 + 1.14206 w_k1_MAT_s006
                                 + 1.01986 w_k1_MSFT_s006
                                 + 1.07952 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.24329 w_k1_ADM_s006
                                 + 1.08733 w_k1_AXP_s006
                                 + 1.19837 w_k1_DIS_s006
                                 + 0.951183 w_k1_FDX_s006
                                 + 1.32141 w_k1_IBM_s006
                                 + 1.17415 w_k1_INTC_s006
                                 + 1.40665 w_k1_KO_s006 + 1.24822 w_k1_MAT_s006
                                 + 1.08915 w_k1_MSFT_s006
                                 + 1.06601 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.895765 w_k1_ADM_s006
                                 + 0.744381 w_k1_AXP_s006
                                 + 0.817517 w_k1_DIS_s006
                                 + 0.768509 w_k1_FDX_s006
                                 + 1.2803 w_k1_IBM_s006
                                 + 0.677102 w_k1_INTC_s006
                                 + 1.09492 w_k1_KO_s006 + 1.35511 w_k1_MAT_s006
                                 + 0.892271 w_k1_MSFT_s006
                                 + 1.04705 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.04392 w_k1_ADM_s007
                                 + 0.949339 w_k1_AXP_s007
                                 + 0.93349 w_k1_DIS_s007
                                 + 0.861156 w_k1_FDX_s007
                                 + 0.968197 w_k1_IBM_s007
                                 + 0.811011 w_k1_INTC_s007
                                 + 0.948924 w_k1_KO_s007
                                 + 0.812589 w_k1_MAT_s007
                                 + 0.821641 w_k1_MSFT_s007
                                 + 0.993686 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.73363 w_k1_ADM_s007
                                 + 1.06235 w_k1_AXP_s007
                                 + 1.10123 w_k1_DIS_s007
                                 + 0.988839 w_k1_FDX_s007
                                 + 1.22383 w_k1_IBM_s007
                                 + 1.04612 w_k1_INTC_s007
                                 + 1.09211 w_k1_KO_s007 + 1.22317 w_k1_MAT_s007
                                 + 0.901621 w_k1_MSFT_s007
                                 + 0.973613 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.784286 w_k1_ADM_s007
                                 + 0.933336 w_k1_AXP_s007
                                 + 1.21285 w_k1_DIS_s007 + 1.0657 w_k1_FDX_s007
                                 + 0.936517 w_k1_IBM_s007
                                 + 1.16869 w_k1_INTC_s007
                                 + 1.10675 w_k1_KO_s007
                                 + 0.660498 w_k1_MAT_s007
                                 + 1.15786 w_k1_MSFT_s007
                                 + 1.04009 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.1397 w_k1_ADM_s007
                                 + 1.17835 w_k1_AXP_s007
                                 + 1.65479 w_k1_DIS_s007 + 1.279 w_k1_FDX_s007
                                 + 0.998737 w_k1_IBM_s007
                                 + 1.32018 w_k1_INTC_s007
                                 + 1.10381 w_k1_KO_s007 + 1.022 w_k1_MAT_s007
                                 + 1.30517 w_k1_MSFT_s007
                                 + 1.10216 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.925172 w_k1_ADM_s007
                                 + 0.952928 w_k1_AXP_s007
                                 + 0.940239 w_k1_DIS_s007
                                 + 0.98165 w_k1_FDX_s007
                                 + 0.74901 w_k1_IBM_s007
                                 + 1.07616 w_k1_INTC_s007
                                 + 0.858837 w_k1_KO_s007 + 0.5621 w_k1_MAT_s007
                                 + 0.853084 w_k1_MSFT_s007
                                 + 0.965433 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.773515 w_k1_ADM_s007
                                 + 0.899575 w_k1_AXP_s007
                                 + 0.697865 w_k1_DIS_s007
                                 + 0.783996 w_k1_FDX_s007
                                 + 1.14666 w_k1_IBM_s007
                                 + 1.06127 w_k1_INTC_s007
                                 + 0.965421 w_k1_KO_s007
                                 + 1.00955 w_k1_MAT_s007
                                 + 0.856649 w_k1_MSFT_s007
                                 + 0.827052 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.702585 w_k1_ADM_s007
                                 + 0.730481 w_k1_AXP_s007
                                 + 0.733902 w_k1_DIS_s007
                                 + 0.714789 w_k1_FDX_s007
                                 + 0.772926 w_k1_IBM_s007
                                 + 0.841537 w_k1_INTC_s007
                                 + 0.768543 w_k1_KO_s007
                                 + 0.739892 w_k1_MAT_s007
                                 + 0.75148 w_k1_MSFT_s007
                                 + 0.889505 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.04519 w_k1_ADM_s007
                                 + 1.4511 w_k1_AXP_s007 + 1.44117 w_k1_DIS_s007
                                 + 1.12719 w_k1_FDX_s007
                                 + 1.54792 w_k1_IBM_s007
                                 + 1.23382 w_k1_INTC_s007
                                 + 1.30737 w_k1_KO_s007 + 1.41405 w_k1_MAT_s007
                                 + 1.08171 w_k1_MSFT_s007 + 1.1932 w_k1_PG_s007
                                 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.814153 w_k1_ADM_s010
                                 + 0.900375 w_k1_AXP_s010
                                 + 0.860417 w_k1_DIS_s010
                                 + 0.785502 w_k1_FDX_s010
                                 + 1.39878 w_k1_IBM_s010
                                 + 0.707691 w_k1_INTC_s010
                                 + 0.987543 w_k1_KO_s010
                                 + 1.31211 w_k1_MAT_s010
                                 + 0.734509 w_k1_MSFT_s010
                                 + 1.03898 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.07156 w_k1_ADM_s010
                                 + 0.833972 w_k1_AXP_s010
                                 + 1.15228 w_k1_DIS_s010
                                 + 0.885326 w_k1_FDX_s010
                                 + 1.20178 w_k1_IBM_s010
                                 + 0.741001 w_k1_INTC_s010
                                 + 1.1553 w_k1_KO_s010 + 0.847761 w_k1_MAT_s010
                                 + 0.914182 w_k1_MSFT_s010
                                 + 1.03197 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.25096 w_k1_ADM_s010
                                 + 1.19786 w_k1_AXP_s010
                                 + 1.07791 w_k1_DIS_s010
                                 + 1.02158 w_k1_FDX_s010
                                 + 0.923097 w_k1_IBM_s010
                                 + 0.782982 w_k1_INTC_s010
                                 + 1.03254 w_k1_KO_s010
                                 + 0.736465 w_k1_MAT_s010
                                 + 0.896832 w_k1_MSFT_s010
                                 + 1.1065 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.12052 w_k1_ADM_s010
                                 + 1.10031 w_k1_AXP_s010
                                 + 1.58025 w_k1_DIS_s010
                                 + 1.18012 w_k1_FDX_s010
                                 + 1.20753 w_k1_IBM_s010
                                 + 1.14203 w_k1_INTC_s010 + 1.3912 w_k1_KO_s010
                                 + 1.07819 w_k1_MAT_s010
                                 + 1.29199 w_k1_MSFT_s010
                                 + 1.27482 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.44074 w_k1_ADM_s010
                                 + 1.41371 w_k1_AXP_s010
                                 + 1.64918 w_k1_DIS_s010
                                 + 1.15811 w_k1_FDX_s010
                                 + 1.59643 w_k1_IBM_s010
                                 + 1.03367 w_k1_INTC_s010 + 1.3191 w_k1_KO_s010
                                 + 1.14402 w_k1_MAT_s010
                                 + 1.22638 w_k1_MSFT_s010
                                 + 1.25874 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.960745 w_k1_ADM_s010
                                 + 0.771827 w_k1_AXP_s010
                                 + 0.812875 w_k1_DIS_s010
                                 + 0.908113 w_k1_FDX_s010
                                 + 0.712413 w_k1_IBM_s010
                                 + 0.811523 w_k1_INTC_s010
                                 + 0.775172 w_k1_KO_s010
                                 + 0.631242 w_k1_MAT_s010
                                 + 0.893284 w_k1_MSFT_s010
                                 + 1.04165 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.10647 w_k1_ADM_s010
                                 + 1.20472 w_k1_AXP_s010 + 1.541 w_k1_DIS_s010
                                 + 1.38574 w_k1_FDX_s010
                                 + 0.970409 w_k1_IBM_s010
                                 + 1.17052 w_k1_INTC_s010 + 1.1666 w_k1_KO_s010
                                 + 0.885749 w_k1_MAT_s010
                                 + 1.17056 w_k1_MSFT_s010
                                 + 1.19799 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.25257 w_k1_ADM_s010
                                 + 1.05745 w_k1_AXP_s010
                                 + 1.21915 w_k1_DIS_s010
                                 + 1.14843 w_k1_FDX_s010
                                 + 0.770966 w_k1_IBM_s010
                                 + 0.963415 w_k1_INTC_s010
                                 + 0.982971 w_k1_KO_s010
                                 + 0.572489 w_k1_MAT_s010
                                 + 1.09332 w_k1_MSFT_s010
                                 + 1.12723 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.03721 w_k1_ADM_s011
                                 + 1.16533 w_k1_AXP_s011
                                 + 1.46812 w_k1_DIS_s011 + 1.1661 w_k1_FDX_s011
                                 + 1.35262 w_k1_IBM_s011
                                 + 1.06938 w_k1_INTC_s011
                                 + 1.36469 w_k1_KO_s011 + 1.66274 w_k1_MAT_s011
                                 + 1.35026 w_k1_MSFT_s011
                                 + 1.14881 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.739213 w_k1_ADM_s011
                                 + 1.16199 w_k1_AXP_s011
                                 + 1.16162 w_k1_DIS_s011
                                 + 1.01827 w_k1_FDX_s011 + 1.6898 w_k1_IBM_s011
                                 + 1.49114 w_k1_INTC_s011
                                 + 1.12767 w_k1_KO_s011 + 1.32316 w_k1_MAT_s011
                                 + 1.06968 w_k1_MSFT_s011
                                 + 1.04539 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.706113 w_k1_ADM_s011
                                 + 0.696401 w_k1_AXP_s011
                                 + 0.520959 w_k1_DIS_s011
                                 + 0.719263 w_k1_FDX_s011
                                 + 0.677115 w_k1_IBM_s011
                                 + 0.894154 w_k1_INTC_s011
                                 + 0.727945 w_k1_KO_s011 + 0.8764 w_k1_MAT_s011
                                 + 0.939988 w_k1_MSFT_s011
                                 + 0.893013 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.683911 w_k1_ADM_s011
                                 + 1.37647 w_k1_AXP_s011
                                 + 1.16721 w_k1_DIS_s011
                                 + 1.14998 w_k1_FDX_s011
                                 + 1.06313 w_k1_IBM_s011
                                 + 1.35738 w_k1_INTC_s011
                                 + 0.919684 w_k1_KO_s011
                                 + 1.34833 w_k1_MAT_s011
                                 + 1.15057 w_k1_MSFT_s011
                                 + 1.01372 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.887672 w_k1_ADM_s011
                                 + 0.718472 w_k1_AXP_s011
                                 + 0.812654 w_k1_DIS_s011
                                 + 0.83733 w_k1_FDX_s011
                                 + 1.02439 w_k1_IBM_s011
                                 + 0.994992 w_k1_INTC_s011
                                 + 0.99786 w_k1_KO_s011
                                 + 0.872286 w_k1_MAT_s011
                                 + 0.915576 w_k1_MSFT_s011
                                 + 0.86062 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.20113 w_k1_ADM_s011
                                 + 1.11053 w_k1_AXP_s011
                                 + 1.23366 w_k1_DIS_s011
                                 + 1.16671 w_k1_FDX_s011
                                 + 0.882134 w_k1_IBM_s011
                                 + 1.31042 w_k1_INTC_s011
                                 + 0.90562 w_k1_KO_s011
                                 + 0.801358 w_k1_MAT_s011
                                 + 1.16722 w_k1_MSFT_s011
                                 + 1.06801 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.00234 w_k1_ADM_s011
                                 + 0.800426 w_k1_AXP_s011
                                 + 0.852424 w_k1_DIS_s011
                                 + 0.870483 w_k1_FDX_s011
                                 + 0.753191 w_k1_IBM_s011
                                 + 0.984135 w_k1_INTC_s011
                                 + 0.855499 w_k1_KO_s011
                                 + 0.795324 w_k1_MAT_s011
                                 + 0.967392 w_k1_MSFT_s011
                                 + 0.982358 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.863847 w_k1_ADM_s011
                                 + 0.766811 w_k1_AXP_s011
                                 + 0.761918 w_k1_DIS_s011
                                 + 0.819173 w_k1_FDX_s011
                                 + 1.13129 w_k1_IBM_s011
                                 + 0.869415 w_k1_INTC_s011
                                 + 0.898598 w_k1_KO_s011
                                 + 1.07866 w_k1_MAT_s011
                                 + 0.90251 w_k1_MSFT_s011
                                 + 0.878633 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.43518 w_k1_ADM_s012
                                 + 1.37377 w_k1_AXP_s012 + 1.3123 w_k1_DIS_s012
                                 + 1.13258 w_k1_FDX_s012
                                 + 0.828854 w_k1_IBM_s012
                                 + 0.919905 w_k1_INTC_s012
                                 + 1.05033 w_k1_KO_s012
                                 + 0.975256 w_k1_MAT_s012
                                 + 1.16712 w_k1_MSFT_s012
                                 + 1.08988 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.07549 w_k1_ADM_s012
                                 + 1.16537 w_k1_AXP_s012
                                 + 0.772233 w_k1_DIS_s012
                                 + 0.939851 w_k1_FDX_s012
                                 + 0.879538 w_k1_IBM_s012
                                 + 0.918494 w_k1_INTC_s012
                                 + 0.86173 w_k1_KO_s012
                                 + 0.839387 w_k1_MAT_s012
                                 + 0.872599 w_k1_MSFT_s012
                                 + 0.972969 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.986141 w_k1_ADM_s012
                                 + 0.979708 w_k1_AXP_s012
                                 + 0.952567 w_k1_DIS_s012
                                 + 0.841707 w_k1_FDX_s012
                                 + 1.31608 w_k1_IBM_s012
                                 + 1.20998 w_k1_INTC_s012
                                 + 1.24621 w_k1_KO_s012
                                 + 0.885689 w_k1_MAT_s012
                                 + 1.13399 w_k1_MSFT_s012
                                 + 0.937957 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.40967 w_k1_ADM_s012
                                 + 0.778938 w_k1_AXP_s012
                                 + 0.931587 w_k1_DIS_s012
                                 + 0.946933 w_k1_FDX_s012
                                 + 0.648322 w_k1_IBM_s012
                                 + 0.788673 w_k1_INTC_s012
                                 + 0.78931 w_k1_KO_s012 + 0.47024 w_k1_MAT_s012
                                 + 0.93738 w_k1_MSFT_s012
                                 + 0.925212 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.05859 w_k1_ADM_s012
                                 + 1.23634 w_k1_AXP_s012
                                 + 1.02963 w_k1_DIS_s012
                                 + 1.04711 w_k1_FDX_s012 + 1.1436 w_k1_IBM_s012
                                 + 1.05604 w_k1_INTC_s012
                                 + 1.19595 w_k1_KO_s012 + 1.11202 w_k1_MAT_s012
                                 + 1.11652 w_k1_MSFT_s012
                                 + 1.07009 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.928873 w_k1_ADM_s012
                                 + 0.746099 w_k1_AXP_s012
                                 + 0.748186 w_k1_DIS_s012
                                 + 0.768096 w_k1_FDX_s012
                                 + 0.801946 w_k1_IBM_s012
                                 + 0.84972 w_k1_INTC_s012
                                 + 0.846433 w_k1_KO_s012
                                 + 0.65566 w_k1_MAT_s012
                                 + 0.862899 w_k1_MSFT_s012
                                 + 0.80145 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.21684 w_k1_ADM_s012
                                 + 1.83655 w_k1_AXP_s012
                                 + 1.64719 w_k1_DIS_s012
                                 + 1.44974 w_k1_FDX_s012
                                 + 1.37318 w_k1_IBM_s012
                                 + 1.20954 w_k1_INTC_s012
                                 + 1.24592 w_k1_KO_s012 + 1.29988 w_k1_MAT_s012
                                 + 1.32876 w_k1_MSFT_s012 + 1.1002 w_k1_PG_s012
                                 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.971562 w_k1_ADM_s012
                                 + 0.997489 w_k1_AXP_s012
                                 + 1.16032 w_k1_DIS_s012
                                 + 0.974978 w_k1_FDX_s012
                                 + 1.84547 w_k1_IBM_s012
                                 + 0.847466 w_k1_INTC_s012
                                 + 1.18148 w_k1_KO_s012 + 1.23939 w_k1_MAT_s012
                                 + 1.10088 w_k1_MSFT_s012
                                 + 1.10285 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.853571 w_k1_ADM_s013
                                 + 0.945433 w_k1_AXP_s013
                                 + 1.04312 w_k1_DIS_s013
                                 + 0.906646 w_k1_FDX_s013
                                 + 1.20878 w_k1_IBM_s013
                                 + 0.992844 w_k1_INTC_s013
                                 + 1.0359 w_k1_KO_s013 + 0.86563 w_k1_MAT_s013
                                 + 1.01526 w_k1_MSFT_s013
                                 + 1.00119 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.904006 w_k1_ADM_s013
                                 + 1.00216 w_k1_AXP_s013
                                 + 0.905511 w_k1_DIS_s013
                                 + 0.942073 w_k1_FDX_s013
                                 + 0.784982 w_k1_IBM_s013
                                 + 0.84439 w_k1_INTC_s013
                                 + 0.812999 w_k1_KO_s013
                                 + 0.680755 w_k1_MAT_s013
                                 + 0.928759 w_k1_MSFT_s013
                                 + 1.06024 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.21967 w_k1_ADM_s013
                                 + 1.32859 w_k1_AXP_s013
                                 + 1.14124 w_k1_DIS_s013
                                 + 1.10431 w_k1_FDX_s013
                                 + 1.61449 w_k1_IBM_s013
                                 + 0.948945 w_k1_INTC_s013
                                 + 1.19274 w_k1_KO_s013 + 1.47131 w_k1_MAT_s013
                                 + 0.946592 w_k1_MSFT_s013
                                 + 1.18067 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.834072 w_k1_ADM_s013
                                 + 0.668747 w_k1_AXP_s013
                                 + 0.73491 w_k1_DIS_s013
                                 + 0.77176 w_k1_FDX_s013
                                 + 0.810716 w_k1_IBM_s013
                                 + 0.831072 w_k1_INTC_s013
                                 + 0.737458 w_k1_KO_s013
                                 + 0.559321 w_k1_MAT_s013
                                 + 0.812869 w_k1_MSFT_s013
                                 + 0.930362 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.636948 w_k1_ADM_s013
                                 + 0.911888 w_k1_AXP_s013
                                 + 0.77578 w_k1_DIS_s013
                                 + 0.765046 w_k1_FDX_s013
                                 + 1.27074 w_k1_IBM_s013
                                 + 0.966248 w_k1_INTC_s013
                                 + 0.988475 w_k1_KO_s013
                                 + 1.18512 w_k1_MAT_s013
                                 + 0.782974 w_k1_MSFT_s013
                                 + 0.978559 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.984026 w_k1_ADM_s013
                                 + 1.32984 w_k1_AXP_s013
                                 + 1.36652 w_k1_DIS_s013
                                 + 1.25505 w_k1_FDX_s013
                                 + 1.13937 w_k1_IBM_s013
                                 + 1.47713 w_k1_INTC_s013
                                 + 1.24544 w_k1_KO_s013
                                 + 0.957524 w_k1_MAT_s013
                                 + 1.20881 w_k1_MSFT_s013
                                 + 1.19657 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.866224 w_k1_ADM_s013
                                 + 0.994679 w_k1_AXP_s013
                                 + 1.11956 w_k1_DIS_s013
                                 + 1.08819 w_k1_FDX_s013
                                 + 0.897833 w_k1_IBM_s013
                                 + 1.13617 w_k1_INTC_s013
                                 + 0.927201 w_k1_KO_s013
                                 + 0.553763 w_k1_MAT_s013
                                 + 1.05063 w_k1_MSFT_s013
                                 + 0.963954 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.644717 w_k1_ADM_s013
                                 + 0.800029 w_k1_AXP_s013
                                 + 0.723533 w_k1_DIS_s013
                                 + 0.738543 w_k1_FDX_s013
                                 + 1.0079 w_k1_IBM_s013
                                 + 1.07391 w_k1_INTC_s013
                                 + 0.81758 w_k1_KO_s013
                                 + 0.839816 w_k1_MAT_s013
                                 + 0.810427 w_k1_MSFT_s013
                                 + 0.961431 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.19219 w_k1_ADM_s014
                                 + 0.972712 w_k1_AXP_s014
                                 + 1.07783 w_k1_DIS_s014
                                 + 1.10874 w_k1_FDX_s014
                                 + 0.680324 w_k1_IBM_s014
                                 + 0.830991 w_k1_INTC_s014
                                 + 0.882839 w_k1_KO_s014
                                 + 0.970746 w_k1_MAT_s014
                                 + 1.11334 w_k1_MSFT_s014
                                 + 0.804123 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.17131 w_k1_ADM_s014
                                 + 1.1411 w_k1_AXP_s014 + 1.36044 w_k1_DIS_s014
                                 + 1.08356 w_k1_FDX_s014
                                 + 1.61101 w_k1_IBM_s014
                                 + 1.05806 w_k1_INTC_s014
                                 + 1.37941 w_k1_KO_s014 + 1.31266 w_k1_MAT_s014
                                 + 1.18916 w_k1_MSFT_s014
                                 + 0.975325 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.993989 w_k1_ADM_s014
                                 + 0.595669 w_k1_AXP_s014
                                 + 0.77498 w_k1_DIS_s014
                                 + 0.855028 w_k1_FDX_s014
                                 + 0.678583 w_k1_IBM_s014
                                 + 0.843366 w_k1_INTC_s014
                                 + 0.724196 w_k1_KO_s014
                                 + 0.508922 w_k1_MAT_s014
                                 + 0.961141 w_k1_MSFT_s014
                                 + 0.808581 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.09088 w_k1_ADM_s014
                                 + 0.693274 w_k1_AXP_s014
                                 + 0.588228 w_k1_DIS_s014
                                 + 0.715063 w_k1_FDX_s014
                                 + 0.80155 w_k1_IBM_s014
                                 + 0.654172 w_k1_INTC_s014
                                 + 0.654173 w_k1_KO_s014
                                 + 0.831907 w_k1_MAT_s014
                                 + 0.718643 w_k1_MSFT_s014
                                 + 0.794569 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.912438 w_k1_ADM_s014
                                 + 1.13197 w_k1_AXP_s014
                                 + 0.781996 w_k1_DIS_s014
                                 + 0.781093 w_k1_FDX_s014
                                 + 1.28631 w_k1_IBM_s014
                                 + 0.75941 w_k1_INTC_s014
                                 + 0.959693 w_k1_KO_s014
                                 + 1.31032 w_k1_MAT_s014
                                 + 0.751606 w_k1_MSFT_s014
                                 + 0.901051 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.19484 w_k1_ADM_s014
                                 + 1.36767 w_k1_AXP_s014 + 1.0611 w_k1_DIS_s014
                                 + 1.09858 w_k1_FDX_s014
                                 + 1.19395 w_k1_IBM_s014
                                 + 0.920581 w_k1_INTC_s014
                                 + 1.19498 w_k1_KO_s014 + 1.34106 w_k1_MAT_s014
                                 + 0.991048 w_k1_MSFT_s014
                                 + 1.04241 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.04648 w_k1_ADM_s014
                                 + 1.25767 w_k1_AXP_s014
                                 + 1.46878 w_k1_DIS_s014
                                 + 1.24034 w_k1_FDX_s014
                                 + 1.13502 w_k1_IBM_s014
                                 + 0.89972 w_k1_INTC_s014
                                 + 1.13662 w_k1_KO_s014 + 1.26005 w_k1_MAT_s014
                                 + 1.19957 w_k1_MSFT_s014
                                 + 1.13556 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.01439 w_k1_ADM_s014
                                 + 0.831277 w_k1_AXP_s014
                                 + 0.988288 w_k1_DIS_s014
                                 + 0.947442 w_k1_FDX_s014
                                 + 0.989852 w_k1_IBM_s014
                                 + 0.756125 w_k1_INTC_s014
                                 + 1.07468 w_k1_KO_s014 + 1.28894 w_k1_MAT_s014
                                 + 0.929584 w_k1_MSFT_s014
                                 + 0.902068 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.15528 w_k1_ADM_s015
                                 + 1.58387 w_k1_AXP_s015
                                 + 1.90954 w_k1_DIS_s015
                                 + 1.27358 w_k1_FDX_s015
                                 + 1.47044 w_k1_IBM_s015
                                 + 1.14848 w_k1_INTC_s015
                                 + 1.69529 w_k1_KO_s015 + 2.13121 w_k1_MAT_s015
                                 + 1.35127 w_k1_MSFT_s015
                                 + 1.22995 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.05322 w_k1_ADM_s015
                                 + 0.993697 w_k1_AXP_s015
                                 + 1.04424 w_k1_DIS_s015 + 1.1193 w_k1_FDX_s015
                                 + 0.641422 w_k1_IBM_s015
                                 + 0.990329 w_k1_INTC_s015
                                 + 1.01158 w_k1_KO_s015
                                 + 0.926701 w_k1_MAT_s015
                                 + 1.05639 w_k1_MSFT_s015
                                 + 0.866942 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.807125 w_k1_ADM_s015
                                 + 0.728286 w_k1_AXP_s015
                                 + 0.724588 w_k1_DIS_s015
                                 + 0.789544 w_k1_FDX_s015
                                 + 0.597792 w_k1_IBM_s015
                                 + 0.934061 w_k1_INTC_s015
                                 + 0.791497 w_k1_KO_s015
                                 + 0.583786 w_k1_MAT_s015
                                 + 0.888096 w_k1_MSFT_s015
                                 + 0.887283 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.29075 w_k1_ADM_s015
                                 + 1.54018 w_k1_AXP_s015
                                 + 1.15031 w_k1_DIS_s015
                                 + 1.18749 w_k1_FDX_s015
                                 + 1.13266 w_k1_IBM_s015
                                 + 1.12369 w_k1_INTC_s015
                                 + 0.984881 w_k1_KO_s015
                                 + 1.12092 w_k1_MAT_s015
                                 + 1.09073 w_k1_MSFT_s015
                                 + 1.06501 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.06405 w_k1_ADM_s015
                                 + 1.41648 w_k1_AXP_s015
                                 + 1.63508 w_k1_DIS_s015 + 1.4887 w_k1_FDX_s015
                                 + 0.834636 w_k1_IBM_s015
                                 + 1.44871 w_k1_INTC_s015
                                 + 1.06466 w_k1_KO_s015 + 1.08295 w_k1_MAT_s015
                                 + 1.38859 w_k1_MSFT_s015
                                 + 1.18547 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.759813 w_k1_ADM_s015
                                 + 0.78969 w_k1_AXP_s015
                                 + 0.892118 w_k1_DIS_s015
                                 + 0.859679 w_k1_FDX_s015
                                 + 0.916363 w_k1_IBM_s015
                                 + 1.04213 w_k1_INTC_s015
                                 + 0.903102 w_k1_KO_s015
                                 + 0.835771 w_k1_MAT_s015
                                 + 0.910288 w_k1_MSFT_s015
                                 + 0.814306 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.661975 w_k1_ADM_s015
                                 + 0.920532 w_k1_AXP_s015
                                 + 0.741336 w_k1_DIS_s015
                                 + 0.728223 w_k1_FDX_s015
                                 + 1.02855 w_k1_IBM_s015
                                 + 0.92192 w_k1_INTC_s015
                                 + 0.926996 w_k1_KO_s015
                                 + 1.08463 w_k1_MAT_s015
                                 + 0.788501 w_k1_MSFT_s015
                                 + 0.849835 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.821376 w_k1_ADM_s015
                                 + 1.0821 w_k1_AXP_s015
                                 + 0.908186 w_k1_DIS_s015
                                 + 0.843317 w_k1_FDX_s015
                                 + 1.42491 w_k1_IBM_s015
                                 + 0.993518 w_k1_INTC_s015
                                 + 1.14115 w_k1_KO_s015 + 1.82449 w_k1_MAT_s015
                                 + 0.843543 w_k1_MSFT_s015
                                 + 0.942584 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.874315 w_k1_ADM_s016
                                 + 0.843586 w_k1_AXP_s016
                                 + 0.6627 w_k1_DIS_s016 + 0.73019 w_k1_FDX_s016
                                 + 0.930648 w_k1_IBM_s016
                                 + 0.939887 w_k1_INTC_s016
                                 + 0.866374 w_k1_KO_s016
                                 + 0.704286 w_k1_MAT_s016
                                 + 0.800243 w_k1_MSFT_s016
                                 + 0.896648 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.00393 w_k1_ADM_s016
                                 + 1.13451 w_k1_AXP_s016
                                 + 1.09965 w_k1_DIS_s016
                                 + 1.11097 w_k1_FDX_s016
                                 + 0.797756 w_k1_IBM_s016
                                 + 1.00539 w_k1_INTC_s016
                                 + 0.927694 w_k1_KO_s016
                                 + 0.984344 w_k1_MAT_s016
                                 + 1.07239 w_k1_MSFT_s016
                                 + 0.977462 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.19973 w_k1_ADM_s016
                                 + 1.29274 w_k1_AXP_s016
                                 + 1.23655 w_k1_DIS_s016
                                 + 1.05638 w_k1_FDX_s016
                                 + 1.60788 w_k1_IBM_s016
                                 + 0.947019 w_k1_INTC_s016
                                 + 1.4093 w_k1_KO_s016 + 1.9966 w_k1_MAT_s016
                                 + 0.975605 w_k1_MSFT_s016
                                 + 1.1037 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.10461 w_k1_ADM_s016
                                 + 1.04065 w_k1_AXP_s016 + 1.0994 w_k1_DIS_s016
                                 + 0.88477 w_k1_FDX_s016
                                 + 1.24486 w_k1_IBM_s016
                                 + 1.09272 w_k1_INTC_s016
                                 + 1.25475 w_k1_KO_s016 + 1.18178 w_k1_MAT_s016
                                 + 1.00347 w_k1_MSFT_s016
                                 + 1.04594 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.802329 w_k1_ADM_s016
                                 + 0.982268 w_k1_AXP_s016
                                 + 1.00936 w_k1_DIS_s016
                                 + 0.945206 w_k1_FDX_s016
                                 + 0.820577 w_k1_IBM_s016
                                 + 1.071 w_k1_INTC_s016 + 1.01248 w_k1_KO_s016
                                 + 0.97659 w_k1_MAT_s016
                                 + 1.01524 w_k1_MSFT_s016
                                 + 0.979196 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.47704 w_k1_ADM_s016
                                 + 1.85692 w_k1_AXP_s016
                                 + 1.68294 w_k1_DIS_s016 + 1.3807 w_k1_FDX_s016
                                 + 1.24216 w_k1_IBM_s016
                                 + 1.30663 w_k1_INTC_s016
                                 + 1.52179 w_k1_KO_s016 + 1.74165 w_k1_MAT_s016
                                 + 1.31386 w_k1_MSFT_s016
                                 + 1.22475 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.26449 w_k1_ADM_s016
                                 + 1.46225 w_k1_AXP_s016
                                 + 1.23256 w_k1_DIS_s016
                                 + 1.17922 w_k1_FDX_s016
                                 + 0.982648 w_k1_IBM_s016
                                 + 1.22093 w_k1_INTC_s016 + 1.175 w_k1_KO_s016
                                 + 1.12968 w_k1_MAT_s016
                                 + 1.36846 w_k1_MSFT_s016
                                 + 1.12502 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.04372 w_k1_ADM_s016
                                 + 0.956102 w_k1_AXP_s016
                                 + 0.74423 w_k1_DIS_s016
                                 + 0.916927 w_k1_FDX_s016
                                 + 0.603036 w_k1_IBM_s016
                                 + 0.960259 w_k1_INTC_s016
                                 + 0.777319 w_k1_KO_s016
                                 + 0.609398 w_k1_MAT_s016
                                 + 0.873264 w_k1_MSFT_s016
                                 + 0.828205 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.780049 w_k1_ADM_s017
                                 + 0.779279 w_k1_AXP_s017
                                 + 1.03678 w_k1_DIS_s017
                                 + 0.85736 w_k1_FDX_s017
                                 + 1.14147 w_k1_IBM_s017
                                 + 1.21858 w_k1_INTC_s017
                                 + 1.09551 w_k1_KO_s017
                                 + 0.857304 w_k1_MAT_s017
                                 + 1.06691 w_k1_MSFT_s017 + 1.0638 w_k1_PG_s017
                                 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.991155 w_k1_ADM_s017
                                 + 1.14768 w_k1_AXP_s017
                                 + 0.953422 w_k1_DIS_s017
                                 + 1.00647 w_k1_FDX_s017
                                 + 0.814198 w_k1_IBM_s017
                                 + 1.08997 w_k1_INTC_s017
                                 + 0.932803 w_k1_KO_s017
                                 + 0.660566 w_k1_MAT_s017
                                 + 1.07979 w_k1_MSFT_s017
                                 + 0.992846 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.982512 w_k1_ADM_s017
                                 + 1.07728 w_k1_AXP_s017
                                 + 1.12106 w_k1_DIS_s017
                                 + 1.07473 w_k1_FDX_s017
                                 + 1.34256 w_k1_IBM_s017
                                 + 0.935903 w_k1_INTC_s017
                                 + 1.21081 w_k1_KO_s017 + 1.42942 w_k1_MAT_s017
                                 + 1.01069 w_k1_MSFT_s017 + 1.0847 w_k1_PG_s017
                                 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.20207 w_k1_ADM_s017
                                 + 1.62172 w_k1_AXP_s017
                                 + 1.59515 w_k1_DIS_s017
                                 + 1.33075 w_k1_FDX_s017 + 1.6055 w_k1_IBM_s017
                                 + 1.10305 w_k1_INTC_s017
                                 + 1.41792 w_k1_KO_s017 + 1.28684 w_k1_MAT_s017
                                 + 1.19175 w_k1_MSFT_s017
                                 + 1.16108 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.635811 w_k1_ADM_s017
                                 + 0.725278 w_k1_AXP_s017
                                 + 0.714375 w_k1_DIS_s017
                                 + 0.747401 w_k1_FDX_s017
                                 + 0.938822 w_k1_IBM_s017
                                 + 0.842092 w_k1_INTC_s017
                                 + 0.903533 w_k1_KO_s017
                                 + 0.674176 w_k1_MAT_s017
                                 + 0.994341 w_k1_MSFT_s017
                                 + 0.990602 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.31465 w_k1_ADM_s017
                                 + 1.26946 w_k1_AXP_s017
                                 + 1.26727 w_k1_DIS_s017
                                 + 1.23159 w_k1_FDX_s017
                                 + 0.945694 w_k1_IBM_s017
                                 + 1.30285 w_k1_INTC_s017
                                 + 1.18627 w_k1_KO_s017
                                 + 0.628221 w_k1_MAT_s017
                                 + 1.35165 w_k1_MSFT_s017
                                 + 1.18659 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.97275 w_k1_ADM_s017
                                 + 0.834421 w_k1_AXP_s017
                                 + 0.589422 w_k1_DIS_s017
                                 + 0.803542 w_k1_FDX_s017
                                 + 0.829449 w_k1_IBM_s017
                                 + 0.791507 w_k1_INTC_s017
                                 + 0.759196 w_k1_KO_s017
                                 + 0.480279 w_k1_MAT_s017
                                 + 0.694755 w_k1_MSFT_s017
                                 + 0.853761 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.818915 w_k1_ADM_s017
                                 + 1.02293 w_k1_AXP_s017
                                 + 0.959788 w_k1_DIS_s017
                                 + 0.909751 w_k1_FDX_s017
                                 + 1.51809 w_k1_IBM_s017
                                 + 1.15536 w_k1_INTC_s017 + 1.2216 w_k1_KO_s017
                                 + 0.978731 w_k1_MAT_s017
                                 + 0.972071 w_k1_MSFT_s017 + 1.031 w_k1_PG_s017
                                 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.37913 w_k1_ADM_s100
                                 + 1.18374 w_k1_AXP_s100
                                 + 1.06259 w_k1_DIS_s100
                                 + 1.15483 w_k1_FDX_s100
                                 + 0.933243 w_k1_IBM_s100
                                 + 1.07941 w_k1_INTC_s100
                                 + 0.904563 w_k1_KO_s100
                                 + 0.630925 w_k1_MAT_s100
                                 + 1.08458 w_k1_MSFT_s100
                                 + 1.10314 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.30161 w_k1_ADM_s100
                                 + 1.03893 w_k1_AXP_s100
                                 + 0.88902 w_k1_DIS_s100
                                 + 1.00655 w_k1_FDX_s100
                                 + 0.703339 w_k1_IBM_s100
                                 + 0.906661 w_k1_INTC_s100
                                 + 0.790733 w_k1_KO_s100
                                 + 0.72098 w_k1_MAT_s100
                                 + 0.932376 w_k1_MSFT_s100
                                 + 1.01469 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.15253 w_k1_ADM_s100
                                 + 0.798035 w_k1_AXP_s100
                                 + 0.715659 w_k1_DIS_s100
                                 + 0.823282 w_k1_FDX_s100
                                 + 0.95031 w_k1_IBM_s100
                                 + 0.871777 w_k1_INTC_s100
                                 + 0.836438 w_k1_KO_s100
                                 + 0.581829 w_k1_MAT_s100
                                 + 0.897135 w_k1_MSFT_s100
                                 + 0.903752 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.36238 w_k1_ADM_s100
                                 + 1.48898 w_k1_AXP_s100
                                 + 1.59387 w_k1_DIS_s100
                                 + 1.27618 w_k1_FDX_s100
                                 + 1.49028 w_k1_IBM_s100
                                 + 1.35173 w_k1_INTC_s100
                                 + 1.23127 w_k1_KO_s100 + 1.57449 w_k1_MAT_s100
                                 + 1.2435 w_k1_MSFT_s100
                                 + 0.942364 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.13845 w_k1_ADM_s100
                                 + 1.13478 w_k1_AXP_s100
                                 + 1.00296 w_k1_DIS_s100
                                 + 1.01771 w_k1_FDX_s100
                                 + 1.19235 w_k1_IBM_s100
                                 + 1.0778 w_k1_INTC_s100 + 1.03681 w_k1_KO_s100
                                 + 1.03418 w_k1_MAT_s100
                                 + 1.13158 w_k1_MSFT_s100
                                 + 0.876583 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.00927 w_k1_ADM_s100
                                 + 0.966592 w_k1_AXP_s100
                                 + 1.1659 w_k1_DIS_s100
                                 + 0.950472 w_k1_FDX_s100
                                 + 0.914386 w_k1_IBM_s100
                                 + 0.997837 w_k1_INTC_s100
                                 + 0.867599 w_k1_KO_s100
                                 + 0.51883 w_k1_MAT_s100
                                 + 0.93943 w_k1_MSFT_s100
                                 + 0.797067 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.93616 w_k1_ADM_s100
                                 + 1.09691 w_k1_AXP_s100
                                 + 0.802944 w_k1_DIS_s100
                                 + 0.885714 w_k1_FDX_s100
                                 + 1.34626 w_k1_IBM_s100
                                 + 0.897765 w_k1_INTC_s100
                                 + 0.939845 w_k1_KO_s100
                                 + 1.26908 w_k1_MAT_s100
                                 + 0.825137 w_k1_MSFT_s100
                                 + 0.869416 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.717552 w_k1_ADM_s100
                                 + 0.701507 w_k1_AXP_s100
                                 + 0.627769 w_k1_DIS_s100
                                 + 0.674518 w_k1_FDX_s100
                                 + 0.828446 w_k1_IBM_s100
                                 + 0.703189 w_k1_INTC_s100
                                 + 0.629775 w_k1_KO_s100
                                 + 0.420345 w_k1_MAT_s100
                                 + 0.746101 w_k1_MSFT_s100
                                 + 0.689114 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.01483 w_k1_ADM_s101
                                 + 1.13561 w_k1_AXP_s101
                                 + 0.898833 w_k1_DIS_s101
                                 + 1.08979 w_k1_FDX_s101
                                 + 0.764774 w_k1_IBM_s101
                                 + 0.921155 w_k1_INTC_s101
                                 + 0.681001 w_k1_KO_s101
                                 + 0.794619 w_k1_MAT_s101
                                 + 0.877759 w_k1_MSFT_s101
                                 + 0.901459 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.777838 w_k1_ADM_s101
                                 + 0.856772 w_k1_AXP_s101
                                 + 1.02989 w_k1_DIS_s101 + 1.0554 w_k1_FDX_s101
                                 + 0.966891 w_k1_IBM_s101
                                 + 0.982031 w_k1_INTC_s101
                                 + 0.886533 w_k1_KO_s101
                                 + 0.890649 w_k1_MAT_s101
                                 + 1.06353 w_k1_MSFT_s101
                                 + 1.06425 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.853946 w_k1_ADM_s101
                                 + 0.933719 w_k1_AXP_s101
                                 + 0.638031 w_k1_DIS_s101
                                 + 0.882356 w_k1_FDX_s101
                                 + 0.772761 w_k1_IBM_s101
                                 + 0.755453 w_k1_INTC_s101
                                 + 0.591984 w_k1_KO_s101
                                 + 0.805502 w_k1_MAT_s101
                                 + 0.708074 w_k1_MSFT_s101
                                 + 0.859184 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.885663 w_k1_ADM_s101
                                 + 1.31625 w_k1_AXP_s101
                                 + 1.44327 w_k1_DIS_s101
                                 + 1.38643 w_k1_FDX_s101
                                 + 0.936793 w_k1_IBM_s101
                                 + 1.03237 w_k1_INTC_s101
                                 + 0.907852 w_k1_KO_s101
                                 + 1.21661 w_k1_MAT_s101
                                 + 1.23359 w_k1_MSFT_s101
                                 + 1.06487 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.13418 w_k1_ADM_s101
                                 + 1.27049 w_k1_AXP_s101
                                 + 1.25013 w_k1_DIS_s101
                                 + 1.13079 w_k1_FDX_s101
                                 + 1.51207 w_k1_IBM_s101
                                 + 0.961838 w_k1_INTC_s101
                                 + 1.1886 w_k1_KO_s101 + 1.78961 w_k1_MAT_s101
                                 + 1.14841 w_k1_MSFT_s101
                                 + 1.24791 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.754173 w_k1_ADM_s101
                                 + 0.732376 w_k1_AXP_s101
                                 + 0.688714 w_k1_DIS_s101
                                 + 0.774096 w_k1_FDX_s101
                                 + 0.896721 w_k1_IBM_s101
                                 + 0.76955 w_k1_INTC_s101
                                 + 0.627812 w_k1_KO_s101
                                 + 0.664692 w_k1_MAT_s101
                                 + 0.774235 w_k1_MSFT_s101
                                 + 0.893793 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.10616 w_k1_ADM_s101
                                 + 1.19359 w_k1_AXP_s101
                                 + 1.13845 w_k1_DIS_s101
                                 + 1.12816 w_k1_FDX_s101
                                 + 1.31348 w_k1_IBM_s101
                                 + 0.828434 w_k1_INTC_s101
                                 + 1.01236 w_k1_KO_s101 + 1.48324 w_k1_MAT_s101
                                 + 0.981026 w_k1_MSFT_s101
                                 + 1.04672 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.00245 w_k1_ADM_s101
                                 + 1.03912 w_k1_AXP_s101
                                 + 0.965632 w_k1_DIS_s101
                                 + 1.02468 w_k1_FDX_s101
                                 + 1.22419 w_k1_IBM_s101
                                 + 0.860285 w_k1_INTC_s101
                                 + 0.951792 w_k1_KO_s101
                                 + 1.21495 w_k1_MAT_s101
                                 + 0.971792 w_k1_MSFT_s101
                                 + 0.968831 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.816806 w_k1_ADM_s102
                                 + 0.75808 w_k1_AXP_s102
                                 + 0.697764 w_k1_DIS_s102
                                 + 0.743766 w_k1_FDX_s102
                                 + 0.702389 w_k1_IBM_s102
                                 + 0.697089 w_k1_INTC_s102
                                 + 0.696674 w_k1_KO_s102
                                 + 0.611211 w_k1_MAT_s102
                                 + 0.724667 w_k1_MSFT_s102
                                 + 0.826934 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.953421 w_k1_ADM_s102
                                 + 1.32426 w_k1_AXP_s102 + 1.5236 w_k1_DIS_s102
                                 + 1.23047 w_k1_FDX_s102
                                 + 1.38022 w_k1_IBM_s102
                                 + 1.0455 w_k1_INTC_s102 + 1.26408 w_k1_KO_s102
                                 + 1.51435 w_k1_MAT_s102
                                 + 1.22062 w_k1_MSFT_s102
                                 + 1.26916 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.01057 w_k1_ADM_s102
                                 + 1.37394 w_k1_AXP_s102
                                 + 1.06103 w_k1_DIS_s102
                                 + 0.989241 w_k1_FDX_s102
                                 + 1.29884 w_k1_IBM_s102
                                 + 0.943403 w_k1_INTC_s102
                                 + 0.929239 w_k1_KO_s102
                                 + 1.58955 w_k1_MAT_s102
                                 + 0.919452 w_k1_MSFT_s102
                                 + 1.14077 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.99456 w_k1_ADM_s102
                                 + 0.840823 w_k1_AXP_s102
                                 + 0.6956 w_k1_DIS_s102
                                 + 0.796355 w_k1_FDX_s102
                                 + 1.08032 w_k1_IBM_s102
                                 + 0.685446 w_k1_INTC_s102
                                 + 0.794094 w_k1_KO_s102
                                 + 1.18144 w_k1_MAT_s102
                                 + 0.813437 w_k1_MSFT_s102
                                 + 0.931643 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.744789 w_k1_ADM_s102
                                 + 0.896206 w_k1_AXP_s102
                                 + 0.820818 w_k1_DIS_s102
                                 + 0.889533 w_k1_FDX_s102
                                 + 0.955933 w_k1_IBM_s102
                                 + 0.854945 w_k1_INTC_s102
                                 + 0.892427 w_k1_KO_s102
                                 + 0.93229 w_k1_MAT_s102
                                 + 0.918689 w_k1_MSFT_s102
                                 + 1.01729 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.89787 w_k1_ADM_s102
                                 + 1.75797 w_k1_AXP_s102 + 1.637 w_k1_DIS_s102
                                 + 1.53056 w_k1_FDX_s102 + 1.1369 w_k1_IBM_s102
                                 + 1.34464 w_k1_INTC_s102
                                 + 1.23226 w_k1_KO_s102 + 1.0769 w_k1_MAT_s102
                                 + 1.36679 w_k1_MSFT_s102
                                 + 1.24471 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.06395 w_k1_ADM_s102
                                 + 1.09346 w_k1_AXP_s102
                                 + 0.980785 w_k1_DIS_s102
                                 + 1.06707 w_k1_FDX_s102
                                 + 0.840275 w_k1_IBM_s102
                                 + 1.01516 w_k1_INTC_s102
                                 + 0.818551 w_k1_KO_s102
                                 + 0.94223 w_k1_MAT_s102
                                 + 1.02729 w_k1_MSFT_s102
                                 + 1.09719 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.19595 w_k1_ADM_s102
                                 + 1.19735 w_k1_AXP_s102
                                 + 1.20434 w_k1_DIS_s102
                                 + 1.18093 w_k1_FDX_s102
                                 + 0.855328 w_k1_IBM_s102
                                 + 1.03762 w_k1_INTC_s102
                                 + 0.965434 w_k1_KO_s102
                                 + 0.54726 w_k1_MAT_s102
                                 + 1.0334 w_k1_MSFT_s102 + 1.11132 w_k1_PG_s102
                                 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.32294 w_k1_ADM_s103
                                 + 1.57378 w_k1_AXP_s103
                                 + 1.27593 w_k1_DIS_s103
                                 + 1.39253 w_k1_FDX_s103
                                 + 0.868652 w_k1_IBM_s103
                                 + 1.22372 w_k1_INTC_s103
                                 + 0.974435 w_k1_KO_s103
                                 + 1.36556 w_k1_MAT_s103
                                 + 1.24279 w_k1_MSFT_s103
                                 + 1.07531 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.18322 w_k1_ADM_s103
                                 + 1.31027 w_k1_AXP_s103 + 1.1579 w_k1_DIS_s103
                                 + 1.04924 w_k1_FDX_s103
                                 + 1.21404 w_k1_IBM_s103
                                 + 1.30087 w_k1_INTC_s103
                                 + 1.14837 w_k1_KO_s103 + 1.33851 w_k1_MAT_s103
                                 + 1.14453 w_k1_MSFT_s103
                                 + 0.97398 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.768023 w_k1_ADM_s103
                                 + 0.853884 w_k1_AXP_s103
                                 + 0.981281 w_k1_DIS_s103
                                 + 0.839471 w_k1_FDX_s103
                                 + 1.21684 w_k1_IBM_s103
                                 + 0.962328 w_k1_INTC_s103
                                 + 1.07424 w_k1_KO_s103 + 1.57519 w_k1_MAT_s103
                                 + 0.920713 w_k1_MSFT_s103
                                 + 0.906252 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.778723 w_k1_ADM_s103
                                 + 0.833839 w_k1_AXP_s103
                                 + 0.613519 w_k1_DIS_s103
                                 + 0.760048 w_k1_FDX_s103
                                 + 0.733495 w_k1_IBM_s103
                                 + 0.834023 w_k1_INTC_s103
                                 + 0.706143 w_k1_KO_s103
                                 + 0.792009 w_k1_MAT_s103
                                 + 0.719099 w_k1_MSFT_s103
                                 + 0.820161 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.933302 w_k1_ADM_s103
                                 + 0.837947 w_k1_AXP_s103
                                 + 0.931253 w_k1_DIS_s103
                                 + 0.957374 w_k1_FDX_s103
                                 + 0.970166 w_k1_IBM_s103
                                 + 1.25295 w_k1_INTC_s103 + 1.0781 w_k1_KO_s103
                                 + 0.81704 w_k1_MAT_s103
                                 + 1.11914 w_k1_MSFT_s103
                                 + 0.958761 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.933574 w_k1_ADM_s103
                                 + 0.986315 w_k1_AXP_s103
                                 + 0.963965 w_k1_DIS_s103
                                 + 1.05918 w_k1_FDX_s103
                                 + 0.862245 w_k1_IBM_s103
                                 + 1.14931 w_k1_INTC_s103
                                 + 1.00209 w_k1_KO_s103
                                 + 0.878686 w_k1_MAT_s103
                                 + 1.00577 w_k1_MSFT_s103
                                 + 0.963198 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.44162 w_k1_ADM_s103
                                 + 1.92244 w_k1_AXP_s103
                                 + 1.70743 w_k1_DIS_s103
                                 + 1.49156 w_k1_FDX_s103
                                 + 1.45718 w_k1_IBM_s103
                                 + 1.35475 w_k1_INTC_s103
                                 + 1.42119 w_k1_KO_s103 + 2.19841 w_k1_MAT_s103
                                 + 1.24984 w_k1_MSFT_s103
                                 + 1.30774 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.722851 w_k1_ADM_s103
                                 + 1.18228 w_k1_AXP_s103
                                 + 0.891688 w_k1_DIS_s103
                                 + 0.820437 w_k1_FDX_s103
                                 + 1.63997 w_k1_IBM_s103
                                 + 1.24692 w_k1_INTC_s103
                                 + 1.21297 w_k1_KO_s103 + 2.4577 w_k1_MAT_s103
                                 + 1.07272 w_k1_MSFT_s103
                                 + 0.968859 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.908311 w_k1_ADM_s104
                                 + 1.01097 w_k1_AXP_s104
                                 + 0.660855 w_k1_DIS_s104
                                 + 0.876382 w_k1_FDX_s104
                                 + 0.815597 w_k1_IBM_s104
                                 + 0.93764 w_k1_INTC_s104
                                 + 0.727043 w_k1_KO_s104
                                 + 0.530563 w_k1_MAT_s104
                                 + 0.794614 w_k1_MSFT_s104
                                 + 0.806152 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.894093 w_k1_ADM_s104
                                 + 0.949823 w_k1_AXP_s104
                                 + 1.05686 w_k1_DIS_s104
                                 + 0.972714 w_k1_FDX_s104
                                 + 0.889382 w_k1_IBM_s104
                                 + 1.00639 w_k1_INTC_s104
                                 + 0.941595 w_k1_KO_s104
                                 + 0.688843 w_k1_MAT_s104
                                 + 0.998669 w_k1_MSFT_s104
                                 + 1.03892 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.14308 w_k1_ADM_s104
                                 + 0.989502 w_k1_AXP_s104
                                 + 1.10686 w_k1_DIS_s104
                                 + 0.974329 w_k1_FDX_s104
                                 + 1.55006 w_k1_IBM_s104
                                 + 1.01622 w_k1_INTC_s104 + 1.2705 w_k1_KO_s104
                                 + 1.21845 w_k1_MAT_s104
                                 + 1.01964 w_k1_MSFT_s104 + 1.1138 w_k1_PG_s104
                                 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.00298 w_k1_ADM_s104
                                 + 1.27965 w_k1_AXP_s104
                                 + 1.09519 w_k1_DIS_s104
                                 + 1.18384 w_k1_FDX_s104
                                 + 0.814562 w_k1_IBM_s104
                                 + 1.08433 w_k1_INTC_s104
                                 + 0.884046 w_k1_KO_s104
                                 + 0.862617 w_k1_MAT_s104
                                 + 1.0782 w_k1_MSFT_s104 + 1.0503 w_k1_PG_s104
                                 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.706245 w_k1_ADM_s104
                                 + 0.658331 w_k1_AXP_s104
                                 + 0.639135 w_k1_DIS_s104
                                 + 0.766633 w_k1_FDX_s104
                                 + 0.692079 w_k1_IBM_s104
                                 + 0.788386 w_k1_INTC_s104
                                 + 0.74075 w_k1_KO_s104
                                 + 0.431385 w_k1_MAT_s104
                                 + 0.636871 w_k1_MSFT_s104
                                 + 0.815026 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.2962 w_k1_ADM_s104
                                 + 1.3726 w_k1_AXP_s104 + 1.8141 w_k1_DIS_s104
                                 + 1.25259 w_k1_FDX_s104 + 1.181 w_k1_IBM_s104
                                 + 1.27248 w_k1_INTC_s104 + 1.3178 w_k1_KO_s104
                                 + 1.15416 w_k1_MAT_s104
                                 + 1.26776 w_k1_MSFT_s104
                                 + 1.13581 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.983566 w_k1_ADM_s104
                                 + 0.806806 w_k1_AXP_s104
                                 + 0.894495 w_k1_DIS_s104
                                 + 0.97134 w_k1_FDX_s104
                                 + 0.566758 w_k1_IBM_s104
                                 + 0.912262 w_k1_INTC_s104
                                 + 0.623027 w_k1_KO_s104
                                 + 0.608716 w_k1_MAT_s104
                                 + 0.910976 w_k1_MSFT_s104
                                 + 0.812474 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.12762 w_k1_ADM_s104
                                 + 1.3689 w_k1_AXP_s104 + 1.24933 w_k1_DIS_s104
                                 + 1.25211 w_k1_FDX_s104
                                 + 0.952738 w_k1_IBM_s104
                                 + 1.35186 w_k1_INTC_s104
                                 + 1.17763 w_k1_KO_s104
                                 + 0.705245 w_k1_MAT_s104
                                 + 1.16805 w_k1_MSFT_s104
                                 + 1.18584 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.685199 w_k1_ADM_s105
                                 + 0.832448 w_k1_AXP_s105
                                 + 0.642834 w_k1_DIS_s105
                                 + 0.721113 w_k1_FDX_s105
                                 + 0.703446 w_k1_IBM_s105
                                 + 0.76151 w_k1_INTC_s105
                                 + 0.646597 w_k1_KO_s105
                                 + 0.660366 w_k1_MAT_s105
                                 + 0.832306 w_k1_MSFT_s105
                                 + 0.718841 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.11096 w_k1_ADM_s105
                                 + 1.03684 w_k1_AXP_s105
                                 + 1.42097 w_k1_DIS_s105
                                 + 1.08106 w_k1_FDX_s105
                                 + 0.903467 w_k1_IBM_s105
                                 + 1.01301 w_k1_INTC_s105
                                 + 1.02163 w_k1_KO_s105
                                 + 0.458996 w_k1_MAT_s105
                                 + 1.01448 w_k1_MSFT_s105
                                 + 0.974253 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.46181 w_k1_ADM_s105
                                 + 1.70142 w_k1_AXP_s105
                                 + 1.40694 w_k1_DIS_s105
                                 + 1.44917 w_k1_FDX_s105
                                 + 0.959108 w_k1_IBM_s105
                                 + 1.14383 w_k1_INTC_s105
                                 + 1.04934 w_k1_KO_s105 + 1.06244 w_k1_MAT_s105
                                 + 1.16722 w_k1_MSFT_s105
                                 + 1.19648 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.919117 w_k1_ADM_s105
                                 + 0.856006 w_k1_AXP_s105
                                 + 0.975221 w_k1_DIS_s105
                                 + 0.913748 w_k1_FDX_s105
                                 + 0.620113 w_k1_IBM_s105
                                 + 0.87389 w_k1_INTC_s105
                                 + 0.850685 w_k1_KO_s105
                                 + 0.550327 w_k1_MAT_s105
                                 + 0.980895 w_k1_MSFT_s105
                                 + 0.890892 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.979753 w_k1_ADM_s105
                                 + 1.10794 w_k1_AXP_s105
                                 + 0.957496 w_k1_DIS_s105
                                 + 0.922187 w_k1_FDX_s105
                                 + 0.913617 w_k1_IBM_s105
                                 + 0.787868 w_k1_INTC_s105
                                 + 0.841286 w_k1_KO_s105
                                 + 0.936172 w_k1_MAT_s105
                                 + 0.792451 w_k1_MSFT_s105
                                 + 0.891242 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.980606 w_k1_ADM_s105
                                 + 0.911817 w_k1_AXP_s105
                                 + 0.7389 w_k1_DIS_s105
                                 + 0.676707 w_k1_FDX_s105
                                 + 1.28241 w_k1_IBM_s105
                                 + 0.701604 w_k1_INTC_s105
                                 + 0.90795 w_k1_KO_s105
                                 + 0.784998 w_k1_MAT_s105
                                 + 0.733599 w_k1_MSFT_s105
                                 + 0.923742 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.26951 w_k1_ADM_s105
                                 + 1.51667 w_k1_AXP_s105
                                 + 1.45752 w_k1_DIS_s105
                                 + 1.17801 w_k1_FDX_s105
                                 + 1.88454 w_k1_IBM_s105
                                 + 1.0608 w_k1_INTC_s105 + 1.47993 w_k1_KO_s105
                                 + 1.88727 w_k1_MAT_s105
                                 + 1.07096 w_k1_MSFT_s105
                                 + 1.17895 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.963435 w_k1_ADM_s105
                                 + 0.978041 w_k1_AXP_s105
                                 + 1.19806 w_k1_DIS_s105
                                 + 0.916248 w_k1_FDX_s105
                                 + 1.51967 w_k1_IBM_s105
                                 + 0.878749 w_k1_INTC_s105
                                 + 1.27374 w_k1_KO_s105 + 0.89337 w_k1_MAT_s105
                                 + 0.941179 w_k1_MSFT_s105
                                 + 1.09359 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.20358 w_k1_ADM_s106
                                 + 0.861378 w_k1_AXP_s106
                                 + 0.81725 w_k1_DIS_s106
                                 + 0.912094 w_k1_FDX_s106
                                 + 0.742707 w_k1_IBM_s106
                                 + 0.895719 w_k1_INTC_s106
                                 + 0.855964 w_k1_KO_s106
                                 + 0.772812 w_k1_MAT_s106
                                 + 0.956016 w_k1_MSFT_s106
                                 + 0.839754 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.31713 w_k1_ADM_s106
                                 + 1.19283 w_k1_AXP_s106
                                 + 1.17283 w_k1_DIS_s106
                                 + 1.00683 w_k1_FDX_s106
                                 + 1.19091 w_k1_IBM_s106
                                 + 1.00304 w_k1_INTC_s106
                                 + 1.18579 w_k1_KO_s106 + 1.27575 w_k1_MAT_s106
                                 + 1.05943 w_k1_MSFT_s106 + 1.1322 w_k1_PG_s106
                                 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.958428 w_k1_ADM_s106
                                 + 0.79378 w_k1_AXP_s106
                                 + 0.688244 w_k1_DIS_s106
                                 + 0.762986 w_k1_FDX_s106
                                 + 0.950624 w_k1_IBM_s106
                                 + 0.727718 w_k1_INTC_s106
                                 + 0.744174 w_k1_KO_s106
                                 + 0.935254 w_k1_MAT_s106
                                 + 0.761165 w_k1_MSFT_s106
                                 + 0.88291 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.50966 w_k1_ADM_s106
                                 + 1.0024 w_k1_AXP_s106
                                 + 0.871697 w_k1_DIS_s106
                                 + 1.05831 w_k1_FDX_s106
                                 + 0.72559 w_k1_IBM_s106
                                 + 0.883516 w_k1_INTC_s106
                                 + 0.768171 w_k1_KO_s106
                                 + 0.630018 w_k1_MAT_s106
                                 + 0.86216 w_k1_MSFT_s106
                                 + 0.903764 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.49623 w_k1_ADM_s106
                                 + 1.47694 w_k1_AXP_s106
                                 + 1.53357 w_k1_DIS_s106
                                 + 1.16617 w_k1_FDX_s106
                                 + 1.30396 w_k1_IBM_s106
                                 + 0.90461 w_k1_INTC_s106
                                 + 1.35641 w_k1_KO_s106 + 1.38509 w_k1_MAT_s106
                                 + 1.07785 w_k1_MSFT_s106
                                 + 1.26547 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.68074 w_k1_ADM_s106
                                 + 1.47579 w_k1_AXP_s106
                                 + 1.23381 w_k1_DIS_s106
                                 + 1.35328 w_k1_FDX_s106
                                 + 0.910629 w_k1_IBM_s106
                                 + 1.18331 w_k1_INTC_s106
                                 + 1.09658 w_k1_KO_s106
                                 + 0.875865 w_k1_MAT_s106
                                 + 1.1373 w_k1_MSFT_s106 + 1.12901 w_k1_PG_s106
                                 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.12971 w_k1_ADM_s106
                                 + 0.908072 w_k1_AXP_s106
                                 + 0.8519 w_k1_DIS_s106
                                 + 0.897597 w_k1_FDX_s106
                                 + 1.10669 w_k1_IBM_s106
                                 + 1.03176 w_k1_INTC_s106
                                 + 0.931366 w_k1_KO_s106
                                 + 0.980966 w_k1_MAT_s106
                                 + 1.07874 w_k1_MSFT_s106
                                 + 0.971816 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.887669 w_k1_ADM_s106
                                 + 0.9508 w_k1_AXP_s106
                                 + 0.908999 w_k1_DIS_s106
                                 + 0.834724 w_k1_FDX_s106
                                 + 1.54107 w_k1_IBM_s106
                                 + 0.885007 w_k1_INTC_s106
                                 + 1.21763 w_k1_KO_s106 + 1.32271 w_k1_MAT_s106
                                 + 1.01687 w_k1_MSFT_s106
                                 + 1.08803 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.05759 w_k1_ADM_s107
                                 + 1.23256 w_k1_AXP_s107 + 1.273 w_k1_DIS_s107
                                 + 1.04607 w_k1_FDX_s107
                                 + 1.06494 w_k1_IBM_s107
                                 + 1.1245 w_k1_INTC_s107 + 1.09307 w_k1_KO_s107
                                 + 1.28385 w_k1_MAT_s107
                                 + 0.965868 w_k1_MSFT_s107
                                 + 1.16166 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.894844 w_k1_ADM_s107
                                 + 0.914332 w_k1_AXP_s107
                                 + 0.958588 w_k1_DIS_s107
                                 + 0.948541 w_k1_FDX_s107
                                 + 0.834783 w_k1_IBM_s107
                                 + 1.13376 w_k1_INTC_s107
                                 + 1.12249 w_k1_KO_s107
                                 + 0.949925 w_k1_MAT_s107
                                 + 0.956495 w_k1_MSFT_s107
                                 + 1.01345 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.25762 w_k1_ADM_s107
                                 + 1.63736 w_k1_AXP_s107
                                 + 1.68858 w_k1_DIS_s107 + 1.3605 w_k1_FDX_s107
                                 + 1.11259 w_k1_IBM_s107
                                 + 1.07989 w_k1_INTC_s107
                                 + 1.28164 w_k1_KO_s107 + 1.61673 w_k1_MAT_s107
                                 + 1.31159 w_k1_MSFT_s107
                                 + 1.30678 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.64367 w_k1_ADM_s107
                                 + 0.712053 w_k1_AXP_s107
                                 + 0.711096 w_k1_DIS_s107
                                 + 0.71549 w_k1_FDX_s107
                                 + 0.990636 w_k1_IBM_s107
                                 + 0.899049 w_k1_INTC_s107
                                 + 0.870513 w_k1_KO_s107
                                 + 0.799731 w_k1_MAT_s107
                                 + 0.719317 w_k1_MSFT_s107
                                 + 1.01198 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.85286 w_k1_ADM_s107
                                 + 1.22426 w_k1_AXP_s107 + 1.157 w_k1_DIS_s107
                                 + 0.944428 w_k1_FDX_s107
                                 + 1.79396 w_k1_IBM_s107
                                 + 1.26256 w_k1_INTC_s107
                                 + 1.36107 w_k1_KO_s107 + 1.94413 w_k1_MAT_s107
                                 + 0.974824 w_k1_MSFT_s107
                                 + 1.07473 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.05283 w_k1_ADM_s107
                                 + 1.28719 w_k1_AXP_s107
                                 + 1.12814 w_k1_DIS_s107
                                 + 1.12565 w_k1_FDX_s107
                                 + 0.796096 w_k1_IBM_s107
                                 + 1.19427 w_k1_INTC_s107 + 1.0023 w_k1_KO_s107
                                 + 0.871404 w_k1_MAT_s107
                                 + 1.01854 w_k1_MSFT_s107
                                 + 1.14301 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.798173 w_k1_ADM_s107
                                 + 0.880133 w_k1_AXP_s107
                                 + 0.773141 w_k1_DIS_s107
                                 + 0.929438 w_k1_FDX_s107
                                 + 0.596147 w_k1_IBM_s107
                                 + 0.981036 w_k1_INTC_s107
                                 + 0.710031 w_k1_KO_s107
                                 + 0.777478 w_k1_MAT_s107
                                 + 0.804186 w_k1_MSFT_s107
                                 + 0.851054 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.721092 w_k1_ADM_s107
                                 + 0.80464 w_k1_AXP_s107
                                 + 0.93331 w_k1_DIS_s107
                                 + 0.777097 w_k1_FDX_s107
                                 + 1.07742 w_k1_IBM_s107
                                 + 1.0541 w_k1_INTC_s107 + 1.15304 w_k1_KO_s107
                                 + 0.949739 w_k1_MAT_s107
                                 + 0.876257 w_k1_MSFT_s107
                                 + 1.10307 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.03324 w_k1_ADM_s110
                                 + 0.890644 w_k1_AXP_s110
                                 + 0.810195 w_k1_DIS_s110
                                 + 0.965608 w_k1_FDX_s110
                                 + 0.731524 w_k1_IBM_s110
                                 + 0.986428 w_k1_INTC_s110
                                 + 0.922563 w_k1_KO_s110
                                 + 0.713435 w_k1_MAT_s110
                                 + 1.10561 w_k1_MSFT_s110
                                 + 0.92937 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.961107 w_k1_ADM_s110
                                 + 0.84735 w_k1_AXP_s110
                                 + 0.822832 w_k1_DIS_s110
                                 + 0.819078 w_k1_FDX_s110
                                 + 1.1449 w_k1_IBM_s110
                                 + 0.885112 w_k1_INTC_s110
                                 + 0.993737 w_k1_KO_s110
                                 + 1.20623 w_k1_MAT_s110
                                 + 0.920105 w_k1_MSFT_s110
                                 + 0.949349 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.28119 w_k1_ADM_s110
                                 + 1.06656 w_k1_AXP_s110
                                 + 1.32457 w_k1_DIS_s110
                                 + 1.32812 w_k1_FDX_s110
                                 + 0.641758 w_k1_IBM_s110
                                 + 1.13384 w_k1_INTC_s110 + 1.0003 w_k1_KO_s110
                                 + 0.790488 w_k1_MAT_s110
                                 + 1.35626 w_k1_MSFT_s110
                                 + 0.989015 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.19967 w_k1_ADM_s110
                                 + 1.03483 w_k1_AXP_s110
                                 + 1.26287 w_k1_DIS_s110
                                 + 1.04644 w_k1_FDX_s110 + 1.1645 w_k1_IBM_s110
                                 + 0.899978 w_k1_INTC_s110
                                 + 1.35327 w_k1_KO_s110 + 1.66541 w_k1_MAT_s110
                                 + 1.21333 w_k1_MSFT_s110
                                 + 1.16394 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.53978 w_k1_ADM_s110
                                 + 1.28266 w_k1_AXP_s110
                                 + 1.71913 w_k1_DIS_s110
                                 + 1.43736 w_k1_FDX_s110
                                 + 1.23328 w_k1_IBM_s110
                                 + 1.15849 w_k1_INTC_s110
                                 + 1.46767 w_k1_KO_s110 + 1.55103 w_k1_MAT_s110
                                 + 1.41432 w_k1_MSFT_s110
                                 + 1.39756 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.985953 w_k1_ADM_s110
                                 + 0.934645 w_k1_AXP_s110
                                 + 1.19872 w_k1_DIS_s110
                                 + 0.97216 w_k1_FDX_s110
                                 + 1.77416 w_k1_IBM_s110
                                 + 0.90609 w_k1_INTC_s110
                                 + 1.43549 w_k1_KO_s110 + 2.13299 w_k1_MAT_s110
                                 + 1.07304 w_k1_MSFT_s110
                                 + 1.16017 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.943843 w_k1_ADM_s110
                                 + 0.512658 w_k1_AXP_s110
                                 + 0.782099 w_k1_DIS_s110
                                 + 0.77684 w_k1_FDX_s110
                                 + 0.729954 w_k1_IBM_s110
                                 + 0.79557 w_k1_INTC_s110
                                 + 0.990343 w_k1_KO_s110
                                 + 0.492617 w_k1_MAT_s110
                                 + 0.966581 w_k1_MSFT_s110
                                 + 0.835798 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.890765 w_k1_ADM_s110
                                 + 0.672576 w_k1_AXP_s110
                                 + 0.531707 w_k1_DIS_s110
                                 + 0.730246 w_k1_FDX_s110
                                 + 0.774792 w_k1_IBM_s110
                                 + 0.635983 w_k1_INTC_s110
                                 + 0.762356 w_k1_KO_s110
                                 + 0.94552 w_k1_MAT_s110
                                 + 0.771787 w_k1_MSFT_s110
                                 + 0.851564 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.04028 w_k1_ADM_s111
                                 + 0.947835 w_k1_AXP_s111
                                 + 0.937967 w_k1_DIS_s111
                                 + 1.14146 w_k1_FDX_s111
                                 + 0.760406 w_k1_IBM_s111
                                 + 1.03727 w_k1_INTC_s111
                                 + 0.843132 w_k1_KO_s111
                                 + 0.541245 w_k1_MAT_s111
                                 + 1.18279 w_k1_MSFT_s111
                                 + 0.931857 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.08452 w_k1_ADM_s111
                                 + 1.24271 w_k1_AXP_s111
                                 + 1.10963 w_k1_DIS_s111
                                 + 1.10653 w_k1_FDX_s111
                                 + 1.42909 w_k1_IBM_s111
                                 + 0.918981 w_k1_INTC_s111
                                 + 1.1166 w_k1_KO_s111 + 1.28685 w_k1_MAT_s111
                                 + 1.03546 w_k1_MSFT_s111
                                 + 1.01299 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.967144 w_k1_ADM_s111
                                 + 0.687809 w_k1_AXP_s111
                                 + 0.810058 w_k1_DIS_s111
                                 + 0.820311 w_k1_FDX_s111
                                 + 1.16397 w_k1_IBM_s111
                                 + 0.763655 w_k1_INTC_s111
                                 + 0.980529 w_k1_KO_s111
                                 + 0.949201 w_k1_MAT_s111
                                 + 0.922025 w_k1_MSFT_s111
                                 + 0.945856 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.3217 w_k1_ADM_s111
                                 + 1.49369 w_k1_AXP_s111
                                 + 1.49431 w_k1_DIS_s111
                                 + 1.37475 w_k1_FDX_s111
                                 + 1.16364 w_k1_IBM_s111
                                 + 1.13306 w_k1_INTC_s111
                                 + 1.15356 w_k1_KO_s111 + 1.06278 w_k1_MAT_s111
                                 + 1.45017 w_k1_MSFT_s111 + 1.0814 w_k1_PG_s111
                                 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.22826 w_k1_ADM_s111
                                 + 1.17067 w_k1_AXP_s111
                                 + 1.29285 w_k1_DIS_s111
                                 + 1.12837 w_k1_FDX_s111 + 1.0444 w_k1_IBM_s111
                                 + 1.03035 w_k1_INTC_s111
                                 + 1.22285 w_k1_KO_s111
                                 + 0.991812 w_k1_MAT_s111
                                 + 1.21749 w_k1_MSFT_s111
                                 + 1.08854 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.997597 w_k1_ADM_s111
                                 + 0.890573 w_k1_AXP_s111
                                 + 1.06584 w_k1_DIS_s111
                                 + 0.993796 w_k1_FDX_s111
                                 + 1.89437 w_k1_IBM_s111
                                 + 1.01441 w_k1_INTC_s111
                                 + 1.45496 w_k1_KO_s111 + 1.73507 w_k1_MAT_s111
                                 + 1.20934 w_k1_MSFT_s111 + 1.1344 w_k1_PG_s111
                                 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.706556 w_k1_ADM_s111
                                 + 0.732531 w_k1_AXP_s111
                                 + 0.658506 w_k1_DIS_s111
                                 + 0.82456 w_k1_FDX_s111
                                 + 0.749359 w_k1_IBM_s111
                                 + 0.873435 w_k1_INTC_s111
                                 + 0.739496 w_k1_KO_s111
                                 + 0.662074 w_k1_MAT_s111
                                 + 0.910585 w_k1_MSFT_s111
                                 + 0.846183 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.971781 w_k1_ADM_s111
                                 + 0.721675 w_k1_AXP_s111
                                 + 1.08055 w_k1_DIS_s111
                                 + 0.943874 w_k1_FDX_s111
                                 + 0.910096 w_k1_IBM_s111
                                 + 0.921729 w_k1_INTC_s111 + 1.105 w_k1_KO_s111
                                 + 0.528596 w_k1_MAT_s111
                                 + 1.16702 w_k1_MSFT_s111
                                 + 0.958174 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.841542 w_k1_ADM_s112
                                 + 0.81241 w_k1_AXP_s112
                                 + 0.837731 w_k1_DIS_s112
                                 + 0.891506 w_k1_FDX_s112
                                 + 0.704796 w_k1_IBM_s112
                                 + 0.85717 w_k1_INTC_s112
                                 + 0.69916 w_k1_KO_s112
                                 + 0.853193 w_k1_MAT_s112
                                 + 0.762263 w_k1_MSFT_s112
                                 + 0.851154 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.970149 w_k1_ADM_s112
                                 + 1.28883 w_k1_AXP_s112
                                 + 1.00066 w_k1_DIS_s112 + 1.1438 w_k1_FDX_s112
                                 + 0.762625 w_k1_IBM_s112
                                 + 1.10534 w_k1_INTC_s112
                                 + 0.850618 w_k1_KO_s112
                                 + 0.882007 w_k1_MAT_s112
                                 + 0.957274 w_k1_MSFT_s112
                                 + 0.947657 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.15912 w_k1_ADM_s112
                                 + 1.4536 w_k1_AXP_s112 + 1.3642 w_k1_DIS_s112
                                 + 1.1828 w_k1_FDX_s112 + 1.55241 w_k1_IBM_s112
                                 + 1.31039 w_k1_INTC_s112
                                 + 1.16129 w_k1_KO_s112 + 1.51069 w_k1_MAT_s112
                                 + 1.10994 w_k1_MSFT_s112
                                 + 1.04814 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.918273 w_k1_ADM_s112
                                 + 1.05142 w_k1_AXP_s112
                                 + 1.38487 w_k1_DIS_s112
                                 + 1.18218 w_k1_FDX_s112
                                 + 0.860603 w_k1_IBM_s112
                                 + 0.947231 w_k1_INTC_s112
                                 + 1.0381 w_k1_KO_s112 + 0.825043 w_k1_MAT_s112
                                 + 1.06624 w_k1_MSFT_s112
                                 + 1.10354 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.865513 w_k1_ADM_s112
                                 + 1.03861 w_k1_AXP_s112
                                 + 0.725504 w_k1_DIS_s112
                                 + 0.848838 w_k1_FDX_s112
                                 + 0.919127 w_k1_IBM_s112
                                 + 0.827454 w_k1_INTC_s112
                                 + 0.883092 w_k1_KO_s112 + 1.5133 w_k1_MAT_s112
                                 + 0.737265 w_k1_MSFT_s112
                                 + 0.851113 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.06214 w_k1_ADM_s112
                                 + 1.41416 w_k1_AXP_s112 + 1.739 w_k1_DIS_s112
                                 + 1.39829 w_k1_FDX_s112
                                 + 1.26441 w_k1_IBM_s112
                                 + 1.33107 w_k1_INTC_s112 + 1.402 w_k1_KO_s112
                                 + 1.56395 w_k1_MAT_s112
                                 + 1.35858 w_k1_MSFT_s112
                                 + 1.15295 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.02194 w_k1_ADM_s112
                                 + 1.12855 w_k1_AXP_s112 + 1.1557 w_k1_DIS_s112
                                 + 1.06594 w_k1_FDX_s112
                                 + 1.11346 w_k1_IBM_s112
                                 + 1.09063 w_k1_INTC_s112
                                 + 0.975289 w_k1_KO_s112
                                 + 1.16051 w_k1_MAT_s112
                                 + 0.956488 w_k1_MSFT_s112
                                 + 1.09919 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.73935 w_k1_ADM_s112
                                 + 0.530915 w_k1_AXP_s112
                                 + 0.75567 w_k1_DIS_s112
                                 + 0.738688 w_k1_FDX_s112
                                 + 0.896716 w_k1_IBM_s112
                                 + 0.66848 w_k1_INTC_s112
                                 + 0.840776 w_k1_KO_s112
                                 + 0.766294 w_k1_MAT_s112
                                 + 0.690336 w_k1_MSFT_s112
                                 + 0.806859 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.9479 w_k1_ADM_s113
                                 + 0.962935 w_k1_AXP_s113
                                 + 1.01702 w_k1_DIS_s113
                                 + 0.947573 w_k1_FDX_s113
                                 + 0.910552 w_k1_IBM_s113
                                 + 0.877364 w_k1_INTC_s113
                                 + 0.88325 w_k1_KO_s113
                                 + 0.757555 w_k1_MAT_s113
                                 + 0.894151 w_k1_MSFT_s113
                                 + 0.895335 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.5286 w_k1_ADM_s113
                                 + 1.5557 w_k1_AXP_s113 + 1.10764 w_k1_DIS_s113
                                 + 1.27317 w_k1_FDX_s113
                                 + 1.26708 w_k1_IBM_s113
                                 + 1.08634 w_k1_INTC_s113
                                 + 1.10895 w_k1_KO_s113 + 1.41471 w_k1_MAT_s113
                                 + 1.00899 w_k1_MSFT_s113
                                 + 1.12892 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.36242 w_k1_ADM_s113
                                 + 1.36975 w_k1_AXP_s113
                                 + 1.46789 w_k1_DIS_s113
                                 + 1.26414 w_k1_FDX_s113
                                 + 1.42792 w_k1_IBM_s113
                                 + 1.27199 w_k1_INTC_s113
                                 + 1.19322 w_k1_KO_s113 + 1.25506 w_k1_MAT_s113
                                 + 1.27586 w_k1_MSFT_s113
                                 + 1.17628 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.34131 w_k1_ADM_s113
                                 + 1.16684 w_k1_AXP_s113 + 1.2187 w_k1_DIS_s113
                                 + 1.00806 w_k1_FDX_s113
                                 + 1.24355 w_k1_IBM_s113
                                 + 0.96867 w_k1_INTC_s113
                                 + 1.11843 w_k1_KO_s113 + 1.47915 w_k1_MAT_s113
                                 + 1.08315 w_k1_MSFT_s113
                                 + 1.09005 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.04358 w_k1_ADM_s113
                                 + 0.768423 w_k1_AXP_s113
                                 + 0.649467 w_k1_DIS_s113
                                 + 0.844431 w_k1_FDX_s113
                                 + 0.740292 w_k1_IBM_s113
                                 + 0.785869 w_k1_INTC_s113
                                 + 0.727495 w_k1_KO_s113
                                 + 0.607126 w_k1_MAT_s113
                                 + 0.769305 w_k1_MSFT_s113
                                 + 0.85055 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.953372 w_k1_ADM_s113
                                 + 1.05882 w_k1_AXP_s113
                                 + 0.774288 w_k1_DIS_s113
                                 + 0.88007 w_k1_FDX_s113
                                 + 1.22199 w_k1_IBM_s113
                                 + 0.957543 w_k1_INTC_s113
                                 + 0.878653 w_k1_KO_s113
                                 + 1.08654 w_k1_MAT_s113
                                 + 0.834422 w_k1_MSFT_s113
                                 + 0.915633 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.16923 w_k1_ADM_s113
                                 + 0.954669 w_k1_AXP_s113
                                 + 1.13124 w_k1_DIS_s113
                                 + 1.15855 w_k1_FDX_s113
                                 + 0.69488 w_k1_IBM_s113
                                 + 1.00613 w_k1_INTC_s113
                                 + 1.02098 w_k1_KO_s113
                                 + 0.688445 w_k1_MAT_s113
                                 + 1.05914 w_k1_MSFT_s113 + 1.0084 w_k1_PG_s113
                                 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.31188 w_k1_ADM_s113
                                 + 1.16315 w_k1_AXP_s113
                                 + 1.41415 w_k1_DIS_s113
                                 + 1.18805 w_k1_FDX_s113
                                 + 0.907091 w_k1_IBM_s113
                                 + 1.3078 w_k1_INTC_s113
                                 + 0.965789 w_k1_KO_s113
                                 + 0.702736 w_k1_MAT_s113
                                 + 1.28392 w_k1_MSFT_s113
                                 + 1.04313 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.897117 w_k1_ADM_s114
                                 + 0.687532 w_k1_AXP_s114
                                 + 0.799801 w_k1_DIS_s114
                                 + 0.917298 w_k1_FDX_s114
                                 + 0.785301 w_k1_IBM_s114
                                 + 0.938859 w_k1_INTC_s114
                                 + 0.831143 w_k1_KO_s114
                                 + 0.542938 w_k1_MAT_s114
                                 + 0.909141 w_k1_MSFT_s114
                                 + 0.833297 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.05005 w_k1_ADM_s114
                                 + 1.08411 w_k1_AXP_s114 + 1.6807 w_k1_DIS_s114
                                 + 1.34416 w_k1_FDX_s114
                                 + 1.11195 w_k1_IBM_s114
                                 + 1.43859 w_k1_INTC_s114
                                 + 1.27526 w_k1_KO_s114 + 1.26586 w_k1_MAT_s114
                                 + 1.51441 w_k1_MSFT_s114 + 1.1572 w_k1_PG_s114
                                 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.05761 w_k1_ADM_s114
                                 + 0.972637 w_k1_AXP_s114
                                 + 0.928369 w_k1_DIS_s114
                                 + 1.01518 w_k1_FDX_s114
                                 + 1.61917 w_k1_IBM_s114
                                 + 1.13102 w_k1_INTC_s114
                                 + 1.31716 w_k1_KO_s114 + 1.67846 w_k1_MAT_s114
                                 + 1.02319 w_k1_MSFT_s114
                                 + 0.989979 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.24272 w_k1_ADM_s114
                                 + 1.13046 w_k1_AXP_s114 + 1.0388 w_k1_DIS_s114
                                 + 1.12987 w_k1_FDX_s114
                                 + 1.26578 w_k1_IBM_s114
                                 + 1.14144 w_k1_INTC_s114
                                 + 1.05953 w_k1_KO_s114 + 1.46115 w_k1_MAT_s114
                                 + 1.00014 w_k1_MSFT_s114
                                 + 1.03493 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.816113 w_k1_ADM_s114
                                 + 0.790937 w_k1_AXP_s114
                                 + 0.674976 w_k1_DIS_s114
                                 + 0.813776 w_k1_FDX_s114
                                 + 1.02808 w_k1_IBM_s114
                                 + 0.895099 w_k1_INTC_s114
                                 + 0.73144 w_k1_KO_s114
                                 + 0.875875 w_k1_MAT_s114
                                 + 0.661674 w_k1_MSFT_s114
                                 + 0.896697 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.990721 w_k1_ADM_s114
                                 + 0.938481 w_k1_AXP_s114
                                 + 1.17448 w_k1_DIS_s114
                                 + 1.12796 w_k1_FDX_s114
                                 + 0.852329 w_k1_IBM_s114
                                 + 1.23988 w_k1_INTC_s114
                                 + 1.06504 w_k1_KO_s114
                                 + 0.794317 w_k1_MAT_s114
                                 + 1.12477 w_k1_MSFT_s114
                                 + 1.03498 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.744041 w_k1_ADM_s114
                                 + 0.726878 w_k1_AXP_s114
                                 + 0.64911 w_k1_DIS_s114
                                 + 0.714937 w_k1_FDX_s114
                                 + 1.27416 w_k1_IBM_s114
                                 + 0.745477 w_k1_INTC_s114
                                 + 0.872288 w_k1_KO_s114
                                 + 1.28935 w_k1_MAT_s114
                                 + 0.695508 w_k1_MSFT_s114
                                 + 0.882255 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.04225 w_k1_ADM_s114
                                 + 0.92989 w_k1_AXP_s114
                                 + 1.14216 w_k1_DIS_s114
                                 + 1.06544 w_k1_FDX_s114
                                 + 1.06558 w_k1_IBM_s114
                                 + 1.15491 w_k1_INTC_s114
                                 + 1.01559 w_k1_KO_s114 + 1.10648 w_k1_MAT_s114
                                 + 1.01267 w_k1_MSFT_s114
                                 + 1.03063 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.945417 w_k1_ADM_s115
                                 + 1.20813 w_k1_AXP_s115
                                 + 1.19061 w_k1_DIS_s115
                                 + 1.04378 w_k1_FDX_s115
                                 + 1.67396 w_k1_IBM_s115
                                 + 1.15919 w_k1_INTC_s115
                                 + 1.30384 w_k1_KO_s115 + 2.15086 w_k1_MAT_s115
                                 + 0.936642 w_k1_MSFT_s115
                                 + 1.11333 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.795993 w_k1_ADM_s115
                                 + 0.664023 w_k1_AXP_s115
                                 + 0.948344 w_k1_DIS_s115
                                 + 0.828285 w_k1_FDX_s115
                                 + 1.17336 w_k1_IBM_s115
                                 + 0.994302 w_k1_INTC_s115
                                 + 0.957594 w_k1_KO_s115
                                 + 0.928479 w_k1_MAT_s115
                                 + 0.964319 w_k1_MSFT_s115
                                 + 1.01313 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.782779 w_k1_ADM_s115
                                 + 1.01491 w_k1_AXP_s115
                                 + 0.799507 w_k1_DIS_s115
                                 + 0.832467 w_k1_FDX_s115
                                 + 1.18265 w_k1_IBM_s115
                                 + 0.979647 w_k1_INTC_s115
                                 + 0.897726 w_k1_KO_s115
                                 + 1.10226 w_k1_MAT_s115
                                 + 0.853675 w_k1_MSFT_s115
                                 + 1.05668 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.10351 w_k1_ADM_s115
                                 + 1.15112 w_k1_AXP_s115
                                 + 1.54431 w_k1_DIS_s115 + 1.2336 w_k1_FDX_s115
                                 + 1.20885 w_k1_IBM_s115
                                 + 1.25187 w_k1_INTC_s115 + 1.2218 w_k1_KO_s115
                                 + 1.76164 w_k1_MAT_s115
                                 + 1.25087 w_k1_MSFT_s115
                                 + 1.15325 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.903224 w_k1_ADM_s115
                                 + 0.880731 w_k1_AXP_s115
                                 + 1.09109 w_k1_DIS_s115
                                 + 1.01669 w_k1_FDX_s115
                                 + 1.12747 w_k1_IBM_s115
                                 + 1.10285 w_k1_INTC_s115
                                 + 1.03615 w_k1_KO_s115 + 1.1872 w_k1_MAT_s115
                                 + 1.07207 w_k1_MSFT_s115
                                 + 1.18453 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.971754 w_k1_ADM_s115
                                 + 0.901565 w_k1_AXP_s115
                                 + 0.950687 w_k1_DIS_s115
                                 + 0.958761 w_k1_FDX_s115
                                 + 0.792324 w_k1_IBM_s115
                                 + 0.98111 w_k1_INTC_s115
                                 + 0.933325 w_k1_KO_s115
                                 + 0.748806 w_k1_MAT_s115
                                 + 0.888787 w_k1_MSFT_s115
                                 + 0.944048 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.957352 w_k1_ADM_s115
                                 + 0.628041 w_k1_AXP_s115
                                 + 0.592453 w_k1_DIS_s115
                                 + 0.72121 w_k1_FDX_s115
                                 + 0.767414 w_k1_IBM_s115
                                 + 0.748382 w_k1_INTC_s115
                                 + 0.638925 w_k1_KO_s115
                                 + 0.664275 w_k1_MAT_s115
                                 + 0.770259 w_k1_MSFT_s115
                                 + 0.879664 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.36992 w_k1_ADM_s115
                                 + 1.16859 w_k1_AXP_s115
                                 + 1.17706 w_k1_DIS_s115
                                 + 1.38275 w_k1_FDX_s115
                                 + 0.655734 w_k1_IBM_s115
                                 + 1.09611 w_k1_INTC_s115
                                 + 0.828001 w_k1_KO_s115
                                 + 0.58009 w_k1_MAT_s115
                                 + 1.10254 w_k1_MSFT_s115
                                 + 1.12193 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.825737 w_k1_ADM_s116
                                 + 0.679367 w_k1_AXP_s116
                                 + 0.71671 w_k1_DIS_s116
                                 + 0.76574 w_k1_FDX_s116
                                 + 0.730709 w_k1_IBM_s116
                                 + 0.725254 w_k1_INTC_s116
                                 + 0.777656 w_k1_KO_s116
                                 + 0.818896 w_k1_MAT_s116
                                 + 0.875519 w_k1_MSFT_s116
                                 + 0.876546 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.07053 w_k1_ADM_s116
                                 + 0.830948 w_k1_AXP_s116
                                 + 0.92572 w_k1_DIS_s116
                                 + 0.885378 w_k1_FDX_s116
                                 + 0.941189 w_k1_IBM_s116
                                 + 0.980382 w_k1_INTC_s116
                                 + 0.938296 w_k1_KO_s116 + 1.2383 w_k1_MAT_s116
                                 + 1.02306 w_k1_MSFT_s116
                                 + 0.957805 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.826281 w_k1_ADM_s116
                                 + 0.654148 w_k1_AXP_s116
                                 + 0.712413 w_k1_DIS_s116
                                 + 0.850769 w_k1_FDX_s116
                                 + 0.548473 w_k1_IBM_s116
                                 + 0.822725 w_k1_INTC_s116
                                 + 0.629395 w_k1_KO_s116
                                 + 0.523709 w_k1_MAT_s116
                                 + 0.889229 w_k1_MSFT_s116
                                 + 0.869523 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.988803 w_k1_ADM_s116
                                 + 0.847425 w_k1_AXP_s116
                                 + 0.916603 w_k1_DIS_s116
                                 + 1.03554 w_k1_FDX_s116
                                 + 0.702539 w_k1_IBM_s116
                                 + 0.861596 w_k1_INTC_s116
                                 + 0.812763 w_k1_KO_s116
                                 + 1.08105 w_k1_MAT_s116
                                 + 1.0195 w_k1_MSFT_s116 + 1.01223 w_k1_PG_s116
                                 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.05898 w_k1_ADM_s116
                                 + 0.939111 w_k1_AXP_s116
                                 + 1.33209 w_k1_DIS_s116
                                 + 1.20328 w_k1_FDX_s116 + 0.8902 w_k1_IBM_s116
                                 + 1.02849 w_k1_INTC_s116
                                 + 1.12323 w_k1_KO_s116
                                 + 0.953876 w_k1_MAT_s116
                                 + 1.2372 w_k1_MSFT_s116 + 1.17787 w_k1_PG_s116
                                 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.30139 w_k1_ADM_s116
                                 + 1.36157 w_k1_AXP_s116
                                 + 1.52805 w_k1_DIS_s116
                                 + 1.30574 w_k1_FDX_s116
                                 + 1.24337 w_k1_IBM_s116
                                 + 1.04908 w_k1_INTC_s116
                                 + 1.27872 w_k1_KO_s116 + 1.62352 w_k1_MAT_s116
                                 + 1.32633 w_k1_MSFT_s116
                                 + 1.26218 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.673906 w_k1_ADM_s116
                                 + 0.752716 w_k1_AXP_s116
                                 + 0.72873 w_k1_DIS_s116
                                 + 0.752614 w_k1_FDX_s116
                                 + 1.38416 w_k1_IBM_s116
                                 + 0.857535 w_k1_INTC_s116
                                 + 0.916897 w_k1_KO_s116
                                 + 1.37876 w_k1_MAT_s116
                                 + 0.820378 w_k1_MSFT_s116
                                 + 0.917448 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.18828 w_k1_ADM_s116
                                 + 0.821558 w_k1_AXP_s116
                                 + 1.27024 w_k1_DIS_s116
                                 + 1.07842 w_k1_FDX_s116
                                 + 1.23299 w_k1_IBM_s116
                                 + 1.21473 w_k1_INTC_s116
                                 + 1.38857 w_k1_KO_s116 + 1.16299 w_k1_MAT_s116
                                 + 1.29276 w_k1_MSFT_s116
                                 + 1.29013 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.787058 w_k1_ADM_s117
                                 + 0.805913 w_k1_AXP_s117
                                 + 0.595049 w_k1_DIS_s117
                                 + 0.717102 w_k1_FDX_s117
                                 + 0.84696 w_k1_IBM_s117
                                 + 0.739446 w_k1_INTC_s117
                                 + 0.80612 w_k1_KO_s117
                                 + 0.637888 w_k1_MAT_s117
                                 + 0.748554 w_k1_MSFT_s117
                                 + 0.78471 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.02608 w_k1_ADM_s117
                                 + 0.697953 w_k1_AXP_s117
                                 + 0.763912 w_k1_DIS_s117
                                 + 1.00225 w_k1_FDX_s117
                                 + 0.624021 w_k1_IBM_s117
                                 + 0.916511 w_k1_INTC_s117
                                 + 0.768893 w_k1_KO_s117
                                 + 0.433174 w_k1_MAT_s117
                                 + 0.988571 w_k1_MSFT_s117
                                 + 0.94174 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.04951 w_k1_ADM_s117
                                 + 1.45026 w_k1_AXP_s117
                                 + 1.29424 w_k1_DIS_s117
                                 + 1.32155 w_k1_FDX_s117
                                 + 0.947221 w_k1_IBM_s117
                                 + 1.10649 w_k1_INTC_s117 + 1.0404 w_k1_KO_s117
                                 + 1.24139 w_k1_MAT_s117
                                 + 1.19762 w_k1_MSFT_s117 + 1.2084 w_k1_PG_s117
                                 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.14438 w_k1_ADM_s117
                                 + 0.941134 w_k1_AXP_s117
                                 + 1.12743 w_k1_DIS_s117
                                 + 1.05077 w_k1_FDX_s117
                                 + 1.06509 w_k1_IBM_s117
                                 + 0.941875 w_k1_INTC_s117
                                 + 1.10191 w_k1_KO_s117
                                 + 0.866288 w_k1_MAT_s117
                                 + 1.04037 w_k1_MSFT_s117
                                 + 1.01996 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.732928 w_k1_ADM_s117
                                 + 0.877209 w_k1_AXP_s117
                                 + 0.73204 w_k1_DIS_s117
                                 + 0.854312 w_k1_FDX_s117
                                 + 0.725798 w_k1_IBM_s117
                                 + 0.836902 w_k1_INTC_s117
                                 + 0.838312 w_k1_KO_s117
                                 + 0.926982 w_k1_MAT_s117
                                 + 0.913451 w_k1_MSFT_s117
                                 + 0.892935 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.978378 w_k1_ADM_s117
                                 + 1.04252 w_k1_AXP_s117
                                 + 0.880158 w_k1_DIS_s117
                                 + 1.11494 w_k1_FDX_s117
                                 + 0.698076 w_k1_IBM_s117
                                 + 1.12945 w_k1_INTC_s117
                                 + 0.836926 w_k1_KO_s117
                                 + 0.783582 w_k1_MAT_s117
                                 + 1.02001 w_k1_MSFT_s117
                                 + 0.993201 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.936776 w_k1_ADM_s117
                                 + 0.793732 w_k1_AXP_s117
                                 + 0.82543 w_k1_DIS_s117
                                 + 0.874746 w_k1_FDX_s117
                                 + 1.28069 w_k1_IBM_s117
                                 + 0.829362 w_k1_INTC_s117
                                 + 1.03351 w_k1_KO_s117
                                 + 0.982906 w_k1_MAT_s117
                                 + 0.928927 w_k1_MSFT_s117
                                 + 1.01881 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.00096 w_k1_ADM_s117
                                 + 1.26252 w_k1_AXP_s117
                                 + 1.37846 w_k1_DIS_s117
                                 + 1.23337 w_k1_FDX_s117
                                 + 1.37439 w_k1_IBM_s117
                                 + 1.11794 w_k1_INTC_s117
                                 + 1.56687 w_k1_KO_s117 + 1.45662 w_k1_MAT_s117
                                 + 1.25771 w_k1_MSFT_s117
                                 + 1.29626 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.105533604319934 w_k1_ADM_s000
                                 + 0.0509224629265921 w_k1_ADM_s001
                                 + 0.125120705240964 w_k1_ADM_s002
                                 + 0.0427494578719902 w_k1_ADM_s003
                                 + 0.0925647859684575 w_k1_ADM_s004
                                 + 0.0236682081265246 w_k1_ADM_s005
                                 + 0.0296735810952728 w_k1_ADM_s006
                                 + 0.0270945434723955 w_k1_ADM_s007
                                 + 0.0119799806652162 w_k1_ADM_s010
                                 + 0.0115484388896935 w_k1_ADM_s011
                                 + 0.0130166760490685 w_k1_ADM_s012
                                 + 0.0181478450469886 w_k1_ADM_s013
                                 + 0.0112755307889348 w_k1_ADM_s014
                                 + 0.008123171668857 w_k1_ADM_s015
                                 + 0.0370749665700157 w_k1_ADM_s016
                                 + 0.0217568610745875 w_k1_ADM_s017
                                 + 0.00565418525451875 w_k1_ADM_s100
                                 + 0.008235537192078 w_k1_ADM_s101
                                 + 0.00557835993005441 w_k1_ADM_s102
                                 + 0.00591215451419917 w_k1_ADM_s103
                                 + 0.0141939513548834 w_k1_ADM_s104
                                 + 0.0125802164448245 w_k1_ADM_s105
                                 + 0.00694915835210501 w_k1_ADM_s106
                                 + 0.0204902099189153 w_k1_ADM_s107
                                 + 0.100564729317459 w_k1_ADM_s110
                                 + 0.023120060225904 w_k1_ADM_s111
                                 + 0.0232586904416939 w_k1_ADM_s112
                                 + 0.0380688994877693 w_k1_ADM_s113
                                 + 0.0242756843740548 w_k1_ADM_s114
                                 + 0.0649505160580211 w_k1_ADM_s115
                                 + 0.0221351829219827 w_k1_ADM_s116
                                 + 0.022951730221438 w_k1_ADM_s117
                                 + 0.107394229409656 w_k1_AXP_s000
                                 + 0.0457096589189144 w_k1_AXP_s001
                                 + 0.116943904147683 w_k1_AXP_s002
                                 + 0.0431118791879775 w_k1_AXP_s003
                                 + 0.0925426573696709 w_k1_AXP_s004
                                 + 0.0250349510117359 w_k1_AXP_s005
                                 + 0.0274110640656442 w_k1_AXP_s006
                                 + 0.0319212435390857 w_k1_AXP_s007
                                 + 0.010890588760283 w_k1_AXP_s010
                                 + 0.0132176125915569 w_k1_AXP_s011
                                 + 0.0132486271082823 w_k1_AXP_s012
                                 + 0.0205440323738071 w_k1_AXP_s013
                                 + 0.0103643968912312 w_k1_AXP_s014
                                 + 0.00942060482062178 w_k1_AXP_s015
                                 + 0.0394939730438601 w_k1_AXP_s016
                                 + 0.0242217924010184 w_k1_AXP_s017
                                 + 0.00516422617346445 w_k1_AXP_s100
                                 + 0.00950753827116936 w_k1_AXP_s101
                                 + 0.00589855513515758 w_k1_AXP_s102
                                 + 0.00672260365569065 w_k1_AXP_s103
                                 + 0.0143724873384656 w_k1_AXP_s104
                                 + 0.0133649983524777 w_k1_AXP_s105
                                 + 0.00579900326206577 w_k1_AXP_s106
                                 + 0.0242330288896947 w_k1_AXP_s107
                                 + 0.0840527848348183 w_k1_AXP_s110
                                 + 0.0216162938852062 w_k1_AXP_s111
                                 + 0.0270206044109254 w_k1_AXP_s112
                                 + 0.0355459287643871 w_k1_AXP_s113
                                 + 0.0222402466991383 w_k1_AXP_s114
                                 + 0.0630158060277479 w_k1_AXP_s115
                                 + 0.0188109009830884 w_k1_AXP_s116
                                 + 0.0244058713947899 w_k1_AXP_s117
                                 + 0.107468637472394 w_k1_DIS_s000
                                 + 0.049695581726792 w_k1_DIS_s001
                                 + 0.125990469772064 w_k1_DIS_s002
                                 + 0.0441975875986577 w_k1_DIS_s003
                                 + 0.0986883182270598 w_k1_DIS_s004
                                 + 0.0265208611698712 w_k1_DIS_s005
                                 + 0.0259968485172407 w_k1_DIS_s006
                                 + 0.0338360803806818 w_k1_DIS_s007
                                 + 0.0125232651781894 w_k1_DIS_s010
                                 + 0.013482701153162 w_k1_DIS_s011
                                 + 0.0121486297194573 w_k1_DIS_s012
                                 + 0.0207061596194067 w_k1_DIS_s013
                                 + 0.0112066437927049 w_k1_DIS_s014
                                 + 0.00898576064481507 w_k1_DIS_s015
                                 + 0.0359697867099281 w_k1_DIS_s016
                                 + 0.0227280359610851 w_k1_DIS_s017
                                 + 0.00487290816040953 w_k1_DIS_s100
                                 + 0.00912913377612037 w_k1_DIS_s101
                                 + 0.00548698025298046 w_k1_DIS_s102
                                 + 0.00599722073140476 w_k1_DIS_s103
                                 + 0.0152418538144336 w_k1_DIS_s104
                                 + 0.0136637074423168 w_k1_DIS_s105
                                 + 0.00538072017008085 w_k1_DIS_s106
                                 + 0.024089841975936 w_k1_DIS_s107
                                 + 0.0965589274471174 w_k1_DIS_s110
                                 + 0.022799994671624 w_k1_DIS_s111
                                 + 0.0265682202723298 w_k1_DIS_s112
                                 + 0.0339255139417877 w_k1_DIS_s113
                                 + 0.0255182772621074 w_k1_DIS_s114
                                 + 0.0703900787621307 w_k1_DIS_s115
                                 + 0.0226881054477167 w_k1_DIS_s116
                                 + 0.0241284324071611 w_k1_DIS_s117
                                 + 0.109239370859171 w_k1_FDX_s000
                                 + 0.0475914833018294 w_k1_FDX_s001
                                 + 0.121941912020993 w_k1_FDX_s002
                                 + 0.0466704329834997 w_k1_FDX_s003
                                 + 0.0897569126694778 w_k1_FDX_s004
                                 + 0.0254015648884794 w_k1_FDX_s005
                                 + 0.0257428946370744 w_k1_FDX_s006
                                 + 0.0302901092830486 w_k1_FDX_s007
                                 + 0.0111835715683224 w_k1_FDX_s010
                                 + 0.0129407533610679 w_k1_FDX_s011
                                 + 0.0116211611402153 w_k1_FDX_s012
                                 + 0.0199322077194263 w_k1_FDX_s013
                                 + 0.0105439846110083 w_k1_FDX_s014
                                 + 0.00863146764371933 w_k1_FDX_s015
                                 + 0.0343435605039551 w_k1_FDX_s016
                                 + 0.0225072053265761 w_k1_FDX_s017
                                 + 0.0048271597016602 w_k1_FDX_s100
                                 + 0.00961576054511254 w_k1_FDX_s101
                                 + 0.00554203557466792 w_k1_FDX_s102
                                 + 0.00614075420890937 w_k1_FDX_s103
                                 + 0.01432561500358 w_k1_FDX_s104
                                 + 0.0119653578246696 w_k1_FDX_s105
                                 + 0.005392961099639 w_k1_FDX_s106
                                 + 0.0218618681134305 w_k1_FDX_s107
                                 + 0.0916226448371765 w_k1_FDX_s110
                                 + 0.0231209207589052 w_k1_FDX_s111
                                 + 0.0252555986041741 w_k1_FDX_s112
                                 + 0.0337725615652125 w_k1_FDX_s113
                                 + 0.0254852517927856 w_k1_FDX_s114
                                 + 0.0668519299333302 w_k1_FDX_s115
                                 + 0.0213132373920705 w_k1_FDX_s116
                                 + 0.0247572698120647 w_k1_FDX_s117
                                 + 0.104202518308152 w_k1_IBM_s000
                                 + 0.0458844190927316 w_k1_IBM_s001
                                 + 0.116989903927916 w_k1_IBM_s002
                                 + 0.0471942880977971 w_k1_IBM_s003
                                 + 0.088847701021538 w_k1_IBM_s004
                                 + 0.0258120786872955 w_k1_IBM_s005
                                 + 0.0256845065866327 w_k1_IBM_s006
                                 + 0.0330586725527724 w_k1_IBM_s007
                                 + 0.0118355092253068 w_k1_IBM_s010
                                 + 0.0136047322076293 w_k1_IBM_s011
                                 + 0.0122015514423228 w_k1_IBM_s012
                                 + 0.0213291780988492 w_k1_IBM_s013
                                 + 0.0112493603669596 w_k1_IBM_s014
                                 + 0.00844189380693097 w_k1_IBM_s015
                                 + 0.0346070871458445 w_k1_IBM_s016
                                 + 0.0243041317902234 w_k1_IBM_s017
                                 + 0.00503270791757991 w_k1_IBM_s100
                                 + 0.00899506735033731 w_k1_IBM_s101
                                 + 0.0050540600149096 w_k1_IBM_s102
                                 + 0.00620766014585047 w_k1_IBM_s103
                                 + 0.0135355603833791 w_k1_IBM_s104
                                 + 0.0130721197386942 w_k1_IBM_s105
                                 + 0.00538167090855988 w_k1_IBM_s106
                                 + 0.0219904412657804 w_k1_IBM_s107
                                 + 0.0922346935329448 w_k1_IBM_s110
                                 + 0.0242333120315302 w_k1_IBM_s111
                                 + 0.0249289036630816 w_k1_IBM_s112
                                 + 0.0340141286401393 w_k1_IBM_s113
                                 + 0.026919387799591 w_k1_IBM_s114
                                 + 0.0673563603731123 w_k1_IBM_s115
                                 + 0.0214466668131773 w_k1_IBM_s116
                                 + 0.0252546397670934 w_k1_IBM_s117
                                 + 0.109221488025592 w_k1_INTC_s000
                                 + 0.0399424598716345 w_k1_INTC_s001
                                 + 0.106589918636444 w_k1_INTC_s002
                                 + 0.0459538570601633 w_k1_INTC_s003
                                 + 0.0780867924668829 w_k1_INTC_s004
                                 + 0.023496699661744 w_k1_INTC_s005
                                 + 0.0232576391195844 w_k1_INTC_s006
                                 + 0.0334490293619422 w_k1_INTC_s007
                                 + 0.0097659683734937 w_k1_INTC_s010
                                 + 0.014569012972817 w_k1_INTC_s011
                                 + 0.0115211155174771 w_k1_INTC_s012
                                 + 0.0214336142557707 w_k1_INTC_s013
                                 + 0.00889296008799001 w_k1_INTC_s014
                                 + 0.00900467959654986 w_k1_INTC_s015
                                 + 0.037233524855089 w_k1_INTC_s016
                                 + 0.0236287568061276 w_k1_INTC_s017
                                 + 0.00487731230022533 w_k1_INTC_s100
                                 + 0.00798439623161501 w_k1_INTC_s101
                                 + 0.00508530735770443 w_k1_INTC_s102
                                 + 0.00676009617060171 w_k1_INTC_s103
                                 + 0.0146026549853101 w_k1_INTC_s104
                                 + 0.0108377723519926 w_k1_INTC_s105
                                 + 0.00502918785410429 w_k1_INTC_s106
                                 + 0.0239858679537391 w_k1_INTC_s107
                                 + 0.0851539415284389 w_k1_INTC_s110
                                 + 0.021419466945409 w_k1_INTC_s111
                                 + 0.0252089419046356 w_k1_INTC_s112
                                 + 0.0329887035221653 w_k1_INTC_s113
                                 + 0.027223575882377 w_k1_INTC_s114
                                 + 0.0690473684615932 w_k1_INTC_s115
                                 + 0.0208363579993865 w_k1_INTC_s116
                                 + 0.0228116224732476 w_k1_INTC_s117
                                 + 0.111178227986098 w_k1_KO_s000
                                 + 0.0477807616566809 w_k1_KO_s001
                                 + 0.113960502476692 w_k1_KO_s002
                                 + 0.0431746552695867 w_k1_KO_s003
                                 + 0.0904054034719778 w_k1_KO_s004
                                 + 0.0266147642341954 w_k1_KO_s005
                                 + 0.0284178405332518 w_k1_KO_s006
                                 + 0.0319996291546474 w_k1_KO_s007
                                 + 0.0116355375934041 w_k1_KO_s010
                                 + 0.0126824249710595 w_k1_KO_s011
                                 + 0.0123170169555155 w_k1_KO_s012
                                 + 0.0197969115591919 w_k1_KO_s013
                                 + 0.0109708064756509 w_k1_KO_s014
                                 + 0.00868431557030561 w_k1_KO_s015
                                 + 0.0375273927947119 w_k1_KO_s016
                                 + 0.0239074699330609 w_k1_KO_s017
                                 + 0.00444190896912908 w_k1_KO_s100
                                 + 0.00748617664967035 w_k1_KO_s101
                                 + 0.00484790193361258 w_k1_KO_s102
                                 + 0.00612747803684103 w_k1_KO_s103
                                 + 0.0138053858867523 w_k1_KO_s104
                                 + 0.0121958905444124 w_k1_KO_s105
                                 + 0.00533144032401763 w_k1_KO_s106
                                 + 0.0234542560645753 w_k1_KO_s107
                                 + 0.101880635353613 w_k1_KO_s110
                                 + 0.0232490958669696 w_k1_KO_s111
                                 + 0.0228159457719212 w_k1_KO_s112
                                 + 0.0311346573170669 w_k1_KO_s113
                                 + 0.0250067177253619 w_k1_KO_s114
                                 + 0.0652715988716244 w_k1_KO_s115
                                 + 0.0222445609139792 w_k1_KO_s116
                                 + 0.0259056120986852 w_k1_KO_s117
                                 + 0.0974335181431904 w_k1_MAT_s000
                                 + 0.044952304338879 w_k1_MAT_s001
                                 + 0.139542311952836 w_k1_MAT_s002
                                 + 0.0530531116426421 w_k1_MAT_s003
                                 + 0.0872048653901433 w_k1_MAT_s004
                                 + 0.0264424200402835 w_k1_MAT_s005
                                 + 0.0297749031488371 w_k1_MAT_s006
                                 + 0.0297223581380784 w_k1_MAT_s007
                                 + 0.00986750259576352 w_k1_MAT_s010
                                 + 0.0146837888439408 w_k1_MAT_s011
                                 + 0.010830227662488 w_k1_MAT_s012
                                 + 0.0168983964228879 w_k1_MAT_s013
                                 + 0.0121238387991758 w_k1_MAT_s014
                                 + 0.00969248489137712 w_k1_MAT_s015
                                 + 0.0367716024804751 w_k1_MAT_s016
                                 + 0.0188405853996258 w_k1_MAT_s017
                                 + 0.00403564834662115 w_k1_MAT_s100
                                 + 0.00955531747684293 w_k1_MAT_s101
                                 + 0.00488632873486908 w_k1_MAT_s102
                                 + 0.00761151742432056 w_k1_MAT_s103
                                 + 0.0111477746196762 w_k1_MAT_s104
                                 + 0.0101910593643575 w_k1_MAT_s105
                                 + 0.00532763259644377 w_k1_MAT_s106
                                 + 0.0244233984331333 w_k1_MAT_s107
                                 + 0.107772601235905 w_k1_MAT_s110
                                 + 0.0203505897688514 w_k1_MAT_s111
                                 + 0.0270242927645666 w_k1_MAT_s112
                                 + 0.0312939360516917 w_k1_MAT_s113
                                 + 0.026305397942365 w_k1_MAT_s114
                                 + 0.0733625146391239 w_k1_MAT_s115
                                 + 0.0246209946559216 w_k1_MAT_s116
                                 + 0.0245875302314865 w_k1_MAT_s117
                                 + 0.107153436687387 w_k1_MSFT_s000
                                 + 0.046873418062903 w_k1_MSFT_s001
                                 + 0.120161790416077 w_k1_MSFT_s002
                                 + 0.045559368868769 w_k1_MSFT_s003
                                 + 0.0888750272259332 w_k1_MSFT_s004
                                 + 0.0241065216463226 w_k1_MSFT_s005
                                 + 0.0260420494399393 w_k1_MSFT_s006
                                 + 0.0299535153762734 w_k1_MSFT_s007
                                 + 0.0109208098798265 w_k1_MSFT_s010
                                 + 0.0140216564229343 w_k1_MSFT_s011
                                 + 0.012374469111384 w_k1_MSFT_s012
                                 + 0.0197500952162489 w_k1_MSFT_s013
                                 + 0.010601014075225 w_k1_MSFT_s014
                                 + 0.00856950109572586 w_k1_MSFT_s015
                                 + 0.0361304641143793 w_k1_MSFT_s016
                                 + 0.0238833976441162 w_k1_MSFT_s017
                                 + 0.00482839770039328 w_k1_MSFT_s100
                                 + 0.00873483216818317 w_k1_MSFT_s101
                                 + 0.00524829429064783 w_k1_MSFT_s102
                                 + 0.00612889634251914 w_k1_MSFT_s103
                                 + 0.0138583439635046 w_k1_MSFT_s104
                                 + 0.0112231423151565 w_k1_MSFT_s105
                                 + 0.00527772081159792 w_k1_MSFT_s106
                                 + 0.0210904078339326 w_k1_MSFT_s107
                                 + 0.101191378957037 w_k1_MSFT_s110
                                 + 0.0249751679151027 w_k1_MSFT_s111
                                 + 0.0226585018482239 w_k1_MSFT_s112
                                 + 0.0324062996049853 w_k1_MSFT_s113
                                 + 0.0250049568865485 w_k1_MSFT_s114
                                 + 0.0652060283434495 w_k1_MSFT_s115
                                 + 0.0233761805813222 w_k1_MSFT_s116
                                 + 0.0248357450274549 w_k1_MSFT_s117
                                 + 0.115429242091529 w_k1_PG_s000
                                 + 0.0446048260590283 w_k1_PG_s001
                                 + 0.117095032741553 w_k1_PG_s002
                                 + 0.0429265719313943 w_k1_PG_s003
                                 + 0.0859269566883665 w_k1_PG_s004
                                 + 0.0272845973976932 w_k1_PG_s005
                                 + 0.027232758335268 w_k1_PG_s006
                                 + 0.03134021697966 w_k1_PG_s007
                                 + 0.0124187708954052 w_k1_PG_s010
                                 + 0.0128927522213405 w_k1_PG_s011
                                 + 0.011529979267028 w_k1_PG_s012
                                 + 0.0210034209115037 w_k1_PG_s013
                                 + 0.00990540050436539 w_k1_PG_s014
                                 + 0.00808584844189841 w_k1_PG_s015
                                 + 0.0352114264922447 w_k1_PG_s016
                                 + 0.0235578076389083 w_k1_PG_s017
                                 + 0.00444270080923963 w_k1_PG_s100
                                 + 0.00883057664472618 w_k1_PG_s101
                                 + 0.0057089559638481 w_k1_PG_s102
                                 + 0.00589176921599305 w_k1_PG_s103
                                 + 0.0141030100340942 w_k1_PG_s104
                                 + 0.0118938809380042 w_k1_PG_s105
                                 + 0.00543805923704491 w_k1_PG_s106
                                 + 0.0242445813316918 w_k1_PG_s107
                                 + 0.0941778771541094 w_k1_PG_s110
                                 + 0.0223106395460038 w_k1_PG_s111
                                 + 0.0235975452376008 w_k1_PG_s112
                                 + 0.0324350837188974 w_k1_PG_s113
                                 + 0.0243497005028917 w_k1_PG_s114
                                 + 0.0689326944111661 w_k1_PG_s115
                                 + 0.0229090261447646 w_k1_PG_s116
                                 + 0.025390914401091 w_k1_PG_s117 >= 1.16
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
