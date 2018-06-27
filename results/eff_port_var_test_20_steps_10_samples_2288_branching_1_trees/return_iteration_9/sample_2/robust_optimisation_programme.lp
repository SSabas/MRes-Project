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
 w_k1_ADM_s0_return_constraint:  1.13364 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.836272 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.971295 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.982467 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.30977 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.99122 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.03077 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.729057 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.69622 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.831406 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.835956 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.33901 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.968328 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.2121 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.905798 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.01255 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.745846 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.937131 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.35413 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.79204 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.35807 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.742538 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.749495 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.3664 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.871195 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.03398 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.702368 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.865712 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.848224 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.23607 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.889597 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.987952 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.05917 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.14708 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.11694 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.632024 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.09344 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.833622 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.18221 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.863905 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.927361 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.05457 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.18693 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.927329 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.728535 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.649232 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.8786 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.607832 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.773193 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.32143 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.965091 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.18474 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.735329 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.07859 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.506385 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.895069 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.33464 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.772743 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.2854 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.668359 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.665855 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.29921 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.949665 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.09419 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.726148 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.877834 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.749764 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.36946 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.64893 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.707739 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.17339 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.13628 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.15481 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.595987 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.802767 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.00221 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.18361 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.805556 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.0897 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.06411 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.23968 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.897623 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.767329 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.458421 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 2.08547 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.636709 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.872915 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.24083 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.840617 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.26646 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.651431 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.919051 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.491697 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.860991 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.40572 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.862911 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.54766 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.60326 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.761322 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.55474 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.883411 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.12072 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.549352 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.02564 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.688613 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.17452 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.650373 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.02814 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.37522 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.17161 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.33753 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.52095 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.830707 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.765908 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.11754 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.868318 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.06305 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.02419 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.21458 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.873325 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.9097 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.684393 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.71737 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.726006 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.872768 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.3285 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.982265 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.1355 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.838345 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.915139 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.608063 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.894705 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.31623 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.885774 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.13836 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.670631 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.748293 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.37914 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.00183 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.969221 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.729361 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.995043 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.881596 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.22819 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.827206 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.914617 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.24802 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.23674 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.09357 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.70435 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.0564 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.94242 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.14782 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.878502 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.745405 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.14948 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.952003 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.0841 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.653371 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.692893 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.39693 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.856411 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.1719 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.866605 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.823466 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.23564 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.635215 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.15751 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.708797 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.853154 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.991791 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.742503 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.67674 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.16975 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.985042 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.925691 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.6914 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.55913 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.717918 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.00976 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.589088 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.22271 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.449612 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.909015 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.26999 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.929022 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.67984 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.700436 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.646458 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.909434 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.11443 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.880557 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.984905 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.931769 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.09151 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.932903 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.954903 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.73024 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.63268 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.840447 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.947223 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.36377 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.00096 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.10174 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.88934 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.979164 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.745545 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.987507 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.19405 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.04459 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.24654 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.894404 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.87716 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.35901 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.08961 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.928602 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.757451 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.11412 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.930182 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.20802 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.770831 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.958675 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.19356 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.20833 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.14697 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.790156 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.11034 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.971835 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.10884 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.820043 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.914261 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.15236 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.15665 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.939169 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.795699 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.608344 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.6161 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.729086 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.10007 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.01173 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.866147 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.18971 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.680267 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.924256 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.590031 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.856133 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.08537 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.816393 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.47493 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.842236 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.822648 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.20439 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.833396 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.16614 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.680698 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.01733 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.687557 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.27675 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.652176 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.0717 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.30051 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.10562 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.55881 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.721858 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.869353 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.898075 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.28677 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.800662 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.706534 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.1819 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.23742 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.1119 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.58684 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.680652 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.60901 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.701514 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.979105 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.02528 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.799508 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.13462 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.533014 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.21578 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.516273 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.71441 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.951343 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.701397 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.56148 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.874876 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.63646 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.74986 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.629017 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.12451 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.637833 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.738135 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.461226 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.2417 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.477755 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.641173 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.4233 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.87744 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.78517 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.630577 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.519772 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.05913 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.09959 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.888176 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.03436 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.04038 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.23734 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.888716 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.909643 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.685243 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.51638 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.759389 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.868298 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.24303 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.96041 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.18279 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.768153 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.9319 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.620517 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.868669 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.27502 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.941224 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.14979 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.686674 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.769767 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.27824 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.933934 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.899527 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.675894 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.961546 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.815694 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.06853 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.857184 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.986454 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.23314 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.15945 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.17683 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.692329 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.10306 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.856173 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.10371 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.9001 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.978591 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.0254 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.11947 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.943227 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.858034 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.753617 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.25288 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.829506 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.87722 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.09253 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.972012 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.10797 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.821365 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.06761 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.698071 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.905251 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.17235 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.867417 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.2393 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.820615 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.862589 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.13941 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.871208 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.08609 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.778663 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.958596 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.784349 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.11517 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.846141 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.987256 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.20007 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.11988 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.19352 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.839296 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.933867 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.98222 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.240000021435335 z_k1_s0000
                                 - 0.104000064155324 z_k1_s0001
                                 - 0.138000012325318 z_k1_s0002
                                 - 0.210000018755918 z_k1_s0003
                                 - 0.226000788318445 z_k1_s0004
                                 - 0.384000363496609 z_k1_s0005
                                 - 0.120000010717668 z_k1_s0006
                                 - 0.223999251872808 z_k1_s0007
                                 - 0.409001678617475 z_k1_s0010
                                 - 0.0880003198955372 z_k1_s0011
                                 - 0.197999975012175 z_k1_s0012
                                 - 0.116000376543948 z_k1_s0013
                                 - 0.204001027440723 z_k1_s0014
                                 - 0.34200206321072 z_k1_s0015
                                 - 0.0990003185073247 z_k1_s0016
                                 - 0.19900125375439 z_k1_s0017
                                 - 0.178000160592991 z_k1_s0020
                                 - 0.25999958127107 z_k1_s0021
                                 - 0.0979998810636867 z_k1_s0022
                                 - 0.0899999036707246 z_k1_s0023
                                 - 0.121999939642494 z_k1_s0024
                                 - 0.287999691746319 z_k1_s0025
                                 - 0.0679999974400591 z_k1_s0026
                                 - 0.159999547859242 z_k1_s0027
                                 - 0.134999855075709 z_k1_s0030
                                 - 0.161999890890808 z_k1_s0031
                                 - 0.284999928048562 z_k1_s0032
                                 - 0.28800016602786 z_k1_s0033
                                 - 0.0929999253632496 z_k1_s0034
                                 - 0.0969998646692345 z_k1_s0035
                                 - 0.386000380023345 z_k1_s0036
                                 - 0.17399922280909 z_k1_s0037
                                 - 0.0859999347102358 z_k1_s0040
                                 - 0.0939998857972656 z_k1_s0041
                                 - 0.177000175387121 z_k1_s0042
                                 - 0.0759998895765258 z_k1_s0043
                                 - 0.244999121977337 z_k1_s0044
                                 - 0.318000074932968 z_k1_s0045
                                 - 0.0759998895765258 z_k1_s0046
                                 - 0.344999573314438 z_k1_s0047
                                 - 0.510002324071062 z_k1_s0050
                                 - 0.17500060829627 z_k1_s0051
                                 - 0.266001019122661 z_k1_s0052
                                 - 0.230999952340092 z_k1_s0053
                                 - 0.140000526017154 z_k1_s0054
                                 - 0.286000222279196 z_k1_s0055
                                 - 0.169000650448619 z_k1_s0056
                                 - 0.192000620131741 z_k1_s0057
                                 - 0.313999954610834 z_k1_s0060
                                 - 0.455999588225372 z_k1_s0061
                                 - 0.273999943099935 z_k1_s0062
                                 - 0.317000023349009 z_k1_s0063
                                 - 0.481998373960006 z_k1_s0064
                                 - 0.140999701201624 z_k1_s0065
                                 - 0.187999511102354 z_k1_s0066
                                 - 0.542998866638134 z_k1_s0067
                                 - 0.916999834576777 z_k1_s0070
                                 - 0.682999764226601 z_k1_s0071
                                 - 0.243000470056513 z_k1_s0072
                                 - 0.338000322172788 z_k1_s0073
                                 - 0.877002028109775 z_k1_s0074
                                 - 0.293000308643908 z_k1_s0075
                                 - 0.427001892820975 z_k1_s0076
                                 - 0.192000190056742 z_k1_s0077
                                 - 1.65099627621199 z_k1_s0100
                                 - 0.343999121680127 z_k1_s0101
                                 - 0.72999808612657 z_k1_s0102
                                 - 0.62900010223165 z_k1_s0103
                                 - 0.385998964446443 z_k1_s0104
                                 - 0.42499873654396 z_k1_s0105
                                 - 0.368998946188873 z_k1_s0106
                                 - 1.19899633035043 z_k1_s0107
                                 - 1.13700239697021 z_k1_s0110
                                 - 0.926999866536164 z_k1_s0111
                                 - 0.350000111120355 z_k1_s0112
                                 - 0.255000297553563 z_k1_s0113
                                 - 0.610998721142151 z_k1_s0114
                                 - 0.23000022193063 z_k1_s0115
                                 - 0.618001490358619 z_k1_s0116
                                 - 0.609999002397445 z_k1_s0117
                                 - 0.622000556344178 z_k1_s0120
                                 - 0.330998871301191 z_k1_s0121
                                 - 0.244000264786493 z_k1_s0122
                                 - 0.400001402601443 z_k1_s0123
                                 - 0.252000292838521 z_k1_s0124
                                 - 0.591999712648585 z_k1_s0125
                                 - 0.273000292625049 z_k1_s0126
                                 - 0.240000250760478 z_k1_s0127
                                 - 0.436999181025502 z_k1_s0130
                                 - 0.492999508269888 z_k1_s0131
                                 - 0.625000238874564 z_k1_s0132
                                 - 0.565999129903735 z_k1_s0133
                                 - 1.4519983364633 z_k1_s0134
                                 - 0.432999279922182 z_k1_s0135
                                 - 0.636999942184524 z_k1_s0136
                                 - 1.06299796867764 z_k1_s0137
                                 - 0.89999876832287 z_k1_s0140
                                 - 1.69299717645419 z_k1_s0141
                                 - 2.2749935765757 z_k1_s0142
                                 - 0.48899987899343 z_k1_s0143
                                 - 1.06199718759863 z_k1_s0144
                                 - 1.33099498226713 z_k1_s0145
                                 - 0.499999096890748 z_k1_s0146
                                 - 1.59799363535344 z_k1_s0147
                                 - 0.314000887636184 z_k1_s0150
                                 - 0.337999659837017 z_k1_s0151
                                 - 0.192000132011725 z_k1_s0152
                                 - 0.537001201008217 z_k1_s0153
                                 - 0.243000011539761 z_k1_s0154
                                 - 0.151000101567554 z_k1_s0155
                                 - 0.138000081358421 z_k1_s0156
                                 - 0.251000323607072 z_k1_s0157
                                 - 0.74400036013321 z_k1_s0160
                                 - 1.23600314640202 z_k1_s0161
                                 - 1.19299996685849 z_k1_s0162
                                 - 0.85000255519259 z_k1_s0163
                                 - 0.519000528588673 z_k1_s0164
                                 - 1.12300358039811 z_k1_s0165
                                 - 1.34600178741406 z_k1_s0166
                                 - 0.548000651337621 z_k1_s0167
                                 - 0.470001077012039 z_k1_s0170
                                 - 0.239999787153269 z_k1_s0171
                                 - 0.303000934584258 z_k1_s0172
                                 - 0.346000852927987 z_k1_s0173
                                 - 0.532999119144303 z_k1_s0174
                                 - 0.209999778471625 z_k1_s0175
                                 - 0.587999605560457 z_k1_s0176
                                 - 0.132999944388661 z_k1_s0177
                                 - 0.228000569548896 z_k1_s1000
                                 - 0.209000012502376 z_k1_s1001
                                 - 0.105000214015865 z_k1_s1002
                                 - 0.0940001682998818 z_k1_s1003
                                 - 0.205000796388696 z_k1_s1004
                                 - 0.208000819974891 z_k1_s1005
                                 - 0.0620000797807253 z_k1_s1006
                                 - 0.11200018751686 z_k1_s1007
                                 - 0.243000468522817 z_k1_s1010
                                 - 0.239000601460624 z_k1_s1011
                                 - 0.167000385335168 z_k1_s1012
                                 - 0.244000500513515 z_k1_s1013
                                 - 0.460001670691079 z_k1_s1014
                                 - 0.808001323827597 z_k1_s1015
                                 - 0.210000456432182 z_k1_s1016
                                 - 0.238000569469926 z_k1_s1017
                                 - 0.135000261415567 z_k1_s1020
                                 - 0.159000180307031 z_k1_s1021
                                 - 0.257000574481602 z_k1_s1022
                                 - 0.113000135131423 z_k1_s1023
                                 - 0.335999605281617 z_k1_s1024
                                 - 0.113000135131423 z_k1_s1025
                                 - 0.32200031622926 z_k1_s1026
                                 - 0.417000651092294 z_k1_s1027
                                 - 1.33400077159526 z_k1_s1030
                                 - 0.47699939742244 z_k1_s1031
                                 - 0.493998528777549 z_k1_s1032
                                 - 0.315999753298646 z_k1_s1033
                                 - 0.277000082131889 z_k1_s1034
                                 - 0.786001040279968 z_k1_s1035
                                 - 0.48299803437184 z_k1_s1036
                                 - 0.322999659738851 z_k1_s1037
                                 - 0.173000108503617 z_k1_s1040
                                 - 0.270000491526146 z_k1_s1041
                                 - 0.14300009384519 z_k1_s1042
                                 - 0.0960001547870573 z_k1_s1043
                                 - 0.323999870630774 z_k1_s1044
                                 - 0.17600007400943 z_k1_s1045
                                 - 0.188000835033432 z_k1_s1046
                                 - 0.427999873499949 z_k1_s1047
                                 - 0.533999452821646 z_k1_s1050
                                 - 0.179999791120809 z_k1_s1051
                                 - 0.201999810469965 z_k1_s1052
                                 - 0.408998777974841 z_k1_s1053
                                 - 0.557000546376302 z_k1_s1054
                                 - 0.374998917544124 z_k1_s1055
                                 - 0.27099967343801 z_k1_s1056
                                 - 0.578998080128424 z_k1_s1057
                                 - 0.151000049343062 z_k1_s1060
                                 - 0.113000017234857 z_k1_s1061
                                 - 0.239999742525098 z_k1_s1062
                                 - 0.138000070577126 z_k1_s1063
                                 - 0.498000209065241 z_k1_s1064
                                 - 0.308999602913506 z_k1_s1065
                                 - 0.153000207522581 z_k1_s1066
                                 - 0.147000082784337 z_k1_s1067
                                 - 0.14600039566218 z_k1_s1070
                                 - 0.339001858375685 z_k1_s1071
                                 - 0.168000487731912 z_k1_s1072
                                 - 0.203999633443301 z_k1_s1073
                                 - 0.290999970618984 z_k1_s1074
                                 - 0.13000029281137 z_k1_s1075
                                 - 0.526001357864932 z_k1_s1076
                                 - 0.170000549938396 z_k1_s1077
                                 - 0.363000669082569 z_k1_s1100
                                 - 0.326999072728023 z_k1_s1101
                                 - 0.268999929155129 z_k1_s1102
                                 - 0.164999911428633 z_k1_s1103
                                 - 0.510999689543845 z_k1_s1104
                                 - 0.104000017726496 z_k1_s1105
                                 - 0.149999863380603 z_k1_s1106
                                 - 0.167999838758279 z_k1_s1107
                                 - 0.51499969563854 z_k1_s1110
                                 - 0.339000724678946 z_k1_s1111
                                 - 0.158000158902799 z_k1_s1112
                                 - 0.11699997757227 z_k1_s1113
                                 - 0.313999968599094 z_k1_s1114
                                 - 0.290000108922786 z_k1_s1115
                                 - 0.240999493166626 z_k1_s1116
                                 - 0.431999880174519 z_k1_s1117
                                 - 0.703998714576519 z_k1_s1120
                                 - 0.146999659588692 z_k1_s1121
                                 - 0.514999858761219 z_k1_s1122
                                 - 0.599999115218246 z_k1_s1123
                                 - 0.318999900510369 z_k1_s1124
                                 - 0.238999709381824 z_k1_s1125
                                 - 0.281999460512798 z_k1_s1126
                                 - 0.284999579728667 z_k1_s1127
                                 - 0.308000074200251 z_k1_s1130
                                 - 0.191999817869144 z_k1_s1131
                                 - 0.315000623386244 z_k1_s1132
                                 - 0.161999838798971 z_k1_s1133
                                 - 0.244000215210481 z_k1_s1134
                                 - 0.376998604752627 z_k1_s1135
                                 - 0.181999824845753 z_k1_s1136
                                 - 0.628999103461622 z_k1_s1137
                                 - 0.290999692139589 z_k1_s1140
                                 - 0.369999630381919 z_k1_s1141
                                 - 0.576000385598532 z_k1_s1142
                                 - 1.05099730836973 z_k1_s1143
                                 - 0.890998612256929 z_k1_s1144
                                 - 0.314999496556509 z_k1_s1145
                                 - 0.298999626945229 z_k1_s1146
                                 - 0.439998530807057 z_k1_s1147
                                 - 0.433000236947821 z_k1_s1150
                                 - 0.240998792427566 z_k1_s1151
                                 - 0.379999205950818 z_k1_s1152
                                 - 0.354999228611985 z_k1_s1153
                                 - 0.136999920776751 z_k1_s1154
                                 - 0.327000422951058 z_k1_s1155
                                 - 0.246998697068997 z_k1_s1156
                                 - 0.127999839014882 z_k1_s1157
                                 - 0.281999977681472 z_k1_s1160
                                 - 0.259001072342314 z_k1_s1161
                                 - 0.331999363766777 z_k1_s1162
                                 - 0.416999992411967 z_k1_s1163
                                 - 0.246000133020023 z_k1_s1164
                                 - 0.459001005683779 z_k1_s1165
                                 - 0.157000079068063 z_k1_s1166
                                 - 0.237000052404649 z_k1_s1167
                                 - 0.478000225834528 z_k1_s1170
                                 - 0.823000542271275 z_k1_s1171
                                 - 0.700997626829697 z_k1_s1172
                                 - 0.238999654518183 z_k1_s1173
                                 - 0.431998961245412 z_k1_s1174
                                 - 0.797000146389831 z_k1_s1175
                                 - 0.555996829488048 z_k1_s1176
                                 - 0.558000033472567 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.03378 w_k1_ADM_s000
                                 + 1.19712 w_k1_AXP_s000
                                 + 1.06757 w_k1_DIS_s000
                                 + 1.10356 w_k1_FDX_s000
                                 + 0.955419 w_k1_IBM_s000
                                 + 0.970181 w_k1_INTC_s000
                                 + 1.03979 w_k1_KO_s000 + 1.35821 w_k1_MAT_s000
                                 + 1.02336 w_k1_MSFT_s000
                                 + 1.01628 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.836295 w_k1_ADM_s000
                                 + 1.24166 w_k1_AXP_s000
                                 + 0.887372 w_k1_DIS_s000
                                 + 0.809338 w_k1_FDX_s000
                                 + 1.43649 w_k1_IBM_s000
                                 + 0.924111 w_k1_INTC_s000
                                 + 1.06094 w_k1_KO_s000 + 1.69805 w_k1_MAT_s000
                                 + 0.830801 w_k1_MSFT_s000
                                 + 0.99112 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.08703 w_k1_ADM_s000
                                 + 1.713 w_k1_AXP_s000 + 1.29427 w_k1_DIS_s000
                                 + 1.27132 w_k1_FDX_s000
                                 + 1.29324 w_k1_IBM_s000
                                 + 1.19636 w_k1_INTC_s000 + 1.0398 w_k1_KO_s000
                                 + 1.53428 w_k1_MAT_s000
                                 + 1.15531 w_k1_MSFT_s000
                                 + 1.01818 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.988972 w_k1_ADM_s000
                                 + 1.15818 w_k1_AXP_s000
                                 + 1.62076 w_k1_DIS_s000
                                 + 1.20856 w_k1_FDX_s000
                                 + 1.30351 w_k1_IBM_s000
                                 + 1.08093 w_k1_INTC_s000
                                 + 1.32553 w_k1_KO_s000 + 1.57039 w_k1_MAT_s000
                                 + 1.26244 w_k1_MSFT_s000
                                 + 1.04624 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.04588 w_k1_ADM_s000
                                 + 1.23894 w_k1_AXP_s000
                                 + 0.94432 w_k1_DIS_s000
                                 + 0.964881 w_k1_FDX_s000
                                 + 1.16098 w_k1_IBM_s000
                                 + 0.98741 w_k1_INTC_s000
                                 + 0.998399 w_k1_KO_s000
                                 + 0.941968 w_k1_MAT_s000
                                 + 0.977431 w_k1_MSFT_s000
                                 + 0.942597 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.802389 w_k1_ADM_s000
                                 + 0.949875 w_k1_AXP_s000
                                 + 0.696465 w_k1_DIS_s000
                                 + 0.811925 w_k1_FDX_s000
                                 + 0.961579 w_k1_IBM_s000
                                 + 0.856868 w_k1_INTC_s000
                                 + 0.776401 w_k1_KO_s000
                                 + 0.783597 w_k1_MAT_s000
                                 + 0.804566 w_k1_MSFT_s000
                                 + 0.806016 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.862896 w_k1_ADM_s000
                                 + 1.22958 w_k1_AXP_s000
                                 + 1.20722 w_k1_DIS_s000
                                 + 1.01122 w_k1_FDX_s000
                                 + 1.38979 w_k1_IBM_s000
                                 + 1.02797 w_k1_INTC_s000
                                 + 1.29945 w_k1_KO_s000 + 1.34482 w_k1_MAT_s000
                                 + 1.08208 w_k1_MSFT_s000
                                 + 1.11113 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.781471 w_k1_ADM_s000
                                 + 0.919421 w_k1_AXP_s000
                                 + 0.772264 w_k1_DIS_s000
                                 + 0.999207 w_k1_FDX_s000
                                 + 0.740128 w_k1_IBM_s000
                                 + 0.817581 w_k1_INTC_s000
                                 + 0.776906 w_k1_KO_s000
                                 + 0.789788 w_k1_MAT_s000
                                 + 0.927723 w_k1_MSFT_s000
                                 + 0.901145 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.19996 w_k1_ADM_s001
                                 + 0.971733 w_k1_AXP_s001
                                 + 1.35382 w_k1_DIS_s001
                                 + 1.11154 w_k1_FDX_s001
                                 + 0.949597 w_k1_IBM_s001
                                 + 1.15672 w_k1_INTC_s001 + 1.2814 w_k1_KO_s001
                                 + 0.784603 w_k1_MAT_s001
                                 + 1.17397 w_k1_MSFT_s001
                                 + 1.03536 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.89489 w_k1_ADM_s001
                                 + 1.38281 w_k1_AXP_s001
                                 + 2.03146 w_k1_DIS_s001
                                 + 1.57687 w_k1_FDX_s001
                                 + 1.18655 w_k1_IBM_s001
                                 + 1.48795 w_k1_INTC_s001
                                 + 1.47072 w_k1_KO_s001 + 0.93978 w_k1_MAT_s001
                                 + 1.55575 w_k1_MSFT_s001
                                 + 1.26271 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.0829 w_k1_ADM_s001
                                 + 1.29239 w_k1_AXP_s001
                                 + 0.964012 w_k1_DIS_s001
                                 + 1.05988 w_k1_FDX_s001
                                 + 1.52224 w_k1_IBM_s001
                                 + 1.01379 w_k1_INTC_s001
                                 + 1.15528 w_k1_KO_s001 + 1.63211 w_k1_MAT_s001
                                 + 1.02969 w_k1_MSFT_s001
                                 + 1.05974 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.14808 w_k1_ADM_s001
                                 + 1.3254 w_k1_AXP_s001 + 1.25472 w_k1_DIS_s001
                                 + 1.32699 w_k1_FDX_s001
                                 + 0.834806 w_k1_IBM_s001
                                 + 1.2658 w_k1_INTC_s001 + 1.05748 w_k1_KO_s001
                                 + 0.755653 w_k1_MAT_s001
                                 + 1.19542 w_k1_MSFT_s001
                                 + 1.08275 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.884378 w_k1_ADM_s001
                                 + 0.907413 w_k1_AXP_s001
                                 + 0.810556 w_k1_DIS_s001
                                 + 0.89971 w_k1_FDX_s001
                                 + 0.606511 w_k1_IBM_s001
                                 + 0.870847 w_k1_INTC_s001
                                 + 0.806202 w_k1_KO_s001
                                 + 0.698339 w_k1_MAT_s001
                                 + 0.8008 w_k1_MSFT_s001 + 0.80992 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.814306 w_k1_ADM_s001
                                 + 0.671324 w_k1_AXP_s001
                                 + 0.827048 w_k1_DIS_s001
                                 + 0.807129 w_k1_FDX_s001
                                 + 0.892227 w_k1_IBM_s001
                                 + 0.992337 w_k1_INTC_s001
                                 + 1.05442 w_k1_KO_s001
                                 + 0.611797 w_k1_MAT_s001
                                 + 0.938563 w_k1_MSFT_s001
                                 + 0.841336 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.699649 w_k1_ADM_s001
                                 + 0.514253 w_k1_AXP_s001
                                 + 0.539264 w_k1_DIS_s001
                                 + 0.591822 w_k1_FDX_s001
                                 + 0.786785 w_k1_IBM_s001
                                 + 0.730412 w_k1_INTC_s001
                                 + 0.761488 w_k1_KO_s001
                                 + 0.452852 w_k1_MAT_s001
                                 + 0.772348 w_k1_MSFT_s001
                                 + 0.801745 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.895876 w_k1_ADM_s001
                                 + 0.756772 w_k1_AXP_s001
                                 + 0.970178 w_k1_DIS_s001
                                 + 0.912513 w_k1_FDX_s001
                                 + 1.20531 w_k1_IBM_s001
                                 + 0.899545 w_k1_INTC_s001
                                 + 1.03871 w_k1_KO_s001
                                 + 0.663873 w_k1_MAT_s001
                                 + 0.927987 w_k1_MSFT_s001
                                 + 0.965472 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.939424 w_k1_ADM_s002
                                 + 1.20074 w_k1_AXP_s002
                                 + 0.969425 w_k1_DIS_s002
                                 + 1.03697 w_k1_FDX_s002
                                 + 1.10204 w_k1_IBM_s002
                                 + 0.917001 w_k1_INTC_s002
                                 + 0.964935 w_k1_KO_s002
                                 + 1.22532 w_k1_MAT_s002
                                 + 0.998441 w_k1_MSFT_s002
                                 + 1.02159 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.05116 w_k1_ADM_s002
                                 + 1.31925 w_k1_AXP_s002
                                 + 1.53822 w_k1_DIS_s002 + 1.2447 w_k1_FDX_s002
                                 + 1.60455 w_k1_IBM_s002
                                 + 1.03584 w_k1_INTC_s002 + 1.5266 w_k1_KO_s002
                                 + 1.79414 w_k1_MAT_s002
                                 + 1.21269 w_k1_MSFT_s002
                                 + 1.47393 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.735536 w_k1_ADM_s002
                                 + 0.640386 w_k1_AXP_s002
                                 + 0.535591 w_k1_DIS_s002
                                 + 0.72771 w_k1_FDX_s002
                                 + 0.733323 w_k1_IBM_s002
                                 + 0.71147 w_k1_INTC_s002
                                 + 0.704239 w_k1_KO_s002
                                 + 0.608945 w_k1_MAT_s002
                                 + 0.819041 w_k1_MSFT_s002
                                 + 0.954099 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.916706 w_k1_ADM_s002
                                 + 0.725761 w_k1_AXP_s002
                                 + 1.29739 w_k1_DIS_s002 + 1.0753 w_k1_FDX_s002
                                 + 0.748591 w_k1_IBM_s002
                                 + 0.895847 w_k1_INTC_s002
                                 + 0.933578 w_k1_KO_s002
                                 + 0.725577 w_k1_MAT_s002
                                 + 1.13401 w_k1_MSFT_s002
                                 + 1.03504 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.21515 w_k1_ADM_s002
                                 + 1.30227 w_k1_AXP_s002
                                 + 1.14875 w_k1_DIS_s002
                                 + 1.34285 w_k1_FDX_s002
                                 + 0.865598 w_k1_IBM_s002
                                 + 1.06627 w_k1_INTC_s002
                                 + 1.01783 w_k1_KO_s002
                                 + 0.959779 w_k1_MAT_s002
                                 + 1.27108 w_k1_MSFT_s002
                                 + 1.33171 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.897034 w_k1_ADM_s002
                                 + 0.760142 w_k1_AXP_s002
                                 + 0.844153 w_k1_DIS_s002
                                 + 0.914023 w_k1_FDX_s002
                                 + 0.762265 w_k1_IBM_s002
                                 + 0.891432 w_k1_INTC_s002
                                 + 0.824779 w_k1_KO_s002
                                 + 0.803205 w_k1_MAT_s002
                                 + 0.999906 w_k1_MSFT_s002
                                 + 1.04993 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.633394 w_k1_ADM_s002
                                 + 0.616325 w_k1_AXP_s002
                                 + 0.77833 w_k1_DIS_s002
                                 + 0.713468 w_k1_FDX_s002
                                 + 1.50278 w_k1_IBM_s002
                                 + 0.839905 w_k1_INTC_s002
                                 + 0.977291 w_k1_KO_s002
                                 + 1.04133 w_k1_MAT_s002
                                 + 0.741662 w_k1_MSFT_s002
                                 + 0.939822 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.932222 w_k1_ADM_s002
                                 + 0.990967 w_k1_AXP_s002
                                 + 0.843075 w_k1_DIS_s002
                                 + 0.810517 w_k1_FDX_s002
                                 + 1.00022 w_k1_IBM_s002
                                 + 0.719011 w_k1_INTC_s002
                                 + 0.957714 w_k1_KO_s002
                                 + 1.30428 w_k1_MAT_s002
                                 + 0.913537 w_k1_MSFT_s002
                                 + 1.08069 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.031 w_k1_ADM_s003
                                 + 1.77426 w_k1_AXP_s003
                                 + 1.63251 w_k1_DIS_s003
                                 + 1.38864 w_k1_FDX_s003
                                 + 1.54782 w_k1_IBM_s003
                                 + 1.21865 w_k1_INTC_s003
                                 + 1.64238 w_k1_KO_s003 + 1.75303 w_k1_MAT_s003
                                 + 1.20436 w_k1_MSFT_s003
                                 + 1.11095 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.955541 w_k1_ADM_s003
                                 + 1.15562 w_k1_AXP_s003
                                 + 1.04355 w_k1_DIS_s003
                                 + 0.960371 w_k1_FDX_s003
                                 + 1.5655 w_k1_IBM_s003
                                 + 0.828638 w_k1_INTC_s003
                                 + 1.35098 w_k1_KO_s003 + 1.09561 w_k1_MAT_s003
                                 + 0.975646 w_k1_MSFT_s003
                                 + 0.936606 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.772157 w_k1_ADM_s003
                                 + 1.08347 w_k1_AXP_s003
                                 + 0.952284 w_k1_DIS_s003
                                 + 0.872124 w_k1_FDX_s003
                                 + 1.1233 w_k1_IBM_s003
                                 + 0.939718 w_k1_INTC_s003
                                 + 1.04573 w_k1_KO_s003 + 1.0049 w_k1_MAT_s003
                                 + 0.944407 w_k1_MSFT_s003
                                 + 0.880452 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.78892 w_k1_ADM_s003
                                 + 0.669749 w_k1_AXP_s003
                                 + 0.668883 w_k1_DIS_s003
                                 + 0.732648 w_k1_FDX_s003
                                 + 0.784517 w_k1_IBM_s003
                                 + 0.589381 w_k1_INTC_s003
                                 + 0.867276 w_k1_KO_s003
                                 + 0.660676 w_k1_MAT_s003
                                 + 0.819719 w_k1_MSFT_s003
                                 + 0.900592 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.128 w_k1_ADM_s003
                                 + 1.12103 w_k1_AXP_s003
                                 + 1.52701 w_k1_DIS_s003
                                 + 1.15833 w_k1_FDX_s003
                                 + 0.964025 w_k1_IBM_s003
                                 + 1.2285 w_k1_INTC_s003 + 1.30627 w_k1_KO_s003
                                 + 0.792144 w_k1_MAT_s003
                                 + 1.21538 w_k1_MSFT_s003
                                 + 1.07551 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.998729 w_k1_ADM_s003
                                 + 1.18332 w_k1_AXP_s003
                                 + 1.21409 w_k1_DIS_s003
                                 + 1.10177 w_k1_FDX_s003
                                 + 1.19741 w_k1_IBM_s003
                                 + 1.16341 w_k1_INTC_s003
                                 + 1.43219 w_k1_KO_s003
                                 + 0.961968 w_k1_MAT_s003
                                 + 1.34214 w_k1_MSFT_s003
                                 + 1.04614 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.938936 w_k1_ADM_s003
                                 + 1.14549 w_k1_AXP_s003
                                 + 1.09692 w_k1_DIS_s003
                                 + 1.07518 w_k1_FDX_s003
                                 + 0.938976 w_k1_IBM_s003
                                 + 0.947447 w_k1_INTC_s003
                                 + 1.08571 w_k1_KO_s003 + 1.08361 w_k1_MAT_s003
                                 + 1.03952 w_k1_MSFT_s003
                                 + 0.98776 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.835632 w_k1_ADM_s003
                                 + 1.00676 w_k1_AXP_s003
                                 + 0.883799 w_k1_DIS_s003
                                 + 0.976014 w_k1_FDX_s003
                                 + 0.680257 w_k1_IBM_s003
                                 + 0.904769 w_k1_INTC_s003
                                 + 0.863212 w_k1_KO_s003
                                 + 0.726872 w_k1_MAT_s003
                                 + 0.986454 w_k1_MSFT_s003
                                 + 0.897249 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.936754 w_k1_ADM_s004
                                 + 0.952122 w_k1_AXP_s004
                                 + 0.775917 w_k1_DIS_s004
                                 + 0.768175 w_k1_FDX_s004
                                 + 1.16845 w_k1_IBM_s004
                                 + 0.71665 w_k1_INTC_s004
                                 + 0.810992 w_k1_KO_s004
                                 + 1.54021 w_k1_MAT_s004
                                 + 0.864812 w_k1_MSFT_s004
                                 + 1.05848 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.3152 w_k1_ADM_s004
                                 + 1.38679 w_k1_AXP_s004 + 1.7604 w_k1_DIS_s004
                                 + 1.34008 w_k1_FDX_s004
                                 + 1.37026 w_k1_IBM_s004
                                 + 1.03131 w_k1_INTC_s004
                                 + 1.27027 w_k1_KO_s004 + 1.54798 w_k1_MAT_s004
                                 + 1.21723 w_k1_MSFT_s004
                                 + 1.33422 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.06604 w_k1_ADM_s004
                                 + 0.89706 w_k1_AXP_s004
                                 + 0.967178 w_k1_DIS_s004
                                 + 0.857894 w_k1_FDX_s004
                                 + 1.2402 w_k1_IBM_s004
                                 + 0.913166 w_k1_INTC_s004
                                 + 1.12724 w_k1_KO_s004 + 1.01756 w_k1_MAT_s004
                                 + 1.0962 w_k1_MSFT_s004 + 1.11785 w_k1_PG_s004
                                 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.888663 w_k1_ADM_s004
                                 + 1.38652 w_k1_AXP_s004
                                 + 0.830632 w_k1_DIS_s004
                                 + 0.923365 w_k1_FDX_s004
                                 + 1.36495 w_k1_IBM_s004
                                 + 1.1405 w_k1_INTC_s004
                                 + 0.979768 w_k1_KO_s004 + 1.4216 w_k1_MAT_s004
                                 + 0.941812 w_k1_MSFT_s004
                                 + 1.10404 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.26234 w_k1_ADM_s004
                                 + 1.0442 w_k1_AXP_s004 + 1.36127 w_k1_DIS_s004
                                 + 1.21669 w_k1_FDX_s004
                                 + 0.803743 w_k1_IBM_s004
                                 + 1.40656 w_k1_INTC_s004
                                 + 1.07621 w_k1_KO_s004
                                 + 0.746599 w_k1_MAT_s004
                                 + 1.35489 w_k1_MSFT_s004
                                 + 1.18835 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.917467 w_k1_ADM_s004
                                 + 0.890889 w_k1_AXP_s004
                                 + 0.774716 w_k1_DIS_s004
                                 + 0.845313 w_k1_FDX_s004
                                 + 0.688837 w_k1_IBM_s004
                                 + 0.793038 w_k1_INTC_s004
                                 + 0.68701 w_k1_KO_s004
                                 + 0.802588 w_k1_MAT_s004
                                 + 0.991661 w_k1_MSFT_s004
                                 + 0.921743 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.37867 w_k1_ADM_s004
                                 + 1.51182 w_k1_AXP_s004
                                 + 1.21556 w_k1_DIS_s004
                                 + 1.14279 w_k1_FDX_s004
                                 + 1.39644 w_k1_IBM_s004
                                 + 0.94178 w_k1_INTC_s004
                                 + 1.07333 w_k1_KO_s004 + 1.43875 w_k1_MAT_s004
                                 + 1.0887 w_k1_MSFT_s004 + 1.16293 w_k1_PG_s004
                                 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.02567 w_k1_ADM_s004
                                 + 0.980219 w_k1_AXP_s004
                                 + 1.14057 w_k1_DIS_s004
                                 + 1.05473 w_k1_FDX_s004 + 1.0132 w_k1_IBM_s004
                                 + 0.983081 w_k1_INTC_s004
                                 + 0.871658 w_k1_KO_s004
                                 + 1.15702 w_k1_MAT_s004
                                 + 1.11409 w_k1_MSFT_s004
                                 + 1.06587 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.936245 w_k1_ADM_s005
                                 + 1.22715 w_k1_AXP_s005
                                 + 1.36665 w_k1_DIS_s005
                                 + 1.13154 w_k1_FDX_s005
                                 + 0.991567 w_k1_IBM_s005
                                 + 1.05559 w_k1_INTC_s005
                                 + 1.04273 w_k1_KO_s005 + 0.83752 w_k1_MAT_s005
                                 + 1.14813 w_k1_MSFT_s005
                                 + 1.01335 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.23375 w_k1_ADM_s005
                                 + 1.76795 w_k1_AXP_s005
                                 + 0.968072 w_k1_DIS_s005
                                 + 1.10044 w_k1_FDX_s005
                                 + 1.23403 w_k1_IBM_s005
                                 + 0.855308 w_k1_INTC_s005
                                 + 1.04526 w_k1_KO_s005 + 1.41253 w_k1_MAT_s005
                                 + 1.03226 w_k1_MSFT_s005
                                 + 1.07545 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.03342 w_k1_ADM_s005
                                 + 0.87932 w_k1_AXP_s005
                                 + 0.905596 w_k1_DIS_s005
                                 + 1.00452 w_k1_FDX_s005
                                 + 0.841971 w_k1_IBM_s005
                                 + 0.794216 w_k1_INTC_s005
                                 + 0.818019 w_k1_KO_s005
                                 + 0.838166 w_k1_MAT_s005
                                 + 1.02584 w_k1_MSFT_s005
                                 + 0.839578 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.79287 w_k1_ADM_s005
                                 + 0.733473 w_k1_AXP_s005
                                 + 0.770408 w_k1_DIS_s005
                                 + 0.852852 w_k1_FDX_s005
                                 + 0.791737 w_k1_IBM_s005
                                 + 0.839606 w_k1_INTC_s005
                                 + 0.84296 w_k1_KO_s005
                                 + 0.893663 w_k1_MAT_s005
                                 + 0.84753 w_k1_MSFT_s005
                                 + 0.850805 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.675071 w_k1_ADM_s005
                                 + 0.913743 w_k1_AXP_s005
                                 + 0.767055 w_k1_DIS_s005
                                 + 0.797663 w_k1_FDX_s005
                                 + 1.1457 w_k1_IBM_s005
                                 + 0.775618 w_k1_INTC_s005
                                 + 0.98606 w_k1_KO_s005
                                 + 0.958817 w_k1_MAT_s005
                                 + 0.761082 w_k1_MSFT_s005
                                 + 0.821189 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.854114 w_k1_ADM_s005
                                 + 0.900574 w_k1_AXP_s005
                                 + 1.06722 w_k1_DIS_s005
                                 + 0.906773 w_k1_FDX_s005
                                 + 1.32986 w_k1_IBM_s005
                                 + 1.0082 w_k1_INTC_s005 + 1.20014 w_k1_KO_s005
                                 + 0.946551 w_k1_MAT_s005
                                 + 1.07276 w_k1_MSFT_s005
                                 + 1.06441 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.743332 w_k1_ADM_s005
                                 + 0.67207 w_k1_AXP_s005
                                 + 0.628022 w_k1_DIS_s005
                                 + 0.713499 w_k1_FDX_s005
                                 + 0.706079 w_k1_IBM_s005
                                 + 0.771843 w_k1_INTC_s005
                                 + 0.80371 w_k1_KO_s005
                                 + 0.666568 w_k1_MAT_s005
                                 + 0.790564 w_k1_MSFT_s005
                                 + 0.915474 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.20128 w_k1_ADM_s005
                                 + 1.42462 w_k1_AXP_s005
                                 + 1.86467 w_k1_DIS_s005
                                 + 1.35192 w_k1_FDX_s005
                                 + 1.48348 w_k1_IBM_s005
                                 + 1.21555 w_k1_INTC_s005
                                 + 1.52711 w_k1_KO_s005 + 1.7497 w_k1_MAT_s005
                                 + 1.40076 w_k1_MSFT_s005
                                 + 1.24234 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.01537 w_k1_ADM_s006
                                 + 1.44095 w_k1_AXP_s006
                                 + 1.16769 w_k1_DIS_s006
                                 + 1.24094 w_k1_FDX_s006
                                 + 1.25358 w_k1_IBM_s006
                                 + 1.17088 w_k1_INTC_s006
                                 + 1.04065 w_k1_KO_s006 + 1.51103 w_k1_MAT_s006
                                 + 1.12261 w_k1_MSFT_s006
                                 + 1.04121 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.00529 w_k1_ADM_s006
                                 + 1.19179 w_k1_AXP_s006
                                 + 1.55268 w_k1_DIS_s006
                                 + 1.24497 w_k1_FDX_s006
                                 + 1.31644 w_k1_IBM_s006
                                 + 1.3028 w_k1_INTC_s006 + 1.24136 w_k1_KO_s006
                                 + 1.84227 w_k1_MAT_s006
                                 + 1.29102 w_k1_MSFT_s006
                                 + 1.02862 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.86372 w_k1_ADM_s006
                                 + 1.05958 w_k1_AXP_s006
                                 + 0.867339 w_k1_DIS_s006
                                 + 0.867045 w_k1_FDX_s006
                                 + 1.62514 w_k1_IBM_s006
                                 + 1.13493 w_k1_INTC_s006
                                 + 0.971931 w_k1_KO_s006
                                 + 1.60206 w_k1_MAT_s006
                                 + 0.991676 w_k1_MSFT_s006
                                 + 0.959172 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.3967 w_k1_ADM_s006
                                 + 1.21938 w_k1_AXP_s006
                                 + 1.00637 w_k1_DIS_s006
                                 + 1.14988 w_k1_FDX_s006
                                 + 0.868933 w_k1_IBM_s006
                                 + 0.9072 w_k1_INTC_s006
                                 + 0.969313 w_k1_KO_s006 + 1.0102 w_k1_MAT_s006
                                 + 1.03197 w_k1_MSFT_s006
                                 + 0.977191 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.01369 w_k1_ADM_s006
                                 + 0.931707 w_k1_AXP_s006
                                 + 1.12119 w_k1_DIS_s006 + 1.0076 w_k1_FDX_s006
                                 + 1.07522 w_k1_IBM_s006
                                 + 1.06465 w_k1_INTC_s006
                                 + 1.15102 w_k1_KO_s006 + 1.09308 w_k1_MAT_s006
                                 + 0.988923 w_k1_MSFT_s006
                                 + 1.01684 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.06327 w_k1_ADM_s006
                                 + 0.75411 w_k1_AXP_s006
                                 + 0.885735 w_k1_DIS_s006
                                 + 1.05284 w_k1_FDX_s006
                                 + 0.764717 w_k1_IBM_s006
                                 + 0.893309 w_k1_INTC_s006
                                 + 0.861753 w_k1_KO_s006
                                 + 0.797293 w_k1_MAT_s006
                                 + 1.04462 w_k1_MSFT_s006
                                 + 0.872449 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.681744 w_k1_ADM_s006
                                 + 0.66918 w_k1_AXP_s006
                                 + 0.521627 w_k1_DIS_s006
                                 + 0.695967 w_k1_FDX_s006
                                 + 0.937251 w_k1_IBM_s006
                                 + 0.816993 w_k1_INTC_s006
                                 + 0.721664 w_k1_KO_s006
                                 + 0.72206 w_k1_MAT_s006
                                 + 0.75296 w_k1_MSFT_s006
                                 + 0.763632 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.918267 w_k1_ADM_s006
                                 + 0.822121 w_k1_AXP_s006
                                 + 0.722988 w_k1_DIS_s006
                                 + 0.828706 w_k1_FDX_s006
                                 + 0.923063 w_k1_IBM_s006
                                 + 0.992982 w_k1_INTC_s006
                                 + 0.864434 w_k1_KO_s006
                                 + 0.811844 w_k1_MAT_s006
                                 + 0.816705 w_k1_MSFT_s006
                                 + 0.894597 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.993499 w_k1_ADM_s007
                                 + 0.963828 w_k1_AXP_s007
                                 + 0.990914 w_k1_DIS_s007
                                 + 0.937687 w_k1_FDX_s007
                                 + 1.37649 w_k1_IBM_s007
                                 + 0.986219 w_k1_INTC_s007
                                 + 1.07979 w_k1_KO_s007 + 1.40063 w_k1_MAT_s007
                                 + 0.910193 w_k1_MSFT_s007
                                 + 0.9943 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.892557 w_k1_ADM_s007
                                 + 0.984463 w_k1_AXP_s007
                                 + 1.01232 w_k1_DIS_s007
                                 + 0.992627 w_k1_FDX_s007
                                 + 0.982484 w_k1_IBM_s007
                                 + 1.03319 w_k1_INTC_s007
                                 + 1.09364 w_k1_KO_s007
                                 + 0.987009 w_k1_MAT_s007
                                 + 0.982625 w_k1_MSFT_s007
                                 + 0.977156 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.91775 w_k1_ADM_s007
                                 + 0.759371 w_k1_AXP_s007
                                 + 0.681149 w_k1_DIS_s007
                                 + 0.786705 w_k1_FDX_s007
                                 + 0.969753 w_k1_IBM_s007
                                 + 0.892489 w_k1_INTC_s007
                                 + 0.915209 w_k1_KO_s007
                                 + 1.07533 w_k1_MAT_s007
                                 + 0.853897 w_k1_MSFT_s007
                                 + 0.925339 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.684365 w_k1_ADM_s007
                                 + 0.707458 w_k1_AXP_s007
                                 + 0.559499 w_k1_DIS_s007
                                 + 0.774843 w_k1_FDX_s007
                                 + 0.734234 w_k1_IBM_s007
                                 + 0.811786 w_k1_INTC_s007
                                 + 0.713118 w_k1_KO_s007
                                 + 0.836921 w_k1_MAT_s007
                                 + 0.767565 w_k1_MSFT_s007
                                 + 0.830703 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.04905 w_k1_ADM_s007
                                 + 1.24019 w_k1_AXP_s007
                                 + 1.24732 w_k1_DIS_s007
                                 + 1.28986 w_k1_FDX_s007
                                 + 1.02056 w_k1_IBM_s007
                                 + 1.32052 w_k1_INTC_s007 + 1.1521 w_k1_KO_s007
                                 + 1.44172 w_k1_MAT_s007
                                 + 1.20906 w_k1_MSFT_s007
                                 + 1.03708 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.975216 w_k1_ADM_s007
                                 + 1.44819 w_k1_AXP_s007
                                 + 1.78837 w_k1_DIS_s007
                                 + 1.39088 w_k1_FDX_s007
                                 + 1.44991 w_k1_IBM_s007
                                 + 1.29392 w_k1_INTC_s007
                                 + 1.62743 w_k1_KO_s007 + 2.57773 w_k1_MAT_s007
                                 + 1.46504 w_k1_MSFT_s007
                                 + 1.23609 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.783353 w_k1_ADM_s007
                                 + 0.75063 w_k1_AXP_s007
                                 + 0.818942 w_k1_DIS_s007
                                 + 0.909933 w_k1_FDX_s007
                                 + 0.751431 w_k1_IBM_s007
                                 + 0.91102 w_k1_INTC_s007
                                 + 0.801203 w_k1_KO_s007
                                 + 0.722899 w_k1_MAT_s007
                                 + 0.829538 w_k1_MSFT_s007
                                 + 0.924678 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.745636 w_k1_ADM_s007
                                 + 0.604285 w_k1_AXP_s007
                                 + 0.781544 w_k1_DIS_s007
                                 + 0.694937 w_k1_FDX_s007
                                 + 1.19365 w_k1_IBM_s007
                                 + 0.738616 w_k1_INTC_s007
                                 + 1.12628 w_k1_KO_s007 + 1.14464 w_k1_MAT_s007
                                 + 0.8538 w_k1_MSFT_s007
                                 + 0.853891 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.04172 w_k1_ADM_s010
                                 + 1.12628 w_k1_AXP_s010 + 1.151 w_k1_DIS_s010
                                 + 0.984613 w_k1_FDX_s010
                                 + 1.20803 w_k1_IBM_s010
                                 + 1.1215 w_k1_INTC_s010 + 1.01329 w_k1_KO_s010
                                 + 0.972226 w_k1_MAT_s010
                                 + 1.08392 w_k1_MSFT_s010
                                 + 1.07612 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.677822 w_k1_ADM_s010
                                 + 0.733939 w_k1_AXP_s010
                                 + 0.55883 w_k1_DIS_s010
                                 + 0.683316 w_k1_FDX_s010
                                 + 1.04339 w_k1_IBM_s010
                                 + 0.684278 w_k1_INTC_s010
                                 + 0.710189 w_k1_KO_s010
                                 + 0.839559 w_k1_MAT_s010
                                 + 0.700204 w_k1_MSFT_s010
                                 + 0.801008 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.07461 w_k1_ADM_s010
                                 + 0.934753 w_k1_AXP_s010
                                 + 0.988069 w_k1_DIS_s010
                                 + 0.98266 w_k1_FDX_s010
                                 + 0.726374 w_k1_IBM_s010
                                 + 1.04331 w_k1_INTC_s010
                                 + 0.912137 w_k1_KO_s010
                                 + 0.838021 w_k1_MAT_s010
                                 + 1.11507 w_k1_MSFT_s010
                                 + 1.07156 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.804369 w_k1_ADM_s010
                                 + 0.734103 w_k1_AXP_s010
                                 + 0.849511 w_k1_DIS_s010
                                 + 0.794156 w_k1_FDX_s010
                                 + 1.00773 w_k1_IBM_s010
                                 + 1.07418 w_k1_INTC_s010
                                 + 0.972996 w_k1_KO_s010
                                 + 0.811343 w_k1_MAT_s010
                                 + 0.911427 w_k1_MSFT_s010
                                 + 0.913619 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.10159 w_k1_ADM_s010
                                 + 1.68211 w_k1_AXP_s010
                                 + 1.78686 w_k1_DIS_s010
                                 + 1.50788 w_k1_FDX_s010
                                 + 1.44804 w_k1_IBM_s010
                                 + 1.41541 w_k1_INTC_s010
                                 + 1.24397 w_k1_KO_s010 + 1.76033 w_k1_MAT_s010
                                 + 1.41482 w_k1_MSFT_s010
                                 + 1.27401 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.87352 w_k1_ADM_s010
                                 + 0.823256 w_k1_AXP_s010
                                 + 0.731991 w_k1_DIS_s010
                                 + 0.750699 w_k1_FDX_s010
                                 + 0.746915 w_k1_IBM_s010
                                 + 0.9455 w_k1_INTC_s010 + 0.80605 w_k1_KO_s010
                                 + 0.737251 w_k1_MAT_s010
                                 + 0.976554 w_k1_MSFT_s010
                                 + 0.878613 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.701185 w_k1_ADM_s010
                                 + 0.739672 w_k1_AXP_s010
                                 + 0.616711 w_k1_DIS_s010
                                 + 0.839814 w_k1_FDX_s010
                                 + 0.554924 w_k1_IBM_s010
                                 + 0.839756 w_k1_INTC_s010
                                 + 0.59583 w_k1_KO_s010
                                 + 0.479886 w_k1_MAT_s010
                                 + 0.834852 w_k1_MSFT_s010
                                 + 0.836949 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.1495 w_k1_ADM_s010
                                 + 1.48329 w_k1_AXP_s010
                                 + 1.25168 w_k1_DIS_s010
                                 + 1.27346 w_k1_FDX_s010
                                 + 0.934162 w_k1_IBM_s010
                                 + 1.31275 w_k1_INTC_s010
                                 + 0.979991 w_k1_KO_s010
                                 + 1.20496 w_k1_MAT_s010
                                 + 1.30451 w_k1_MSFT_s010 + 1.109 w_k1_PG_s010
                                 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.950805 w_k1_ADM_s011
                                 + 0.834123 w_k1_AXP_s011
                                 + 0.826798 w_k1_DIS_s011
                                 + 0.906917 w_k1_FDX_s011
                                 + 0.765022 w_k1_IBM_s011
                                 + 0.969075 w_k1_INTC_s011
                                 + 0.834563 w_k1_KO_s011 + 0.8006 w_k1_MAT_s011
                                 + 0.950903 w_k1_MSFT_s011
                                 + 0.896139 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.873424 w_k1_ADM_s011
                                 + 0.954548 w_k1_AXP_s011
                                 + 1.09947 w_k1_DIS_s011 + 1.0721 w_k1_FDX_s011
                                 + 1.00493 w_k1_IBM_s011
                                 + 1.08751 w_k1_INTC_s011
                                 + 1.08014 w_k1_KO_s011
                                 + 0.676441 w_k1_MAT_s011
                                 + 1.03857 w_k1_MSFT_s011
                                 + 0.954059 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.04723 w_k1_ADM_s011
                                 + 1.57678 w_k1_AXP_s011
                                 + 1.57094 w_k1_DIS_s011
                                 + 1.29726 w_k1_FDX_s011
                                 + 1.49842 w_k1_IBM_s011
                                 + 1.47673 w_k1_INTC_s011
                                 + 1.38754 w_k1_KO_s011 + 1.43467 w_k1_MAT_s011
                                 + 1.28237 w_k1_MSFT_s011
                                 + 1.13604 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.883708 w_k1_ADM_s011
                                 + 0.909464 w_k1_AXP_s011
                                 + 0.783539 w_k1_DIS_s011
                                 + 0.749418 w_k1_FDX_s011
                                 + 1.25056 w_k1_IBM_s011
                                 + 0.819554 w_k1_INTC_s011
                                 + 1.02104 w_k1_KO_s011 + 1.05307 w_k1_MAT_s011
                                 + 0.836574 w_k1_MSFT_s011
                                 + 0.840984 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.10521 w_k1_ADM_s011
                                 + 1.16852 w_k1_AXP_s011
                                 + 1.07673 w_k1_DIS_s011
                                 + 1.07841 w_k1_FDX_s011
                                 + 1.38406 w_k1_IBM_s011
                                 + 0.8918 w_k1_INTC_s011 + 1.00757 w_k1_KO_s011
                                 + 1.29705 w_k1_MAT_s011
                                 + 0.914688 w_k1_MSFT_s011
                                 + 0.979158 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.08334 w_k1_ADM_s011
                                 + 1.36003 w_k1_AXP_s011
                                 + 0.803621 w_k1_DIS_s011
                                 + 1.08216 w_k1_FDX_s011
                                 + 0.888582 w_k1_IBM_s011
                                 + 1.14142 w_k1_INTC_s011
                                 + 0.870269 w_k1_KO_s011
                                 + 1.11221 w_k1_MAT_s011
                                 + 0.95009 w_k1_MSFT_s011
                                 + 0.853341 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.759536 w_k1_ADM_s011
                                 + 0.642002 w_k1_AXP_s011
                                 + 0.849032 w_k1_DIS_s011
                                 + 0.828027 w_k1_FDX_s011
                                 + 0.965238 w_k1_IBM_s011
                                 + 0.842914 w_k1_INTC_s011
                                 + 1.0987 w_k1_KO_s011 + 0.758063 w_k1_MAT_s011
                                 + 0.906141 w_k1_MSFT_s011
                                 + 0.849443 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.33088 w_k1_ADM_s011
                                 + 1.26245 w_k1_AXP_s011
                                 + 1.42847 w_k1_DIS_s011
                                 + 1.37847 w_k1_FDX_s011
                                 + 1.02045 w_k1_IBM_s011
                                 + 1.16572 w_k1_INTC_s011
                                 + 1.24405 w_k1_KO_s011
                                 + 0.936383 w_k1_MAT_s011
                                 + 1.22804 w_k1_MSFT_s011 + 1.0705 w_k1_PG_s011
                                 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.945751 w_k1_ADM_s012
                                 + 1.01288 w_k1_AXP_s012
                                 + 1.07229 w_k1_DIS_s012
                                 + 0.957873 w_k1_FDX_s012
                                 + 1.2016 w_k1_IBM_s012
                                 + 1.03068 w_k1_INTC_s012
                                 + 1.21376 w_k1_KO_s012
                                 + 0.965832 w_k1_MAT_s012
                                 + 1.07234 w_k1_MSFT_s012
                                 + 1.05716 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.93312 w_k1_ADM_s012
                                 + 0.820939 w_k1_AXP_s012
                                 + 0.764681 w_k1_DIS_s012
                                 + 0.82147 w_k1_FDX_s012
                                 + 0.714007 w_k1_IBM_s012
                                 + 0.78314 w_k1_INTC_s012
                                 + 0.766155 w_k1_KO_s012
                                 + 0.899261 w_k1_MAT_s012
                                 + 0.890906 w_k1_MSFT_s012
                                 + 0.883663 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.846764 w_k1_ADM_s012
                                 + 1.13957 w_k1_AXP_s012
                                 + 0.699161 w_k1_DIS_s012
                                 + 0.89849 w_k1_FDX_s012
                                 + 0.905781 w_k1_IBM_s012
                                 + 0.936716 w_k1_INTC_s012
                                 + 0.888407 w_k1_KO_s012
                                 + 1.22815 w_k1_MAT_s012
                                 + 0.837872 w_k1_MSFT_s012
                                 + 0.966241 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.01929 w_k1_ADM_s012
                                 + 1.04941 w_k1_AXP_s012
                                 + 1.15812 w_k1_DIS_s012 + 1.0987 w_k1_FDX_s012
                                 + 0.794006 w_k1_IBM_s012
                                 + 1.01483 w_k1_INTC_s012
                                 + 1.01195 w_k1_KO_s012
                                 + 0.766853 w_k1_MAT_s012
                                 + 1.17064 w_k1_MSFT_s012
                                 + 0.972705 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.853413 w_k1_ADM_s012
                                 + 1.09432 w_k1_AXP_s012
                                 + 0.752933 w_k1_DIS_s012
                                 + 0.821098 w_k1_FDX_s012
                                 + 1.30165 w_k1_IBM_s012
                                 + 0.802414 w_k1_INTC_s012
                                 + 1.01541 w_k1_KO_s012 + 1.80314 w_k1_MAT_s012
                                 + 0.735446 w_k1_MSFT_s012
                                 + 1.04228 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.10741 w_k1_ADM_s012
                                 + 1.53608 w_k1_AXP_s012 + 1.5498 w_k1_DIS_s012
                                 + 1.24845 w_k1_FDX_s012
                                 + 1.25721 w_k1_IBM_s012
                                 + 1.50561 w_k1_INTC_s012
                                 + 1.42012 w_k1_KO_s012 + 1.41117 w_k1_MAT_s012
                                 + 1.36689 w_k1_MSFT_s012
                                 + 1.33276 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.939101 w_k1_ADM_s012
                                 + 1.31243 w_k1_AXP_s012
                                 + 1.26669 w_k1_DIS_s012
                                 + 1.15883 w_k1_FDX_s012
                                 + 1.01986 w_k1_IBM_s012
                                 + 1.23574 w_k1_INTC_s012
                                 + 1.21762 w_k1_KO_s012
                                 + 0.902995 w_k1_MAT_s012
                                 + 1.21805 w_k1_MSFT_s012
                                 + 1.05816 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.820781 w_k1_ADM_s012
                                 + 0.736192 w_k1_AXP_s012
                                 + 0.512001 w_k1_DIS_s012
                                 + 0.725833 w_k1_FDX_s012
                                 + 0.627482 w_k1_IBM_s012
                                 + 0.751247 w_k1_INTC_s012
                                 + 0.834948 w_k1_KO_s012
                                 + 0.641662 w_k1_MAT_s012
                                 + 0.80667 w_k1_MSFT_s012
                                 + 0.828073 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.17831 w_k1_ADM_s013
                                 + 1.56225 w_k1_AXP_s013
                                 + 1.46048 w_k1_DIS_s013 + 1.3991 w_k1_FDX_s013
                                 + 1.16663 w_k1_IBM_s013
                                 + 1.23773 w_k1_INTC_s013
                                 + 1.23619 w_k1_KO_s013 + 1.51448 w_k1_MAT_s013
                                 + 1.41234 w_k1_MSFT_s013
                                 + 1.15887 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.08407 w_k1_ADM_s013
                                 + 0.963236 w_k1_AXP_s013
                                 + 0.835286 w_k1_DIS_s013
                                 + 0.875674 w_k1_FDX_s013
                                 + 1.40777 w_k1_IBM_s013
                                 + 0.865291 w_k1_INTC_s013
                                 + 1.04865 w_k1_KO_s013 + 1.48382 w_k1_MAT_s013
                                 + 0.776573 w_k1_MSFT_s013
                                 + 0.998694 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.26537 w_k1_ADM_s013
                                 + 1.14438 w_k1_AXP_s013
                                 + 1.45786 w_k1_DIS_s013
                                 + 1.24106 w_k1_FDX_s013
                                 + 0.967658 w_k1_IBM_s013
                                 + 1.21899 w_k1_INTC_s013
                                 + 1.11465 w_k1_KO_s013
                                 + 0.938397 w_k1_MAT_s013
                                 + 1.24544 w_k1_MSFT_s013 + 1.239 w_k1_PG_s013
                                 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.945873 w_k1_ADM_s013
                                 + 0.815909 w_k1_AXP_s013
                                 + 1.02581 w_k1_DIS_s013
                                 + 0.963049 w_k1_FDX_s013
                                 + 1.1842 w_k1_IBM_s013
                                 + 1.07318 w_k1_INTC_s013
                                 + 1.14155 w_k1_KO_s013 + 1.01747 w_k1_MAT_s013
                                 + 1.05575 w_k1_MSFT_s013 + 1.0703 w_k1_PG_s013
                                 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.882571 w_k1_ADM_s013
                                 + 0.693368 w_k1_AXP_s013
                                 + 0.718966 w_k1_DIS_s013
                                 + 0.822582 w_k1_FDX_s013
                                 + 0.925554 w_k1_IBM_s013
                                 + 0.820497 w_k1_INTC_s013
                                 + 0.833959 w_k1_KO_s013
                                 + 1.00431 w_k1_MAT_s013
                                 + 0.859157 w_k1_MSFT_s013
                                 + 0.970267 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.16967 w_k1_ADM_s013
                                 + 1.32438 w_k1_AXP_s013
                                 + 1.29422 w_k1_DIS_s013
                                 + 1.15761 w_k1_FDX_s013
                                 + 1.61703 w_k1_IBM_s013
                                 + 1.25124 w_k1_INTC_s013
                                 + 1.46914 w_k1_KO_s013 + 1.73366 w_k1_MAT_s013
                                 + 1.20385 w_k1_MSFT_s013
                                 + 1.11567 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.163 w_k1_ADM_s013
                                 + 0.93642 w_k1_AXP_s013
                                 + 1.12523 w_k1_DIS_s013
                                 + 1.14154 w_k1_FDX_s013
                                 + 1.04946 w_k1_IBM_s013
                                 + 1.0869 w_k1_INTC_s013 + 1.18573 w_k1_KO_s013
                                 + 1.05556 w_k1_MAT_s013
                                 + 1.09641 w_k1_MSFT_s013
                                 + 1.15006 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.06298 w_k1_ADM_s013
                                 + 0.949756 w_k1_AXP_s013
                                 + 0.783078 w_k1_DIS_s013
                                 + 1.00131 w_k1_FDX_s013
                                 + 0.804413 w_k1_IBM_s013
                                 + 1.05399 w_k1_INTC_s013
                                 + 0.915423 w_k1_KO_s013
                                 + 0.872264 w_k1_MAT_s013
                                 + 1.05528 w_k1_MSFT_s013
                                 + 0.992022 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.865547 w_k1_ADM_s014
                                 + 0.597374 w_k1_AXP_s014
                                 + 0.567378 w_k1_DIS_s014
                                 + 0.737288 w_k1_FDX_s014
                                 + 0.708329 w_k1_IBM_s014
                                 + 0.813144 w_k1_INTC_s014
                                 + 0.681642 w_k1_KO_s014
                                 + 0.562374 w_k1_MAT_s014
                                 + 0.726278 w_k1_MSFT_s014
                                 + 0.756168 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.816684 w_k1_ADM_s014
                                 + 0.873119 w_k1_AXP_s014
                                 + 0.907952 w_k1_DIS_s014
                                 + 0.903157 w_k1_FDX_s014
                                 + 0.696248 w_k1_IBM_s014
                                 + 1.11459 w_k1_INTC_s014 + 0.9246 w_k1_KO_s014
                                 + 0.656784 w_k1_MAT_s014
                                 + 0.893699 w_k1_MSFT_s014
                                 + 0.909907 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.963994 w_k1_ADM_s014
                                 + 1.06792 w_k1_AXP_s014
                                 + 1.13247 w_k1_DIS_s014
                                 + 1.11743 w_k1_FDX_s014
                                 + 1.00354 w_k1_IBM_s014
                                 + 1.17862 w_k1_INTC_s014 + 1.1131 w_k1_KO_s014
                                 + 1.03137 w_k1_MAT_s014
                                 + 1.11827 w_k1_MSFT_s014
                                 + 1.03547 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.917235 w_k1_ADM_s014
                                 + 0.926745 w_k1_AXP_s014
                                 + 0.722524 w_k1_DIS_s014
                                 + 0.81972 w_k1_FDX_s014
                                 + 1.10128 w_k1_IBM_s014
                                 + 0.993481 w_k1_INTC_s014
                                 + 0.958851 w_k1_KO_s014
                                 + 1.04914 w_k1_MAT_s014
                                 + 0.826975 w_k1_MSFT_s014
                                 + 0.970159 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.09741 w_k1_ADM_s014
                                 + 1.69414 w_k1_AXP_s014
                                 + 1.53521 w_k1_DIS_s014
                                 + 1.30069 w_k1_FDX_s014
                                 + 1.39968 w_k1_IBM_s014
                                 + 1.50912 w_k1_INTC_s014
                                 + 1.38862 w_k1_KO_s014 + 1.67036 w_k1_MAT_s014
                                 + 1.19613 w_k1_MSFT_s014
                                 + 1.25581 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.907127 w_k1_ADM_s014
                                 + 1.00853 w_k1_AXP_s014
                                 + 1.02592 w_k1_DIS_s014
                                 + 0.957904 w_k1_FDX_s014
                                 + 1.39637 w_k1_IBM_s014
                                 + 0.940482 w_k1_INTC_s014
                                 + 1.18426 w_k1_KO_s014 + 1.22015 w_k1_MAT_s014
                                 + 0.930874 w_k1_MSFT_s014
                                 + 1.02442 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.3504 w_k1_ADM_s014
                                 + 1.65779 w_k1_AXP_s014
                                 + 2.04027 w_k1_DIS_s014
                                 + 1.60901 w_k1_FDX_s014
                                 + 1.04726 w_k1_IBM_s014
                                 + 1.5177 w_k1_INTC_s014 + 1.44928 w_k1_KO_s014
                                 + 1.23288 w_k1_MAT_s014
                                 + 1.48438 w_k1_MSFT_s014
                                 + 1.43661 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.719605 w_k1_ADM_s014
                                 + 0.82014 w_k1_AXP_s014
                                 + 0.783302 w_k1_DIS_s014
                                 + 0.81411 w_k1_FDX_s014
                                 + 0.910124 w_k1_IBM_s014
                                 + 0.821268 w_k1_INTC_s014
                                 + 0.861255 w_k1_KO_s014
                                 + 1.04143 w_k1_MAT_s014
                                 + 0.859382 w_k1_MSFT_s014
                                 + 0.915212 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.10986 w_k1_ADM_s015
                                 + 1.10955 w_k1_AXP_s015
                                 + 0.918649 w_k1_DIS_s015
                                 + 0.913441 w_k1_FDX_s015
                                 + 1.11871 w_k1_IBM_s015
                                 + 0.849426 w_k1_INTC_s015
                                 + 0.984702 w_k1_KO_s015
                                 + 1.17798 w_k1_MAT_s015
                                 + 0.872497 w_k1_MSFT_s015
                                 + 1.12243 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.1906 w_k1_ADM_s015
                                 + 1.51208 w_k1_AXP_s015 + 1.4246 w_k1_DIS_s015
                                 + 1.30358 w_k1_FDX_s015 + 1.0935 w_k1_IBM_s015
                                 + 1.08631 w_k1_INTC_s015
                                 + 1.19859 w_k1_KO_s015 + 1.2574 w_k1_MAT_s015
                                 + 1.19515 w_k1_MSFT_s015
                                 + 1.29328 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.18769 w_k1_ADM_s015
                                 + 1.32245 w_k1_AXP_s015
                                 + 1.23474 w_k1_DIS_s015
                                 + 1.07812 w_k1_FDX_s015
                                 + 0.979712 w_k1_IBM_s015
                                 + 1.03161 w_k1_INTC_s015
                                 + 0.996388 w_k1_KO_s015
                                 + 1.31435 w_k1_MAT_s015
                                 + 1.0693 w_k1_MSFT_s015 + 1.14611 w_k1_PG_s015
                                 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.831767 w_k1_ADM_s015
                                 + 0.714871 w_k1_AXP_s015
                                 + 0.70674 w_k1_DIS_s015
                                 + 0.793509 w_k1_FDX_s015
                                 + 0.871761 w_k1_IBM_s015
                                 + 0.852104 w_k1_INTC_s015
                                 + 0.859116 w_k1_KO_s015
                                 + 0.53922 w_k1_MAT_s015
                                 + 0.81113 w_k1_MSFT_s015
                                 + 0.865635 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.918684 w_k1_ADM_s015
                                 + 0.885201 w_k1_AXP_s015
                                 + 1.12623 w_k1_DIS_s015
                                 + 1.02159 w_k1_FDX_s015
                                 + 0.880798 w_k1_IBM_s015
                                 + 0.961244 w_k1_INTC_s015
                                 + 0.924655 w_k1_KO_s015
                                 + 0.727511 w_k1_MAT_s015
                                 + 1.12041 w_k1_MSFT_s015
                                 + 1.07516 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.1466 w_k1_ADM_s015
                                 + 1.2338 w_k1_AXP_s015 + 1.57179 w_k1_DIS_s015
                                 + 1.15705 w_k1_FDX_s015
                                 + 1.13973 w_k1_IBM_s015
                                 + 0.945362 w_k1_INTC_s015
                                 + 1.23948 w_k1_KO_s015
                                 + 0.851074 w_k1_MAT_s015
                                 + 1.1999 w_k1_MSFT_s015 + 1.2635 w_k1_PG_s015
                                 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.06165 w_k1_ADM_s015
                                 + 0.833595 w_k1_AXP_s015
                                 + 1.01796 w_k1_DIS_s015
                                 + 0.765994 w_k1_FDX_s015
                                 + 1.4433 w_k1_IBM_s015
                                 + 0.850879 w_k1_INTC_s015
                                 + 1.2525 w_k1_KO_s015 + 1.31553 w_k1_MAT_s015
                                 + 1.01874 w_k1_MSFT_s015
                                 + 1.03407 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.22686 w_k1_ADM_s015
                                 + 0.951394 w_k1_AXP_s015
                                 + 1.01986 w_k1_DIS_s015
                                 + 1.07179 w_k1_FDX_s015
                                 + 0.710894 w_k1_IBM_s015
                                 + 0.832566 w_k1_INTC_s015
                                 + 0.921345 w_k1_KO_s015
                                 + 0.807097 w_k1_MAT_s015
                                 + 0.988846 w_k1_MSFT_s015
                                 + 1.01541 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.640745 w_k1_ADM_s016
                                 + 0.579992 w_k1_AXP_s016
                                 + 0.655404 w_k1_DIS_s016
                                 + 0.710678 w_k1_FDX_s016
                                 + 0.664317 w_k1_IBM_s016
                                 + 0.690646 w_k1_INTC_s016
                                 + 0.639878 w_k1_KO_s016
                                 + 0.550233 w_k1_MAT_s016
                                 + 0.829582 w_k1_MSFT_s016
                                 + 0.731835 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.732504 w_k1_ADM_s016
                                 + 1.06856 w_k1_AXP_s016
                                 + 0.830988 w_k1_DIS_s016
                                 + 0.966489 w_k1_FDX_s016
                                 + 1.48891 w_k1_IBM_s016
                                 + 0.81895 w_k1_INTC_s016
                                 + 1.06657 w_k1_KO_s016 + 1.52532 w_k1_MAT_s016
                                 + 0.792918 w_k1_MSFT_s016
                                 + 0.985151 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.893353 w_k1_ADM_s016
                                 + 1.13994 w_k1_AXP_s016
                                 + 1.01047 w_k1_DIS_s016
                                 + 1.09711 w_k1_FDX_s016
                                 + 0.971862 w_k1_IBM_s016
                                 + 1.16728 w_k1_INTC_s016
                                 + 0.945396 w_k1_KO_s016
                                 + 1.02616 w_k1_MAT_s016
                                 + 0.999458 w_k1_MSFT_s016
                                 + 1.01023 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.901791 w_k1_ADM_s016
                                 + 0.877536 w_k1_AXP_s016
                                 + 0.735151 w_k1_DIS_s016
                                 + 0.872877 w_k1_FDX_s016
                                 + 0.72745 w_k1_IBM_s016
                                 + 0.884269 w_k1_INTC_s016
                                 + 0.731558 w_k1_KO_s016
                                 + 0.706318 w_k1_MAT_s016
                                 + 0.817276 w_k1_MSFT_s016
                                 + 0.866062 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.24403 w_k1_ADM_s016
                                 + 1.79662 w_k1_AXP_s016
                                 + 2.05164 w_k1_DIS_s016 + 1.7289 w_k1_FDX_s016
                                 + 1.40691 w_k1_IBM_s016
                                 + 1.47681 w_k1_INTC_s016
                                 + 1.49852 w_k1_KO_s016 + 1.40513 w_k1_MAT_s016
                                 + 1.4499 w_k1_MSFT_s016 + 1.30351 w_k1_PG_s016
                                 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.937667 w_k1_ADM_s016
                                 + 1.39508 w_k1_AXP_s016
                                 + 1.42898 w_k1_DIS_s016
                                 + 1.24001 w_k1_FDX_s016
                                 + 1.07798 w_k1_IBM_s016
                                 + 1.2586 w_k1_INTC_s016 + 1.02724 w_k1_KO_s016
                                 + 1.15521 w_k1_MAT_s016
                                 + 1.29569 w_k1_MSFT_s016
                                 + 1.19446 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.580592 w_k1_ADM_s016
                                 + 0.779245 w_k1_AXP_s016
                                 + 0.646958 w_k1_DIS_s016
                                 + 0.73408 w_k1_FDX_s016
                                 + 0.984656 w_k1_IBM_s016
                                 + 0.730035 w_k1_INTC_s016
                                 + 0.887092 w_k1_KO_s016
                                 + 1.00487 w_k1_MAT_s016
                                 + 0.760142 w_k1_MSFT_s016
                                 + 0.902982 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.885727 w_k1_ADM_s016
                                 + 1.02832 w_k1_AXP_s016
                                 + 1.05621 w_k1_DIS_s016
                                 + 1.09434 w_k1_FDX_s016
                                 + 0.653982 w_k1_IBM_s016
                                 + 1.01113 w_k1_INTC_s016
                                 + 0.816018 w_k1_KO_s016
                                 + 0.731091 w_k1_MAT_s016
                                 + 1.05974 w_k1_MSFT_s016
                                 + 0.873814 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.793151 w_k1_ADM_s017
                                 + 1.02489 w_k1_AXP_s017
                                 + 0.954864 w_k1_DIS_s017
                                 + 0.874049 w_k1_FDX_s017
                                 + 1.05689 w_k1_IBM_s017
                                 + 0.974169 w_k1_INTC_s017
                                 + 0.9992 w_k1_KO_s017 + 0.958488 w_k1_MAT_s017
                                 + 0.989705 w_k1_MSFT_s017
                                 + 0.946875 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.963248 w_k1_ADM_s017
                                 + 0.918877 w_k1_AXP_s017
                                 + 0.779699 w_k1_DIS_s017
                                 + 0.797429 w_k1_FDX_s017
                                 + 0.90485 w_k1_IBM_s017
                                 + 0.781208 w_k1_INTC_s017
                                 + 0.839637 w_k1_KO_s017
                                 + 0.865001 w_k1_MAT_s017
                                 + 0.859685 w_k1_MSFT_s017
                                 + 0.929734 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.31072 w_k1_ADM_s017
                                 + 1.60941 w_k1_AXP_s017
                                 + 1.44975 w_k1_DIS_s017
                                 + 1.31143 w_k1_FDX_s017 + 1.1254 w_k1_IBM_s017
                                 + 1.16533 w_k1_INTC_s017
                                 + 1.06451 w_k1_KO_s017
                                 + 0.954374 w_k1_MAT_s017
                                 + 1.25738 w_k1_MSFT_s017
                                 + 1.15987 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.41024 w_k1_ADM_s017
                                 + 1.3529 w_k1_AXP_s017 + 1.20718 w_k1_DIS_s017
                                 + 1.07458 w_k1_FDX_s017
                                 + 0.984559 w_k1_IBM_s017
                                 + 1.00108 w_k1_INTC_s017
                                 + 1.08169 w_k1_KO_s017 + 1.3124 w_k1_MAT_s017
                                 + 1.10355 w_k1_MSFT_s017
                                 + 1.11505 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.859101 w_k1_ADM_s017
                                 + 0.967983 w_k1_AXP_s017
                                 + 1.10878 w_k1_DIS_s017
                                 + 0.986985 w_k1_FDX_s017
                                 + 0.746433 w_k1_IBM_s017
                                 + 0.953318 w_k1_INTC_s017
                                 + 0.989588 w_k1_KO_s017
                                 + 0.862789 w_k1_MAT_s017
                                 + 1.07496 w_k1_MSFT_s017
                                 + 1.10619 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.660771 w_k1_ADM_s017
                                 + 0.659356 w_k1_AXP_s017
                                 + 0.653014 w_k1_DIS_s017
                                 + 0.738347 w_k1_FDX_s017
                                 + 0.695669 w_k1_IBM_s017
                                 + 0.782255 w_k1_INTC_s017
                                 + 0.702773 w_k1_KO_s017
                                 + 0.734223 w_k1_MAT_s017
                                 + 0.888452 w_k1_MSFT_s017
                                 + 0.801184 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.15219 w_k1_ADM_s017
                                 + 1.32201 w_k1_AXP_s017
                                 + 1.30741 w_k1_DIS_s017
                                 + 0.950587 w_k1_FDX_s017
                                 + 1.4538 w_k1_IBM_s017
                                 + 0.825576 w_k1_INTC_s017
                                 + 1.14371 w_k1_KO_s017 + 1.35156 w_k1_MAT_s017
                                 + 0.92868 w_k1_MSFT_s017 + 1.0983 w_k1_PG_s017
                                 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.02874 w_k1_ADM_s017
                                 + 0.991622 w_k1_AXP_s017
                                 + 0.692121 w_k1_DIS_s017
                                 + 0.915456 w_k1_FDX_s017
                                 + 0.597449 w_k1_IBM_s017
                                 + 0.784912 w_k1_INTC_s017
                                 + 0.643103 w_k1_KO_s017
                                 + 0.632861 w_k1_MAT_s017
                                 + 0.808899 w_k1_MSFT_s017
                                 + 0.904918 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.09869 w_k1_ADM_s100
                                 + 1.10421 w_k1_AXP_s100
                                 + 1.12179 w_k1_DIS_s100
                                 + 1.12258 w_k1_FDX_s100
                                 + 0.82085 w_k1_IBM_s100
                                 + 1.22265 w_k1_INTC_s100
                                 + 1.07199 w_k1_KO_s100
                                 + 0.865421 w_k1_MAT_s100
                                 + 1.26466 w_k1_MSFT_s100
                                 + 1.18756 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.876155 w_k1_ADM_s100
                                 + 0.743628 w_k1_AXP_s100
                                 + 0.810763 w_k1_DIS_s100
                                 + 0.902444 w_k1_FDX_s100
                                 + 0.77566 w_k1_IBM_s100
                                 + 0.902415 w_k1_INTC_s100
                                 + 0.887182 w_k1_KO_s100
                                 + 0.787806 w_k1_MAT_s100
                                 + 0.919824 w_k1_MSFT_s100
                                 + 0.911087 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.47422 w_k1_ADM_s100
                                 + 1.5892 w_k1_AXP_s100 + 1.57424 w_k1_DIS_s100
                                 + 1.33349 w_k1_FDX_s100
                                 + 1.16397 w_k1_IBM_s100
                                 + 1.34716 w_k1_INTC_s100 + 1.3218 w_k1_KO_s100
                                 + 1.0812 w_k1_MAT_s100 + 1.2547 w_k1_MSFT_s100
                                 + 1.30472 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.951017 w_k1_ADM_s100
                                 + 1.35897 w_k1_AXP_s100
                                 + 1.07345 w_k1_DIS_s100
                                 + 1.09591 w_k1_FDX_s100
                                 + 1.30357 w_k1_IBM_s100
                                 + 1.08392 w_k1_INTC_s100
                                 + 1.01856 w_k1_KO_s100 + 1.5628 w_k1_MAT_s100
                                 + 1.04169 w_k1_MSFT_s100
                                 + 1.04736 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.12244 w_k1_ADM_s100
                                 + 1.03823 w_k1_AXP_s100
                                 + 0.915379 w_k1_DIS_s100
                                 + 0.973978 w_k1_FDX_s100
                                 + 1.07426 w_k1_IBM_s100
                                 + 1.01495 w_k1_INTC_s100
                                 + 1.05936 w_k1_KO_s100 + 1.21978 w_k1_MAT_s100
                                 + 0.980018 w_k1_MSFT_s100
                                 + 1.01069 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.856748 w_k1_ADM_s100
                                 + 0.676175 w_k1_AXP_s100
                                 + 0.730319 w_k1_DIS_s100
                                 + 0.756737 w_k1_FDX_s100
                                 + 0.857028 w_k1_IBM_s100
                                 + 0.874884 w_k1_INTC_s100
                                 + 0.82583 w_k1_KO_s100
                                 + 0.719679 w_k1_MAT_s100
                                 + 0.799784 w_k1_MSFT_s100
                                 + 0.775191 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.20693 w_k1_ADM_s100
                                 + 1.13727 w_k1_AXP_s100 + 1.1941 w_k1_DIS_s100
                                 + 1.27801 w_k1_FDX_s100
                                 + 0.849607 w_k1_IBM_s100
                                 + 1.12261 w_k1_INTC_s100
                                 + 1.12192 w_k1_KO_s100 + 1.68183 w_k1_MAT_s100
                                 + 1.28388 w_k1_MSFT_s100
                                 + 1.13771 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.21832 w_k1_ADM_s100
                                 + 1.07696 w_k1_AXP_s100 + 1.1399 w_k1_DIS_s100
                                 + 1.06896 w_k1_FDX_s100
                                 + 1.46319 w_k1_IBM_s100
                                 + 1.14082 w_k1_INTC_s100
                                 + 1.36628 w_k1_KO_s100 + 1.40376 w_k1_MAT_s100
                                 + 1.16279 w_k1_MSFT_s100
                                 + 1.10373 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.705059 w_k1_ADM_s101
                                 + 1.04222 w_k1_AXP_s101
                                 + 1.02594 w_k1_DIS_s101
                                 + 0.94825 w_k1_FDX_s101
                                 + 1.55315 w_k1_IBM_s101
                                 + 1.10154 w_k1_INTC_s101
                                 + 1.06424 w_k1_KO_s101 + 1.67648 w_k1_MAT_s101
                                 + 0.960723 w_k1_MSFT_s101
                                 + 1.00331 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.17336 w_k1_ADM_s101
                                 + 1.36907 w_k1_AXP_s101
                                 + 1.27215 w_k1_DIS_s101
                                 + 1.14803 w_k1_FDX_s101
                                 + 1.29452 w_k1_IBM_s101
                                 + 0.931839 w_k1_INTC_s101
                                 + 1.05597 w_k1_KO_s101 + 1.70654 w_k1_MAT_s101
                                 + 0.950032 w_k1_MSFT_s101
                                 + 1.06079 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.08049 w_k1_ADM_s101
                                 + 1.96321 w_k1_AXP_s101
                                 + 1.81028 w_k1_DIS_s101
                                 + 1.26131 w_k1_FDX_s101
                                 + 2.42372 w_k1_IBM_s101
                                 + 1.37546 w_k1_INTC_s101 + 1.4456 w_k1_KO_s101
                                 + 3.1801 w_k1_MAT_s101
                                 + 1.07515 w_k1_MSFT_s101
                                 + 1.24886 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.051 w_k1_ADM_s101
                                 + 1.16414 w_k1_AXP_s101 + 1.3505 w_k1_DIS_s101
                                 + 1.21938 w_k1_FDX_s101
                                 + 0.902844 w_k1_IBM_s101
                                 + 1.04553 w_k1_INTC_s101
                                 + 1.07587 w_k1_KO_s101 + 1.21207 w_k1_MAT_s101
                                 + 1.10846 w_k1_MSFT_s101 + 1.1518 w_k1_PG_s101
                                 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.91242 w_k1_ADM_s101
                                 + 1.03256 w_k1_AXP_s101
                                 + 0.739491 w_k1_DIS_s101
                                 + 0.922333 w_k1_FDX_s101
                                 + 0.963235 w_k1_IBM_s101
                                 + 0.734298 w_k1_INTC_s101
                                 + 0.722952 w_k1_KO_s101
                                 + 1.30705 w_k1_MAT_s101
                                 + 0.832048 w_k1_MSFT_s101
                                 + 0.872911 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.733892 w_k1_ADM_s101
                                 + 0.704123 w_k1_AXP_s101
                                 + 0.707564 w_k1_DIS_s101 + 0.843 w_k1_FDX_s101
                                 + 0.760056 w_k1_IBM_s101
                                 + 0.968906 w_k1_INTC_s101
                                 + 0.755241 w_k1_KO_s101
                                 + 0.748938 w_k1_MAT_s101
                                 + 0.914531 w_k1_MSFT_s101
                                 + 0.816657 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.00598 w_k1_ADM_s101
                                 + 0.972867 w_k1_AXP_s101
                                 + 1.14858 w_k1_DIS_s101
                                 + 1.01418 w_k1_FDX_s101
                                 + 1.10836 w_k1_IBM_s101
                                 + 1.04817 w_k1_INTC_s101
                                 + 0.934821 w_k1_KO_s101
                                 + 0.757323 w_k1_MAT_s101
                                 + 1.07302 w_k1_MSFT_s101
                                 + 1.01013 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.866162 w_k1_ADM_s101
                                 + 1.58452 w_k1_AXP_s101 + 1.5757 w_k1_DIS_s101
                                 + 1.36101 w_k1_FDX_s101
                                 + 1.49108 w_k1_IBM_s101
                                 + 1.33016 w_k1_INTC_s101
                                 + 1.13651 w_k1_KO_s101 + 1.67556 w_k1_MAT_s101
                                 + 1.30957 w_k1_MSFT_s101
                                 + 1.18402 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.355 w_k1_ADM_s102
                                 + 0.944721 w_k1_AXP_s102
                                 + 1.37607 w_k1_DIS_s102
                                 + 1.23791 w_k1_FDX_s102
                                 + 0.961227 w_k1_IBM_s102
                                 + 0.932747 w_k1_INTC_s102
                                 + 1.06771 w_k1_KO_s102
                                 + 0.847094 w_k1_MAT_s102
                                 + 1.21016 w_k1_MSFT_s102
                                 + 1.13042 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.10997 w_k1_ADM_s102
                                 + 1.56326 w_k1_AXP_s102
                                 + 1.63846 w_k1_DIS_s102
                                 + 1.40875 w_k1_FDX_s102
                                 + 1.32235 w_k1_IBM_s102
                                 + 1.20117 w_k1_INTC_s102
                                 + 1.16455 w_k1_KO_s102 + 2.01963 w_k1_MAT_s102
                                 + 1.30353 w_k1_MSFT_s102
                                 + 1.29877 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.07849 w_k1_ADM_s102
                                 + 1.26174 w_k1_AXP_s102 + 1.0381 w_k1_DIS_s102
                                 + 1.03828 w_k1_FDX_s102
                                 + 1.31284 w_k1_IBM_s102
                                 + 0.959184 w_k1_INTC_s102
                                 + 1.1162 w_k1_KO_s102 + 1.32172 w_k1_MAT_s102
                                 + 0.920627 w_k1_MSFT_s102
                                 + 1.29762 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.3308 w_k1_ADM_s102
                                 + 0.931244 w_k1_AXP_s102
                                 + 1.05779 w_k1_DIS_s102
                                 + 0.974847 w_k1_FDX_s102
                                 + 1.32145 w_k1_IBM_s102
                                 + 0.965573 w_k1_INTC_s102
                                 + 1.10961 w_k1_KO_s102 + 1.06463 w_k1_MAT_s102
                                 + 1.06554 w_k1_MSFT_s102
                                 + 1.10544 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.916848 w_k1_ADM_s102
                                 + 0.80474 w_k1_AXP_s102
                                 + 0.891194 w_k1_DIS_s102
                                 + 0.891726 w_k1_FDX_s102
                                 + 0.932309 w_k1_IBM_s102
                                 + 0.947368 w_k1_INTC_s102
                                 + 0.863738 w_k1_KO_s102
                                 + 0.901222 w_k1_MAT_s102
                                 + 0.974845 w_k1_MSFT_s102
                                 + 0.935751 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.14567 w_k1_ADM_s102
                                 + 1.01501 w_k1_AXP_s102
                                 + 1.48339 w_k1_DIS_s102 + 1.0651 w_k1_FDX_s102
                                 + 1.51554 w_k1_IBM_s102
                                 + 1.04187 w_k1_INTC_s102
                                 + 1.37143 w_k1_KO_s102 + 1.12914 w_k1_MAT_s102
                                 + 1.12178 w_k1_MSFT_s102
                                 + 1.13999 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.0989 w_k1_ADM_s102
                                 + 1.01123 w_k1_AXP_s102
                                 + 0.944587 w_k1_DIS_s102
                                 + 1.03645 w_k1_FDX_s102
                                 + 0.848344 w_k1_IBM_s102
                                 + 1.03151 w_k1_INTC_s102
                                 + 0.946021 w_k1_KO_s102
                                 + 0.902961 w_k1_MAT_s102
                                 + 1.03652 w_k1_MSFT_s102
                                 + 0.987331 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.721945 w_k1_ADM_s102
                                 + 0.765064 w_k1_AXP_s102
                                 + 0.647713 w_k1_DIS_s102
                                 + 0.769862 w_k1_FDX_s102
                                 + 0.999882 w_k1_IBM_s102
                                 + 0.901486 w_k1_INTC_s102
                                 + 0.840443 w_k1_KO_s102
                                 + 1.10016 w_k1_MAT_s102
                                 + 0.817263 w_k1_MSFT_s102
                                 + 0.961614 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.953637 w_k1_ADM_s103
                                 + 0.976657 w_k1_AXP_s103
                                 + 0.883643 w_k1_DIS_s103
                                 + 1.01435 w_k1_FDX_s103
                                 + 0.722835 w_k1_IBM_s103
                                 + 1.15043 w_k1_INTC_s103
                                 + 0.978552 w_k1_KO_s103
                                 + 1.09755 w_k1_MAT_s103
                                 + 1.04767 w_k1_MSFT_s103
                                 + 0.940527 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.975886 w_k1_ADM_s103
                                 + 0.914894 w_k1_AXP_s103
                                 + 1.09425 w_k1_DIS_s103
                                 + 1.02642 w_k1_FDX_s103
                                 + 1.01166 w_k1_IBM_s103
                                 + 1.22887 w_k1_INTC_s103 + 1.1436 w_k1_KO_s103
                                 + 0.950831 w_k1_MAT_s103
                                 + 1.21785 w_k1_MSFT_s103
                                 + 1.01157 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.797257 w_k1_ADM_s103
                                 + 0.926543 w_k1_AXP_s103
                                 + 0.758226 w_k1_DIS_s103
                                 + 0.813663 w_k1_FDX_s103
                                 + 1.27756 w_k1_IBM_s103
                                 + 0.959424 w_k1_INTC_s103
                                 + 0.96436 w_k1_KO_s103 + 1.39599 w_k1_MAT_s103
                                 + 0.976349 w_k1_MSFT_s103
                                 + 0.875343 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.13281 w_k1_ADM_s103
                                 + 1.68425 w_k1_AXP_s103
                                 + 1.69423 w_k1_DIS_s103
                                 + 1.32645 w_k1_FDX_s103
                                 + 1.80021 w_k1_IBM_s103
                                 + 1.58674 w_k1_INTC_s103
                                 + 1.72624 w_k1_KO_s103 + 2.44114 w_k1_MAT_s103
                                 + 1.43588 w_k1_MSFT_s103
                                 + 1.14952 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.92428 w_k1_ADM_s103
                                 + 0.990819 w_k1_AXP_s103
                                 + 1.32106 w_k1_DIS_s103
                                 + 1.03609 w_k1_FDX_s103
                                 + 1.37788 w_k1_IBM_s103
                                 + 1.28237 w_k1_INTC_s103
                                 + 1.28245 w_k1_KO_s103 + 1.26883 w_k1_MAT_s103
                                 + 1.20754 w_k1_MSFT_s103
                                 + 1.04822 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.784014 w_k1_ADM_s103
                                 + 0.699825 w_k1_AXP_s103
                                 + 0.591509 w_k1_DIS_s103
                                 + 0.737288 w_k1_FDX_s103
                                 + 0.677329 w_k1_IBM_s103
                                 + 0.859235 w_k1_INTC_s103
                                 + 0.653791 w_k1_KO_s103
                                 + 0.861673 w_k1_MAT_s103
                                 + 0.651165 w_k1_MSFT_s103
                                 + 0.710946 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.13838 w_k1_ADM_s103
                                 + 1.41721 w_k1_AXP_s103
                                 + 1.65918 w_k1_DIS_s103
                                 + 1.33987 w_k1_FDX_s103
                                 + 1.07645 w_k1_IBM_s103
                                 + 1.68233 w_k1_INTC_s103
                                 + 1.53006 w_k1_KO_s103 + 1.31566 w_k1_MAT_s103
                                 + 1.39152 w_k1_MSFT_s103
                                 + 1.24988 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.08342 w_k1_ADM_s103
                                 + 1.54021 w_k1_AXP_s103 + 1.1362 w_k1_DIS_s103
                                 + 1.16911 w_k1_FDX_s103
                                 + 1.19432 w_k1_IBM_s103
                                 + 1.44383 w_k1_INTC_s103 + 1.2078 w_k1_KO_s103
                                 + 1.91708 w_k1_MAT_s103
                                 + 1.11624 w_k1_MSFT_s103
                                 + 0.986644 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.878956 w_k1_ADM_s104
                                 + 0.563704 w_k1_AXP_s104
                                 + 0.863722 w_k1_DIS_s104
                                 + 0.843996 w_k1_FDX_s104
                                 + 0.901484 w_k1_IBM_s104
                                 + 0.981635 w_k1_INTC_s104
                                 + 1.03109 w_k1_KO_s104 + 0.93079 w_k1_MAT_s104
                                 + 1.08427 w_k1_MSFT_s104
                                 + 0.920678 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.39864 w_k1_ADM_s104
                                 + 1.26854 w_k1_AXP_s104
                                 + 1.71278 w_k1_DIS_s104 + 1.3469 w_k1_FDX_s104
                                 + 1.36054 w_k1_IBM_s104
                                 + 1.19351 w_k1_INTC_s104
                                 + 1.29552 w_k1_KO_s104 + 1.21084 w_k1_MAT_s104
                                 + 1.09013 w_k1_MSFT_s104
                                 + 1.13714 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.788514 w_k1_ADM_s104
                                 + 0.867539 w_k1_AXP_s104
                                 + 0.812841 w_k1_DIS_s104
                                 + 0.833183 w_k1_FDX_s104 + 1.541 w_k1_IBM_s104
                                 + 1.07172 w_k1_INTC_s104
                                 + 1.18806 w_k1_KO_s104 + 1.35703 w_k1_MAT_s104
                                 + 0.988053 w_k1_MSFT_s104
                                 + 0.961777 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.766842 w_k1_ADM_s104
                                 + 0.590114 w_k1_AXP_s104
                                 + 0.5664 w_k1_DIS_s104
                                 + 0.662788 w_k1_FDX_s104
                                 + 0.843247 w_k1_IBM_s104
                                 + 0.741993 w_k1_INTC_s104
                                 + 0.704643 w_k1_KO_s104
                                 + 0.709149 w_k1_MAT_s104
                                 + 0.75594 w_k1_MSFT_s104
                                 + 0.814786 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.21703 w_k1_ADM_s104
                                 + 0.85267 w_k1_AXP_s104
                                 + 0.855898 w_k1_DIS_s104
                                 + 0.917163 w_k1_FDX_s104
                                 + 0.89889 w_k1_IBM_s104
                                 + 1.05659 w_k1_INTC_s104
                                 + 0.862751 w_k1_KO_s104
                                 + 0.806453 w_k1_MAT_s104
                                 + 0.956864 w_k1_MSFT_s104
                                 + 0.891376 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.885903 w_k1_ADM_s104
                                 + 0.554005 w_k1_AXP_s104
                                 + 0.675153 w_k1_DIS_s104
                                 + 0.776185 w_k1_FDX_s104
                                 + 0.658306 w_k1_IBM_s104
                                 + 0.93788 w_k1_INTC_s104
                                 + 0.716294 w_k1_KO_s104
                                 + 0.530403 w_k1_MAT_s104
                                 + 0.880811 w_k1_MSFT_s104
                                 + 0.828015 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.12033 w_k1_ADM_s104
                                 + 0.807789 w_k1_AXP_s104
                                 + 1.18774 w_k1_DIS_s104
                                 + 1.07481 w_k1_FDX_s104
                                 + 0.868057 w_k1_IBM_s104
                                 + 1.36818 w_k1_INTC_s104
                                 + 1.09056 w_k1_KO_s104
                                 + 0.803943 w_k1_MAT_s104
                                 + 1.26237 w_k1_MSFT_s104
                                 + 1.01652 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.981417 w_k1_ADM_s104
                                 + 1.0412 w_k1_AXP_s104 + 1.26536 w_k1_DIS_s104
                                 + 1.03571 w_k1_FDX_s104
                                 + 1.26781 w_k1_IBM_s104
                                 + 1.06259 w_k1_INTC_s104
                                 + 1.09976 w_k1_KO_s104 + 1.05004 w_k1_MAT_s104
                                 + 0.950287 w_k1_MSFT_s104
                                 + 1.03897 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.04474 w_k1_ADM_s105
                                 + 0.750912 w_k1_AXP_s105
                                 + 0.798708 w_k1_DIS_s105
                                 + 0.859139 w_k1_FDX_s105
                                 + 0.739288 w_k1_IBM_s105
                                 + 0.880478 w_k1_INTC_s105
                                 + 0.81895 w_k1_KO_s105
                                 + 0.839998 w_k1_MAT_s105
                                 + 0.920764 w_k1_MSFT_s105
                                 + 0.96228 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.14604 w_k1_ADM_s105
                                 + 0.971383 w_k1_AXP_s105
                                 + 1.22427 w_k1_DIS_s105
                                 + 1.01637 w_k1_FDX_s105
                                 + 1.25615 w_k1_IBM_s105
                                 + 1.04963 w_k1_INTC_s105
                                 + 1.24919 w_k1_KO_s105 + 1.07424 w_k1_MAT_s105
                                 + 1.07982 w_k1_MSFT_s105 + 1.074 w_k1_PG_s105
                                 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.927535 w_k1_ADM_s105
                                 + 0.911232 w_k1_AXP_s105
                                 + 1.01111 w_k1_DIS_s105
                                 + 1.02181 w_k1_FDX_s105
                                 + 0.744594 w_k1_IBM_s105
                                 + 1.04179 w_k1_INTC_s105
                                 + 0.892873 w_k1_KO_s105
                                 + 0.800126 w_k1_MAT_s105
                                 + 1.09458 w_k1_MSFT_s105
                                 + 0.864964 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.03187 w_k1_ADM_s105
                                 + 1.25877 w_k1_AXP_s105
                                 + 0.933243 w_k1_DIS_s105
                                 + 1.0443 w_k1_FDX_s105
                                 + 0.917138 w_k1_IBM_s105
                                 + 0.98579 w_k1_INTC_s105
                                 + 0.953244 w_k1_KO_s105
                                 + 1.09437 w_k1_MAT_s105
                                 + 0.901654 w_k1_MSFT_s105
                                 + 0.986967 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.54647 w_k1_ADM_s105
                                 + 1.17611 w_k1_AXP_s105
                                 + 1.56826 w_k1_DIS_s105
                                 + 1.18833 w_k1_FDX_s105
                                 + 1.30315 w_k1_IBM_s105
                                 + 0.940394 w_k1_INTC_s105
                                 + 1.34002 w_k1_KO_s105 + 1.3119 w_k1_MAT_s105
                                 + 1.1801 w_k1_MSFT_s105 + 1.15452 w_k1_PG_s105
                                 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.915516 w_k1_ADM_s105
                                 + 0.906503 w_k1_AXP_s105
                                 + 0.624996 w_k1_DIS_s105
                                 + 0.810048 w_k1_FDX_s105
                                 + 0.97441 w_k1_IBM_s105
                                 + 0.79734 w_k1_INTC_s105
                                 + 0.854488 w_k1_KO_s105 + 1.1171 w_k1_MAT_s105
                                 + 0.784141 w_k1_MSFT_s105
                                 + 0.921943 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.10727 w_k1_ADM_s105
                                 + 1.00331 w_k1_AXP_s105
                                 + 1.14477 w_k1_DIS_s105
                                 + 1.03316 w_k1_FDX_s105
                                 + 0.901805 w_k1_IBM_s105
                                 + 1.09008 w_k1_INTC_s105
                                 + 0.901544 w_k1_KO_s105
                                 + 1.08491 w_k1_MAT_s105
                                 + 1.07831 w_k1_MSFT_s105
                                 + 1.11045 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.47766 w_k1_ADM_s105
                                 + 1.27891 w_k1_AXP_s105
                                 + 1.18043 w_k1_DIS_s105
                                 + 1.16276 w_k1_FDX_s105
                                 + 1.14392 w_k1_IBM_s105
                                 + 0.955644 w_k1_INTC_s105
                                 + 1.1567 w_k1_KO_s105 + 1.95523 w_k1_MAT_s105
                                 + 1.14213 w_k1_MSFT_s105
                                 + 1.20282 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.700101 w_k1_ADM_s106
                                 + 0.641094 w_k1_AXP_s106
                                 + 0.61157 w_k1_DIS_s106
                                 + 0.67894 w_k1_FDX_s106
                                 + 0.62642 w_k1_IBM_s106
                                 + 0.754216 w_k1_INTC_s106
                                 + 0.689053 w_k1_KO_s106
                                 + 0.558327 w_k1_MAT_s106
                                 + 0.79838 w_k1_MSFT_s106
                                 + 0.825446 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.45036 w_k1_ADM_s106
                                 + 1.43758 w_k1_AXP_s106
                                 + 1.56696 w_k1_DIS_s106
                                 + 1.38704 w_k1_FDX_s106
                                 + 0.969084 w_k1_IBM_s106
                                 + 1.25676 w_k1_INTC_s106
                                 + 1.07561 w_k1_KO_s106 + 1.22698 w_k1_MAT_s106
                                 + 1.29278 w_k1_MSFT_s106
                                 + 1.16208 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.14305 w_k1_ADM_s106
                                 + 1.17813 w_k1_AXP_s106
                                 + 0.98504 w_k1_DIS_s106
                                 + 1.03732 w_k1_FDX_s106
                                 + 1.10679 w_k1_IBM_s106 + 1.024 w_k1_INTC_s106
                                 + 1.02387 w_k1_KO_s106 + 1.3426 w_k1_MAT_s106
                                 + 1.04176 w_k1_MSFT_s106
                                 + 1.10379 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.91587 w_k1_ADM_s106
                                 + 1.10799 w_k1_AXP_s106
                                 + 1.53355 w_k1_DIS_s106
                                 + 1.20499 w_k1_FDX_s106
                                 + 1.35901 w_k1_IBM_s106
                                 + 1.25345 w_k1_INTC_s106
                                 + 1.40537 w_k1_KO_s106 + 1.55031 w_k1_MAT_s106
                                 + 1.21054 w_k1_MSFT_s106
                                 + 1.31718 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.888413 w_k1_ADM_s106
                                 + 0.921019 w_k1_AXP_s106
                                 + 1.0534 w_k1_DIS_s106
                                 + 0.971226 w_k1_FDX_s106
                                 + 1.08624 w_k1_IBM_s106
                                 + 0.882231 w_k1_INTC_s106
                                 + 0.921472 w_k1_KO_s106
                                 + 1.21547 w_k1_MAT_s106
                                 + 0.940782 w_k1_MSFT_s106
                                 + 0.991673 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.90137 w_k1_ADM_s106
                                 + 0.746798 w_k1_AXP_s106
                                 + 0.743526 w_k1_DIS_s106
                                 + 0.884449 w_k1_FDX_s106
                                 + 0.70918 w_k1_IBM_s106
                                 + 0.911126 w_k1_INTC_s106
                                 + 0.798906 w_k1_KO_s106
                                 + 0.85351 w_k1_MAT_s106
                                 + 0.965238 w_k1_MSFT_s106
                                 + 0.974582 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.24242 w_k1_ADM_s106
                                 + 0.915845 w_k1_AXP_s106
                                 + 1.34807 w_k1_DIS_s106
                                 + 1.19537 w_k1_FDX_s106
                                 + 0.79981 w_k1_IBM_s106
                                 + 0.977316 w_k1_INTC_s106
                                 + 0.94909 w_k1_KO_s106
                                 + 0.882537 w_k1_MAT_s106
                                 + 1.13754 w_k1_MSFT_s106
                                 + 1.07319 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.685179 w_k1_ADM_s106
                                 + 0.856877 w_k1_AXP_s106
                                 + 0.631662 w_k1_DIS_s106
                                 + 0.751443 w_k1_FDX_s106
                                 + 1.16978 w_k1_IBM_s106
                                 + 0.789053 w_k1_INTC_s106
                                 + 0.850056 w_k1_KO_s106
                                 + 0.858258 w_k1_MAT_s106
                                 + 0.749208 w_k1_MSFT_s106
                                 + 0.925763 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.00786 w_k1_ADM_s107
                                 + 0.780153 w_k1_AXP_s107
                                 + 0.876437 w_k1_DIS_s107
                                 + 0.96985 w_k1_FDX_s107
                                 + 0.625394 w_k1_IBM_s107
                                 + 0.834865 w_k1_INTC_s107
                                 + 0.769379 w_k1_KO_s107
                                 + 0.536986 w_k1_MAT_s107
                                 + 0.878538 w_k1_MSFT_s107
                                 + 0.957219 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.27577 w_k1_ADM_s107
                                 + 1.36527 w_k1_AXP_s107
                                 + 1.69757 w_k1_DIS_s107
                                 + 1.30228 w_k1_FDX_s107
                                 + 1.56601 w_k1_IBM_s107
                                 + 1.23219 w_k1_INTC_s107
                                 + 1.40878 w_k1_KO_s107 + 1.48583 w_k1_MAT_s107
                                 + 1.3694 w_k1_MSFT_s107 + 1.12746 w_k1_PG_s107
                                 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.883611 w_k1_ADM_s107
                                 + 0.774603 w_k1_AXP_s107
                                 + 0.77898 w_k1_DIS_s107
                                 + 0.817145 w_k1_FDX_s107
                                 + 0.767154 w_k1_IBM_s107
                                 + 0.857961 w_k1_INTC_s107
                                 + 0.739443 w_k1_KO_s107
                                 + 0.643719 w_k1_MAT_s107
                                 + 0.759893 w_k1_MSFT_s107
                                 + 0.884773 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.754151 w_k1_ADM_s107
                                 + 1.09518 w_k1_AXP_s107
                                 + 0.931617 w_k1_DIS_s107
                                 + 0.9762 w_k1_FDX_s107
                                 + 0.774251 w_k1_IBM_s107
                                 + 0.953257 w_k1_INTC_s107
                                 + 0.712989 w_k1_KO_s107
                                 + 0.925609 w_k1_MAT_s107
                                 + 0.930053 w_k1_MSFT_s107
                                 + 0.96794 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.913965 w_k1_ADM_s107
                                 + 0.939502 w_k1_AXP_s107
                                 + 1.17455 w_k1_DIS_s107 + 1.0446 w_k1_FDX_s107
                                 + 0.78245 w_k1_IBM_s107
                                 + 0.945297 w_k1_INTC_s107
                                 + 0.881775 w_k1_KO_s107
                                 + 0.71313 w_k1_MAT_s107
                                 + 0.988874 w_k1_MSFT_s107
                                 + 1.11304 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.609648 w_k1_ADM_s107
                                 + 0.691525 w_k1_AXP_s107
                                 + 0.830247 w_k1_DIS_s107
                                 + 0.75388 w_k1_FDX_s107
                                 + 1.06827 w_k1_IBM_s107
                                 + 0.85558 w_k1_INTC_s107
                                 + 0.734164 w_k1_KO_s107
                                 + 0.988148 w_k1_MAT_s107
                                 + 0.801787 w_k1_MSFT_s107
                                 + 0.916067 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.874306 w_k1_ADM_s107
                                 + 1.14605 w_k1_AXP_s107
                                 + 1.12435 w_k1_DIS_s107
                                 + 0.92897 w_k1_FDX_s107
                                 + 1.32559 w_k1_IBM_s107
                                 + 1.13182 w_k1_INTC_s107
                                 + 1.01417 w_k1_KO_s107 + 1.23296 w_k1_MAT_s107
                                 + 0.929562 w_k1_MSFT_s107
                                 + 1.10085 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.659179 w_k1_ADM_s107
                                 + 0.686718 w_k1_AXP_s107
                                 + 0.616178 w_k1_DIS_s107
                                 + 0.701813 w_k1_FDX_s107
                                 + 0.674636 w_k1_IBM_s107
                                 + 0.792881 w_k1_INTC_s107
                                 + 0.609468 w_k1_KO_s107
                                 + 0.61752 w_k1_MAT_s107
                                 + 0.642095 w_k1_MSFT_s107
                                 + 0.772393 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.747477 w_k1_ADM_s110
                                 + 0.767013 w_k1_AXP_s110
                                 + 0.850401 w_k1_DIS_s110
                                 + 0.971072 w_k1_FDX_s110
                                 + 0.707611 w_k1_IBM_s110
                                 + 0.911075 w_k1_INTC_s110
                                 + 0.833251 w_k1_KO_s110
                                 + 0.663663 w_k1_MAT_s110
                                 + 0.878631 w_k1_MSFT_s110
                                 + 0.86667 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.02585 w_k1_ADM_s110
                                 + 1.00786 w_k1_AXP_s110
                                 + 0.984213 w_k1_DIS_s110
                                 + 0.938729 w_k1_FDX_s110
                                 + 1.44865 w_k1_IBM_s110
                                 + 0.75453 w_k1_INTC_s110
                                 + 1.10093 w_k1_KO_s110 + 1.80317 w_k1_MAT_s110
                                 + 0.827714 w_k1_MSFT_s110
                                 + 0.956066 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.980544 w_k1_ADM_s110
                                 + 0.845652 w_k1_AXP_s110
                                 + 1.12917 w_k1_DIS_s110
                                 + 1.07323 w_k1_FDX_s110
                                 + 1.07058 w_k1_IBM_s110
                                 + 1.10958 w_k1_INTC_s110
                                 + 1.08156 w_k1_KO_s110 + 0.78078 w_k1_MAT_s110
                                 + 1.04513 w_k1_MSFT_s110
                                 + 0.97861 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.24529 w_k1_ADM_s110
                                 + 1.29609 w_k1_AXP_s110 + 1.1533 w_k1_DIS_s110
                                 + 1.31233 w_k1_FDX_s110
                                 + 1.01293 w_k1_IBM_s110
                                 + 1.09804 w_k1_INTC_s110
                                 + 0.940054 w_k1_KO_s110
                                 + 1.02057 w_k1_MAT_s110
                                 + 1.04544 w_k1_MSFT_s110
                                 + 1.05566 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.760798 w_k1_ADM_s110
                                 + 0.824272 w_k1_AXP_s110
                                 + 0.738536 w_k1_DIS_s110
                                 + 0.878591 w_k1_FDX_s110
                                 + 0.970338 w_k1_IBM_s110
                                 + 0.814085 w_k1_INTC_s110
                                 + 0.798953 w_k1_KO_s110
                                 + 0.956952 w_k1_MAT_s110
                                 + 0.759772 w_k1_MSFT_s110
                                 + 0.924248 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.18891 w_k1_ADM_s110
                                 + 1.38596 w_k1_AXP_s110
                                 + 1.45191 w_k1_DIS_s110
                                 + 1.39551 w_k1_FDX_s110
                                 + 1.09236 w_k1_IBM_s110
                                 + 1.3729 w_k1_INTC_s110 + 1.3978 w_k1_KO_s110
                                 + 1.17964 w_k1_MAT_s110
                                 + 1.40992 w_k1_MSFT_s110
                                 + 1.16229 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.33322 w_k1_ADM_s110
                                 + 1.46813 w_k1_AXP_s110
                                 + 1.44917 w_k1_DIS_s110
                                 + 1.30685 w_k1_FDX_s110
                                 + 1.54479 w_k1_IBM_s110
                                 + 1.2685 w_k1_INTC_s110 + 1.38997 w_k1_KO_s110
                                 + 1.67716 w_k1_MAT_s110
                                 + 1.28442 w_k1_MSFT_s110
                                 + 1.18727 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.0855 w_k1_ADM_s110
                                 + 1.09063 w_k1_AXP_s110
                                 + 0.899446 w_k1_DIS_s110
                                 + 1.09179 w_k1_FDX_s110
                                 + 0.939766 w_k1_IBM_s110
                                 + 1.08065 w_k1_INTC_s110
                                 + 1.00061 w_k1_KO_s110
                                 + 0.845737 w_k1_MAT_s110
                                 + 0.958801 w_k1_MSFT_s110
                                 + 0.990982 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.933743 w_k1_ADM_s111
                                 + 1.09223 w_k1_AXP_s111
                                 + 0.906879 w_k1_DIS_s111
                                 + 1.01579 w_k1_FDX_s111
                                 + 1.07109 w_k1_IBM_s111
                                 + 0.965906 w_k1_INTC_s111
                                 + 0.963019 w_k1_KO_s111
                                 + 1.23386 w_k1_MAT_s111
                                 + 1.06931 w_k1_MSFT_s111
                                 + 1.07256 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.979934 w_k1_ADM_s111
                                 + 1.19719 w_k1_AXP_s111
                                 + 1.19561 w_k1_DIS_s111
                                 + 1.09501 w_k1_FDX_s111
                                 + 1.32876 w_k1_IBM_s111
                                 + 1.10636 w_k1_INTC_s111 + 1.2359 w_k1_KO_s111
                                 + 1.60807 w_k1_MAT_s111
                                 + 1.10648 w_k1_MSFT_s111
                                 + 1.06235 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.774583 w_k1_ADM_s111
                                 + 1.12874 w_k1_AXP_s111 + 1.1276 w_k1_DIS_s111
                                 + 1.18069 w_k1_FDX_s111
                                 + 0.689468 w_k1_IBM_s111
                                 + 0.950356 w_k1_INTC_s111
                                 + 0.941196 w_k1_KO_s111 + 1.062 w_k1_MAT_s111
                                 + 1.25921 w_k1_MSFT_s111
                                 + 1.00995 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.797334 w_k1_ADM_s111
                                 + 1.39125 w_k1_AXP_s111
                                 + 1.19353 w_k1_DIS_s111 + 1.2532 w_k1_FDX_s111
                                 + 0.976178 w_k1_IBM_s111
                                 + 1.17338 w_k1_INTC_s111
                                 + 1.02782 w_k1_KO_s111 + 1.54775 w_k1_MAT_s111
                                 + 1.30348 w_k1_MSFT_s111
                                 + 1.13549 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.640519 w_k1_ADM_s111
                                 + 0.760645 w_k1_AXP_s111
                                 + 0.677855 w_k1_DIS_s111
                                 + 0.759469 w_k1_FDX_s111
                                 + 1.25726 w_k1_IBM_s111
                                 + 0.949012 w_k1_INTC_s111
                                 + 1.0243 w_k1_KO_s111 + 1.68455 w_k1_MAT_s111
                                 + 0.877776 w_k1_MSFT_s111
                                 + 0.885367 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.768104 w_k1_ADM_s111
                                 + 1.02882 w_k1_AXP_s111
                                 + 0.894183 w_k1_DIS_s111
                                 + 1.03995 w_k1_FDX_s111
                                 + 0.802101 w_k1_IBM_s111
                                 + 1.07734 w_k1_INTC_s111
                                 + 0.844592 w_k1_KO_s111
                                 + 1.18727 w_k1_MAT_s111
                                 + 1.04871 w_k1_MSFT_s111
                                 + 0.936478 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.98111 w_k1_ADM_s111
                                 + 1.29951 w_k1_AXP_s111
                                 + 1.64725 w_k1_DIS_s111
                                 + 1.42604 w_k1_FDX_s111
                                 + 1.27847 w_k1_IBM_s111
                                 + 1.34897 w_k1_INTC_s111
                                 + 1.41938 w_k1_KO_s111 + 1.5502 w_k1_MAT_s111
                                 + 1.52318 w_k1_MSFT_s111
                                 + 1.25641 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.696399 w_k1_ADM_s111
                                 + 0.673721 w_k1_AXP_s111
                                 + 0.709229 w_k1_DIS_s111
                                 + 0.821033 w_k1_FDX_s111
                                 + 0.742368 w_k1_IBM_s111
                                 + 0.880457 w_k1_INTC_s111
                                 + 0.765149 w_k1_KO_s111
                                 + 1.01455 w_k1_MAT_s111
                                 + 0.906682 w_k1_MSFT_s111
                                 + 0.872054 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.3422 w_k1_ADM_s112
                                 + 1.16934 w_k1_AXP_s112
                                 + 1.14413 w_k1_DIS_s112 + 1.1235 w_k1_FDX_s112
                                 + 1.22374 w_k1_IBM_s112
                                 + 1.22665 w_k1_INTC_s112 + 1.1431 w_k1_KO_s112
                                 + 1.05683 w_k1_MAT_s112
                                 + 1.13546 w_k1_MSFT_s112
                                 + 1.00463 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.30438 w_k1_ADM_s112
                                 + 1.39501 w_k1_AXP_s112
                                 + 1.27621 w_k1_DIS_s112
                                 + 1.29253 w_k1_FDX_s112
                                 + 1.00522 w_k1_IBM_s112
                                 + 1.21193 w_k1_INTC_s112
                                 + 1.22409 w_k1_KO_s112
                                 + 0.996615 w_k1_MAT_s112
                                 + 1.09695 w_k1_MSFT_s112
                                 + 1.12891 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.935588 w_k1_ADM_s112
                                 + 0.818431 w_k1_AXP_s112
                                 + 0.648972 w_k1_DIS_s112
                                 + 0.77664 w_k1_FDX_s112
                                 + 0.902529 w_k1_IBM_s112
                                 + 0.826448 w_k1_INTC_s112
                                 + 0.789878 w_k1_KO_s112
                                 + 0.758953 w_k1_MAT_s112
                                 + 0.793113 w_k1_MSFT_s112
                                 + 0.885281 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.17121 w_k1_ADM_s112
                                 + 1.24239 w_k1_AXP_s112
                                 + 0.921007 w_k1_DIS_s112
                                 + 1.02309 w_k1_FDX_s112
                                 + 0.887389 w_k1_IBM_s112
                                 + 0.868745 w_k1_INTC_s112
                                 + 0.976623 w_k1_KO_s112
                                 + 1.05536 w_k1_MAT_s112
                                 + 1.02239 w_k1_MSFT_s112
                                 + 0.975663 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.00386 w_k1_ADM_s112
                                 + 0.88535 w_k1_AXP_s112
                                 + 0.774955 w_k1_DIS_s112
                                 + 0.863554 w_k1_FDX_s112
                                 + 0.703534 w_k1_IBM_s112
                                 + 0.86574 w_k1_INTC_s112
                                 + 0.765907 w_k1_KO_s112
                                 + 0.468102 w_k1_MAT_s112
                                 + 0.919988 w_k1_MSFT_s112
                                 + 0.847757 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.21359 w_k1_ADM_s112
                                 + 1.34861 w_k1_AXP_s112
                                 + 1.47382 w_k1_DIS_s112 + 1.1156 w_k1_FDX_s112
                                 + 1.5389 w_k1_IBM_s112
                                 + 1.17376 w_k1_INTC_s112
                                 + 1.64567 w_k1_KO_s112 + 1.70964 w_k1_MAT_s112
                                 + 1.31593 w_k1_MSFT_s112 + 1.0857 w_k1_PG_s112
                                 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.6128 w_k1_ADM_s112
                                 + 1.84485 w_k1_AXP_s112 + 1.8289 w_k1_DIS_s112
                                 + 1.51365 w_k1_FDX_s112
                                 + 1.04133 w_k1_IBM_s112
                                 + 1.21811 w_k1_INTC_s112
                                 + 1.66331 w_k1_KO_s112 + 1.31054 w_k1_MAT_s112
                                 + 1.47543 w_k1_MSFT_s112
                                 + 1.22567 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.956355 w_k1_ADM_s112
                                 + 0.979603 w_k1_AXP_s112
                                 + 1.27121 w_k1_DIS_s112
                                 + 1.10472 w_k1_FDX_s112
                                 + 0.755126 w_k1_IBM_s112
                                 + 1.07067 w_k1_INTC_s112
                                 + 1.00827 w_k1_KO_s112
                                 + 0.645362 w_k1_MAT_s112
                                 + 1.06134 w_k1_MSFT_s112
                                 + 1.06776 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.862538 w_k1_ADM_s113
                                 + 0.674781 w_k1_AXP_s113
                                 + 0.595459 w_k1_DIS_s113
                                 + 0.772647 w_k1_FDX_s113
                                 + 0.835385 w_k1_IBM_s113
                                 + 0.844819 w_k1_INTC_s113
                                 + 0.915442 w_k1_KO_s113
                                 + 0.782867 w_k1_MAT_s113
                                 + 0.831568 w_k1_MSFT_s113
                                 + 0.840655 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.24092 w_k1_ADM_s113
                                 + 1.33493 w_k1_AXP_s113 + 1.463 w_k1_DIS_s113
                                 + 1.45391 w_k1_FDX_s113
                                 + 0.86778 w_k1_IBM_s113
                                 + 1.24825 w_k1_INTC_s113
                                 + 1.15388 w_k1_KO_s113 + 1.14539 w_k1_MAT_s113
                                 + 1.45332 w_k1_MSFT_s113
                                 + 1.04069 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.15438 w_k1_ADM_s113
                                 + 1.25821 w_k1_AXP_s113
                                 + 1.03294 w_k1_DIS_s113
                                 + 1.01182 w_k1_FDX_s113
                                 + 1.10133 w_k1_IBM_s113
                                 + 1.03326 w_k1_INTC_s113
                                 + 1.07412 w_k1_KO_s113 + 1.13826 w_k1_MAT_s113
                                 + 1.03954 w_k1_MSFT_s113
                                 + 0.968598 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.19574 w_k1_ADM_s113
                                 + 1.41718 w_k1_AXP_s113
                                 + 1.20252 w_k1_DIS_s113
                                 + 1.19024 w_k1_FDX_s113
                                 + 1.00622 w_k1_IBM_s113
                                 + 1.36326 w_k1_INTC_s113
                                 + 1.15063 w_k1_KO_s113 + 1.09143 w_k1_MAT_s113
                                 + 1.18118 w_k1_MSFT_s113
                                 + 1.02105 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.906061 w_k1_ADM_s113
                                 + 1.20378 w_k1_AXP_s113
                                 + 1.51606 w_k1_DIS_s113 + 1.1855 w_k1_FDX_s113
                                 + 1.1839 w_k1_IBM_s113 + 1.349 w_k1_INTC_s113
                                 + 1.34772 w_k1_KO_s113 + 1.55211 w_k1_MAT_s113
                                 + 1.25399 w_k1_MSFT_s113
                                 + 1.12605 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.827789 w_k1_ADM_s113
                                 + 0.858631 w_k1_AXP_s113
                                 + 0.910963 w_k1_DIS_s113
                                 + 0.840886 w_k1_FDX_s113
                                 + 1.15414 w_k1_IBM_s113
                                 + 0.751002 w_k1_INTC_s113
                                 + 1.01045 w_k1_KO_s113 + 1.31497 w_k1_MAT_s113
                                 + 0.841167 w_k1_MSFT_s113
                                 + 1.02118 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.23366 w_k1_ADM_s113
                                 + 1.39068 w_k1_AXP_s113
                                 + 1.26932 w_k1_DIS_s113
                                 + 1.02737 w_k1_FDX_s113
                                 + 1.59329 w_k1_IBM_s113
                                 + 0.997737 w_k1_INTC_s113
                                 + 1.52575 w_k1_KO_s113 + 2.29462 w_k1_MAT_s113
                                 + 1.0892 w_k1_MSFT_s113 + 1.33643 w_k1_PG_s113
                                 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.979478 w_k1_ADM_s113
                                 + 1.08709 w_k1_AXP_s113
                                 + 1.00298 w_k1_DIS_s113
                                 + 0.996371 w_k1_FDX_s113
                                 + 0.788714 w_k1_IBM_s113
                                 + 0.956838 w_k1_INTC_s113
                                 + 0.91929 w_k1_KO_s113 + 0.99868 w_k1_MAT_s113
                                 + 1.05937 w_k1_MSFT_s113
                                 + 0.919391 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.1332 w_k1_ADM_s114
                                 + 1.57189 w_k1_AXP_s114
                                 + 1.36934 w_k1_DIS_s114
                                 + 1.23266 w_k1_FDX_s114
                                 + 1.48225 w_k1_IBM_s114
                                 + 1.17618 w_k1_INTC_s114 + 1.1296 w_k1_KO_s114
                                 + 1.25774 w_k1_MAT_s114
                                 + 1.06217 w_k1_MSFT_s114
                                 + 1.07237 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.05268 w_k1_ADM_s114
                                 + 1.10929 w_k1_AXP_s114
                                 + 1.16886 w_k1_DIS_s114
                                 + 1.27483 w_k1_FDX_s114
                                 + 0.759127 w_k1_IBM_s114
                                 + 1.16121 w_k1_INTC_s114
                                 + 0.878773 w_k1_KO_s114
                                 + 0.851019 w_k1_MAT_s114
                                 + 1.35645 w_k1_MSFT_s114
                                 + 0.971846 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.10587 w_k1_ADM_s114
                                 + 1.48824 w_k1_AXP_s114
                                 + 1.54801 w_k1_DIS_s114
                                 + 1.35442 w_k1_FDX_s114
                                 + 1.21623 w_k1_IBM_s114
                                 + 1.40105 w_k1_INTC_s114
                                 + 1.14458 w_k1_KO_s114 + 1.52859 w_k1_MAT_s114
                                 + 1.16358 w_k1_MSFT_s114
                                 + 1.18089 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.991136 w_k1_ADM_s114
                                 + 0.873746 w_k1_AXP_s114
                                 + 0.769335 w_k1_DIS_s114
                                 + 0.87677 w_k1_FDX_s114
                                 + 1.02268 w_k1_IBM_s114
                                 + 0.910113 w_k1_INTC_s114
                                 + 0.851843 w_k1_KO_s114
                                 + 0.829621 w_k1_MAT_s114
                                 + 0.856723 w_k1_MSFT_s114
                                 + 0.949991 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.738871 w_k1_ADM_s114
                                 + 0.635898 w_k1_AXP_s114
                                 + 0.600779 w_k1_DIS_s114
                                 + 0.742619 w_k1_FDX_s114
                                 + 0.674427 w_k1_IBM_s114
                                 + 0.727086 w_k1_INTC_s114
                                 + 0.766754 w_k1_KO_s114
                                 + 0.599152 w_k1_MAT_s114
                                 + 0.70258 w_k1_MSFT_s114
                                 + 0.783518 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.859516 w_k1_ADM_s114
                                 + 1.3461 w_k1_AXP_s114 + 1.08353 w_k1_DIS_s114
                                 + 1.09265 w_k1_FDX_s114
                                 + 1.01156 w_k1_IBM_s114
                                 + 1.05728 w_k1_INTC_s114
                                 + 1.06578 w_k1_KO_s114 + 1.21994 w_k1_MAT_s114
                                 + 1.04546 w_k1_MSFT_s114
                                 + 0.983811 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.914762 w_k1_ADM_s114
                                 + 0.957959 w_k1_AXP_s114
                                 + 0.560832 w_k1_DIS_s114
                                 + 0.764293 w_k1_FDX_s114
                                 + 0.811697 w_k1_IBM_s114
                                 + 0.807379 w_k1_INTC_s114
                                 + 0.782306 w_k1_KO_s114
                                 + 0.909449 w_k1_MAT_s114
                                 + 0.747152 w_k1_MSFT_s114
                                 + 0.88242 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.28419 w_k1_ADM_s114
                                 + 1.17207 w_k1_AXP_s114
                                 + 1.00544 w_k1_DIS_s114
                                 + 0.907872 w_k1_FDX_s114
                                 + 1.67224 w_k1_IBM_s114
                                 + 0.958984 w_k1_INTC_s114
                                 + 1.24476 w_k1_KO_s114 + 1.21222 w_k1_MAT_s114
                                 + 1.02355 w_k1_MSFT_s114
                                 + 1.15746 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.968603 w_k1_ADM_s115
                                 + 0.90073 w_k1_AXP_s115
                                 + 0.703987 w_k1_DIS_s115
                                 + 0.866205 w_k1_FDX_s115
                                 + 0.678826 w_k1_IBM_s115
                                 + 0.868539 w_k1_INTC_s115
                                 + 0.721455 w_k1_KO_s115
                                 + 0.737374 w_k1_MAT_s115
                                 + 0.962914 w_k1_MSFT_s115
                                 + 0.741809 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.57674 w_k1_ADM_s115
                                 + 1.19727 w_k1_AXP_s115
                                 + 1.77774 w_k1_DIS_s115
                                 + 1.28329 w_k1_FDX_s115
                                 + 1.06672 w_k1_IBM_s115
                                 + 1.02793 w_k1_INTC_s115
                                 + 1.33608 w_k1_KO_s115 + 1.44741 w_k1_MAT_s115
                                 + 1.39391 w_k1_MSFT_s115
                                 + 1.18303 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.01489 w_k1_ADM_s115
                                 + 0.940379 w_k1_AXP_s115
                                 + 1.06056 w_k1_DIS_s115
                                 + 0.919777 w_k1_FDX_s115
                                 + 1.17479 w_k1_IBM_s115
                                 + 0.756024 w_k1_INTC_s115
                                 + 1.15336 w_k1_KO_s115 + 1.43726 w_k1_MAT_s115
                                 + 0.924551 w_k1_MSFT_s115
                                 + 0.919013 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.06203 w_k1_ADM_s115
                                 + 1.32704 w_k1_AXP_s115
                                 + 1.37345 w_k1_DIS_s115
                                 + 1.09387 w_k1_FDX_s115
                                 + 1.55069 w_k1_IBM_s115
                                 + 1.10261 w_k1_INTC_s115
                                 + 1.26942 w_k1_KO_s115 + 1.40415 w_k1_MAT_s115
                                 + 1.15614 w_k1_MSFT_s115
                                 + 1.03979 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.0796 w_k1_ADM_s115
                                 + 1.16982 w_k1_AXP_s115
                                 + 0.806153 w_k1_DIS_s115
                                 + 0.864257 w_k1_FDX_s115
                                 + 1.03801 w_k1_IBM_s115
                                 + 0.953014 w_k1_INTC_s115
                                 + 0.865487 w_k1_KO_s115
                                 + 0.753996 w_k1_MAT_s115
                                 + 0.953721 w_k1_MSFT_s115
                                 + 0.835625 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.29435 w_k1_ADM_s115
                                 + 1.2608 w_k1_AXP_s115 + 1.1534 w_k1_DIS_s115
                                 + 1.15033 w_k1_FDX_s115
                                 + 0.837611 w_k1_IBM_s115
                                 + 1.15741 w_k1_INTC_s115
                                 + 0.992602 w_k1_KO_s115
                                 + 1.10231 w_k1_MAT_s115
                                 + 1.26968 w_k1_MSFT_s115
                                 + 0.931344 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.04069 w_k1_ADM_s115
                                 + 0.792994 w_k1_AXP_s115
                                 + 0.999205 w_k1_DIS_s115
                                 + 0.878372 w_k1_FDX_s115
                                 + 1.00285 w_k1_IBM_s115
                                 + 1.00377 w_k1_INTC_s115
                                 + 1.04321 w_k1_KO_s115 + 1.00548 w_k1_MAT_s115
                                 + 0.994069 w_k1_MSFT_s115
                                 + 0.898963 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.693666 w_k1_ADM_s115
                                 + 0.517885 w_k1_AXP_s115
                                 + 0.5902 w_k1_DIS_s115
                                 + 0.616858 w_k1_FDX_s115
                                 + 0.842127 w_k1_IBM_s115
                                 + 0.788956 w_k1_INTC_s115
                                 + 0.712896 w_k1_KO_s115
                                 + 0.826108 w_k1_MAT_s115
                                 + 0.792466 w_k1_MSFT_s115
                                 + 0.73973 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.688147 w_k1_ADM_s116
                                 + 0.630286 w_k1_AXP_s116
                                 + 0.718867 w_k1_DIS_s116
                                 + 0.815811 w_k1_FDX_s116
                                 + 0.719359 w_k1_IBM_s116
                                 + 0.791772 w_k1_INTC_s116
                                 + 0.734551 w_k1_KO_s116
                                 + 0.781879 w_k1_MAT_s116
                                 + 0.769534 w_k1_MSFT_s116
                                 + 0.810528 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.854868 w_k1_ADM_s116
                                 + 1.17295 w_k1_AXP_s116
                                 + 1.34455 w_k1_DIS_s116
                                 + 1.02028 w_k1_FDX_s116
                                 + 1.64108 w_k1_IBM_s116
                                 + 0.99022 w_k1_INTC_s116
                                 + 1.34973 w_k1_KO_s116 + 1.82813 w_k1_MAT_s116
                                 + 1.05778 w_k1_MSFT_s116 + 1.1265 w_k1_PG_s116
                                 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.06055 w_k1_ADM_s116
                                 + 1.2133 w_k1_AXP_s116 + 1.25462 w_k1_DIS_s116
                                 + 1.07732 w_k1_FDX_s116
                                 + 1.00041 w_k1_IBM_s116
                                 + 1.0109 w_k1_INTC_s116
                                 + 0.978624 w_k1_KO_s116
                                 + 1.01144 w_k1_MAT_s116
                                 + 0.946323 w_k1_MSFT_s116
                                 + 1.04599 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.927095 w_k1_ADM_s116
                                 + 0.889203 w_k1_AXP_s116
                                 + 1.09136 w_k1_DIS_s116
                                 + 0.964598 w_k1_FDX_s116
                                 + 0.849428 w_k1_IBM_s116
                                 + 0.740027 w_k1_INTC_s116
                                 + 0.842839 w_k1_KO_s116
                                 + 1.08147 w_k1_MAT_s116
                                 + 0.892281 w_k1_MSFT_s116
                                 + 1.00742 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.12958 w_k1_ADM_s116
                                 + 1.18239 w_k1_AXP_s116
                                 + 1.65168 w_k1_DIS_s116
                                 + 1.31169 w_k1_FDX_s116
                                 + 1.14267 w_k1_IBM_s116
                                 + 0.986137 w_k1_INTC_s116
                                 + 1.31043 w_k1_KO_s116 + 1.3863 w_k1_MAT_s116
                                 + 1.27792 w_k1_MSFT_s116
                                 + 1.18298 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.792512 w_k1_ADM_s116
                                 + 0.745175 w_k1_AXP_s116
                                 + 0.771302 w_k1_DIS_s116
                                 + 0.783973 w_k1_FDX_s116
                                 + 1.0116 w_k1_IBM_s116
                                 + 0.74926 w_k1_INTC_s116
                                 + 0.98153 w_k1_KO_s116 + 1.21927 w_k1_MAT_s116
                                 + 0.715824 w_k1_MSFT_s116
                                 + 0.980437 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.946674 w_k1_ADM_s116
                                 + 0.967493 w_k1_AXP_s116
                                 + 1.47878 w_k1_DIS_s116
                                 + 1.15488 w_k1_FDX_s116
                                 + 0.88089 w_k1_IBM_s116
                                 + 1.00318 w_k1_INTC_s116
                                 + 1.03274 w_k1_KO_s116
                                 + 0.978172 w_k1_MAT_s116
                                 + 1.1402 w_k1_MSFT_s116 + 1.11073 w_k1_PG_s116
                                 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.03247 w_k1_ADM_s116
                                 + 0.876128 w_k1_AXP_s116
                                 + 1.11921 w_k1_DIS_s116
                                 + 0.928371 w_k1_FDX_s116
                                 + 1.30346 w_k1_IBM_s116
                                 + 1.08747 w_k1_INTC_s116
                                 + 1.25044 w_k1_KO_s116 + 1.56575 w_k1_MAT_s116
                                 + 0.932177 w_k1_MSFT_s116
                                 + 1.07019 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.09507 w_k1_ADM_s117
                                 + 0.907611 w_k1_AXP_s117
                                 + 1.27762 w_k1_DIS_s117
                                 + 1.02924 w_k1_FDX_s117
                                 + 0.956679 w_k1_IBM_s117
                                 + 0.971251 w_k1_INTC_s117
                                 + 0.988954 w_k1_KO_s117
                                 + 0.733446 w_k1_MAT_s117
                                 + 1.05527 w_k1_MSFT_s117
                                 + 0.979187 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.02135 w_k1_ADM_s117
                                 + 0.885281 w_k1_AXP_s117
                                 + 0.876555 w_k1_DIS_s117
                                 + 1.00453 w_k1_FDX_s117
                                 + 0.692481 w_k1_IBM_s117
                                 + 0.766348 w_k1_INTC_s117
                                 + 0.878977 w_k1_KO_s117
                                 + 0.78908 w_k1_MAT_s117
                                 + 0.936558 w_k1_MSFT_s117
                                 + 0.887855 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.4586 w_k1_ADM_s117
                                 + 1.41874 w_k1_AXP_s117
                                 + 1.53281 w_k1_DIS_s117
                                 + 1.23366 w_k1_FDX_s117
                                 + 1.46279 w_k1_IBM_s117
                                 + 1.45187 w_k1_INTC_s117
                                 + 1.37936 w_k1_KO_s117 + 1.46064 w_k1_MAT_s117
                                 + 1.40374 w_k1_MSFT_s117
                                 + 1.13782 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.699932 w_k1_ADM_s117
                                 + 0.671761 w_k1_AXP_s117
                                 + 0.605313 w_k1_DIS_s117
                                 + 0.696816 w_k1_FDX_s117
                                 + 0.983812 w_k1_IBM_s117
                                 + 0.906519 w_k1_INTC_s117
                                 + 0.840486 w_k1_KO_s117
                                 + 1.04943 w_k1_MAT_s117
                                 + 0.743357 w_k1_MSFT_s117
                                 + 0.856985 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.787504 w_k1_ADM_s117
                                 + 0.563593 w_k1_AXP_s117
                                 + 0.680538 w_k1_DIS_s117
                                 + 0.775677 w_k1_FDX_s117
                                 + 0.740601 w_k1_IBM_s117
                                 + 0.976105 w_k1_INTC_s117
                                 + 0.835369 w_k1_KO_s117
                                 + 0.640599 w_k1_MAT_s117
                                 + 0.911739 w_k1_MSFT_s117
                                 + 0.827112 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.948462 w_k1_ADM_s117
                                 + 0.952156 w_k1_AXP_s117
                                 + 1.02609 w_k1_DIS_s117
                                 + 0.863455 w_k1_FDX_s117
                                 + 1.32268 w_k1_IBM_s117
                                 + 1.01847 w_k1_INTC_s117
                                 + 1.09329 w_k1_KO_s117 + 1.16813 w_k1_MAT_s117
                                 + 1.05447 w_k1_MSFT_s117
                                 + 0.886787 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.49677 w_k1_ADM_s117
                                 + 1.27748 w_k1_AXP_s117 + 1.3929 w_k1_DIS_s117
                                 + 1.38737 w_k1_FDX_s117
                                 + 0.790248 w_k1_IBM_s117
                                 + 1.14733 w_k1_INTC_s117
                                 + 1.20701 w_k1_KO_s117 + 1.14073 w_k1_MAT_s117
                                 + 1.33488 w_k1_MSFT_s117
                                 + 1.15227 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.17488 w_k1_ADM_s117
                                 + 1.01977 w_k1_AXP_s117
                                 + 0.91318 w_k1_DIS_s117
                                 + 0.938466 w_k1_FDX_s117
                                 + 1.07935 w_k1_IBM_s117
                                 + 0.998611 w_k1_INTC_s117
                                 + 1.05396 w_k1_KO_s117 + 1.11494 w_k1_MAT_s117
                                 + 0.980262 w_k1_MSFT_s117
                                 + 0.88678 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0151586006851997 w_k1_ADM_s000
                                 + 0.0171158273946658 w_k1_ADM_s001
                                 + 0.0119392430595177 w_k1_ADM_s002
                                 + 0.0145086563377263 w_k1_ADM_s003
                                 + 0.0152007823598983 w_k1_ADM_s004
                                 + 0.018464950171809 w_k1_ADM_s005
                                 + 0.0272195428639046 w_k1_ADM_s006
                                 + 0.0365839694513232 w_k1_ADM_s007
                                 + 0.0567689774348717 w_k1_ADM_s010
                                 + 0.0468828810411136 w_k1_ADM_s011
                                 + 0.0283545660614121 w_k1_ADM_s012
                                 + 0.0603432073425134 w_k1_ADM_s013
                                 + 0.0900119371117836 w_k1_ADM_s014
                                 + 0.0227644291735535 w_k1_ADM_s015
                                 + 0.0617989898161519 w_k1_ADM_s016
                                 + 0.0290002623521799 w_k1_ADM_s017
                                 + 0.0129739473042988 w_k1_ADM_s100
                                 + 0.0231875392578986 w_k1_ADM_s101
                                 + 0.0187938302723547 w_k1_ADM_s102
                                 + 0.0426150204259603 w_k1_ADM_s103
                                 + 0.0189697338101615 w_k1_ADM_s104
                                 + 0.037339072676293 w_k1_ADM_s105
                                 + 0.0168209254024138 w_k1_ADM_s106
                                 + 0.0179909510851416 w_k1_ADM_s107
                                 + 0.0197078650182751 w_k1_ADM_s110
                                 + 0.0198991248098869 w_k1_ADM_s111
                                 + 0.0365885154269384 w_k1_ADM_s112
                                 + 0.024350293536638 w_k1_ADM_s113
                                 + 0.0416555022963316 w_k1_ADM_s114
                                 + 0.0247907131167094 w_k1_ADM_s115
                                 + 0.0218913367279186 w_k1_ADM_s116
                                 + 0.0513768167245428 w_k1_ADM_s117
                                 + 0.0189430637944574 w_k1_AXP_s000
                                 + 0.0154498540229558 w_k1_AXP_s001
                                 + 0.0126307442579244 w_k1_AXP_s002
                                 + 0.0176478447107622 w_k1_AXP_s003
                                 + 0.0146859981742937 w_k1_AXP_s004
                                 + 0.0211117097702132 w_k1_AXP_s005
                                 + 0.0280040915978794 w_k1_AXP_s006
                                 + 0.0392837918319441 w_k1_AXP_s007
                                 + 0.0630665015369661 w_k1_AXP_s010
                                 + 0.0481067534564816 w_k1_AXP_s011
                                 + 0.0331967057728578 w_k1_AXP_s012
                                 + 0.0552093135584891 w_k1_AXP_s013
                                 + 0.101795054868886 w_k1_AXP_s014
                                 + 0.0225252352889479 w_k1_AXP_s015
                                 + 0.0796962576071964 w_k1_AXP_s016
                                 + 0.0322161270597658 w_k1_AXP_s017
                                 + 0.0124640200298875 w_k1_AXP_s100
                                 + 0.0281770688851774 w_k1_AXP_s101
                                 + 0.0183533196194015 w_k1_AXP_s102
                                 + 0.0473571561424196 w_k1_AXP_s103
                                 + 0.0159200483595218 w_k1_AXP_s104
                                 + 0.0328215409062434 w_k1_AXP_s105
                                 + 0.0165041879027404 w_k1_AXP_s106
                                 + 0.0201314151493121 w_k1_AXP_s107
                                 + 0.0201811814516593 w_k1_AXP_s110
                                 + 0.0245089277411606 w_k1_AXP_s111
                                 + 0.0359938171020178 w_k1_AXP_s112
                                 + 0.0264436620069892 w_k1_AXP_s113
                                 + 0.0443613142360917 w_k1_AXP_s114
                                 + 0.0234170343925734 w_k1_AXP_s115
                                 + 0.0224759144915072 w_k1_AXP_s116
                                 + 0.0459915561185937 w_k1_AXP_s117
                                 + 0.0166618532293892 w_k1_DIS_s000
                                 + 0.0176356645206656 w_k1_DIS_s001
                                 + 0.0131282848217779 w_k1_DIS_s002
                                 + 0.0169045314378363 w_k1_DIS_s003
                                 + 0.0153227311614211 w_k1_DIS_s004
                                 + 0.0216202871880412 w_k1_DIS_s005
                                 + 0.0278729361060619 w_k1_DIS_s006
                                 + 0.0407235417715526 w_k1_DIS_s007
                                 + 0.0607731152354485 w_k1_DIS_s010
                                 + 0.0494769368145927 w_k1_DIS_s011
                                 + 0.0312982621818234 w_k1_DIS_s012
                                 + 0.0569531009915608 w_k1_DIS_s013
                                 + 0.102452036403177 w_k1_DIS_s014
                                 + 0.0229403949412343 w_k1_DIS_s015
                                 + 0.0746425779274106 w_k1_DIS_s016
                                 + 0.0308179579447449 w_k1_DIS_s017
                                 + 0.0123268193748193 w_k1_DIS_s100
                                 + 0.0271328821504889 w_k1_DIS_s101
                                 + 0.0187392611257094 w_k1_DIS_s102
                                 + 0.0460990913485678 w_k1_DIS_s103
                                 + 0.0194349411952105 w_k1_DIS_s104
                                 + 0.0333441075437474 w_k1_DIS_s105
                                 + 0.0177090497682542 w_k1_DIS_s106
                                 + 0.0217024467411041 w_k1_DIS_s107
                                 + 0.0202144633565137 w_k1_DIS_s110
                                 + 0.0236569190207167 w_k1_DIS_s111
                                 + 0.0335738860698179 w_k1_DIS_s112
                                 + 0.0255972222451531 w_k1_DIS_s113
                                 + 0.0401786399891822 w_k1_DIS_s114
                                 + 0.024337984025158 w_k1_DIS_s115
                                 + 0.026803538222301 w_k1_DIS_s116
                                 + 0.0494706170277506 w_k1_DIS_s117
                                 + 0.0165327681898506 w_k1_FDX_s000
                                 + 0.0165693768871282 w_k1_FDX_s001
                                 + 0.0128155883210796 w_k1_FDX_s002
                                 + 0.0160204592934336 w_k1_FDX_s003
                                 + 0.0143168428613914 w_k1_FDX_s004
                                 + 0.019850396068593 w_k1_FDX_s005
                                 + 0.0277438274007617 w_k1_FDX_s006
                                 + 0.0405159895396094 w_k1_FDX_s007
                                 + 0.0581536721771795 w_k1_FDX_s010
                                 + 0.0493053773283379 w_k1_FDX_s011
                                 + 0.0296297652947925 w_k1_FDX_s012
                                 + 0.0585104350014314 w_k1_FDX_s013
                                 + 0.0989733028067261 w_k1_FDX_s014
                                 + 0.0215823327017546 w_k1_FDX_s015
                                 + 0.0765173571540681 w_k1_FDX_s016
                                 + 0.0273317209399733 w_k1_FDX_s017
                                 + 0.0124362077105543 w_k1_FDX_s100
                                 + 0.0265529289599371 w_k1_FDX_s101
                                 + 0.0184285228777656 w_k1_FDX_s102
                                 + 0.0455513446044677 w_k1_FDX_s103
                                 + 0.0177156629667166 w_k1_FDX_s104
                                 + 0.0319413840646811 w_k1_FDX_s105
                                 + 0.0172481993748811 w_k1_FDX_s106
                                 + 0.0192943029199642 w_k1_FDX_s107
                                 + 0.0213823708439291 w_k1_FDX_s110
                                 + 0.0246593423415278 w_k1_FDX_s111
                                 + 0.0327856154592659 w_k1_FDX_s112
                                 + 0.0244864113707047 w_k1_FDX_s113
                                 + 0.0416586202802008 w_k1_FDX_s114
                                 + 0.0221265184773435 w_k1_FDX_s115
                                 + 0.0233807974801058 w_k1_FDX_s116
                                 + 0.0466834000314585 w_k1_FDX_s117
                                 + 0.0179509190696441 w_k1_IBM_s000
                                 + 0.0163766734765175 w_k1_IBM_s001
                                 + 0.0133994308247485 w_k1_IBM_s002
                                 + 0.0169525989344661 w_k1_IBM_s003
                                 + 0.0142419197441179 w_k1_IBM_s004
                                 + 0.0207340957694484 w_k1_IBM_s005
                                 + 0.0301816426261148 w_k1_IBM_s006
                                 + 0.0428699995954112 w_k1_IBM_s007
                                 + 0.057186952309067 w_k1_IBM_s010
                                 + 0.049137671139615 w_k1_IBM_s011
                                 + 0.0302364775281331 w_k1_IBM_s012
                                 + 0.0604655996132015 w_k1_IBM_s013
                                 + 0.0996083306677255 w_k1_IBM_s014
                                 + 0.0214086350976357 w_k1_IBM_s015
                                 + 0.0773680895574324 w_k1_IBM_s016
                                 + 0.0287379172895661 w_k1_IBM_s017
                                 + 0.0120906005413555 w_k1_IBM_s100
                                 + 0.0295671375658682 w_k1_IBM_s101
                                 + 0.0200137460956479 w_k1_IBM_s102
                                 + 0.044665446939418 w_k1_IBM_s103
                                 + 0.0193753839460515 w_k1_IBM_s104
                                 + 0.031443765947851 w_k1_IBM_s105
                                 + 0.017116816888632 w_k1_IBM_s106
                                 + 0.020875362088311 w_k1_IBM_s107
                                 + 0.0218473647913449 w_k1_IBM_s110
                                 + 0.0248141297155293 w_k1_IBM_s111
                                 + 0.0310760232680126 w_k1_IBM_s112
                                 + 0.0244389928257954 w_k1_IBM_s113
                                 + 0.0438562853454156 w_k1_IBM_s114
                                 + 0.0231952427449962 w_k1_IBM_s115
                                 + 0.0250688993211026 w_k1_IBM_s116
                                 + 0.0470351579818686 w_k1_IBM_s117
                                 + 0.0157973108432714 w_k1_INTC_s000
                                 + 0.0171996075656997 w_k1_INTC_s001
                                 + 0.0114186640022297 w_k1_INTC_s002
                                 + 0.0148656325295292 w_k1_INTC_s003
                                 + 0.014144131016233 w_k1_INTC_s004
                                 + 0.0185400465987043 w_k1_INTC_s005
                                 + 0.0289218396308274 w_k1_INTC_s006
                                 + 0.0416932971661015 w_k1_INTC_s007
                                 + 0.0635629159920146 w_k1_INTC_s010
                                 + 0.0487523005880615 w_k1_INTC_s011
                                 + 0.031459804652455 w_k1_INTC_s012
                                 + 0.0588265378550672 w_k1_INTC_s013
                                 + 0.107116790243111 w_k1_INTC_s014
                                 + 0.0199227082378837 w_k1_INTC_s015
                                 + 0.0738686735063862 w_k1_INTC_s016
                                 + 0.026070434915453 w_k1_INTC_s017
                                 + 0.0129812650454176 w_k1_INTC_s100
                                 + 0.0263254766423035 w_k1_INTC_s101
                                 + 0.0181664157391684 w_k1_INTC_s102
                                 + 0.0540570313125377 w_k1_INTC_s103
                                 + 0.0193596809235908 w_k1_INTC_s104
                                 + 0.0294426521289614 w_k1_INTC_s105
                                 + 0.0166104568147038 w_k1_INTC_s106
                                 + 0.0199464337787282 w_k1_INTC_s107
                                 + 0.0198770754811836 w_k1_INTC_s110
                                 + 0.0247581683081206 w_k1_INTC_s111
                                 + 0.0319392629092957 w_k1_INTC_s112
                                 + 0.0244191473158761 w_k1_INTC_s113
                                 + 0.0417967979783325 w_k1_INTC_s114
                                 + 0.0216047422581181 w_k1_INTC_s115
                                 + 0.0212568785140269 w_k1_INTC_s116
                                 + 0.0475791206815262 w_k1_INTC_s117
                                 + 0.0163547879200864 w_k1_KO_s000
                                 + 0.018121040694191 w_k1_KO_s001
                                 + 0.0130311210284155 w_k1_KO_s002
                                 + 0.0181807621972462 w_k1_KO_s003
                                 + 0.0132756876759443 w_k1_KO_s004
                                 + 0.0203735627192476 w_k1_KO_s005
                                 + 0.0274776082930346 w_k1_KO_s006
                                 + 0.0424614614494619 w_k1_KO_s007
                                 + 0.0541272287498382 w_k1_KO_s010
                                 + 0.0494984796625115 w_k1_KO_s011
                                 + 0.0325950139306652 w_k1_KO_s012
                                 + 0.05975417148653 w_k1_KO_s013
                                 + 0.103318688747469 w_k1_KO_s014
                                 + 0.0218292904229788 w_k1_KO_s015
                                 + 0.0711656958737185 w_k1_KO_s016
                                 + 0.0280101636365775 w_k1_KO_s017
                                 + 0.0127589518048262 w_k1_KO_s100
                                 + 0.0242451457880208 w_k1_KO_s101
                                 + 0.0184182665202761 w_k1_KO_s102
                                 + 0.0487102359224385 w_k1_KO_s103
                                 + 0.0184703058815573 w_k1_KO_s104
                                 + 0.0321327775492825 w_k1_KO_s105
                                 + 0.0164118483584603 w_k1_KO_s106
                                 + 0.0184868768209558 w_k1_KO_s107
                                 + 0.0203875573826862 w_k1_KO_s110
                                 + 0.0242306559878972 w_k1_KO_s111
                                 + 0.0337148969573587 w_k1_KO_s112
                                 + 0.0259395232667317 w_k1_KO_s113
                                 + 0.0400892084567931 w_k1_KO_s114
                                 + 0.0231537786575295 w_k1_KO_s115
                                 + 0.0246447510355775 w_k1_KO_s116
                                 + 0.0485535701189739 w_k1_KO_s117
                                 + 0.0189615770537429 w_k1_MAT_s000
                                 + 0.0134306041777184 w_k1_MAT_s001
                                 + 0.0143747162329485 w_k1_MAT_s002
                                 + 0.0160254785536074 w_k1_MAT_s003
                                 + 0.0151277303519937 w_k1_MAT_s004
                                 + 0.0195726246766032 w_k1_MAT_s005
                                 + 0.032895936414973 w_k1_MAT_s006
                                 + 0.0505080704412388 w_k1_MAT_s007
                                 + 0.0563067396742356 w_k1_MAT_s010
                                 + 0.0439599385623826 w_k1_MAT_s011
                                 + 0.0319513460079643 w_k1_MAT_s012
                                 + 0.0636427103394914 w_k1_MAT_s013
                                 + 0.101560273711284 w_k1_MAT_s014
                                 + 0.0202622699606255 w_k1_MAT_s015
                                 + 0.0789901815935049 w_k1_MAT_s016
                                 + 0.028942978505693 w_k1_MAT_s017
                                 + 0.0128364231010603 w_k1_MAT_s100
                                 + 0.0340628361293672 w_k1_MAT_s101
                                 + 0.0207538995146763 w_k1_MAT_s102
                                 + 0.0566211427505696 w_k1_MAT_s103
                                 + 0.0170528857468522 w_k1_MAT_s104
                                 + 0.0382686772048806 w_k1_MAT_s105
                                 + 0.0188935361581892 w_k1_MAT_s106
                                 + 0.0196856601454911 w_k1_MAT_s107
                                 + 0.0221431096162737 w_k1_MAT_s110
                                 + 0.0321459727299962 w_k1_MAT_s111
                                 + 0.0302601220763956 w_k1_MAT_s112
                                 + 0.0291664815116857 w_k1_MAT_s113
                                 + 0.0415670055272946 w_k1_MAT_s114
                                 + 0.0253058417389415 w_k1_MAT_s115
                                 + 0.0290607996321327 w_k1_MAT_s116
                                 + 0.047388359068254 w_k1_MAT_s117
                                 + 0.0162406767968063 w_k1_MSFT_s000
                                 + 0.0170509774241654 w_k1_MSFT_s001
                                 + 0.0131499108601232 w_k1_MSFT_s002
                                 + 0.0164199309818573 w_k1_MSFT_s003
                                 + 0.0156879524388667 w_k1_MSFT_s004
                                 + 0.0205076306899062 w_k1_MSFT_s005
                                 + 0.0274903414620366 w_k1_MSFT_s006
                                 + 0.0398046336614396 w_k1_MSFT_s007
                                 + 0.0625101956895664 w_k1_MSFT_s010
                                 + 0.0479258327557497 w_k1_MSFT_s011
                                 + 0.0315524369212616 w_k1_MSFT_s012
                                 + 0.0596492973025683 w_k1_MSFT_s013
                                 + 0.0973987787577312 w_k1_MSFT_s014
                                 + 0.0216104018492886 w_k1_MSFT_s015
                                 + 0.0729575257519664 w_k1_MSFT_s016
                                 + 0.0284747764132526 w_k1_MSFT_s017
                                 + 0.0128734279037737 w_k1_MSFT_s100
                                 + 0.0256922844191794 w_k1_MSFT_s101
                                 + 0.0185650905095158 w_k1_MSFT_s102
                                 + 0.0479351061141547 w_k1_MSFT_s103
                                 + 0.0180487243551773 w_k1_MSFT_s104
                                 + 0.0318081628942894 w_k1_MSFT_s105
                                 + 0.0173466160437454 w_k1_MSFT_s106
                                 + 0.018999911332214 w_k1_MSFT_s107
                                 + 0.0193185895376418 w_k1_MSFT_s110
                                 + 0.0261577413126729 w_k1_MSFT_s111
                                 + 0.0330903472143972 w_k1_MSFT_s112
                                 + 0.0254163650332752 w_k1_MSFT_s113
                                 + 0.0401068908147303 w_k1_MSFT_s114
                                 + 0.0240744493761865 w_k1_MSFT_s115
                                 + 0.0222010444315995 w_k1_MSFT_s116
                                 + 0.04960345836904 w_k1_MSFT_s117
                                 + 0.0156493255733084 w_k1_PG_s000
                                 + 0.015944770719557 w_k1_PG_s001
                                 + 0.0145338531664803 w_k1_PG_s002
                                 + 0.0155090171316821 w_k1_PG_s003
                                 + 0.0153857859912968 w_k1_PG_s004
                                 + 0.0193751391781228 w_k1_PG_s005
                                 + 0.0261103070707136 w_k1_PG_s006
                                 + 0.0391528845208522 w_k1_PG_s007
                                 + 0.0591281553976454 w_k1_PG_s010
                                 + 0.0448788393038084 w_k1_PG_s011
                                 + 0.031141561074618 w_k1_PG_s012
                                 + 0.0605796095747695 w_k1_PG_s013
                                 + 0.0992908670429795 w_k1_PG_s014
                                 + 0.0232409502578006 w_k1_PG_s015
                                 + 0.0741566577152746 w_k1_PG_s016
                                 + 0.0292941894554312 w_k1_PG_s017
                                 + 0.0125921814254014 w_k1_PG_s100
                                 + 0.0254225977661551 w_k1_PG_s101
                                 + 0.0197966116353801 w_k1_PG_s102
                                 + 0.0430438479590361 w_k1_PG_s103
                                 + 0.0175238158357013 w_k1_PG_s104
                                 + 0.0327172558603886 w_k1_PG_s105
                                 + 0.0179792203820371 w_k1_PG_s106
                                 + 0.0202140650381351 w_k1_PG_s107
                                 + 0.0200240808818293 w_k1_PG_s110
                                 + 0.0243403538735183 w_k1_PG_s111
                                 + 0.0309438995419707 w_k1_PG_s112
                                 + 0.0241051902023976 w_k1_PG_s113
                                 + 0.0413141045063483 w_k1_PG_s114
                                 + 0.0206041958369226 w_k1_PG_s115
                                 + 0.0245675032304172 w_k1_PG_s116
                                 + 0.0440074762232634 w_k1_PG_s117 >= 1.12
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
