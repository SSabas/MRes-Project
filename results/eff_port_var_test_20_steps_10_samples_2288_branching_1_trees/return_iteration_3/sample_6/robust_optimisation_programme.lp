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
 w_k1_ADM_s0_return_constraint:  1.16907 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.866057 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.04371 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.888725 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.15357 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.971942 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.04136 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.08379 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.16189 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.09164 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.02233 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.11583 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.816586 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.71988 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.09562 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.12553 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.747543 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.944649 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.12421 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.45095 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.883003 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.03407 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.21857 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.917583 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.931501 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.11948 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.852004 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.08261 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.953354 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.739169 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.951655 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.24861 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.12642 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.967926 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.761959 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.973046 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.723661 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.10888 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.16077 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.952851 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.16734 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.807126 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.17511 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.7544 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.01584 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.02282 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.59573 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.894887 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.32232 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.51191 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.847979 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.682678 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.22024 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.21657 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.650518 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.852461 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.18868 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.56342 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.775782 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.979345 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.50627 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.10363 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.790922 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.00309 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.729317 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.10311 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.757671 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.525087 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.930978 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.60957 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.24486 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.87303 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.702222 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.742779 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.718299 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.36808 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.20552 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.922836 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.21368 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.783941 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.20401 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.775203 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.91303 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.61728 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.55091 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.00361 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.12574 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.7912 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.821129 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.619029 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.30853 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.3094 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.780557 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.869371 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.86709 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.5448 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.64912 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.13328 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.59976 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.834355 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.992066 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.95529 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.595846 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.965076 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.856084 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.475687 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.937018 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.76126 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.20548 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.812196 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.503345 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.698288 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.687828 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.36929 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.16666 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.911354 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.09378 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.90756 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.09219 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.881943 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.995275 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.28952 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.36393 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.88141 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.22647 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.29601 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.821199 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.743318 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.32255 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.06873 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.74838 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.983518 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.31817 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.46423 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.733304 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.00823 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.37183 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.928271 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.969892 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.906425 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.75948 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.15857 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.999275 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.617499 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.986176 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.57887 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.21719 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.809448 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.629288 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.891556 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.804154 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.24537 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.973148 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.07443 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.2546 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.747315 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.25829 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.716193 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.891477 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.943576 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.22968 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.49062 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.771631 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.90205 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.19583 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.771506 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.826964 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.63171 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.852708 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.713837 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.0767 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.1537 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.03535 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.14548 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.22049 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.973859 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.962519 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.56994 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.867235 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.771676 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.550032 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.735276 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.857441 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.899356 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.831933 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.26059 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.728709 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.584027 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.904982 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.42832 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.1111 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.950497 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.18044 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.961284 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.938021 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.839506 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.845105 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.06986 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.09755 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.750475 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.07542 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.10766 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.915424 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.714145 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.11317 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.17998 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.758144 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.932127 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.32671 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.15311 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.708264 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.94926 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.42223 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.06578 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.35572 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.00878 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.919587 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.19773 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.01222 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.82439 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.13453 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.43072 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.14378 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.942089 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.755664 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.88748 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.996426 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.29721 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.07599 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.967398 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.24091 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.858313 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.23715 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.809177 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.849529 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.0617 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.19655 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.13386 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.937664 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.59179 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.930096 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.711942 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.897538 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.34956 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.831135 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.789436 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.18384 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.20931 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.806131 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.994885 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.19011 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.874413 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.999955 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.11422 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.713219 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.854619 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.77635 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.616883 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.92469 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.19029 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.06259 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.10388 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.724449 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.764078 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.842245 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.22327 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.957184 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.09261 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.57129 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.71907 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.35355 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.658556 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.811396 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.559246 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.55779 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.16132 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.867506 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.60763 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.846999 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.718758 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.709613 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.70173 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.887278 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.56873 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.28581 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.35303 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.12951 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.938759 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.18173 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.987341 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.757672 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.23256 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.824927 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.749087 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.444388 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.47677 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.859076 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.12898 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.877258 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.11201 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.648501 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.436303 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.851634 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.43543 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.07221 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.915611 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.207 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.939801 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.11018 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.876205 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.873095 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.18747 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.32894 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.849258 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.08443 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.19819 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.845253 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.675144 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.29671 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.25335 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.850858 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.00155 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.48214 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.29157 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.729974 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.12448 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.30831 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.897085 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.11128 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.910142 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.739286 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.05388 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.975411 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.711924 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.00877 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.43984 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.15376 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.926371 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.679639 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.907898 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.915341 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.25213 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.06115 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.947582 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.12725 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.877314 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.13686 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.924158 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.881267 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.08158 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.19307 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.06585 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.06024 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.2157 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.959036 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.794358 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.10556 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.20881 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.880557 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.918801 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.2806 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.18521 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.896776 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.04365 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.12936 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.987965 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.992833 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.01766 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.807299 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.02031 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.929776 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.701845 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.940973 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.11581 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.08223 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.998167 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.737514 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.804274 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.827017 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.1007 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.411000809658356 z_k1_s0000
                                 - 1.17700187448053 z_k1_s0001
                                 - 0.250000460189199 z_k1_s0002
                                 - 0.789998931600531 z_k1_s0003
                                 - 0.569000595978462 z_k1_s0004
                                 - 0.920001206679223 z_k1_s0005
                                 - 0.840002785406331 z_k1_s0006
                                 - 0.575002828678904 z_k1_s0007
                                 - 0.108000075299311 z_k1_s0010
                                 - 0.470000077246732 z_k1_s0011
                                 - 0.130000169349255 z_k1_s0012
                                 - 0.222999407722773 z_k1_s0013
                                 - 0.344999542872049 z_k1_s0014
                                 - 0.386000490947776 z_k1_s0015
                                 - 0.158000095848977 z_k1_s0016
                                 - 0.112000092399301 z_k1_s0017
                                 - 0.129999776746 z_k1_s0020
                                 - 0.303998277745686 z_k1_s0021
                                 - 0.380999591186111 z_k1_s0022
                                 - 0.610000043499378 z_k1_s0023
                                 - 0.46199901565136 z_k1_s0024
                                 - 0.639000384665405 z_k1_s0025
                                 - 0.152999700878021 z_k1_s0026
                                 - 0.275999680583662 z_k1_s0027
                                 - 0.262000117163235 z_k1_s0030
                                 - 0.149000045037108 z_k1_s0031
                                 - 0.207999991396771 z_k1_s0032
                                 - 0.371000432654799 z_k1_s0033
                                 - 0.220999158859104 z_k1_s0034
                                 - 0.0890000347188091 z_k1_s0035
                                 - 0.173999480803089 z_k1_s0036
                                 - 0.19000050414126 z_k1_s0037
                                 - 0.339000285859755 z_k1_s0040
                                 - 0.390001375470311 z_k1_s0041
                                 - 0.316000046526386 z_k1_s0042
                                 - 0.403000392916043 z_k1_s0043
                                 - 0.347001263369865 z_k1_s0044
                                 - 0.285001400691104 z_k1_s0045
                                 - 0.208000733584166 z_k1_s0046
                                 - 0.283001156313577 z_k1_s0047
                                 - 0.148000125257107 z_k1_s0050
                                 - 0.285999713020494 z_k1_s0051
                                 - 0.193999653244478 z_k1_s0052
                                 - 0.106000076146289 z_k1_s0053
                                 - 0.296999701982821 z_k1_s0054
                                 - 0.232000223478742 z_k1_s0055
                                 - 0.158000160850189 z_k1_s0056
                                 - 0.252000963865756 z_k1_s0057
                                 - 0.182000193937352 z_k1_s0060
                                 - 0.539002239066902 z_k1_s0061
                                 - 0.233000486098716 z_k1_s0062
                                 - 0.323001352854579 z_k1_s0063
                                 - 0.829001383608231 z_k1_s0064
                                 - 0.392999705327963 z_k1_s0065
                                 - 0.239000344882149 z_k1_s0066
                                 - 0.24700045509383 z_k1_s0067
                                 - 1.02899995755386 z_k1_s0070
                                 - 0.78000093298127 z_k1_s0071
                                 - 0.388999465386826 z_k1_s0072
                                 - 0.90999641265087 z_k1_s0073
                                 - 0.414999683519292 z_k1_s0074
                                 - 0.505999324784379 z_k1_s0075
                                 - 0.378999554135653 z_k1_s0076
                                 - 1.2029968422463 z_k1_s0077
                                 - 0.162000166370374 z_k1_s0100
                                 - 0.122999847429795 z_k1_s0101
                                 - 0.0719998396760887 z_k1_s0102
                                 - 0.186999727230728 z_k1_s0103
                                 - 0.151999338027805 z_k1_s0104
                                 - 0.159000219067132 z_k1_s0105
                                 - 0.0789999175166733 z_k1_s0106
                                 - 0.0699998748072604 z_k1_s0107
                                 - 0.220999561862803 z_k1_s0110
                                 - 0.23199944822252 z_k1_s0111
                                 - 0.148999936690229 z_k1_s0112
                                 - 0.249999262265694 z_k1_s0113
                                 - 0.28000079733704 z_k1_s0114
                                 - 0.354000032847867 z_k1_s0115
                                 - 0.176000026754868 z_k1_s0116
                                 - 0.182999954438325 z_k1_s0117
                                 - 0.181999369186333 z_k1_s0120
                                 - 0.192999419422566 z_k1_s0121
                                 - 0.190999345788927 z_k1_s0122
                                 - 0.453998031400231 z_k1_s0123
                                 - 0.237998592937972 z_k1_s0124
                                 - 0.177999458418243 z_k1_s0125
                                 - 0.158999350146641 z_k1_s0126
                                 - 0.769996421560685 z_k1_s0127
                                 - 0.475000621012911 z_k1_s0130
                                 - 0.540999209622776 z_k1_s0131
                                 - 0.542997722279315 z_k1_s0132
                                 - 0.734000527308878 z_k1_s0133
                                 - 0.20699970707155 z_k1_s0134
                                 - 0.227999858046713 z_k1_s0135
                                 - 0.254999580686761 z_k1_s0136
                                 - 0.683997373942719 z_k1_s0137
                                 - 0.156000309605782 z_k1_s0140
                                 - 0.115999646630481 z_k1_s0141
                                 - 0.0619998896127706 z_k1_s0142
                                 - 0.145999574862484 z_k1_s0143
                                 - 0.107999855435105 z_k1_s0144
                                 - 0.236999660632611 z_k1_s0145
                                 - 0.145000411787311 z_k1_s0146
                                 - 0.167999370499426 z_k1_s0147
                                 - 0.522997731267569 z_k1_s0150
                                 - 0.271999211767329 z_k1_s0151
                                 - 0.111999675433606 z_k1_s0152
                                 - 0.284998722220471 z_k1_s0153
                                 - 0.441000077641743 z_k1_s0154
                                 - 0.178999692147873 z_k1_s0155
                                 - 0.153999613971071 z_k1_s0156
                                 - 0.443998110827451 z_k1_s0157
                                 - 0.242999806857149 z_k1_s0160
                                 - 0.498000081677564 z_k1_s0161
                                 - 0.703999189447897 z_k1_s0162
                                 - 0.73799828394405 z_k1_s0163
                                 - 0.595999440578036 z_k1_s0164
                                 - 0.300999837279835 z_k1_s0165
                                 - 0.325000029406445 z_k1_s0166
                                 - 0.313999972340079 z_k1_s0167
                                 - 0.267999340519493 z_k1_s0170
                                 - 0.327999148546342 z_k1_s0171
                                 - 0.330999171947612 z_k1_s0172
                                 - 0.302999393534792 z_k1_s0173
                                 - 0.246999506709878 z_k1_s0174
                                 - 0.307999212537392 z_k1_s0175
                                 - 0.586999408859848 z_k1_s0176
                                 - 0.927997558814112 z_k1_s0177
                                 - 1.36799661174308 z_k1_s1000
                                 - 0.64899877114242 z_k1_s1001
                                 - 0.784998614906578 z_k1_s1002
                                 - 1.69299964471316 z_k1_s1003
                                 - 1.40699760921823 z_k1_s1004
                                 - 0.591998714345185 z_k1_s1005
                                 - 0.783998447194677 z_k1_s1006
                                 - 0.640999013244065 z_k1_s1007
                                 - 0.313000620244741 z_k1_s1010
                                 - 0.434002222794994 z_k1_s1011
                                 - 0.398000011960198 z_k1_s1012
                                 - 0.34800147836499 z_k1_s1013
                                 - 0.389999869664245 z_k1_s1014
                                 - 0.459001489592598 z_k1_s1015
                                 - 0.141000387787135 z_k1_s1016
                                 - 0.657999925069695 z_k1_s1017
                                 - 0.428996622251779 z_k1_s1020
                                 - 0.36199852042714 z_k1_s1021
                                 - 0.386998757104599 z_k1_s1022
                                 - 0.423998949107804 z_k1_s1023
                                 - 0.481996490890254 z_k1_s1024
                                 - 0.278999159172886 z_k1_s1025
                                 - 0.908994835281616 z_k1_s1026
                                 - 0.684997779593485 z_k1_s1027
                                 - 0.392000670366687 z_k1_s1030
                                 - 0.592999739334151 z_k1_s1031
                                 - 0.620001312702847 z_k1_s1032
                                 - 0.379000829042592 z_k1_s1033
                                 - 0.627999946438572 z_k1_s1034
                                 - 0.349000592624416 z_k1_s1035
                                 - 0.731000909317905 z_k1_s1036
                                 - 0.431002668143219 z_k1_s1037
                                 - 0.218000380431836 z_k1_s1040
                                 - 0.218000380431836 z_k1_s1041
                                 - 0.496000481066452 z_k1_s1042
                                 - 0.417002138104389 z_k1_s1043
                                 - 0.303000285136149 z_k1_s1044
                                 - 0.548999643788177 z_k1_s1045
                                 - 1.03400213921655 z_k1_s1046
                                 - 0.377999754739401 z_k1_s1047
                                 - 0.424998442652044 z_k1_s1050
                                 - 0.727995652953718 z_k1_s1051
                                 - 0.362998582976067 z_k1_s1052
                                 - 0.295998814009798 z_k1_s1053
                                 - 0.430999646435535 z_k1_s1054
                                 - 0.657999384080518 z_k1_s1055
                                 - 0.714996463076833 z_k1_s1056
                                 - 0.485999058827228 z_k1_s1057
                                 - 0.114000045588047 z_k1_s1060
                                 - 0.133000053186055 z_k1_s1061
                                 - 0.173999356907757 z_k1_s1062
                                 - 0.137999995796031 z_k1_s1063
                                 - 0.187000142654108 z_k1_s1064
                                 - 0.445000856690485 z_k1_s1065
                                 - 0.294999354389785 z_k1_s1066
                                 - 0.126000101292075 z_k1_s1067
                                 - 0.120999924933945 z_k1_s1070
                                 - 0.383000244344481 z_k1_s1071
                                 - 0.217000457939198 z_k1_s1072
                                 - 0.109000025608235 z_k1_s1073
                                 - 0.249999655285072 z_k1_s1074
                                 - 0.474999918641451 z_k1_s1075
                                 - 0.137999893845333 z_k1_s1076
                                 - 0.219000409293493 z_k1_s1077
                                 - 0.569998356117487 z_k1_s1100
                                 - 0.321999800502894 z_k1_s1101
                                 - 0.241000104670378 z_k1_s1102
                                 - 0.362999729300798 z_k1_s1103
                                 - 0.569001769552634 z_k1_s1104
                                 - 0.818000933830324 z_k1_s1105
                                 - 0.665000316636093 z_k1_s1106
                                 - 0.920997931857999 z_k1_s1107
                                 - 0.125000134200699 z_k1_s1110
                                 - 0.514001069379074 z_k1_s1111
                                 - 0.180000093720969 z_k1_s1112
                                 - 0.242999619313113 z_k1_s1113
                                 - 0.210999441024879 z_k1_s1114
                                 - 0.104000100936578 z_k1_s1115
                                 - 0.352000047169842 z_k1_s1116
                                 - 0.185000037840973 z_k1_s1117
                                 - 0.403000337216063 z_k1_s1120
                                 - 0.151999987779672 z_k1_s1121
                                 - 0.136000013402879 z_k1_s1122
                                 - 0.281999432790908 z_k1_s1123
                                 - 0.351001114091901 z_k1_s1124
                                 - 0.454000775742402 z_k1_s1125
                                 - 0.226000216072548 z_k1_s1126
                                 - 0.308000200353681 z_k1_s1127
                                 - 0.648999882299183 z_k1_s1130
                                 - 0.218000680882437 z_k1_s1131
                                 - 0.451999788499286 z_k1_s1132
                                 - 0.696999966113512 z_k1_s1133
                                 - 0.422001237243796 z_k1_s1134
                                 - 0.406001209305687 z_k1_s1135
                                 - 0.220000584299469 z_k1_s1136
                                 - 0.939001889805885 z_k1_s1137
                                 - 0.240999805798716 z_k1_s1140
                                 - 0.233999747742303 z_k1_s1141
                                 - 0.10900002682103 z_k1_s1142
                                 - 0.502999785766671 z_k1_s1143
                                 - 0.10299997705839 z_k1_s1144
                                 - 0.303000320012659 z_k1_s1145
                                 - 0.224999673098344 z_k1_s1146
                                 - 0.474999553541019 z_k1_s1147
                                 - 0.103999967436781 z_k1_s1150
                                 - 0.193000148621065 z_k1_s1151
                                 - 0.16300000771628 z_k1_s1152
                                 - 0.29400052826721 z_k1_s1153
                                 - 0.23299986282762 z_k1_s1154
                                 - 0.13099995215114 z_k1_s1155
                                 - 0.187999651470442 z_k1_s1156
                                 - 0.114999924368571 z_k1_s1157
                                 - 0.116000085599593 z_k1_s1160
                                 - 0.161999979889104 z_k1_s1161
                                 - 0.161000035013238 z_k1_s1162
                                 - 0.240000945205959 z_k1_s1163
                                 - 0.251999068716496 z_k1_s1164
                                 - 0.51499996106721 z_k1_s1165
                                 - 0.431000946494961 z_k1_s1166
                                 - 0.147999941627088 z_k1_s1167
                                 - 1.00199734276948 z_k1_s1170
                                 - 0.631999952193063 z_k1_s1171
                                 - 1.18400271063869 z_k1_s1172
                                 - 1.15500269095579 z_k1_s1173
                                 - 0.731996444377359 z_k1_s1174
                                 - 0.946002072344814 z_k1_s1175
                                 - 0.464999934198393 z_k1_s1176
                                 - 0.796999349044412 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.18122 w_k1_ADM_s000
                                 + 1.13441 w_k1_AXP_s000
                                 + 1.29656 w_k1_DIS_s000
                                 + 0.986766 w_k1_FDX_s000
                                 + 1.65574 w_k1_IBM_s000
                                 + 1.08564 w_k1_INTC_s000
                                 + 1.43654 w_k1_KO_s000 + 1.41692 w_k1_MAT_s000
                                 + 1.21325 w_k1_MSFT_s000 + 1.2588 w_k1_PG_s000
                                 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.785455 w_k1_ADM_s000
                                 + 0.791109 w_k1_AXP_s000
                                 + 0.82596 w_k1_DIS_s000
                                 + 0.878315 w_k1_FDX_s000
                                 + 0.700038 w_k1_IBM_s000
                                 + 0.877654 w_k1_INTC_s000
                                 + 0.787192 w_k1_KO_s000
                                 + 0.748045 w_k1_MAT_s000
                                 + 0.951696 w_k1_MSFT_s000
                                 + 0.860116 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.854516 w_k1_ADM_s000
                                 + 1.23728 w_k1_AXP_s000
                                 + 0.905707 w_k1_DIS_s000
                                 + 0.921957 w_k1_FDX_s000
                                 + 1.34556 w_k1_IBM_s000
                                 + 0.851166 w_k1_INTC_s000
                                 + 0.925766 w_k1_KO_s000
                                 + 1.46242 w_k1_MAT_s000
                                 + 0.864035 w_k1_MSFT_s000
                                 + 1.07164 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.02143 w_k1_ADM_s000
                                 + 1.05444 w_k1_AXP_s000
                                 + 1.09756 w_k1_DIS_s000
                                 + 1.09673 w_k1_FDX_s000
                                 + 0.992249 w_k1_IBM_s000
                                 + 1.14212 w_k1_INTC_s000
                                 + 1.07075 w_k1_KO_s000
                                 + 0.924472 w_k1_MAT_s000
                                 + 1.12482 w_k1_MSFT_s000
                                 + 1.14615 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.01096 w_k1_ADM_s000
                                 + 1.1965 w_k1_AXP_s000 + 1.35262 w_k1_DIS_s000
                                 + 1.25173 w_k1_FDX_s000
                                 + 0.81067 w_k1_IBM_s000
                                 + 1.26098 w_k1_INTC_s000
                                 + 0.989005 w_k1_KO_s000
                                 + 0.758828 w_k1_MAT_s000
                                 + 1.16081 w_k1_MSFT_s000 + 1.1756 w_k1_PG_s000
                                 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.711883 w_k1_ADM_s000
                                 + 0.847288 w_k1_AXP_s000
                                 + 0.854599 w_k1_DIS_s000
                                 + 0.863688 w_k1_FDX_s000
                                 + 0.994365 w_k1_IBM_s000
                                 + 0.804236 w_k1_INTC_s000
                                 + 0.977642 w_k1_KO_s000 + 1.023 w_k1_MAT_s000
                                 + 0.963376 w_k1_MSFT_s000
                                 + 0.945831 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.921886 w_k1_ADM_s000
                                 + 1.22442 w_k1_AXP_s000
                                 + 1.58597 w_k1_DIS_s000
                                 + 1.19397 w_k1_FDX_s000 + 1.2615 w_k1_IBM_s000
                                 + 1.36398 w_k1_INTC_s000
                                 + 1.29293 w_k1_KO_s000 + 1.19744 w_k1_MAT_s000
                                 + 1.19345 w_k1_MSFT_s000
                                 + 1.35499 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.778183 w_k1_ADM_s000
                                 + 0.774874 w_k1_AXP_s000
                                 + 0.792753 w_k1_DIS_s000
                                 + 0.80845 w_k1_FDX_s000
                                 + 1.20047 w_k1_IBM_s000
                                 + 0.923153 w_k1_INTC_s000
                                 + 0.992292 w_k1_KO_s000
                                 + 1.11156 w_k1_MAT_s000
                                 + 0.857656 w_k1_MSFT_s000
                                 + 1.06342 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.29918 w_k1_ADM_s001
                                 + 0.967465 w_k1_AXP_s001
                                 + 1.25257 w_k1_DIS_s001
                                 + 1.12808 w_k1_FDX_s001
                                 + 0.911809 w_k1_IBM_s001
                                 + 0.997563 w_k1_INTC_s001
                                 + 1.14821 w_k1_KO_s001 + 1.05738 w_k1_MAT_s001
                                 + 1.16444 w_k1_MSFT_s001
                                 + 1.11196 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.886723 w_k1_ADM_s001
                                 + 0.850426 w_k1_AXP_s001
                                 + 0.680372 w_k1_DIS_s001
                                 + 0.793279 w_k1_FDX_s001
                                 + 0.789347 w_k1_IBM_s001
                                 + 0.989653 w_k1_INTC_s001
                                 + 0.871756 w_k1_KO_s001
                                 + 0.811625 w_k1_MAT_s001
                                 + 0.908036 w_k1_MSFT_s001
                                 + 0.976992 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.75529 w_k1_ADM_s001
                                 + 1.56579 w_k1_AXP_s001
                                 + 1.86278 w_k1_DIS_s001
                                 + 1.27778 w_k1_FDX_s001
                                 + 1.40507 w_k1_IBM_s001
                                 + 1.16163 w_k1_INTC_s001
                                 + 1.64369 w_k1_KO_s001 + 2.52278 w_k1_MAT_s001
                                 + 1.33752 w_k1_MSFT_s001
                                 + 1.24093 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.981663 w_k1_ADM_s001
                                 + 1.24959 w_k1_AXP_s001
                                 + 1.13711 w_k1_DIS_s001
                                 + 1.09322 w_k1_FDX_s001
                                 + 1.08747 w_k1_IBM_s001
                                 + 1.29753 w_k1_INTC_s001
                                 + 1.26677 w_k1_KO_s001 + 1.62443 w_k1_MAT_s001
                                 + 1.10025 w_k1_MSFT_s001
                                 + 1.21707 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.27087 w_k1_ADM_s001
                                 + 1.04674 w_k1_AXP_s001
                                 + 0.91091 w_k1_DIS_s001
                                 + 0.94758 w_k1_FDX_s001
                                 + 0.874452 w_k1_IBM_s001
                                 + 1.12148 w_k1_INTC_s001
                                 + 0.994305 w_k1_KO_s001
                                 + 1.13962 w_k1_MAT_s001
                                 + 1.08487 w_k1_MSFT_s001
                                 + 0.945512 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.713771 w_k1_ADM_s001
                                 + 0.988039 w_k1_AXP_s001
                                 + 1.00231 w_k1_DIS_s001
                                 + 0.880698 w_k1_FDX_s001
                                 + 1.37009 w_k1_IBM_s001
                                 + 0.983492 w_k1_INTC_s001
                                 + 1.24984 w_k1_KO_s001 + 2.68407 w_k1_MAT_s001
                                 + 0.961196 w_k1_MSFT_s001
                                 + 1.03229 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.35705 w_k1_ADM_s001
                                 + 1.44279 w_k1_AXP_s001
                                 + 1.10133 w_k1_DIS_s001
                                 + 1.28086 w_k1_FDX_s001
                                 + 0.825838 w_k1_IBM_s001
                                 + 1.13242 w_k1_INTC_s001
                                 + 1.00529 w_k1_KO_s001 + 1.37332 w_k1_MAT_s001
                                 + 1.2269 w_k1_MSFT_s001 + 1.14218 w_k1_PG_s001
                                 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.57034 w_k1_ADM_s001
                                 + 1.54134 w_k1_AXP_s001
                                 + 1.49969 w_k1_DIS_s001
                                 + 1.41687 w_k1_FDX_s001
                                 + 1.04614 w_k1_IBM_s001
                                 + 1.49013 w_k1_INTC_s001
                                 + 1.42312 w_k1_KO_s001 + 1.92182 w_k1_MAT_s001
                                 + 1.56744 w_k1_MSFT_s001
                                 + 1.38836 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.39615 w_k1_ADM_s002
                                 + 0.893811 w_k1_AXP_s002
                                 + 1.59902 w_k1_DIS_s002
                                 + 1.30436 w_k1_FDX_s002
                                 + 1.27663 w_k1_IBM_s002
                                 + 1.26026 w_k1_INTC_s002
                                 + 1.31894 w_k1_KO_s002
                                 + 0.635736 w_k1_MAT_s002
                                 + 1.20304 w_k1_MSFT_s002
                                 + 1.11313 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.19473 w_k1_ADM_s002
                                 + 1.11383 w_k1_AXP_s002
                                 + 1.13269 w_k1_DIS_s002
                                 + 0.94439 w_k1_FDX_s002
                                 + 1.53163 w_k1_IBM_s002
                                 + 1.29264 w_k1_INTC_s002
                                 + 1.30364 w_k1_KO_s002 + 1.21467 w_k1_MAT_s002
                                 + 1.08959 w_k1_MSFT_s002
                                 + 1.08716 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.33042 w_k1_ADM_s002
                                 + 1.535 w_k1_AXP_s002 + 1.58853 w_k1_DIS_s002
                                 + 1.45025 w_k1_FDX_s002
                                 + 1.26883 w_k1_IBM_s002
                                 + 1.52963 w_k1_INTC_s002
                                 + 1.39072 w_k1_KO_s002 + 1.00801 w_k1_MAT_s002
                                 + 1.15442 w_k1_MSFT_s002
                                 + 1.21053 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.08656 w_k1_ADM_s002
                                 + 0.752825 w_k1_AXP_s002
                                 + 0.73738 w_k1_DIS_s002
                                 + 0.802254 w_k1_FDX_s002
                                 + 0.888047 w_k1_IBM_s002
                                 + 1.01638 w_k1_INTC_s002
                                 + 0.811942 w_k1_KO_s002
                                 + 0.536649 w_k1_MAT_s002
                                 + 0.744923 w_k1_MSFT_s002
                                 + 0.84306 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.08704 w_k1_ADM_s002
                                 + 1.02748 w_k1_AXP_s002
                                 + 1.20627 w_k1_DIS_s002
                                 + 1.09955 w_k1_FDX_s002
                                 + 1.00678 w_k1_IBM_s002
                                 + 1.11606 w_k1_INTC_s002
                                 + 1.07906 w_k1_KO_s002 + 1.02083 w_k1_MAT_s002
                                 + 1.07152 w_k1_MSFT_s002
                                 + 1.14692 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.22429 w_k1_ADM_s002
                                 + 0.858526 w_k1_AXP_s002
                                 + 0.940542 w_k1_DIS_s002
                                 + 0.935157 w_k1_FDX_s002
                                 + 0.759502 w_k1_IBM_s002
                                 + 0.799693 w_k1_INTC_s002
                                 + 0.814109 w_k1_KO_s002
                                 + 0.63365 w_k1_MAT_s002
                                 + 0.928048 w_k1_MSFT_s002
                                 + 0.848022 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.0187 w_k1_ADM_s002
                                 + 0.751063 w_k1_AXP_s002
                                 + 0.980204 w_k1_DIS_s002
                                 + 0.886006 w_k1_FDX_s002
                                 + 1.25811 w_k1_IBM_s002
                                 + 1.00207 w_k1_INTC_s002
                                 + 1.04098 w_k1_KO_s002
                                 + 0.646838 w_k1_MAT_s002
                                 + 0.868482 w_k1_MSFT_s002
                                 + 1.01085 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.04177 w_k1_ADM_s002
                                 + 0.717244 w_k1_AXP_s002
                                 + 0.749842 w_k1_DIS_s002
                                 + 0.757755 w_k1_FDX_s002
                                 + 1.16278 w_k1_IBM_s002
                                 + 0.897365 w_k1_INTC_s002
                                 + 0.923532 w_k1_KO_s002
                                 + 0.648721 w_k1_MAT_s002
                                 + 0.796617 w_k1_MSFT_s002
                                 + 0.948114 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.769728 w_k1_ADM_s003
                                 + 0.923227 w_k1_AXP_s003
                                 + 0.869346 w_k1_DIS_s003
                                 + 0.996869 w_k1_FDX_s003
                                 + 0.786705 w_k1_IBM_s003
                                 + 1.02575 w_k1_INTC_s003
                                 + 0.887294 w_k1_KO_s003
                                 + 0.812364 w_k1_MAT_s003
                                 + 0.975117 w_k1_MSFT_s003
                                 + 0.936292 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.9264 w_k1_ADM_s003
                                 + 1.46545 w_k1_AXP_s003
                                 + 1.71806 w_k1_DIS_s003
                                 + 1.26195 w_k1_FDX_s003
                                 + 1.40855 w_k1_IBM_s003
                                 + 1.24409 w_k1_INTC_s003
                                 + 1.32873 w_k1_KO_s003 + 1.33823 w_k1_MAT_s003
                                 + 1.13652 w_k1_MSFT_s003
                                 + 1.05551 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.617728 w_k1_ADM_s003
                                 + 0.921166 w_k1_AXP_s003
                                 + 0.871418 w_k1_DIS_s003
                                 + 0.794737 w_k1_FDX_s003
                                 + 1.1017 w_k1_IBM_s003
                                 + 1.07738 w_k1_INTC_s003
                                 + 0.921027 w_k1_KO_s003
                                 + 0.982639 w_k1_MAT_s003
                                 + 0.891219 w_k1_MSFT_s003
                                 + 0.788531 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.00507 w_k1_ADM_s003
                                 + 1.10945 w_k1_AXP_s003 + 1.3837 w_k1_DIS_s003
                                 + 1.23316 w_k1_FDX_s003
                                 + 0.999383 w_k1_IBM_s003
                                 + 1.23917 w_k1_INTC_s003
                                 + 1.12683 w_k1_KO_s003
                                 + 0.806387 w_k1_MAT_s003
                                 + 1.15639 w_k1_MSFT_s003
                                 + 1.10211 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.672528 w_k1_ADM_s003
                                 + 0.884494 w_k1_AXP_s003
                                 + 0.622192 w_k1_DIS_s003
                                 + 0.748533 w_k1_FDX_s003
                                 + 0.920359 w_k1_IBM_s003
                                 + 1.15396 w_k1_INTC_s003
                                 + 0.787689 w_k1_KO_s003
                                 + 1.16008 w_k1_MAT_s003
                                 + 1.0064 w_k1_MSFT_s003
                                 + 0.847801 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.770038 w_k1_ADM_s003
                                 + 0.716484 w_k1_AXP_s003
                                 + 0.551361 w_k1_DIS_s003
                                 + 0.691944 w_k1_FDX_s003
                                 + 0.653737 w_k1_IBM_s003
                                 + 1.12593 w_k1_INTC_s003
                                 + 0.633815 w_k1_KO_s003
                                 + 0.447608 w_k1_MAT_s003
                                 + 0.779726 w_k1_MSFT_s003
                                 + 0.83169 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.946498 w_k1_ADM_s003
                                 + 1.52611 w_k1_AXP_s003
                                 + 1.05202 w_k1_DIS_s003
                                 + 1.01883 w_k1_FDX_s003
                                 + 1.13438 w_k1_IBM_s003 + 1.128 w_k1_INTC_s003
                                 + 1.00802 w_k1_KO_s003 + 1.29491 w_k1_MAT_s003
                                 + 1.03519 w_k1_MSFT_s003
                                 + 1.04544 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.705202 w_k1_ADM_s003
                                 + 1.00209 w_k1_AXP_s003
                                 + 0.820221 w_k1_DIS_s003
                                 + 0.840222 w_k1_FDX_s003
                                 + 1.46687 w_k1_IBM_s003
                                 + 1.19234 w_k1_INTC_s003 + 1.111 w_k1_KO_s003
                                 + 1.59121 w_k1_MAT_s003
                                 + 1.05737 w_k1_MSFT_s003
                                 + 1.06106 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.29233 w_k1_ADM_s004
                                 + 1.30019 w_k1_AXP_s004
                                 + 0.862866 w_k1_DIS_s004
                                 + 1.05429 w_k1_FDX_s004
                                 + 1.15597 w_k1_IBM_s004
                                 + 0.996353 w_k1_INTC_s004
                                 + 0.873753 w_k1_KO_s004
                                 + 1.22554 w_k1_MAT_s004
                                 + 0.936632 w_k1_MSFT_s004
                                 + 0.994188 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.95725 w_k1_ADM_s004
                                 + 0.959484 w_k1_AXP_s004
                                 + 0.736849 w_k1_DIS_s004
                                 + 0.860813 w_k1_FDX_s004
                                 + 1.21879 w_k1_IBM_s004
                                 + 0.861725 w_k1_INTC_s004
                                 + 0.912935 w_k1_KO_s004
                                 + 1.45608 w_k1_MAT_s004
                                 + 0.83964 w_k1_MSFT_s004
                                 + 0.949558 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.43625 w_k1_ADM_s004
                                 + 1.44006 w_k1_AXP_s004
                                 + 1.33665 w_k1_DIS_s004
                                 + 1.22027 w_k1_FDX_s004
                                 + 1.48946 w_k1_IBM_s004
                                 + 1.08545 w_k1_INTC_s004
                                 + 1.34328 w_k1_KO_s004 + 2.14514 w_k1_MAT_s004
                                 + 1.18698 w_k1_MSFT_s004
                                 + 1.25682 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.07439 w_k1_ADM_s004
                                 + 0.90696 w_k1_AXP_s004
                                 + 0.964945 w_k1_DIS_s004
                                 + 1.03855 w_k1_FDX_s004
                                 + 0.885309 w_k1_IBM_s004
                                 + 0.869008 w_k1_INTC_s004
                                 + 0.942341 w_k1_KO_s004
                                 + 0.687857 w_k1_MAT_s004
                                 + 0.99347 w_k1_MSFT_s004 + 0.9651 w_k1_PG_s004
                                 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.09638 w_k1_ADM_s004
                                 + 0.658125 w_k1_AXP_s004
                                 + 0.589055 w_k1_DIS_s004
                                 + 0.762362 w_k1_FDX_s004
                                 + 0.727699 w_k1_IBM_s004
                                 + 0.808942 w_k1_INTC_s004
                                 + 0.608956 w_k1_KO_s004
                                 + 0.586799 w_k1_MAT_s004
                                 + 0.673407 w_k1_MSFT_s004
                                 + 0.81306 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.20311 w_k1_ADM_s004
                                 + 0.79074 w_k1_AXP_s004
                                 + 0.890887 w_k1_DIS_s004
                                 + 0.950693 w_k1_FDX_s004
                                 + 0.997444 w_k1_IBM_s004
                                 + 0.945874 w_k1_INTC_s004
                                 + 0.796483 w_k1_KO_s004
                                 + 0.724156 w_k1_MAT_s004
                                 + 1.01268 w_k1_MSFT_s004
                                 + 0.954363 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.56668 w_k1_ADM_s004
                                 + 1.40061 w_k1_AXP_s004 + 1.6 w_k1_DIS_s004
                                 + 1.41573 w_k1_FDX_s004
                                 + 1.01059 w_k1_IBM_s004
                                 + 1.1762 w_k1_INTC_s004 + 1.03269 w_k1_KO_s004
                                 + 0.994158 w_k1_MAT_s004
                                 + 1.30608 w_k1_MSFT_s004
                                 + 1.11779 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.18157 w_k1_ADM_s004
                                 + 1.07328 w_k1_AXP_s004
                                 + 1.16514 w_k1_DIS_s004
                                 + 1.10277 w_k1_FDX_s004
                                 + 1.28805 w_k1_IBM_s004
                                 + 1.12013 w_k1_INTC_s004
                                 + 1.16656 w_k1_KO_s004 + 1.11334 w_k1_MAT_s004
                                 + 1.1128 w_k1_MSFT_s004
                                 + 0.962344 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.48526 w_k1_ADM_s005
                                 + 2.06074 w_k1_AXP_s005
                                 + 1.71192 w_k1_DIS_s005
                                 + 1.43957 w_k1_FDX_s005
                                 + 1.51852 w_k1_IBM_s005
                                 + 1.46243 w_k1_INTC_s005
                                 + 1.34555 w_k1_KO_s005 + 2.18518 w_k1_MAT_s005
                                 + 1.37472 w_k1_MSFT_s005
                                 + 1.36172 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.36723 w_k1_ADM_s005
                                 + 1.12554 w_k1_AXP_s005
                                 + 1.41041 w_k1_DIS_s005
                                 + 1.27768 w_k1_FDX_s005
                                 + 1.01642 w_k1_IBM_s005
                                 + 1.15813 w_k1_INTC_s005
                                 + 1.13906 w_k1_KO_s005 + 1.09894 w_k1_MAT_s005
                                 + 1.18741 w_k1_MSFT_s005
                                 + 1.18952 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.845891 w_k1_ADM_s005
                                 + 0.589529 w_k1_AXP_s005
                                 + 0.607286 w_k1_DIS_s005
                                 + 0.753889 w_k1_FDX_s005
                                 + 0.720856 w_k1_IBM_s005
                                 + 0.916689 w_k1_INTC_s005
                                 + 0.747652 w_k1_KO_s005
                                 + 0.702134 w_k1_MAT_s005
                                 + 0.795632 w_k1_MSFT_s005
                                 + 0.930835 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.934039 w_k1_ADM_s005
                                 + 0.771192 w_k1_AXP_s005
                                 + 0.52772 w_k1_DIS_s005
                                 + 0.701242 w_k1_FDX_s005
                                 + 0.89703 w_k1_IBM_s005
                                 + 0.88115 w_k1_INTC_s005
                                 + 0.67274 w_k1_KO_s005
                                 + 0.868453 w_k1_MAT_s005
                                 + 0.739524 w_k1_MSFT_s005
                                 + 0.898176 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.31298 w_k1_ADM_s005
                                 + 1.31061 w_k1_AXP_s005
                                 + 0.955855 w_k1_DIS_s005
                                 + 1.00078 w_k1_FDX_s005
                                 + 1.32394 w_k1_IBM_s005
                                 + 1.0532 w_k1_INTC_s005
                                 + 0.975225 w_k1_KO_s005
                                 + 1.34464 w_k1_MAT_s005
                                 + 0.962094 w_k1_MSFT_s005
                                 + 1.01616 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.862479 w_k1_ADM_s005
                                 + 0.845629 w_k1_AXP_s005
                                 + 0.916773 w_k1_DIS_s005
                                 + 0.943879 w_k1_FDX_s005
                                 + 1.20854 w_k1_IBM_s005
                                 + 0.872368 w_k1_INTC_s005
                                 + 0.948521 w_k1_KO_s005
                                 + 1.17619 w_k1_MAT_s005
                                 + 0.961397 w_k1_MSFT_s005
                                 + 0.949468 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.24764 w_k1_ADM_s005
                                 + 1.30465 w_k1_AXP_s005
                                 + 0.960261 w_k1_DIS_s005
                                 + 1.11835 w_k1_FDX_s005
                                 + 0.715416 w_k1_IBM_s005
                                 + 1.11177 w_k1_INTC_s005
                                 + 0.770133 w_k1_KO_s005
                                 + 0.954908 w_k1_MAT_s005
                                 + 1.13237 w_k1_MSFT_s005 + 1.0829 w_k1_PG_s005
                                 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.22724 w_k1_ADM_s005
                                 + 0.846885 w_k1_AXP_s005
                                 + 0.71601 w_k1_DIS_s005
                                 + 0.840258 w_k1_FDX_s005
                                 + 0.935877 w_k1_IBM_s005
                                 + 0.917342 w_k1_INTC_s005
                                 + 0.871992 w_k1_KO_s005
                                 + 1.02368 w_k1_MAT_s005
                                 + 0.918528 w_k1_MSFT_s005
                                 + 0.830483 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.894084 w_k1_ADM_s006
                                 + 0.683479 w_k1_AXP_s006
                                 + 0.731128 w_k1_DIS_s006
                                 + 0.787879 w_k1_FDX_s006
                                 + 0.911834 w_k1_IBM_s006
                                 + 0.766248 w_k1_INTC_s006
                                 + 0.909577 w_k1_KO_s006
                                 + 0.820952 w_k1_MAT_s006
                                 + 0.935712 w_k1_MSFT_s006
                                 + 0.826824 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.03231 w_k1_ADM_s006
                                 + 0.995099 w_k1_AXP_s006
                                 + 1.11338 w_k1_DIS_s006
                                 + 1.00834 w_k1_FDX_s006
                                 + 1.12415 w_k1_IBM_s006
                                 + 0.92173 w_k1_INTC_s006
                                 + 1.01753 w_k1_KO_s006 + 1.09367 w_k1_MAT_s006
                                 + 1.08426 w_k1_MSFT_s006
                                 + 0.936058 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.06027 w_k1_ADM_s006
                                 + 0.629722 w_k1_AXP_s006
                                 + 0.92672 w_k1_DIS_s006
                                 + 0.913389 w_k1_FDX_s006
                                 + 0.719134 w_k1_IBM_s006
                                 + 0.764476 w_k1_INTC_s006
                                 + 0.89058 w_k1_KO_s006
                                 + 0.727703 w_k1_MAT_s006
                                 + 0.93712 w_k1_MSFT_s006
                                 + 0.900145 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.961153 w_k1_ADM_s006
                                 + 1.15869 w_k1_AXP_s006
                                 + 1.19719 w_k1_DIS_s006
                                 + 1.12316 w_k1_FDX_s006
                                 + 0.786609 w_k1_IBM_s006
                                 + 0.792956 w_k1_INTC_s006
                                 + 1.0495 w_k1_KO_s006 + 1.31006 w_k1_MAT_s006
                                 + 1.11987 w_k1_MSFT_s006
                                 + 1.02221 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.50135 w_k1_ADM_s006
                                 + 1.53646 w_k1_AXP_s006
                                 + 1.43725 w_k1_DIS_s006 + 1.3142 w_k1_FDX_s006
                                 + 1.19146 w_k1_IBM_s006
                                 + 1.13934 w_k1_INTC_s006
                                 + 1.25409 w_k1_KO_s006 + 1.40155 w_k1_MAT_s006
                                 + 1.20659 w_k1_MSFT_s006
                                 + 1.11264 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.810762 w_k1_ADM_s006
                                 + 0.901526 w_k1_AXP_s006
                                 + 0.8764 w_k1_DIS_s006
                                 + 0.830594 w_k1_FDX_s006
                                 + 1.30709 w_k1_IBM_s006
                                 + 0.782785 w_k1_INTC_s006
                                 + 1.11748 w_k1_KO_s006 + 1.53014 w_k1_MAT_s006
                                 + 0.851561 w_k1_MSFT_s006
                                 + 1.0398 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.894476 w_k1_ADM_s006
                                 + 0.826163 w_k1_AXP_s006
                                 + 0.8012 w_k1_DIS_s006
                                 + 0.906506 w_k1_FDX_s006
                                 + 0.872868 w_k1_IBM_s006
                                 + 0.744591 w_k1_INTC_s006
                                 + 0.896213 w_k1_KO_s006 + 1.0238 w_k1_MAT_s006
                                 + 0.754268 w_k1_MSFT_s006
                                 + 0.845254 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.819396 w_k1_ADM_s006
                                 + 0.725634 w_k1_AXP_s006
                                 + 0.770594 w_k1_DIS_s006
                                 + 0.89132 w_k1_FDX_s006
                                 + 0.655553 w_k1_IBM_s006
                                 + 0.73672 w_k1_INTC_s006
                                 + 0.726197 w_k1_KO_s006
                                 + 0.749547 w_k1_MAT_s006
                                 + 0.940855 w_k1_MSFT_s006
                                 + 0.78512 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.47167 w_k1_ADM_s007
                                 + 1.47153 w_k1_AXP_s007
                                 + 1.40142 w_k1_DIS_s007
                                 + 1.36199 w_k1_FDX_s007
                                 + 1.16568 w_k1_IBM_s007
                                 + 1.42174 w_k1_INTC_s007
                                 + 1.22618 w_k1_KO_s007 + 1.37227 w_k1_MAT_s007
                                 + 1.51489 w_k1_MSFT_s007 + 1.2306 w_k1_PG_s007
                                 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.10711 w_k1_ADM_s007
                                 + 1.18317 w_k1_AXP_s007
                                 + 0.989071 w_k1_DIS_s007
                                 + 1.06779 w_k1_FDX_s007
                                 + 0.840723 w_k1_IBM_s007
                                 + 0.894315 w_k1_INTC_s007
                                 + 0.887855 w_k1_KO_s007
                                 + 0.921888 w_k1_MAT_s007
                                 + 0.964127 w_k1_MSFT_s007
                                 + 1.03632 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.630856 w_k1_ADM_s007
                                 + 0.731746 w_k1_AXP_s007
                                 + 0.574222 w_k1_DIS_s007
                                 + 0.603804 w_k1_FDX_s007
                                 + 1.04169 w_k1_IBM_s007
                                 + 0.862173 w_k1_INTC_s007
                                 + 0.771673 w_k1_KO_s007
                                 + 0.820878 w_k1_MAT_s007
                                 + 0.818877 w_k1_MSFT_s007
                                 + 0.760943 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.966397 w_k1_ADM_s007
                                 + 1.05167 w_k1_AXP_s007
                                 + 0.960016 w_k1_DIS_s007
                                 + 0.914626 w_k1_FDX_s007
                                 + 1.26907 w_k1_IBM_s007
                                 + 0.921977 w_k1_INTC_s007
                                 + 1.01679 w_k1_KO_s007 + 1.15051 w_k1_MAT_s007
                                 + 1.00252 w_k1_MSFT_s007 + 1.0098 w_k1_PG_s007
                                 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.49083 w_k1_ADM_s007
                                 + 1.19383 w_k1_AXP_s007
                                 + 1.30577 w_k1_DIS_s007
                                 + 1.31078 w_k1_FDX_s007
                                 + 0.726618 w_k1_IBM_s007
                                 + 1.21798 w_k1_INTC_s007
                                 + 0.90305 w_k1_KO_s007 + 0.69742 w_k1_MAT_s007
                                 + 1.35913 w_k1_MSFT_s007
                                 + 0.99914 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.9388 w_k1_ADM_s007
                                 + 0.794153 w_k1_AXP_s007
                                 + 1.11057 w_k1_DIS_s007
                                 + 0.928564 w_k1_FDX_s007 + 1.267 w_k1_IBM_s007
                                 + 0.948493 w_k1_INTC_s007
                                 + 1.18165 w_k1_KO_s007 + 1.02304 w_k1_MAT_s007
                                 + 1.23592 w_k1_MSFT_s007
                                 + 0.993298 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.13651 w_k1_ADM_s007
                                 + 0.818809 w_k1_AXP_s007
                                 + 0.924831 w_k1_DIS_s007
                                 + 1.04631 w_k1_FDX_s007
                                 + 0.62682 w_k1_IBM_s007
                                 + 1.01082 w_k1_INTC_s007
                                 + 0.713456 w_k1_KO_s007 + 0.4426 w_k1_MAT_s007
                                 + 1.12441 w_k1_MSFT_s007
                                 + 0.931058 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.03594 w_k1_ADM_s007
                                 + 0.819736 w_k1_AXP_s007
                                 + 0.75815 w_k1_DIS_s007
                                 + 0.847805 w_k1_FDX_s007
                                 + 0.754934 w_k1_IBM_s007
                                 + 0.883168 w_k1_INTC_s007
                                 + 0.908854 w_k1_KO_s007
                                 + 0.711627 w_k1_MAT_s007
                                 + 0.909235 w_k1_MSFT_s007
                                 + 0.902682 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.858102 w_k1_ADM_s010
                                 + 0.795599 w_k1_AXP_s010
                                 + 0.783165 w_k1_DIS_s010
                                 + 0.835479 w_k1_FDX_s010
                                 + 1.00755 w_k1_IBM_s010
                                 + 0.781849 w_k1_INTC_s010
                                 + 0.954845 w_k1_KO_s010 + 1.5523 w_k1_MAT_s010
                                 + 0.792183 w_k1_MSFT_s010
                                 + 0.971681 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.41013 w_k1_ADM_s010
                                 + 0.98079 w_k1_AXP_s010 + 1.2803 w_k1_DIS_s010
                                 + 1.22425 w_k1_FDX_s010
                                 + 0.862172 w_k1_IBM_s010
                                 + 1.26458 w_k1_INTC_s010
                                 + 1.16031 w_k1_KO_s010 + 1.00783 w_k1_MAT_s010
                                 + 1.23589 w_k1_MSFT_s010
                                 + 1.06854 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.1197 w_k1_ADM_s010
                                 + 1.03635 w_k1_AXP_s010
                                 + 0.822624 w_k1_DIS_s010
                                 + 1.10558 w_k1_FDX_s010
                                 + 0.685542 w_k1_IBM_s010
                                 + 0.886102 w_k1_INTC_s010
                                 + 0.735764 w_k1_KO_s010 + 1.5818 w_k1_MAT_s010
                                 + 0.855225 w_k1_MSFT_s010
                                 + 0.88589 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.18606 w_k1_ADM_s010
                                 + 1.38946 w_k1_AXP_s010
                                 + 1.09204 w_k1_DIS_s010
                                 + 1.13124 w_k1_FDX_s010
                                 + 1.18207 w_k1_IBM_s010
                                 + 1.12885 w_k1_INTC_s010
                                 + 1.21584 w_k1_KO_s010 + 1.42942 w_k1_MAT_s010
                                 + 0.907048 w_k1_MSFT_s010
                                 + 1.06347 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.13681 w_k1_ADM_s010
                                 + 0.8456 w_k1_AXP_s010
                                 + 0.990315 w_k1_DIS_s010
                                 + 0.937435 w_k1_FDX_s010
                                 + 1.0537 w_k1_IBM_s010
                                 + 0.991609 w_k1_INTC_s010
                                 + 1.0728 w_k1_KO_s010 + 0.980078 w_k1_MAT_s010
                                 + 0.916918 w_k1_MSFT_s010
                                 + 0.993111 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.911011 w_k1_ADM_s010
                                 + 0.889961 w_k1_AXP_s010
                                 + 1.13531 w_k1_DIS_s010 + 1.0572 w_k1_FDX_s010
                                 + 0.938047 w_k1_IBM_s010
                                 + 1.00791 w_k1_INTC_s010
                                 + 1.09586 w_k1_KO_s010 + 1.10171 w_k1_MAT_s010
                                 + 1.1257 w_k1_MSFT_s010
                                 + 0.899681 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.840711 w_k1_ADM_s010
                                 + 0.58035 w_k1_AXP_s010
                                 + 0.601725 w_k1_DIS_s010
                                 + 0.727885 w_k1_FDX_s010
                                 + 0.579111 w_k1_IBM_s010
                                 + 0.730625 w_k1_INTC_s010
                                 + 0.773442 w_k1_KO_s010
                                 + 0.898702 w_k1_MAT_s010
                                 + 0.769483 w_k1_MSFT_s010
                                 + 0.725928 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.20809 w_k1_ADM_s010
                                 + 1.3397 w_k1_AXP_s010 + 1.85362 w_k1_DIS_s010
                                 + 1.46438 w_k1_FDX_s010 + 1.0773 w_k1_IBM_s010
                                 + 1.26034 w_k1_INTC_s010
                                 + 1.30617 w_k1_KO_s010 + 1.60846 w_k1_MAT_s010
                                 + 1.38419 w_k1_MSFT_s010
                                 + 1.27557 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.27969 w_k1_ADM_s011
                                 + 1.17198 w_k1_AXP_s011 + 1.2082 w_k1_DIS_s011
                                 + 1.08176 w_k1_FDX_s011
                                 + 1.31664 w_k1_IBM_s011
                                 + 1.02897 w_k1_INTC_s011
                                 + 1.14217 w_k1_KO_s011 + 1.02662 w_k1_MAT_s011
                                 + 0.953331 w_k1_MSFT_s011
                                 + 1.08399 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.672174 w_k1_ADM_s011
                                 + 0.70462 w_k1_AXP_s011
                                 + 0.659749 w_k1_DIS_s011
                                 + 0.693392 w_k1_FDX_s011
                                 + 0.892441 w_k1_IBM_s011
                                 + 0.81505 w_k1_INTC_s011
                                 + 0.864986 w_k1_KO_s011
                                 + 0.741114 w_k1_MAT_s011
                                 + 0.726607 w_k1_MSFT_s011
                                 + 0.822529 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.813928 w_k1_ADM_s011
                                 + 0.723539 w_k1_AXP_s011
                                 + 0.820131 w_k1_DIS_s011
                                 + 0.86603 w_k1_FDX_s011
                                 + 0.577337 w_k1_IBM_s011
                                 + 0.779283 w_k1_INTC_s011
                                 + 0.742578 w_k1_KO_s011
                                 + 0.58822 w_k1_MAT_s011
                                 + 0.913265 w_k1_MSFT_s011
                                 + 0.819669 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.732767 w_k1_ADM_s011
                                 + 1.03173 w_k1_AXP_s011
                                 + 1.10988 w_k1_DIS_s011
                                 + 0.885313 w_k1_FDX_s011
                                 + 1.33899 w_k1_IBM_s011
                                 + 0.732717 w_k1_INTC_s011
                                 + 1.12433 w_k1_KO_s011 + 1.18648 w_k1_MAT_s011
                                 + 0.720318 w_k1_MSFT_s011
                                 + 1.00876 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.96367 w_k1_ADM_s011
                                 + 0.946839 w_k1_AXP_s011
                                 + 0.773711 w_k1_DIS_s011
                                 + 0.958618 w_k1_FDX_s011
                                 + 0.806089 w_k1_IBM_s011
                                 + 0.819207 w_k1_INTC_s011
                                 + 0.711999 w_k1_KO_s011
                                 + 0.685728 w_k1_MAT_s011
                                 + 0.774246 w_k1_MSFT_s011
                                 + 0.834588 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.02853 w_k1_ADM_s011
                                 + 0.935874 w_k1_AXP_s011
                                 + 1.19118 w_k1_DIS_s011
                                 + 1.08709 w_k1_FDX_s011
                                 + 0.973543 w_k1_IBM_s011
                                 + 0.9176 w_k1_INTC_s011 + 1.07959 w_k1_KO_s011
                                 + 1.02769 w_k1_MAT_s011
                                 + 0.960461 w_k1_MSFT_s011
                                 + 0.989833 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.49103 w_k1_ADM_s011
                                 + 1.20387 w_k1_AXP_s011
                                 + 1.78603 w_k1_DIS_s011
                                 + 1.31596 w_k1_FDX_s011
                                 + 1.19538 w_k1_IBM_s011
                                 + 1.03775 w_k1_INTC_s011
                                 + 1.24104 w_k1_KO_s011 + 1.33733 w_k1_MAT_s011
                                 + 1.21467 w_k1_MSFT_s011
                                 + 1.09705 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.24987 w_k1_ADM_s011
                                 + 1.33621 w_k1_AXP_s011
                                 + 1.36764 w_k1_DIS_s011
                                 + 1.24131 w_k1_FDX_s011
                                 + 0.862979 w_k1_IBM_s011
                                 + 0.877258 w_k1_INTC_s011
                                 + 1.02063 w_k1_KO_s011 + 1.07429 w_k1_MAT_s011
                                 + 1.04521 w_k1_MSFT_s011 + 1.102 w_k1_PG_s011
                                 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.887992 w_k1_ADM_s012
                                 + 0.544057 w_k1_AXP_s012
                                 + 0.53238 w_k1_DIS_s012 + 0.7814 w_k1_FDX_s012
                                 + 0.512112 w_k1_IBM_s012
                                 + 0.836614 w_k1_INTC_s012
                                 + 0.513743 w_k1_KO_s012
                                 + 0.403734 w_k1_MAT_s012
                                 + 0.746439 w_k1_MSFT_s012
                                 + 0.749972 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.50241 w_k1_ADM_s012
                                 + 1.17997 w_k1_AXP_s012
                                 + 1.43846 w_k1_DIS_s012
                                 + 1.20642 w_k1_FDX_s012
                                 + 1.16032 w_k1_IBM_s012
                                 + 1.13574 w_k1_INTC_s012
                                 + 1.12205 w_k1_KO_s012
                                 + 0.817714 w_k1_MAT_s012
                                 + 1.24592 w_k1_MSFT_s012
                                 + 1.14752 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.3177 w_k1_ADM_s012
                                 + 1.39395 w_k1_AXP_s012
                                 + 1.70048 w_k1_DIS_s012
                                 + 1.39946 w_k1_FDX_s012
                                 + 1.62825 w_k1_IBM_s012
                                 + 1.26647 w_k1_INTC_s012
                                 + 1.36709 w_k1_KO_s012 + 1.44883 w_k1_MAT_s012
                                 + 1.29951 w_k1_MSFT_s012 + 1.1763 w_k1_PG_s012
                                 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.01212 w_k1_ADM_s012
                                 + 1.04562 w_k1_AXP_s012
                                 + 1.19864 w_k1_DIS_s012
                                 + 1.08296 w_k1_FDX_s012
                                 + 1.38491 w_k1_IBM_s012
                                 + 1.20052 w_k1_INTC_s012
                                 + 1.21198 w_k1_KO_s012
                                 + 0.942524 w_k1_MAT_s012
                                 + 1.16141 w_k1_MSFT_s012
                                 + 1.15526 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.799116 w_k1_ADM_s012
                                 + 0.614081 w_k1_AXP_s012
                                 + 0.653027 w_k1_DIS_s012
                                 + 0.777385 w_k1_FDX_s012
                                 + 0.711679 w_k1_IBM_s012
                                 + 0.702427 w_k1_INTC_s012
                                 + 0.778518 w_k1_KO_s012
                                 + 0.766588 w_k1_MAT_s012
                                 + 0.864672 w_k1_MSFT_s012
                                 + 0.84027 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.27238 w_k1_ADM_s012
                                 + 1.23681 w_k1_AXP_s012
                                 + 1.28166 w_k1_DIS_s012
                                 + 1.29527 w_k1_FDX_s012
                                 + 0.719697 w_k1_IBM_s012
                                 + 1.18374 w_k1_INTC_s012
                                 + 1.00922 w_k1_KO_s012
                                 + 0.914813 w_k1_MAT_s012
                                 + 1.2278 w_k1_MSFT_s012 + 1.0722 w_k1_PG_s012
                                 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.786233 w_k1_ADM_s012
                                 + 0.806703 w_k1_AXP_s012
                                 + 0.736242 w_k1_DIS_s012
                                 + 0.821856 w_k1_FDX_s012
                                 + 1.14967 w_k1_IBM_s012
                                 + 0.945297 w_k1_INTC_s012
                                 + 0.915986 w_k1_KO_s012
                                 + 1.04927 w_k1_MAT_s012
                                 + 0.81108 w_k1_MSFT_s012
                                 + 0.90442 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.989698 w_k1_ADM_s012
                                 + 1.06284 w_k1_AXP_s012
                                 + 0.899421 w_k1_DIS_s012
                                 + 0.996099 w_k1_FDX_s012
                                 + 0.981901 w_k1_IBM_s012
                                 + 1.23403 w_k1_INTC_s012
                                 + 0.919284 w_k1_KO_s012
                                 + 0.694866 w_k1_MAT_s012
                                 + 0.950066 w_k1_MSFT_s012
                                 + 0.960785 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.3537 w_k1_ADM_s013
                                 + 1.39736 w_k1_AXP_s013
                                 + 1.65259 w_k1_DIS_s013
                                 + 1.30301 w_k1_FDX_s013
                                 + 1.18502 w_k1_IBM_s013
                                 + 1.13697 w_k1_INTC_s013
                                 + 1.27904 w_k1_KO_s013 + 2.477 w_k1_MAT_s013
                                 + 1.22171 w_k1_MSFT_s013
                                 + 1.16095 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.05585 w_k1_ADM_s013
                                 + 1.34145 w_k1_AXP_s013
                                 + 1.05481 w_k1_DIS_s013
                                 + 1.14851 w_k1_FDX_s013
                                 + 0.785292 w_k1_IBM_s013
                                 + 1.25586 w_k1_INTC_s013
                                 + 1.00143 w_k1_KO_s013 + 1.05083 w_k1_MAT_s013
                                 + 1.06158 w_k1_MSFT_s013
                                 + 1.07193 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.2098 w_k1_ADM_s013
                                 + 1.09519 w_k1_AXP_s013
                                 + 1.01581 w_k1_DIS_s013
                                 + 1.00182 w_k1_FDX_s013 + 1.032 w_k1_IBM_s013
                                 + 1.07894 w_k1_INTC_s013
                                 + 0.995874 w_k1_KO_s013
                                 + 1.17071 w_k1_MAT_s013
                                 + 0.914507 w_k1_MSFT_s013
                                 + 1.06296 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.756033 w_k1_ADM_s013
                                 + 0.695935 w_k1_AXP_s013
                                 + 0.710044 w_k1_DIS_s013
                                 + 0.800862 w_k1_FDX_s013
                                 + 0.708723 w_k1_IBM_s013
                                 + 0.805834 w_k1_INTC_s013
                                 + 0.670106 w_k1_KO_s013
                                 + 0.744529 w_k1_MAT_s013
                                 + 0.829174 w_k1_MSFT_s013
                                 + 0.826161 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.00378 w_k1_ADM_s013
                                 + 1.4583 w_k1_AXP_s013 + 1.5473 w_k1_DIS_s013
                                 + 1.14013 w_k1_FDX_s013
                                 + 1.58499 w_k1_IBM_s013
                                 + 1.25494 w_k1_INTC_s013
                                 + 1.44036 w_k1_KO_s013 + 2.13056 w_k1_MAT_s013
                                 + 1.15497 w_k1_MSFT_s013
                                 + 1.39518 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.727807 w_k1_ADM_s013
                                 + 0.729932 w_k1_AXP_s013
                                 + 0.670343 w_k1_DIS_s013
                                 + 0.738142 w_k1_FDX_s013
                                 + 1.04776 w_k1_IBM_s013
                                 + 0.7483 w_k1_INTC_s013
                                 + 0.875795 w_k1_KO_s013
                                 + 1.04255 w_k1_MAT_s013
                                 + 0.763042 w_k1_MSFT_s013
                                 + 0.952076 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.03065 w_k1_ADM_s013
                                 + 1.18221 w_k1_AXP_s013
                                 + 1.26676 w_k1_DIS_s013
                                 + 1.03826 w_k1_FDX_s013
                                 + 1.17482 w_k1_IBM_s013
                                 + 1.13752 w_k1_INTC_s013
                                 + 1.20831 w_k1_KO_s013 + 2.20572 w_k1_MAT_s013
                                 + 1.13832 w_k1_MSFT_s013
                                 + 1.12339 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.884935 w_k1_ADM_s013
                                 + 1.07577 w_k1_AXP_s013
                                 + 0.906037 w_k1_DIS_s013
                                 + 0.850795 w_k1_FDX_s013
                                 + 1.1485 w_k1_IBM_s013
                                 + 1.06118 w_k1_INTC_s013
                                 + 1.01492 w_k1_KO_s013 + 1.14666 w_k1_MAT_s013
                                 + 0.882511 w_k1_MSFT_s013
                                 + 0.966505 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.07808 w_k1_ADM_s014
                                 + 0.849715 w_k1_AXP_s014
                                 + 0.834907 w_k1_DIS_s014
                                 + 0.950023 w_k1_FDX_s014
                                 + 0.87157 w_k1_IBM_s014
                                 + 0.809337 w_k1_INTC_s014
                                 + 0.961331 w_k1_KO_s014
                                 + 0.750019 w_k1_MAT_s014
                                 + 0.937822 w_k1_MSFT_s014
                                 + 0.935607 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.18258 w_k1_ADM_s014
                                 + 0.858963 w_k1_AXP_s014
                                 + 0.991727 w_k1_DIS_s014
                                 + 1.07437 w_k1_FDX_s014
                                 + 0.922343 w_k1_IBM_s014
                                 + 1.09337 w_k1_INTC_s014
                                 + 1.04623 w_k1_KO_s014
                                 + 0.472315 w_k1_MAT_s014
                                 + 1.05087 w_k1_MSFT_s014
                                 + 0.821613 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.825467 w_k1_ADM_s014
                                 + 0.664589 w_k1_AXP_s014
                                 + 0.577271 w_k1_DIS_s014
                                 + 0.678694 w_k1_FDX_s014
                                 + 0.765634 w_k1_IBM_s014
                                 + 0.785746 w_k1_INTC_s014
                                 + 0.724252 w_k1_KO_s014
                                 + 0.517157 w_k1_MAT_s014
                                 + 0.710437 w_k1_MSFT_s014
                                 + 0.785848 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.963878 w_k1_ADM_s014
                                 + 1.2533 w_k1_AXP_s014 + 1.04036 w_k1_DIS_s014
                                 + 1.11986 w_k1_FDX_s014
                                 + 1.05722 w_k1_IBM_s014
                                 + 0.846206 w_k1_INTC_s014
                                 + 1.11628 w_k1_KO_s014 + 1.07221 w_k1_MAT_s014
                                 + 0.984449 w_k1_MSFT_s014
                                 + 1.0627 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.968487 w_k1_ADM_s014
                                 + 0.936892 w_k1_AXP_s014
                                 + 0.798771 w_k1_DIS_s014
                                 + 0.798386 w_k1_FDX_s014
                                 + 1.2919 w_k1_IBM_s014
                                 + 0.743094 w_k1_INTC_s014
                                 + 1.16176 w_k1_KO_s014 + 1.42412 w_k1_MAT_s014
                                 + 0.754732 w_k1_MSFT_s014
                                 + 0.953958 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.21004 w_k1_ADM_s014
                                 + 1.08477 w_k1_AXP_s014
                                 + 1.28744 w_k1_DIS_s014
                                 + 1.19375 w_k1_FDX_s014
                                 + 1.02385 w_k1_IBM_s014
                                 + 0.910164 w_k1_INTC_s014
                                 + 1.22452 w_k1_KO_s014
                                 + 0.791629 w_k1_MAT_s014
                                 + 1.0821 w_k1_MSFT_s014 + 1.11164 w_k1_PG_s014
                                 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.05991 w_k1_ADM_s014
                                 + 1.21234 w_k1_AXP_s014
                                 + 1.49621 w_k1_DIS_s014
                                 + 1.18464 w_k1_FDX_s014
                                 + 1.64088 w_k1_IBM_s014
                                 + 1.30781 w_k1_INTC_s014
                                 + 1.34947 w_k1_KO_s014
                                 + 0.965733 w_k1_MAT_s014
                                 + 1.08022 w_k1_MSFT_s014
                                 + 1.11988 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.828779 w_k1_ADM_s014
                                 + 0.763535 w_k1_AXP_s014
                                 + 0.678763 w_k1_DIS_s014
                                 + 0.84215 w_k1_FDX_s014
                                 + 0.768031 w_k1_IBM_s014
                                 + 0.774724 w_k1_INTC_s014
                                 + 0.851768 w_k1_KO_s014
                                 + 0.692655 w_k1_MAT_s014
                                 + 0.745053 w_k1_MSFT_s014
                                 + 0.889465 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.815764 w_k1_ADM_s015
                                 + 0.813819 w_k1_AXP_s015
                                 + 0.833609 w_k1_DIS_s015
                                 + 0.845878 w_k1_FDX_s015
                                 + 0.824718 w_k1_IBM_s015
                                 + 0.835547 w_k1_INTC_s015
                                 + 0.868627 w_k1_KO_s015
                                 + 0.715913 w_k1_MAT_s015
                                 + 0.808841 w_k1_MSFT_s015
                                 + 0.959179 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.937187 w_k1_ADM_s015
                                 + 1.06448 w_k1_AXP_s015
                                 + 1.37864 w_k1_DIS_s015
                                 + 1.19769 w_k1_FDX_s015
                                 + 0.936216 w_k1_IBM_s015
                                 + 1.38413 w_k1_INTC_s015
                                 + 1.07416 w_k1_KO_s015
                                 + 0.700589 w_k1_MAT_s015
                                 + 1.20885 w_k1_MSFT_s015
                                 + 1.06453 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.30729 w_k1_ADM_s015
                                 + 1.57321 w_k1_AXP_s015
                                 + 1.98069 w_k1_DIS_s015 + 1.468 w_k1_FDX_s015
                                 + 1.53463 w_k1_IBM_s015 + 1.397 w_k1_INTC_s015
                                 + 1.60552 w_k1_KO_s015 + 1.41459 w_k1_MAT_s015
                                 + 1.415 w_k1_MSFT_s015 + 1.5435 w_k1_PG_s015
                                 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.860082 w_k1_ADM_s015
                                 + 0.803271 w_k1_AXP_s015
                                 + 0.64112 w_k1_DIS_s015
                                 + 0.742219 w_k1_FDX_s015
                                 + 0.783492 w_k1_IBM_s015
                                 + 0.777823 w_k1_INTC_s015
                                 + 0.673918 w_k1_KO_s015
                                 + 0.661671 w_k1_MAT_s015
                                 + 0.745169 w_k1_MSFT_s015
                                 + 0.812316 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.903517 w_k1_ADM_s015
                                 + 1.03955 w_k1_AXP_s015
                                 + 0.926539 w_k1_DIS_s015
                                 + 0.847671 w_k1_FDX_s015
                                 + 1.10695 w_k1_IBM_s015
                                 + 0.885252 w_k1_INTC_s015
                                 + 1.0591 w_k1_KO_s015 + 1.19153 w_k1_MAT_s015
                                 + 0.935594 w_k1_MSFT_s015
                                 + 1.03884 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.07021 w_k1_ADM_s015
                                 + 1.18939 w_k1_AXP_s015
                                 + 1.01323 w_k1_DIS_s015
                                 + 1.09317 w_k1_FDX_s015
                                 + 1.19431 w_k1_IBM_s015
                                 + 1.17132 w_k1_INTC_s015
                                 + 1.03043 w_k1_KO_s015 + 1.264 w_k1_MAT_s015
                                 + 1.0144 w_k1_MSFT_s015 + 1.05406 w_k1_PG_s015
                                 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.53917 w_k1_ADM_s015
                                 + 1.58809 w_k1_AXP_s015
                                 + 1.68366 w_k1_DIS_s015
                                 + 1.43009 w_k1_FDX_s015
                                 + 1.18198 w_k1_IBM_s015
                                 + 1.40212 w_k1_INTC_s015
                                 + 1.29478 w_k1_KO_s015 + 1.32916 w_k1_MAT_s015
                                 + 1.45126 w_k1_MSFT_s015
                                 + 1.11524 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.871294 w_k1_ADM_s015
                                 + 1.21147 w_k1_AXP_s015
                                 + 1.32514 w_k1_DIS_s015
                                 + 1.03893 w_k1_FDX_s015
                                 + 1.49784 w_k1_IBM_s015
                                 + 1.18126 w_k1_INTC_s015 + 1.2696 w_k1_KO_s015
                                 + 1.62414 w_k1_MAT_s015
                                 + 1.29905 w_k1_MSFT_s015
                                 + 1.16371 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.823599 w_k1_ADM_s016
                                 + 0.860507 w_k1_AXP_s016
                                 + 0.619558 w_k1_DIS_s016
                                 + 0.777849 w_k1_FDX_s016
                                 + 0.653154 w_k1_IBM_s016
                                 + 0.745081 w_k1_INTC_s016
                                 + 0.663126 w_k1_KO_s016
                                 + 0.513407 w_k1_MAT_s016
                                 + 0.763385 w_k1_MSFT_s016
                                 + 0.781334 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.817717 w_k1_ADM_s016
                                 + 0.880801 w_k1_AXP_s016
                                 + 0.783706 w_k1_DIS_s016
                                 + 0.770224 w_k1_FDX_s016
                                 + 0.961977 w_k1_IBM_s016
                                 + 0.867875 w_k1_INTC_s016
                                 + 0.787911 w_k1_KO_s016
                                 + 0.822682 w_k1_MAT_s016
                                 + 0.862944 w_k1_MSFT_s016
                                 + 0.913366 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.06535 w_k1_ADM_s016
                                 + 1.0547 w_k1_AXP_s016 + 1.23823 w_k1_DIS_s016
                                 + 1.16353 w_k1_FDX_s016
                                 + 0.803023 w_k1_IBM_s016
                                 + 1.08339 w_k1_INTC_s016
                                 + 1.00237 w_k1_KO_s016
                                 + 0.669349 w_k1_MAT_s016
                                 + 1.24701 w_k1_MSFT_s016
                                 + 0.991628 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.13257 w_k1_ADM_s016
                                 + 1.46141 w_k1_AXP_s016
                                 + 1.52117 w_k1_DIS_s016
                                 + 1.23711 w_k1_FDX_s016 + 1.4897 w_k1_IBM_s016
                                 + 1.27999 w_k1_INTC_s016
                                 + 1.22427 w_k1_KO_s016 + 1.29857 w_k1_MAT_s016
                                 + 1.29154 w_k1_MSFT_s016
                                 + 1.12389 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.958411 w_k1_ADM_s016
                                 + 1.02554 w_k1_AXP_s016
                                 + 0.832312 w_k1_DIS_s016
                                 + 1.00388 w_k1_FDX_s016
                                 + 0.893768 w_k1_IBM_s016
                                 + 1.10831 w_k1_INTC_s016
                                 + 0.88854 w_k1_KO_s016
                                 + 0.643566 w_k1_MAT_s016
                                 + 1.10124 w_k1_MSFT_s016
                                 + 0.916684 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.788128 w_k1_ADM_s016
                                 + 0.914034 w_k1_AXP_s016
                                 + 1.04111 w_k1_DIS_s016
                                 + 0.84511 w_k1_FDX_s016
                                 + 1.36726 w_k1_IBM_s016
                                 + 0.936103 w_k1_INTC_s016
                                 + 1.03032 w_k1_KO_s016 + 1.14403 w_k1_MAT_s016
                                 + 1.04315 w_k1_MSFT_s016
                                 + 0.994124 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.642506 w_k1_ADM_s016
                                 + 0.555225 w_k1_AXP_s016
                                 + 0.544282 w_k1_DIS_s016
                                 + 0.749939 w_k1_FDX_s016
                                 + 0.548694 w_k1_IBM_s016
                                 + 0.796502 w_k1_INTC_s016
                                 + 0.590094 w_k1_KO_s016
                                 + 0.425771 w_k1_MAT_s016
                                 + 0.732991 w_k1_MSFT_s016
                                 + 0.718198 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.905173 w_k1_ADM_s016
                                 + 0.967683 w_k1_AXP_s016
                                 + 1.32775 w_k1_DIS_s016
                                 + 1.05875 w_k1_FDX_s016
                                 + 1.31174 w_k1_IBM_s016
                                 + 1.24018 w_k1_INTC_s016
                                 + 1.12217 w_k1_KO_s016 + 1.20134 w_k1_MAT_s016
                                 + 1.20353 w_k1_MSFT_s016
                                 + 1.14535 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.08419 w_k1_ADM_s017
                                 + 1.36767 w_k1_AXP_s017
                                 + 1.75308 w_k1_DIS_s017
                                 + 1.29485 w_k1_FDX_s017
                                 + 1.54559 w_k1_IBM_s017
                                 + 1.22704 w_k1_INTC_s017
                                 + 1.34658 w_k1_KO_s017 + 2.10528 w_k1_MAT_s017
                                 + 1.1764 w_k1_MSFT_s017 + 1.23321 w_k1_PG_s017
                                 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.05298 w_k1_ADM_s017
                                 + 1.28372 w_k1_AXP_s017
                                 + 1.27477 w_k1_DIS_s017
                                 + 1.27339 w_k1_FDX_s017
                                 + 1.06935 w_k1_IBM_s017
                                 + 1.19536 w_k1_INTC_s017
                                 + 1.19335 w_k1_KO_s017 + 1.24775 w_k1_MAT_s017
                                 + 1.01544 w_k1_MSFT_s017
                                 + 1.20528 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.978091 w_k1_ADM_s017
                                 + 1.16207 w_k1_AXP_s017
                                 + 1.07048 w_k1_DIS_s017
                                 + 1.06011 w_k1_FDX_s017
                                 + 0.967425 w_k1_IBM_s017
                                 + 0.970809 w_k1_INTC_s017
                                 + 1.06874 w_k1_KO_s017 + 1.86267 w_k1_MAT_s017
                                 + 1.00854 w_k1_MSFT_s017
                                 + 1.04234 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.03846 w_k1_ADM_s017
                                 + 0.88151 w_k1_AXP_s017
                                 + 1.28209 w_k1_DIS_s017
                                 + 1.10645 w_k1_FDX_s017
                                 + 0.859339 w_k1_IBM_s017
                                 + 1.10867 w_k1_INTC_s017
                                 + 1.20992 w_k1_KO_s017 + 1.46787 w_k1_MAT_s017
                                 + 1.11535 w_k1_MSFT_s017 + 1.0709 w_k1_PG_s017
                                 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.239 w_k1_ADM_s017
                                 + 1.3624 w_k1_AXP_s017 + 1.17665 w_k1_DIS_s017
                                 + 1.07979 w_k1_FDX_s017
                                 + 0.965788 w_k1_IBM_s017
                                 + 1.08638 w_k1_INTC_s017
                                 + 1.19431 w_k1_KO_s017 + 1.49369 w_k1_MAT_s017
                                 + 0.989172 w_k1_MSFT_s017
                                 + 1.15099 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.911892 w_k1_ADM_s017
                                 + 1.185 w_k1_AXP_s017 + 0.914127 w_k1_DIS_s017
                                 + 0.995055 w_k1_FDX_s017
                                 + 1.45887 w_k1_IBM_s017
                                 + 1.10685 w_k1_INTC_s017
                                 + 1.32198 w_k1_KO_s017 + 2.09806 w_k1_MAT_s017
                                 + 0.971613 w_k1_MSFT_s017
                                 + 1.11391 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.74998 w_k1_ADM_s017
                                 + 0.765516 w_k1_AXP_s017
                                 + 0.693285 w_k1_DIS_s017
                                 + 0.767005 w_k1_FDX_s017
                                 + 0.865595 w_k1_IBM_s017
                                 + 0.937554 w_k1_INTC_s017
                                 + 0.873509 w_k1_KO_s017
                                 + 1.07303 w_k1_MAT_s017
                                 + 0.782104 w_k1_MSFT_s017
                                 + 0.867926 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.959331 w_k1_ADM_s017
                                 + 0.915582 w_k1_AXP_s017
                                 + 0.811757 w_k1_DIS_s017
                                 + 0.956048 w_k1_FDX_s017
                                 + 0.835013 w_k1_IBM_s017
                                 + 0.966198 w_k1_INTC_s017
                                 + 0.972261 w_k1_KO_s017
                                 + 1.00316 w_k1_MAT_s017
                                 + 0.85886 w_k1_MSFT_s017
                                 + 1.01577 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.11139 w_k1_ADM_s100
                                 + 0.975599 w_k1_AXP_s100
                                 + 1.00074 w_k1_DIS_s100
                                 + 1.02956 w_k1_FDX_s100
                                 + 0.648119 w_k1_IBM_s100
                                 + 1.0849 w_k1_INTC_s100
                                 + 0.926579 w_k1_KO_s100
                                 + 0.748465 w_k1_MAT_s100
                                 + 0.998387 w_k1_MSFT_s100
                                 + 1.08294 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.731909 w_k1_ADM_s100
                                 + 0.798966 w_k1_AXP_s100
                                 + 0.848654 w_k1_DIS_s100
                                 + 0.812621 w_k1_FDX_s100
                                 + 1.35234 w_k1_IBM_s100
                                 + 1.02708 w_k1_INTC_s100
                                 + 1.07107 w_k1_KO_s100 + 1.30145 w_k1_MAT_s100
                                 + 0.833709 w_k1_MSFT_s100
                                 + 0.906867 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.992941 w_k1_ADM_s100
                                 + 1.2967 w_k1_AXP_s100 + 1.25859 w_k1_DIS_s100
                                 + 1.1559 w_k1_FDX_s100 + 1.28222 w_k1_IBM_s100
                                 + 1.16106 w_k1_INTC_s100
                                 + 1.38643 w_k1_KO_s100 + 1.73375 w_k1_MAT_s100
                                 + 1.219 w_k1_MSFT_s100 + 1.22694 w_k1_PG_s100
                                 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.997688 w_k1_ADM_s100
                                 + 0.632365 w_k1_AXP_s100
                                 + 0.705766 w_k1_DIS_s100
                                 + 0.770748 w_k1_FDX_s100
                                 + 0.830857 w_k1_IBM_s100
                                 + 0.862187 w_k1_INTC_s100
                                 + 0.901091 w_k1_KO_s100
                                 + 0.840197 w_k1_MAT_s100
                                 + 0.859999 w_k1_MSFT_s100
                                 + 0.991054 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.27127 w_k1_ADM_s100
                                 + 1.09564 w_k1_AXP_s100
                                 + 0.92588 w_k1_DIS_s100
                                 + 1.00646 w_k1_FDX_s100
                                 + 0.98368 w_k1_IBM_s100
                                 + 0.911601 w_k1_INTC_s100
                                 + 1.13504 w_k1_KO_s100 + 1.30284 w_k1_MAT_s100
                                 + 0.939905 w_k1_MSFT_s100
                                 + 1.1509 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.56601 w_k1_ADM_s100
                                 + 1.49133 w_k1_AXP_s100
                                 + 1.77409 w_k1_DIS_s100
                                 + 1.57732 w_k1_FDX_s100
                                 + 0.951461 w_k1_IBM_s100
                                 + 1.65504 w_k1_INTC_s100
                                 + 1.49982 w_k1_KO_s100 + 1.18342 w_k1_MAT_s100
                                 + 1.61437 w_k1_MSFT_s100
                                 + 1.40428 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.10263 w_k1_ADM_s100
                                 + 0.826205 w_k1_AXP_s100
                                 + 1.20113 w_k1_DIS_s100
                                 + 0.95624 w_k1_FDX_s100
                                 + 1.34008 w_k1_IBM_s100
                                 + 1.0778 w_k1_INTC_s100 + 1.34601 w_k1_KO_s100
                                 + 1.04379 w_k1_MAT_s100
                                 + 0.995702 w_k1_MSFT_s100
                                 + 1.19006 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.14347 w_k1_ADM_s100
                                 + 1.32708 w_k1_AXP_s100
                                 + 1.58065 w_k1_DIS_s100
                                 + 1.21746 w_k1_FDX_s100
                                 + 1.92175 w_k1_IBM_s100
                                 + 1.35961 w_k1_INTC_s100
                                 + 1.59004 w_k1_KO_s100 + 1.84749 w_k1_MAT_s100
                                 + 1.09541 w_k1_MSFT_s100
                                 + 1.30646 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.903789 w_k1_ADM_s101
                                 + 0.906913 w_k1_AXP_s101
                                 + 0.787944 w_k1_DIS_s101
                                 + 0.880185 w_k1_FDX_s101
                                 + 0.863358 w_k1_IBM_s101
                                 + 0.97876 w_k1_INTC_s101
                                 + 0.904512 w_k1_KO_s101
                                 + 0.841479 w_k1_MAT_s101
                                 + 1.00138 w_k1_MSFT_s101
                                 + 0.858363 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.863493 w_k1_ADM_s101
                                 + 0.900011 w_k1_AXP_s101
                                 + 0.89487 w_k1_DIS_s101
                                 + 0.908446 w_k1_FDX_s101
                                 + 1.17692 w_k1_IBM_s101
                                 + 0.904783 w_k1_INTC_s101
                                 + 1.07462 w_k1_KO_s101 + 1.30228 w_k1_MAT_s101
                                 + 1.00817 w_k1_MSFT_s101 + 0.9669 w_k1_PG_s101
                                 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.15306 w_k1_ADM_s101
                                 + 1.15489 w_k1_AXP_s101
                                 + 1.27683 w_k1_DIS_s101
                                 + 1.14252 w_k1_FDX_s101
                                 + 1.07596 w_k1_IBM_s101
                                 + 1.11657 w_k1_INTC_s101 + 1.0521 w_k1_KO_s101
                                 + 1.07688 w_k1_MAT_s101
                                 + 1.2017 w_k1_MSFT_s101 + 1.05163 w_k1_PG_s101
                                 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.908388 w_k1_ADM_s101
                                 + 0.610204 w_k1_AXP_s101
                                 + 0.78748 w_k1_DIS_s101
                                 + 0.755198 w_k1_FDX_s101
                                 + 0.913007 w_k1_IBM_s101
                                 + 0.92313 w_k1_INTC_s101
                                 + 0.872816 w_k1_KO_s101
                                 + 0.850233 w_k1_MAT_s101
                                 + 0.794647 w_k1_MSFT_s101
                                 + 0.848253 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.44676 w_k1_ADM_s101
                                 + 1.21628 w_k1_AXP_s101
                                 + 1.32667 w_k1_DIS_s101 + 1.2428 w_k1_FDX_s101
                                 + 1.36248 w_k1_IBM_s101
                                 + 1.04519 w_k1_INTC_s101
                                 + 1.45225 w_k1_KO_s101 + 1.65583 w_k1_MAT_s101
                                 + 1.20546 w_k1_MSFT_s101
                                 + 1.20779 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.4019 w_k1_ADM_s101
                                 + 1.18253 w_k1_AXP_s101 + 1.3378 w_k1_DIS_s101
                                 + 1.28904 w_k1_FDX_s101
                                 + 0.890533 w_k1_IBM_s101
                                 + 1.09467 w_k1_INTC_s101
                                 + 1.07193 w_k1_KO_s101 + 1.29734 w_k1_MAT_s101
                                 + 1.22409 w_k1_MSFT_s101
                                 + 1.17355 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.10953 w_k1_ADM_s101
                                 + 0.954979 w_k1_AXP_s101
                                 + 1.02645 w_k1_DIS_s101
                                 + 0.919085 w_k1_FDX_s101
                                 + 1.70598 w_k1_IBM_s101
                                 + 0.879095 w_k1_INTC_s101
                                 + 1.23815 w_k1_KO_s101 + 1.72076 w_k1_MAT_s101
                                 + 0.896163 w_k1_MSFT_s101
                                 + 0.99615 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.06053 w_k1_ADM_s101
                                 + 0.77625 w_k1_AXP_s101
                                 + 0.854263 w_k1_DIS_s101
                                 + 0.960613 w_k1_FDX_s101
                                 + 0.715713 w_k1_IBM_s101
                                 + 0.979117 w_k1_INTC_s101
                                 + 0.914774 w_k1_KO_s101
                                 + 0.756006 w_k1_MAT_s101
                                 + 1.03679 w_k1_MSFT_s101
                                 + 0.940745 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.945485 w_k1_ADM_s102
                                 + 0.964132 w_k1_AXP_s102
                                 + 0.934811 w_k1_DIS_s102
                                 + 1.01479 w_k1_FDX_s102
                                 + 0.738635 w_k1_IBM_s102
                                 + 0.983875 w_k1_INTC_s102
                                 + 0.846363 w_k1_KO_s102
                                 + 0.817747 w_k1_MAT_s102
                                 + 0.867741 w_k1_MSFT_s102
                                 + 0.900762 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.913873 w_k1_ADM_s102
                                 + 0.646906 w_k1_AXP_s102
                                 + 0.595986 w_k1_DIS_s102
                                 + 0.73074 w_k1_FDX_s102
                                 + 0.744843 w_k1_IBM_s102
                                 + 0.676828 w_k1_INTC_s102
                                 + 0.699947 w_k1_KO_s102
                                 + 0.703251 w_k1_MAT_s102
                                 + 0.716434 w_k1_MSFT_s102
                                 + 0.749745 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.06238 w_k1_ADM_s102
                                 + 0.795426 w_k1_AXP_s102
                                 + 0.832918 w_k1_DIS_s102
                                 + 0.849563 w_k1_FDX_s102
                                 + 0.693258 w_k1_IBM_s102
                                 + 0.703754 w_k1_INTC_s102
                                 + 0.889142 w_k1_KO_s102
                                 + 0.847092 w_k1_MAT_s102
                                 + 0.87584 w_k1_MSFT_s102
                                 + 0.907418 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.08181 w_k1_ADM_s102
                                 + 1.58449 w_k1_AXP_s102
                                 + 1.67877 w_k1_DIS_s102
                                 + 1.24659 w_k1_FDX_s102
                                 + 1.48343 w_k1_IBM_s102
                                 + 1.07719 w_k1_INTC_s102
                                 + 1.52578 w_k1_KO_s102 + 1.93904 w_k1_MAT_s102
                                 + 1.20688 w_k1_MSFT_s102
                                 + 1.10385 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.23707 w_k1_ADM_s102
                                 + 1.23673 w_k1_AXP_s102
                                 + 1.31038 w_k1_DIS_s102
                                 + 1.21786 w_k1_FDX_s102
                                 + 1.00455 w_k1_IBM_s102
                                 + 0.961261 w_k1_INTC_s102
                                 + 1.10604 w_k1_KO_s102 + 1.25643 w_k1_MAT_s102
                                 + 1.10423 w_k1_MSFT_s102
                                 + 0.987401 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.29288 w_k1_ADM_s102
                                 + 1.50171 w_k1_AXP_s102
                                 + 1.36952 w_k1_DIS_s102
                                 + 1.19866 w_k1_FDX_s102
                                 + 0.87376 w_k1_IBM_s102
                                 + 0.954586 w_k1_INTC_s102
                                 + 1.13622 w_k1_KO_s102 + 1.17246 w_k1_MAT_s102
                                 + 1.04112 w_k1_MSFT_s102
                                 + 1.05104 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.04573 w_k1_ADM_s102
                                 + 0.927577 w_k1_AXP_s102
                                 + 0.909408 w_k1_DIS_s102
                                 + 0.925898 w_k1_FDX_s102
                                 + 0.983542 w_k1_IBM_s102
                                 + 0.7946 w_k1_INTC_s102
                                 + 0.875727 w_k1_KO_s102
                                 + 0.97929 w_k1_MAT_s102
                                 + 0.870181 w_k1_MSFT_s102
                                 + 0.930581 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.92721 w_k1_ADM_s102
                                 + 1.04669 w_k1_AXP_s102
                                 + 1.10452 w_k1_DIS_s102
                                 + 0.904602 w_k1_FDX_s102
                                 + 1.3831 w_k1_IBM_s102
                                 + 0.813826 w_k1_INTC_s102
                                 + 1.04317 w_k1_KO_s102 + 1.48257 w_k1_MAT_s102
                                 + 0.838425 w_k1_MSFT_s102
                                 + 0.903319 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.832556 w_k1_ADM_s103
                                 + 0.937028 w_k1_AXP_s103
                                 + 0.98712 w_k1_DIS_s103
                                 + 0.882558 w_k1_FDX_s103
                                 + 1.14885 w_k1_IBM_s103
                                 + 0.954925 w_k1_INTC_s103
                                 + 1.03793 w_k1_KO_s103 + 1.16868 w_k1_MAT_s103
                                 + 1.06616 w_k1_MSFT_s103
                                 + 1.05653 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.970975 w_k1_ADM_s103
                                 + 1.30598 w_k1_AXP_s103
                                 + 1.19984 w_k1_DIS_s103
                                 + 1.24475 w_k1_FDX_s103
                                 + 0.870309 w_k1_IBM_s103
                                 + 1.27269 w_k1_INTC_s103
                                 + 1.03501 w_k1_KO_s103 + 1.29535 w_k1_MAT_s103
                                 + 1.12038 w_k1_MSFT_s103
                                 + 1.12447 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.10028 w_k1_ADM_s103
                                 + 1.28118 w_k1_AXP_s103
                                 + 1.58787 w_k1_DIS_s103
                                 + 1.08583 w_k1_FDX_s103
                                 + 1.62438 w_k1_IBM_s103
                                 + 0.993858 w_k1_INTC_s103
                                 + 1.48949 w_k1_KO_s103 + 2.04452 w_k1_MAT_s103
                                 + 1.14881 w_k1_MSFT_s103
                                 + 1.11442 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.891783 w_k1_ADM_s103
                                 + 0.761726 w_k1_AXP_s103
                                 + 0.985719 w_k1_DIS_s103
                                 + 0.975312 w_k1_FDX_s103
                                 + 0.69446 w_k1_IBM_s103
                                 + 1.02858 w_k1_INTC_s103
                                 + 0.94964 w_k1_KO_s103
                                 + 0.523886 w_k1_MAT_s103
                                 + 1.08929 w_k1_MSFT_s103
                                 + 0.92905 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.920744 w_k1_ADM_s103
                                 + 0.815227 w_k1_AXP_s103
                                 + 0.846797 w_k1_DIS_s103
                                 + 0.859597 w_k1_FDX_s103
                                 + 0.835828 w_k1_IBM_s103
                                 + 1.13424 w_k1_INTC_s103
                                 + 0.997191 w_k1_KO_s103
                                 + 0.889708 w_k1_MAT_s103
                                 + 0.998157 w_k1_MSFT_s103
                                 + 0.894781 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.558044 w_k1_ADM_s103
                                 + 0.835055 w_k1_AXP_s103
                                 + 0.685787 w_k1_DIS_s103
                                 + 0.718086 w_k1_FDX_s103
                                 + 1.02488 w_k1_IBM_s103
                                 + 0.935327 w_k1_INTC_s103
                                 + 0.945828 w_k1_KO_s103
                                 + 1.33431 w_k1_MAT_s103
                                 + 0.809792 w_k1_MSFT_s103
                                 + 0.899133 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.00139 w_k1_ADM_s103
                                 + 1.23412 w_k1_AXP_s103
                                 + 1.38661 w_k1_DIS_s103
                                 + 1.26567 w_k1_FDX_s103 + 1.1106 w_k1_IBM_s103
                                 + 1.30294 w_k1_INTC_s103
                                 + 1.31687 w_k1_KO_s103 + 1.43397 w_k1_MAT_s103
                                 + 1.37447 w_k1_MSFT_s103
                                 + 1.07831 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.667457 w_k1_ADM_s103
                                 + 0.614524 w_k1_AXP_s103
                                 + 0.542311 w_k1_DIS_s103
                                 + 0.715944 w_k1_FDX_s103
                                 + 0.607603 w_k1_IBM_s103
                                 + 0.810077 w_k1_INTC_s103
                                 + 0.662132 w_k1_KO_s103
                                 + 0.612021 w_k1_MAT_s103
                                 + 0.749891 w_k1_MSFT_s103
                                 + 0.726214 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.604326 w_k1_ADM_s104
                                 + 0.694224 w_k1_AXP_s104
                                 + 0.692382 w_k1_DIS_s104
                                 + 0.753054 w_k1_FDX_s104
                                 + 0.637462 w_k1_IBM_s104
                                 + 0.822113 w_k1_INTC_s104
                                 + 0.674991 w_k1_KO_s104
                                 + 0.564817 w_k1_MAT_s104
                                 + 0.796214 w_k1_MSFT_s104
                                 + 0.808002 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.959053 w_k1_ADM_s104
                                 + 1.11498 w_k1_AXP_s104
                                 + 1.02165 w_k1_DIS_s104
                                 + 0.98857 w_k1_FDX_s104
                                 + 1.21732 w_k1_IBM_s104
                                 + 1.00064 w_k1_INTC_s104
                                 + 1.14337 w_k1_KO_s104 + 1.02947 w_k1_MAT_s104
                                 + 0.993045 w_k1_MSFT_s104
                                 + 1.08612 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.05347 w_k1_ADM_s104
                                 + 1.33177 w_k1_AXP_s104
                                 + 1.35257 w_k1_DIS_s104
                                 + 1.10328 w_k1_FDX_s104
                                 + 1.62041 w_k1_IBM_s104
                                 + 0.925715 w_k1_INTC_s104
                                 + 1.36563 w_k1_KO_s104 + 1.23074 w_k1_MAT_s104
                                 + 1.05354 w_k1_MSFT_s104
                                 + 1.15565 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.977669 w_k1_ADM_s104
                                 + 0.901653 w_k1_AXP_s104
                                 + 1.39189 w_k1_DIS_s104
                                 + 1.12431 w_k1_FDX_s104
                                 + 0.938215 w_k1_IBM_s104
                                 + 1.02751 w_k1_INTC_s104
                                 + 1.10281 w_k1_KO_s104 + 1.02525 w_k1_MAT_s104
                                 + 1.20411 w_k1_MSFT_s104
                                 + 1.02987 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.01293 w_k1_ADM_s104
                                 + 1.20087 w_k1_AXP_s104
                                 + 1.25846 w_k1_DIS_s104
                                 + 1.20493 w_k1_FDX_s104
                                 + 0.693692 w_k1_IBM_s104
                                 + 1.4275 w_k1_INTC_s104
                                 + 0.990095 w_k1_KO_s104
                                 + 0.580289 w_k1_MAT_s104
                                 + 1.20927 w_k1_MSFT_s104
                                 + 1.01198 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.786288 w_k1_ADM_s104
                                 + 0.871913 w_k1_AXP_s104
                                 + 0.91742 w_k1_DIS_s104
                                 + 0.882989 w_k1_FDX_s104
                                 + 1.18163 w_k1_IBM_s104
                                 + 1.0186 w_k1_INTC_s104 + 1.10013 w_k1_KO_s104
                                 + 1.02899 w_k1_MAT_s104
                                 + 1.06301 w_k1_MSFT_s104
                                 + 1.02312 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.878422 w_k1_ADM_s104
                                 + 0.889292 w_k1_AXP_s104
                                 + 0.893847 w_k1_DIS_s104
                                 + 0.87212 w_k1_FDX_s104
                                 + 0.798841 w_k1_IBM_s104
                                 + 0.878443 w_k1_INTC_s104
                                 + 0.874209 w_k1_KO_s104
                                 + 0.700067 w_k1_MAT_s104
                                 + 0.882878 w_k1_MSFT_s104
                                 + 0.919701 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.22636 w_k1_ADM_s104
                                 + 1.43485 w_k1_AXP_s104 + 1.4373 w_k1_DIS_s104
                                 + 1.24746 w_k1_FDX_s104
                                 + 1.15293 w_k1_IBM_s104
                                 + 1.0848 w_k1_INTC_s104 + 1.22673 w_k1_KO_s104
                                 + 0.98145 w_k1_MAT_s104
                                 + 1.16883 w_k1_MSFT_s104 + 1.0238 w_k1_PG_s104
                                 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.959832 w_k1_ADM_s105
                                 + 0.838445 w_k1_AXP_s105
                                 + 0.877215 w_k1_DIS_s105
                                 + 0.917452 w_k1_FDX_s105
                                 + 0.915714 w_k1_IBM_s105
                                 + 0.975568 w_k1_INTC_s105
                                 + 0.88196 w_k1_KO_s105
                                 + 0.927789 w_k1_MAT_s105
                                 + 0.839638 w_k1_MSFT_s105
                                 + 0.850846 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.19458 w_k1_ADM_s105
                                 + 1.51391 w_k1_AXP_s105
                                 + 1.57209 w_k1_DIS_s105
                                 + 1.22595 w_k1_FDX_s105
                                 + 1.22538 w_k1_IBM_s105
                                 + 1.33492 w_k1_INTC_s105
                                 + 1.26243 w_k1_KO_s105 + 1.60236 w_k1_MAT_s105
                                 + 1.31163 w_k1_MSFT_s105
                                 + 1.08574 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.920414 w_k1_ADM_s105
                                 + 0.825761 w_k1_AXP_s105
                                 + 0.810844 w_k1_DIS_s105
                                 + 0.88314 w_k1_FDX_s105
                                 + 0.678581 w_k1_IBM_s105
                                 + 1.02511 w_k1_INTC_s105
                                 + 0.796115 w_k1_KO_s105
                                 + 0.652982 w_k1_MAT_s105
                                 + 0.86041 w_k1_MSFT_s105
                                 + 0.86211 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.927567 w_k1_ADM_s105
                                 + 1.1404 w_k1_AXP_s105
                                 + 0.985681 w_k1_DIS_s105
                                 + 1.01093 w_k1_FDX_s105
                                 + 0.724223 w_k1_IBM_s105
                                 + 1.01089 w_k1_INTC_s105
                                 + 0.779731 w_k1_KO_s105
                                 + 0.988524 w_k1_MAT_s105
                                 + 0.945718 w_k1_MSFT_s105
                                 + 0.859375 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.37692 w_k1_ADM_s105
                                 + 1.20139 w_k1_AXP_s105 + 1.0961 w_k1_DIS_s105
                                 + 1.1074 w_k1_FDX_s105 + 1.28248 w_k1_IBM_s105
                                 + 0.961707 w_k1_INTC_s105
                                 + 1.10961 w_k1_KO_s105 + 1.19351 w_k1_MAT_s105
                                 + 1.05432 w_k1_MSFT_s105
                                 + 1.05996 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.00351 w_k1_ADM_s105
                                 + 0.882647 w_k1_AXP_s105
                                 + 1.20817 w_k1_DIS_s105
                                 + 0.972603 w_k1_FDX_s105
                                 + 0.981034 w_k1_IBM_s105
                                 + 1.23547 w_k1_INTC_s105
                                 + 1.11832 w_k1_KO_s105 + 1.16775 w_k1_MAT_s105
                                 + 1.11139 w_k1_MSFT_s105
                                 + 1.07958 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.787626 w_k1_ADM_s105
                                 + 0.743788 w_k1_AXP_s105
                                 + 0.681838 w_k1_DIS_s105
                                 + 0.757698 w_k1_FDX_s105
                                 + 0.904601 w_k1_IBM_s105
                                 + 0.829359 w_k1_INTC_s105
                                 + 0.764578 w_k1_KO_s105
                                 + 0.828186 w_k1_MAT_s105
                                 + 0.781061 w_k1_MSFT_s105
                                 + 0.755868 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.20105 w_k1_ADM_s105
                                 + 1.22637 w_k1_AXP_s105
                                 + 0.912751 w_k1_DIS_s105
                                 + 1.06336 w_k1_FDX_s105
                                 + 0.951113 w_k1_IBM_s105
                                 + 1.22019 w_k1_INTC_s105
                                 + 0.933778 w_k1_KO_s105
                                 + 1.18289 w_k1_MAT_s105
                                 + 1.15127 w_k1_MSFT_s105
                                 + 1.02539 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.962596 w_k1_ADM_s106
                                 + 0.800538 w_k1_AXP_s106
                                 + 0.887092 w_k1_DIS_s106
                                 + 0.77919 w_k1_FDX_s106
                                 + 1.19014 w_k1_IBM_s106
                                 + 0.906794 w_k1_INTC_s106
                                 + 0.942735 w_k1_KO_s106
                                 + 1.33646 w_k1_MAT_s106
                                 + 0.834105 w_k1_MSFT_s106
                                 + 0.919457 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.34972 w_k1_ADM_s106
                                 + 1.35502 w_k1_AXP_s106
                                 + 2.00201 w_k1_DIS_s106
                                 + 1.27495 w_k1_FDX_s106
                                 + 1.79345 w_k1_IBM_s106
                                 + 1.35678 w_k1_INTC_s106
                                 + 1.58324 w_k1_KO_s106 + 1.91028 w_k1_MAT_s106
                                 + 1.41514 w_k1_MSFT_s106 + 1.252 w_k1_PG_s106
                                 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.22036 w_k1_ADM_s106
                                 + 1.04793 w_k1_AXP_s106
                                 + 0.928985 w_k1_DIS_s106
                                 + 1.08172 w_k1_FDX_s106
                                 + 0.779782 w_k1_IBM_s106
                                 + 1.14386 w_k1_INTC_s106
                                 + 0.725175 w_k1_KO_s106
                                 + 0.766114 w_k1_MAT_s106
                                 + 0.939451 w_k1_MSFT_s106
                                 + 0.847932 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.852058 w_k1_ADM_s106
                                 + 1.16347 w_k1_AXP_s106
                                 + 0.94213 w_k1_DIS_s106
                                 + 0.873614 w_k1_FDX_s106
                                 + 1.3509 w_k1_IBM_s106
                                 + 0.929419 w_k1_INTC_s106
                                 + 1.11016 w_k1_KO_s106 + 2.13652 w_k1_MAT_s106
                                 + 0.957236 w_k1_MSFT_s106
                                 + 0.950591 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.00048 w_k1_ADM_s106
                                 + 0.966607 w_k1_AXP_s106
                                 + 0.836171 w_k1_DIS_s106
                                 + 0.914639 w_k1_FDX_s106
                                 + 1.05868 w_k1_IBM_s106
                                 + 0.951903 w_k1_INTC_s106
                                 + 0.858665 w_k1_KO_s106
                                 + 1.09262 w_k1_MAT_s106
                                 + 0.867136 w_k1_MSFT_s106
                                 + 0.939908 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.947265 w_k1_ADM_s106
                                 + 1.10271 w_k1_AXP_s106
                                 + 1.24949 w_k1_DIS_s106
                                 + 1.03821 w_k1_FDX_s106
                                 + 1.12003 w_k1_IBM_s106
                                 + 1.17427 w_k1_INTC_s106
                                 + 0.983681 w_k1_KO_s106
                                 + 1.04299 w_k1_MAT_s106
                                 + 1.06769 w_k1_MSFT_s106
                                 + 1.03946 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.862433 w_k1_ADM_s106
                                 + 0.801008 w_k1_AXP_s106
                                 + 0.720957 w_k1_DIS_s106
                                 + 0.783882 w_k1_FDX_s106
                                 + 0.729441 w_k1_IBM_s106
                                 + 0.70326 w_k1_INTC_s106
                                 + 0.684798 w_k1_KO_s106
                                 + 0.964494 w_k1_MAT_s106
                                 + 0.802186 w_k1_MSFT_s106
                                 + 0.794112 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.22788 w_k1_ADM_s106
                                 + 1.62538 w_k1_AXP_s106
                                 + 1.36197 w_k1_DIS_s106
                                 + 1.39282 w_k1_FDX_s106
                                 + 1.17299 w_k1_IBM_s106
                                 + 1.30252 w_k1_INTC_s106 + 1.1425 w_k1_KO_s106
                                 + 1.15757 w_k1_MAT_s106
                                 + 1.21439 w_k1_MSFT_s106 + 1.1587 w_k1_PG_s106
                                 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.12112 w_k1_ADM_s107
                                 + 1.61018 w_k1_AXP_s107 + 1.8442 w_k1_DIS_s107
                                 + 1.32704 w_k1_FDX_s107
                                 + 1.55231 w_k1_IBM_s107
                                 + 1.03214 w_k1_INTC_s107
                                 + 1.55756 w_k1_KO_s107 + 2.3144 w_k1_MAT_s107
                                 + 1.1304 w_k1_MSFT_s107 + 1.40797 w_k1_PG_s107
                                 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.671621 w_k1_ADM_s107
                                 + 1.12695 w_k1_AXP_s107
                                 + 0.79869 w_k1_DIS_s107
                                 + 0.899463 w_k1_FDX_s107
                                 + 0.885101 w_k1_IBM_s107
                                 + 0.916814 w_k1_INTC_s107
                                 + 0.911626 w_k1_KO_s107
                                 + 1.42894 w_k1_MAT_s107
                                 + 0.874169 w_k1_MSFT_s107
                                 + 0.93598 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.959472 w_k1_ADM_s107
                                 + 1.00643 w_k1_AXP_s107 + 1.0582 w_k1_DIS_s107
                                 + 0.999957 w_k1_FDX_s107
                                 + 0.950557 w_k1_IBM_s107
                                 + 0.932222 w_k1_INTC_s107
                                 + 0.935673 w_k1_KO_s107 + 1.0637 w_k1_MAT_s107
                                 + 0.941838 w_k1_MSFT_s107
                                 + 1.02994 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.761254 w_k1_ADM_s107
                                 + 1.11788 w_k1_AXP_s107
                                 + 1.30009 w_k1_DIS_s107
                                 + 1.04063 w_k1_FDX_s107
                                 + 1.36064 w_k1_IBM_s107
                                 + 1.12029 w_k1_INTC_s107
                                 + 1.25215 w_k1_KO_s107 + 1.36685 w_k1_MAT_s107
                                 + 1.10299 w_k1_MSFT_s107 + 1.0706 w_k1_PG_s107
                                 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.17642 w_k1_ADM_s107
                                 + 1.08664 w_k1_AXP_s107
                                 + 1.30202 w_k1_DIS_s107
                                 + 1.16011 w_k1_FDX_s107
                                 + 1.10147 w_k1_IBM_s107
                                 + 0.998372 w_k1_INTC_s107
                                 + 1.26765 w_k1_KO_s107 + 1.20146 w_k1_MAT_s107
                                 + 1.05887 w_k1_MSFT_s107
                                 + 1.28093 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.08604 w_k1_ADM_s107
                                 + 1.15866 w_k1_AXP_s107 + 1.2514 w_k1_DIS_s107
                                 + 1.14723 w_k1_FDX_s107
                                 + 0.767975 w_k1_IBM_s107
                                 + 1.16074 w_k1_INTC_s107
                                 + 1.07586 w_k1_KO_s107
                                 + 0.929678 w_k1_MAT_s107
                                 + 0.982234 w_k1_MSFT_s107 + 1.022 w_k1_PG_s107
                                 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.939971 w_k1_ADM_s107
                                 + 0.952472 w_k1_AXP_s107
                                 + 1.24208 w_k1_DIS_s107
                                 + 0.922263 w_k1_FDX_s107
                                 + 1.81794 w_k1_IBM_s107
                                 + 0.856135 w_k1_INTC_s107
                                 + 1.47763 w_k1_KO_s107 + 1.91121 w_k1_MAT_s107
                                 + 0.920112 w_k1_MSFT_s107
                                 + 1.12291 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.674839 w_k1_ADM_s107
                                 + 0.772034 w_k1_AXP_s107
                                 + 0.773756 w_k1_DIS_s107
                                 + 0.699075 w_k1_FDX_s107
                                 + 1.10093 w_k1_IBM_s107
                                 + 0.72141 w_k1_INTC_s107
                                 + 0.854194 w_k1_KO_s107
                                 + 1.31174 w_k1_MAT_s107
                                 + 0.680421 w_k1_MSFT_s107
                                 + 0.949893 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.10438 w_k1_ADM_s110
                                 + 1.27268 w_k1_AXP_s110
                                 + 1.05246 w_k1_DIS_s110
                                 + 0.963017 w_k1_FDX_s110
                                 + 1.39301 w_k1_IBM_s110
                                 + 0.952067 w_k1_INTC_s110
                                 + 1.01748 w_k1_KO_s110 + 1.38103 w_k1_MAT_s110
                                 + 0.99636 w_k1_MSFT_s110
                                 + 0.931694 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.22586 w_k1_ADM_s110
                                 + 1.19685 w_k1_AXP_s110
                                 + 1.43828 w_k1_DIS_s110
                                 + 1.37292 w_k1_FDX_s110
                                 + 1.02476 w_k1_IBM_s110
                                 + 0.996031 w_k1_INTC_s110
                                 + 1.20019 w_k1_KO_s110 + 1.22683 w_k1_MAT_s110
                                 + 1.33582 w_k1_MSFT_s110
                                 + 1.21281 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.05536 w_k1_ADM_s110
                                 + 0.723845 w_k1_AXP_s110
                                 + 1.06151 w_k1_DIS_s110
                                 + 0.918306 w_k1_FDX_s110
                                 + 0.87345 w_k1_IBM_s110
                                 + 0.875866 w_k1_INTC_s110
                                 + 1.01106 w_k1_KO_s110
                                 + 0.796059 w_k1_MAT_s110
                                 + 1.12033 w_k1_MSFT_s110
                                 + 0.914397 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.05576 w_k1_ADM_s110
                                 + 1.01788 w_k1_AXP_s110
                                 + 1.03672 w_k1_DIS_s110
                                 + 1.04321 w_k1_FDX_s110
                                 + 1.10511 w_k1_IBM_s110
                                 + 1.00175 w_k1_INTC_s110
                                 + 1.09726 w_k1_KO_s110 + 1.11222 w_k1_MAT_s110
                                 + 1.00648 w_k1_MSFT_s110
                                 + 0.995024 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.0572 w_k1_ADM_s110
                                 + 1.35185 w_k1_AXP_s110
                                 + 1.04808 w_k1_DIS_s110
                                 + 1.11186 w_k1_FDX_s110
                                 + 0.845378 w_k1_IBM_s110
                                 + 1.13829 w_k1_INTC_s110
                                 + 0.881553 w_k1_KO_s110
                                 + 0.900937 w_k1_MAT_s110
                                 + 1.06372 w_k1_MSFT_s110
                                 + 1.00362 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.721383 w_k1_ADM_s110
                                 + 0.940008 w_k1_AXP_s110
                                 + 0.765715 w_k1_DIS_s110
                                 + 0.900869 w_k1_FDX_s110
                                 + 0.707516 w_k1_IBM_s110
                                 + 0.85577 w_k1_INTC_s110
                                 + 0.671216 w_k1_KO_s110
                                 + 0.686394 w_k1_MAT_s110
                                 + 0.840998 w_k1_MSFT_s110
                                 + 0.949336 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.827117 w_k1_ADM_s110
                                 + 1.03031 w_k1_AXP_s110
                                 + 0.749434 w_k1_DIS_s110
                                 + 0.857279 w_k1_FDX_s110
                                 + 0.995748 w_k1_IBM_s110
                                 + 0.697607 w_k1_INTC_s110
                                 + 0.843504 w_k1_KO_s110
                                 + 1.06269 w_k1_MAT_s110
                                 + 0.816814 w_k1_MSFT_s110
                                 + 0.967299 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.25761 w_k1_ADM_s110
                                 + 1.30898 w_k1_AXP_s110
                                 + 1.44185 w_k1_DIS_s110
                                 + 1.23273 w_k1_FDX_s110
                                 + 1.31132 w_k1_IBM_s110
                                 + 0.938997 w_k1_INTC_s110
                                 + 1.25705 w_k1_KO_s110 + 1.66829 w_k1_MAT_s110
                                 + 1.1851 w_k1_MSFT_s110 + 1.18387 w_k1_PG_s110
                                 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.41818 w_k1_ADM_s111
                                 + 1.05304 w_k1_AXP_s111
                                 + 0.876664 w_k1_DIS_s111
                                 + 1.10912 w_k1_FDX_s111
                                 + 0.749171 w_k1_IBM_s111
                                 + 0.948683 w_k1_INTC_s111
                                 + 0.872629 w_k1_KO_s111
                                 + 0.577276 w_k1_MAT_s111
                                 + 1.12207 w_k1_MSFT_s111
                                 + 1.03178 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.06047 w_k1_ADM_s111
                                 + 0.976601 w_k1_AXP_s111
                                 + 1.11847 w_k1_DIS_s111
                                 + 0.994894 w_k1_FDX_s111
                                 + 1.09976 w_k1_IBM_s111
                                 + 1.07366 w_k1_INTC_s111
                                 + 1.09933 w_k1_KO_s111 + 1.10412 w_k1_MAT_s111
                                 + 1.08289 w_k1_MSFT_s111
                                 + 1.08513 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.14424 w_k1_ADM_s111
                                 + 1.13262 w_k1_AXP_s111
                                 + 0.690719 w_k1_DIS_s111
                                 + 0.802016 w_k1_FDX_s111
                                 + 1.0542 w_k1_IBM_s111 + 0.848 w_k1_INTC_s111
                                 + 0.790258 w_k1_KO_s111 + 0.8864 w_k1_MAT_s111
                                 + 0.711557 w_k1_MSFT_s111
                                 + 0.90168 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.931422 w_k1_ADM_s111
                                 + 0.598629 w_k1_AXP_s111
                                 + 0.568536 w_k1_DIS_s111
                                 + 0.785194 w_k1_FDX_s111
                                 + 0.663676 w_k1_IBM_s111
                                 + 0.639419 w_k1_INTC_s111
                                 + 0.693215 w_k1_KO_s111
                                 + 0.611159 w_k1_MAT_s111
                                 + 0.768242 w_k1_MSFT_s111
                                 + 0.822447 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.00317 w_k1_ADM_s111
                                 + 1.33049 w_k1_AXP_s111
                                 + 1.15436 w_k1_DIS_s111
                                 + 1.09475 w_k1_FDX_s111
                                 + 1.45619 w_k1_IBM_s111
                                 + 1.04557 w_k1_INTC_s111
                                 + 1.12281 w_k1_KO_s111 + 1.62486 w_k1_MAT_s111
                                 + 0.997955 w_k1_MSFT_s111
                                 + 1.06968 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.3977 w_k1_ADM_s111
                                 + 1.51976 w_k1_AXP_s111
                                 + 1.39933 w_k1_DIS_s111 + 1.3811 w_k1_FDX_s111
                                 + 1.01631 w_k1_IBM_s111
                                 + 1.22118 w_k1_INTC_s111
                                 + 1.14877 w_k1_KO_s111
                                 + 0.901886 w_k1_MAT_s111
                                 + 1.13559 w_k1_MSFT_s111
                                 + 1.16108 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.17461 w_k1_ADM_s111
                                 + 0.721245 w_k1_AXP_s111
                                 + 0.877551 w_k1_DIS_s111
                                 + 0.863991 w_k1_FDX_s111
                                 + 0.949439 w_k1_IBM_s111
                                 + 0.948988 w_k1_INTC_s111
                                 + 0.937334 w_k1_KO_s111
                                 + 0.532028 w_k1_MAT_s111
                                 + 0.941579 w_k1_MSFT_s111
                                 + 1.02923 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.32617 w_k1_ADM_s111
                                 + 1.65108 w_k1_AXP_s111
                                 + 1.73667 w_k1_DIS_s111
                                 + 1.41535 w_k1_FDX_s111
                                 + 1.50133 w_k1_IBM_s111
                                 + 1.37046 w_k1_INTC_s111
                                 + 1.55443 w_k1_KO_s111 + 2.00854 w_k1_MAT_s111
                                 + 1.59293 w_k1_MSFT_s111
                                 + 1.34707 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.11448 w_k1_ADM_s112
                                 + 0.825617 w_k1_AXP_s112
                                 + 1.00173 w_k1_DIS_s112
                                 + 1.08679 w_k1_FDX_s112
                                 + 0.727943 w_k1_IBM_s112
                                 + 1.20571 w_k1_INTC_s112
                                 + 0.828834 w_k1_KO_s112
                                 + 0.585808 w_k1_MAT_s112
                                 + 1.18474 w_k1_MSFT_s112
                                 + 0.934839 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.851298 w_k1_ADM_s112
                                 + 0.855238 w_k1_AXP_s112
                                 + 0.74192 w_k1_DIS_s112
                                 + 0.820148 w_k1_FDX_s112
                                 + 1.20101 w_k1_IBM_s112
                                 + 0.847722 w_k1_INTC_s112
                                 + 0.841925 w_k1_KO_s112
                                 + 1.05695 w_k1_MAT_s112
                                 + 0.836849 w_k1_MSFT_s112
                                 + 0.861216 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.958693 w_k1_ADM_s112
                                 + 1.08684 w_k1_AXP_s112
                                 + 1.31894 w_k1_DIS_s112
                                 + 1.00452 w_k1_FDX_s112
                                 + 1.67613 w_k1_IBM_s112
                                 + 1.23758 w_k1_INTC_s112
                                 + 1.17424 w_k1_KO_s112 + 1.37263 w_k1_MAT_s112
                                 + 1.15061 w_k1_MSFT_s112
                                 + 1.07637 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.09531 w_k1_ADM_s112
                                 + 1.1852 w_k1_AXP_s112
                                 + 0.779234 w_k1_DIS_s112
                                 + 1.03569 w_k1_FDX_s112
                                 + 1.07984 w_k1_IBM_s112
                                 + 0.872831 w_k1_INTC_s112
                                 + 0.798537 w_k1_KO_s112
                                 + 1.27074 w_k1_MAT_s112
                                 + 0.905694 w_k1_MSFT_s112
                                 + 0.948602 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.840336 w_k1_ADM_s112
                                 + 0.717013 w_k1_AXP_s112
                                 + 0.606473 w_k1_DIS_s112
                                 + 0.785566 w_k1_FDX_s112
                                 + 0.758413 w_k1_IBM_s112
                                 + 0.756681 w_k1_INTC_s112
                                 + 0.666076 w_k1_KO_s112
                                 + 0.79716 w_k1_MAT_s112
                                 + 0.772846 w_k1_MSFT_s112
                                 + 0.802407 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.21921 w_k1_ADM_s112
                                 + 1.48488 w_k1_AXP_s112
                                 + 1.40291 w_k1_DIS_s112 + 1.3135 w_k1_FDX_s112
                                 + 1.15795 w_k1_IBM_s112
                                 + 1.33753 w_k1_INTC_s112
                                 + 1.14389 w_k1_KO_s112 + 1.32728 w_k1_MAT_s112
                                 + 1.31143 w_k1_MSFT_s112
                                 + 1.05706 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.06058 w_k1_ADM_s112
                                 + 0.775716 w_k1_AXP_s112
                                 + 0.736477 w_k1_DIS_s112
                                 + 0.957649 w_k1_FDX_s112
                                 + 0.782887 w_k1_IBM_s112
                                 + 0.953862 w_k1_INTC_s112
                                 + 0.712458 w_k1_KO_s112
                                 + 0.620934 w_k1_MAT_s112
                                 + 0.89914 w_k1_MSFT_s112
                                 + 0.913138 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.952284 w_k1_ADM_s112
                                 + 0.807768 w_k1_AXP_s112
                                 + 1.05077 w_k1_DIS_s112 + 1.0026 w_k1_FDX_s112
                                 + 1.13676 w_k1_IBM_s112
                                 + 1.00495 w_k1_INTC_s112
                                 + 0.996472 w_k1_KO_s112
                                 + 1.16776 w_k1_MAT_s112
                                 + 1.12816 w_k1_MSFT_s112 + 1.0022 w_k1_PG_s112
                                 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.02371 w_k1_ADM_s113
                                 + 1.2421 w_k1_AXP_s113 + 1.34596 w_k1_DIS_s113
                                 + 1.11295 w_k1_FDX_s113
                                 + 1.43713 w_k1_IBM_s113
                                 + 1.0486 w_k1_INTC_s113 + 1.27627 w_k1_KO_s113
                                 + 1.76688 w_k1_MAT_s113
                                 + 1.12909 w_k1_MSFT_s113
                                 + 1.15592 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.16313 w_k1_ADM_s113
                                 + 1.12199 w_k1_AXP_s113
                                 + 1.55902 w_k1_DIS_s113
                                 + 1.27364 w_k1_FDX_s113
                                 + 0.953171 w_k1_IBM_s113
                                 + 1.18554 w_k1_INTC_s113
                                 + 1.05031 w_k1_KO_s113
                                 + 0.922518 w_k1_MAT_s113
                                 + 1.17919 w_k1_MSFT_s113 + 1.0327 w_k1_PG_s113
                                 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.810554 w_k1_ADM_s113
                                 + 1.03746 w_k1_AXP_s113
                                 + 0.872571 w_k1_DIS_s113
                                 + 1.00456 w_k1_FDX_s113
                                 + 0.681659 w_k1_IBM_s113
                                 + 0.913544 w_k1_INTC_s113
                                 + 0.683742 w_k1_KO_s113
                                 + 0.976386 w_k1_MAT_s113
                                 + 0.852176 w_k1_MSFT_s113
                                 + 0.844565 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.01819 w_k1_ADM_s113
                                 + 1.08096 w_k1_AXP_s113
                                 + 1.16686 w_k1_DIS_s113
                                 + 1.16603 w_k1_FDX_s113
                                 + 0.857558 w_k1_IBM_s113
                                 + 1.0609 w_k1_INTC_s113
                                 + 0.842677 w_k1_KO_s113 + 1.066 w_k1_MAT_s113
                                 + 0.986351 w_k1_MSFT_s113
                                 + 0.922223 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.747771 w_k1_ADM_s113
                                 + 0.77697 w_k1_AXP_s113
                                 + 0.804084 w_k1_DIS_s113
                                 + 0.737514 w_k1_FDX_s113
                                 + 1.31155 w_k1_IBM_s113
                                 + 0.77737 w_k1_INTC_s113
                                 + 0.878393 w_k1_KO_s113
                                 + 1.45739 w_k1_MAT_s113
                                 + 0.810575 w_k1_MSFT_s113
                                 + 0.826014 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.85318 w_k1_ADM_s113
                                 + 1.17521 w_k1_AXP_s113
                                 + 1.06986 w_k1_DIS_s113
                                 + 1.04252 w_k1_FDX_s113
                                 + 1.11051 w_k1_IBM_s113
                                 + 0.909857 w_k1_INTC_s113
                                 + 1.00181 w_k1_KO_s113 + 1.58152 w_k1_MAT_s113
                                 + 0.954796 w_k1_MSFT_s113
                                 + 1.04898 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.982212 w_k1_ADM_s113
                                 + 1.57561 w_k1_AXP_s113
                                 + 1.56283 w_k1_DIS_s113
                                 + 1.38112 w_k1_FDX_s113
                                 + 1.09297 w_k1_IBM_s113
                                 + 1.33329 w_k1_INTC_s113
                                 + 1.09623 w_k1_KO_s113 + 1.72377 w_k1_MAT_s113
                                 + 1.23935 w_k1_MSFT_s113
                                 + 1.08098 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.813852 w_k1_ADM_s113
                                 + 0.747956 w_k1_AXP_s113
                                 + 0.823419 w_k1_DIS_s113
                                 + 0.827712 w_k1_FDX_s113
                                 + 0.845483 w_k1_IBM_s113
                                 + 0.817241 w_k1_INTC_s113
                                 + 0.72029 w_k1_KO_s113
                                 + 0.792044 w_k1_MAT_s113
                                 + 0.777204 w_k1_MSFT_s113
                                 + 0.852033 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.829011 w_k1_ADM_s114
                                 + 0.831195 w_k1_AXP_s114
                                 + 1.35204 w_k1_DIS_s114
                                 + 1.09392 w_k1_FDX_s114
                                 + 1.11723 w_k1_IBM_s114
                                 + 1.03968 w_k1_INTC_s114 + 1.1456 w_k1_KO_s114
                                 + 1.06467 w_k1_MAT_s114
                                 + 1.15087 w_k1_MSFT_s114
                                 + 1.10028 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.04449 w_k1_ADM_s114
                                 + 1.12408 w_k1_AXP_s114
                                 + 0.94988 w_k1_DIS_s114
                                 + 1.24626 w_k1_FDX_s114
                                 + 0.832974 w_k1_IBM_s114
                                 + 1.07739 w_k1_INTC_s114
                                 + 0.803828 w_k1_KO_s114
                                 + 1.08562 w_k1_MAT_s114
                                 + 1.1827 w_k1_MSFT_s114 + 1.11314 w_k1_PG_s114
                                 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.09493 w_k1_ADM_s114
                                 + 1.15587 w_k1_AXP_s114
                                 + 1.74673 w_k1_DIS_s114
                                 + 1.47506 w_k1_FDX_s114
                                 + 0.98046 w_k1_IBM_s114
                                 + 1.38879 w_k1_INTC_s114
                                 + 1.03283 w_k1_KO_s114
                                 + 0.781512 w_k1_MAT_s114
                                 + 1.35828 w_k1_MSFT_s114 + 1.2047 w_k1_PG_s114
                                 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.865245 w_k1_ADM_s114
                                 + 0.75993 w_k1_AXP_s114
                                 + 0.761698 w_k1_DIS_s114
                                 + 0.829256 w_k1_FDX_s114
                                 + 0.746895 w_k1_IBM_s114
                                 + 0.877791 w_k1_INTC_s114
                                 + 0.775378 w_k1_KO_s114
                                 + 0.629651 w_k1_MAT_s114
                                 + 0.840898 w_k1_MSFT_s114
                                 + 0.822417 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.09286 w_k1_ADM_s114
                                 + 1.71709 w_k1_AXP_s114
                                 + 1.67725 w_k1_DIS_s114
                                 + 1.38069 w_k1_FDX_s114
                                 + 1.69894 w_k1_IBM_s114
                                 + 1.32987 w_k1_INTC_s114
                                 + 1.38673 w_k1_KO_s114 + 1.75279 w_k1_MAT_s114
                                 + 1.27397 w_k1_MSFT_s114 + 1.2708 w_k1_PG_s114
                                 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.840799 w_k1_ADM_s114
                                 + 1.22778 w_k1_AXP_s114
                                 + 1.34389 w_k1_DIS_s114
                                 + 1.18266 w_k1_FDX_s114
                                 + 1.44211 w_k1_IBM_s114
                                 + 1.06352 w_k1_INTC_s114
                                 + 1.14512 w_k1_KO_s114 + 1.93474 w_k1_MAT_s114
                                 + 1.17592 w_k1_MSFT_s114
                                 + 1.17272 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.22008 w_k1_ADM_s114
                                 + 0.82307 w_k1_AXP_s114
                                 + 1.06758 w_k1_DIS_s114
                                 + 1.10167 w_k1_FDX_s114
                                 + 0.750975 w_k1_IBM_s114
                                 + 0.964684 w_k1_INTC_s114
                                 + 0.874639 w_k1_KO_s114
                                 + 0.602991 w_k1_MAT_s114
                                 + 1.07289 w_k1_MSFT_s114
                                 + 0.970981 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.883175 w_k1_ADM_s114
                                 + 0.746458 w_k1_AXP_s114
                                 + 0.905475 w_k1_DIS_s114
                                 + 0.886511 w_k1_FDX_s114
                                 + 1.2738 w_k1_IBM_s114
                                 + 0.830408 w_k1_INTC_s114
                                 + 1.03071 w_k1_KO_s114
                                 + 0.942105 w_k1_MAT_s114
                                 + 0.943243 w_k1_MSFT_s114
                                 + 0.950636 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.773528 w_k1_ADM_s115
                                 + 0.949909 w_k1_AXP_s115
                                 + 0.874578 w_k1_DIS_s115
                                 + 1.05725 w_k1_FDX_s115
                                 + 0.74046 w_k1_IBM_s115
                                 + 1.07583 w_k1_INTC_s115
                                 + 0.752167 w_k1_KO_s115
                                 + 0.781798 w_k1_MAT_s115
                                 + 1.02674 w_k1_MSFT_s115
                                 + 0.991254 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.835122 w_k1_ADM_s115
                                 + 0.753999 w_k1_AXP_s115
                                 + 0.676618 w_k1_DIS_s115
                                 + 0.71183 w_k1_FDX_s115
                                 + 0.863928 w_k1_IBM_s115
                                 + 0.937232 w_k1_INTC_s115
                                 + 0.79926 w_k1_KO_s115 + 1.01894 w_k1_MAT_s115
                                 + 0.802096 w_k1_MSFT_s115
                                 + 0.820422 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.673367 w_k1_ADM_s115
                                 + 0.762983 w_k1_AXP_s115
                                 + 0.852774 w_k1_DIS_s115
                                 + 0.838281 w_k1_FDX_s115
                                 + 0.609195 w_k1_IBM_s115
                                 + 0.803413 w_k1_INTC_s115
                                 + 0.734713 w_k1_KO_s115
                                 + 0.808432 w_k1_MAT_s115
                                 + 0.770372 w_k1_MSFT_s115
                                 + 0.962749 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.857634 w_k1_ADM_s115
                                 + 0.966512 w_k1_AXP_s115
                                 + 0.923139 w_k1_DIS_s115
                                 + 0.987684 w_k1_FDX_s115
                                 + 0.943664 w_k1_IBM_s115
                                 + 1.00418 w_k1_INTC_s115
                                 + 0.804084 w_k1_KO_s115
                                 + 1.30515 w_k1_MAT_s115
                                 + 0.912319 w_k1_MSFT_s115
                                 + 0.914818 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.00724 w_k1_ADM_s115
                                 + 1.50732 w_k1_AXP_s115
                                 + 1.73312 w_k1_DIS_s115
                                 + 1.39824 w_k1_FDX_s115
                                 + 1.02536 w_k1_IBM_s115
                                 + 1.24229 w_k1_INTC_s115
                                 + 1.06275 w_k1_KO_s115 + 1.58753 w_k1_MAT_s115
                                 + 1.24784 w_k1_MSFT_s115
                                 + 1.30652 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.845311 w_k1_ADM_s115
                                 + 1.01899 w_k1_AXP_s115
                                 + 1.19483 w_k1_DIS_s115
                                 + 0.915173 w_k1_FDX_s115
                                 + 1.38232 w_k1_IBM_s115
                                 + 1.15983 w_k1_INTC_s115
                                 + 1.11847 w_k1_KO_s115 + 1.3657 w_k1_MAT_s115
                                 + 1.03208 w_k1_MSFT_s115
                                 + 1.00202 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.05356 w_k1_ADM_s115
                                 + 1.30456 w_k1_AXP_s115
                                 + 1.03816 w_k1_DIS_s115
                                 + 1.12621 w_k1_FDX_s115
                                 + 0.930506 w_k1_IBM_s115
                                 + 0.854893 w_k1_INTC_s115
                                 + 0.925568 w_k1_KO_s115
                                 + 1.49501 w_k1_MAT_s115
                                 + 1.09074 w_k1_MSFT_s115
                                 + 1.04448 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.00948 w_k1_ADM_s115
                                 + 1.59355 w_k1_AXP_s115
                                 + 1.08318 w_k1_DIS_s115
                                 + 1.09736 w_k1_FDX_s115
                                 + 1.42853 w_k1_IBM_s115
                                 + 0.955781 w_k1_INTC_s115
                                 + 1.06217 w_k1_KO_s115 + 2.74994 w_k1_MAT_s115
                                 + 0.9823 w_k1_MSFT_s115 + 1.04344 w_k1_PG_s115
                                 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.35015 w_k1_ADM_s116
                                 + 1.79073 w_k1_AXP_s116
                                 + 1.36839 w_k1_DIS_s116
                                 + 1.28316 w_k1_FDX_s116
                                 + 1.05016 w_k1_IBM_s116
                                 + 0.867741 w_k1_INTC_s116
                                 + 1.20307 w_k1_KO_s116 + 2.0035 w_k1_MAT_s116
                                 + 1.05607 w_k1_MSFT_s116
                                 + 1.04522 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.41553 w_k1_ADM_s116
                                 + 1.47974 w_k1_AXP_s116
                                 + 1.49316 w_k1_DIS_s116
                                 + 1.32954 w_k1_FDX_s116
                                 + 1.34921 w_k1_IBM_s116
                                 + 1.04837 w_k1_INTC_s116
                                 + 1.23222 w_k1_KO_s116 + 1.69327 w_k1_MAT_s116
                                 + 1.19476 w_k1_MSFT_s116
                                 + 1.16589 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.3858 w_k1_ADM_s116
                                 + 1.26544 w_k1_AXP_s116
                                 + 1.35371 w_k1_DIS_s116
                                 + 1.08193 w_k1_FDX_s116
                                 + 1.17715 w_k1_IBM_s116
                                 + 1.11096 w_k1_INTC_s116
                                 + 1.21448 w_k1_KO_s116 + 1.14093 w_k1_MAT_s116
                                 + 1.08766 w_k1_MSFT_s116
                                 + 1.10806 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.787301 w_k1_ADM_s116
                                 + 1.12761 w_k1_AXP_s116
                                 + 1.20256 w_k1_DIS_s116
                                 + 0.89387 w_k1_FDX_s116
                                 + 1.47862 w_k1_IBM_s116
                                 + 1.01227 w_k1_INTC_s116
                                 + 1.14946 w_k1_KO_s116 + 1.30112 w_k1_MAT_s116
                                 + 0.954861 w_k1_MSFT_s116
                                 + 1.02818 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.02446 w_k1_ADM_s116
                                 + 0.866152 w_k1_AXP_s116
                                 + 0.855836 w_k1_DIS_s116
                                 + 1.00266 w_k1_FDX_s116
                                 + 0.679776 w_k1_IBM_s116
                                 + 0.860069 w_k1_INTC_s116
                                 + 0.785291 w_k1_KO_s116
                                 + 0.536711 w_k1_MAT_s116
                                 + 1.00082 w_k1_MSFT_s116 + 0.9214 w_k1_PG_s116
                                 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.909175 w_k1_ADM_s116
                                 + 1.1206 w_k1_AXP_s116
                                 + 0.727952 w_k1_DIS_s116
                                 + 0.82598 w_k1_FDX_s116
                                 + 0.937153 w_k1_IBM_s116
                                 + 0.765927 w_k1_INTC_s116
                                 + 0.764876 w_k1_KO_s116
                                 + 1.12022 w_k1_MAT_s116
                                 + 0.711613 w_k1_MSFT_s116
                                 + 0.88691 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.10119 w_k1_ADM_s116
                                 + 0.902715 w_k1_AXP_s116
                                 + 1.16585 w_k1_DIS_s116
                                 + 1.01605 w_k1_FDX_s116
                                 + 0.876913 w_k1_IBM_s116
                                 + 0.839294 w_k1_INTC_s116
                                 + 1.04363 w_k1_KO_s116
                                 + 0.802195 w_k1_MAT_s116
                                 + 1.05152 w_k1_MSFT_s116
                                 + 0.942333 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.97629 w_k1_ADM_s116
                                 + 0.957452 w_k1_AXP_s116
                                 + 1.16929 w_k1_DIS_s116
                                 + 0.890017 w_k1_FDX_s116
                                 + 1.18605 w_k1_IBM_s116
                                 + 0.952646 w_k1_INTC_s116
                                 + 1.17765 w_k1_KO_s116 + 1.03344 w_k1_MAT_s116
                                 + 0.942079 w_k1_MSFT_s116
                                 + 1.13114 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.841254 w_k1_ADM_s117
                                 + 0.794153 w_k1_AXP_s117
                                 + 0.970488 w_k1_DIS_s117
                                 + 0.848126 w_k1_FDX_s117
                                 + 0.976953 w_k1_IBM_s117
                                 + 0.918369 w_k1_INTC_s117
                                 + 1.02444 w_k1_KO_s117
                                 + 0.830443 w_k1_MAT_s117
                                 + 0.924052 w_k1_MSFT_s117
                                 + 1.04207 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.30032 w_k1_ADM_s117
                                 + 0.96314 w_k1_AXP_s117
                                 + 1.30223 w_k1_DIS_s117
                                 + 1.20546 w_k1_FDX_s117
                                 + 0.814485 w_k1_IBM_s117
                                 + 0.969413 w_k1_INTC_s117
                                 + 1.05082 w_k1_KO_s117
                                 + 0.948452 w_k1_MAT_s117
                                 + 1.12182 w_k1_MSFT_s117
                                 + 1.05888 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.807563 w_k1_ADM_s117
                                 + 0.689428 w_k1_AXP_s117
                                 + 0.633135 w_k1_DIS_s117
                                 + 0.75314 w_k1_FDX_s117
                                 + 0.787678 w_k1_IBM_s117
                                 + 0.780853 w_k1_INTC_s117
                                 + 0.701885 w_k1_KO_s117
                                 + 0.682061 w_k1_MAT_s117
                                 + 0.720838 w_k1_MSFT_s117
                                 + 0.817649 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.34782 w_k1_ADM_s117
                                 + 1.498 w_k1_AXP_s117 + 1.70968 w_k1_DIS_s117
                                 + 1.38485 w_k1_FDX_s117
                                 + 1.27456 w_k1_IBM_s117
                                 + 1.35956 w_k1_INTC_s117
                                 + 1.48201 w_k1_KO_s117 + 1.28588 w_k1_MAT_s117
                                 + 1.44838 w_k1_MSFT_s117
                                 + 1.21452 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.27338 w_k1_ADM_s117
                                 + 1.10825 w_k1_AXP_s117
                                 + 1.19736 w_k1_DIS_s117
                                 + 1.04235 w_k1_FDX_s117
                                 + 1.02585 w_k1_IBM_s117
                                 + 0.904804 w_k1_INTC_s117
                                 + 1.04385 w_k1_KO_s117 + 1.12295 w_k1_MAT_s117
                                 + 1.14532 w_k1_MSFT_s117
                                 + 1.05972 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.997653 w_k1_ADM_s117
                                 + 0.953117 w_k1_AXP_s117
                                 + 0.877338 w_k1_DIS_s117
                                 + 0.943437 w_k1_FDX_s117
                                 + 0.730211 w_k1_IBM_s117
                                 + 0.820296 w_k1_INTC_s117
                                 + 0.865769 w_k1_KO_s117
                                 + 0.950537 w_k1_MAT_s117
                                 + 0.91396 w_k1_MSFT_s117
                                 + 0.937293 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.791155 w_k1_ADM_s117
                                 + 1.0628 w_k1_AXP_s117
                                 + 0.766321 w_k1_DIS_s117
                                 + 0.773538 w_k1_FDX_s117
                                 + 1.2298 w_k1_IBM_s117
                                 + 0.905965 w_k1_INTC_s117
                                 + 0.912673 w_k1_KO_s117
                                 + 1.30094 w_k1_MAT_s117
                                 + 0.765312 w_k1_MSFT_s117
                                 + 0.888622 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.848557 w_k1_ADM_s117
                                 + 1.00457 w_k1_AXP_s117
                                 + 1.09612 w_k1_DIS_s117
                                 + 0.852865 w_k1_FDX_s117
                                 + 1.60039 w_k1_IBM_s117
                                 + 0.738649 w_k1_INTC_s117
                                 + 1.27258 w_k1_KO_s117 + 1.44212 w_k1_MAT_s117
                                 + 0.851977 w_k1_MSFT_s117
                                 + 1.12852 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0488253668123026 w_k1_ADM_s000
                                 + 0.0210842746180849 w_k1_ADM_s001
                                 + 0.0344230834748729 w_k1_ADM_s002
                                 + 0.0135691171858669 w_k1_ADM_s003
                                 + 0.0308185326093869 w_k1_ADM_s004
                                 + 0.0197039979251148 w_k1_ADM_s005
                                 + 0.0325605971926576 w_k1_ADM_s006
                                 + 0.0627341112669481 w_k1_ADM_s007
                                 + 0.0108349712303074 w_k1_ADM_s010
                                 + 0.0186829685141373 w_k1_ADM_s011
                                 + 0.0246650445605008 w_k1_ADM_s012
                                 + 0.0366790021381631 w_k1_ADM_s013
                                 + 0.0118156157907366 w_k1_ADM_s014
                                 + 0.0228699797672613 w_k1_ADM_s015
                                 + 0.0349467536909803 w_k1_ADM_s016
                                 + 0.0319172785921781 w_k1_ADM_s017
                                 + 0.0877710164195879 w_k1_ADM_s100
                                 + 0.0349466588014907 w_k1_ADM_s101
                                 + 0.041489323801681 w_k1_ADM_s102
                                 + 0.0371498959053143 w_k1_ADM_s103
                                 + 0.0338147282400157 w_k1_ADM_s104
                                 + 0.0428686383089947 w_k1_ADM_s105
                                 + 0.0163692849709753 w_k1_ADM_s106
                                 + 0.0177154843335977 w_k1_ADM_s107
                                 + 0.0456173383904445 w_k1_ADM_s110
                                 + 0.0217048742894603 w_k1_ADM_s111
                                 + 0.0239926829095713 w_k1_ADM_s112
                                 + 0.0363624761238636 w_k1_ADM_s113
                                 + 0.0206011958837397 w_k1_ADM_s114
                                 + 0.0126311075102608 w_k1_ADM_s115
                                 + 0.0214349303061013 w_k1_ADM_s116
                                 + 0.0709770679575551 w_k1_ADM_s117
                                 + 0.0547409339509273 w_k1_AXP_s000
                                 + 0.0212949691489387 w_k1_AXP_s001
                                 + 0.0283502000946051 w_k1_AXP_s002
                                 + 0.0177862613198647 w_k1_AXP_s003
                                 + 0.0268432872732501 w_k1_AXP_s004
                                 + 0.0182799636994885 w_k1_AXP_s005
                                 + 0.031864499576561 w_k1_AXP_s006
                                 + 0.0587249076085537 w_k1_AXP_s007
                                 + 0.00993630934178698 w_k1_AXP_s010
                                 + 0.0184104014478951 w_k1_AXP_s011
                                 + 0.0238065468896175 w_k1_AXP_s012
                                 + 0.0400055156441911 w_k1_AXP_s013
                                 + 0.011187184403616 w_k1_AXP_s014
                                 + 0.0257409114242188 w_k1_AXP_s015
                                 + 0.0383941395011007 w_k1_AXP_s016
                                 + 0.0343984142332547 w_k1_AXP_s017
                                 + 0.0786447421367159 w_k1_AXP_s100
                                 + 0.03009026158092 w_k1_AXP_s101
                                 + 0.0420265773024974 w_k1_AXP_s102
                                 + 0.0419519099909462 w_k1_AXP_s103
                                 + 0.0373540360707863 w_k1_AXP_s104
                                 + 0.0432216514123088 w_k1_AXP_s105
                                 + 0.0172693453082708 w_k1_AXP_s106
                                 + 0.0208923833203605 w_k1_AXP_s107
                                 + 0.0508360466737584 w_k1_AXP_s110
                                 + 0.0198105840112346 w_k1_AXP_s111
                                 + 0.0229466980042358 w_k1_AXP_s112
                                 + 0.0412706432230282 w_k1_AXP_s113
                                 + 0.0206022241940883 w_k1_AXP_s114
                                 + 0.0156604164687166 w_k1_AXP_s115
                                 + 0.0224795834692249 w_k1_AXP_s116
                                 + 0.0695865156435351 w_k1_AXP_s117
                                 + 0.0594247020455445 w_k1_DIS_s000
                                 + 0.0199392046866133 w_k1_DIS_s001
                                 + 0.0312246848858599 w_k1_DIS_s002
                                 + 0.0170383640049246 w_k1_DIS_s003
                                 + 0.0251198263355815 w_k1_DIS_s004
                                 + 0.0165922965688928 w_k1_DIS_s005
                                 + 0.0325353127617922 w_k1_DIS_s006
                                 + 0.0567692498987528 w_k1_DIS_s007
                                 + 0.0105612069965463 w_k1_DIS_s010
                                 + 0.0202267856439833 w_k1_DIS_s011
                                 + 0.0243665278096876 w_k1_DIS_s012
                                 + 0.038442667923831 w_k1_DIS_s013
                                 + 0.0115534140647573 w_k1_DIS_s014
                                 + 0.0265313390265323 w_k1_DIS_s015
                                 + 0.0393840844854198 w_k1_DIS_s016
                                 + 0.0336319700976485 w_k1_DIS_s017
                                 + 0.0841048641706557 w_k1_DIS_s100
                                 + 0.0325551324718304 w_k1_DIS_s101
                                 + 0.0424784617108646 w_k1_DIS_s102
                                 + 0.0447500551047074 w_k1_DIS_s103
                                 + 0.0397747185441034 w_k1_DIS_s104
                                 + 0.0430188703858721 w_k1_DIS_s105
                                 + 0.0175573111774506 w_k1_DIS_s106
                                 + 0.0216116549280583 w_k1_DIS_s107
                                 + 0.0474420911844401 w_k1_DIS_s110
                                 + 0.0196624319511427 w_k1_DIS_s111
                                 + 0.0225546502104346 w_k1_DIS_s112
                                 + 0.0431180299789711 w_k1_DIS_s113
                                 + 0.0237190178881744 w_k1_DIS_s114
                                 + 0.0151202771166558 w_k1_DIS_s115
                                 + 0.0217329057301863 w_k1_DIS_s116
                                 + 0.0745612722164351 w_k1_DIS_s117
                                 + 0.0551087350959215 w_k1_FDX_s000
                                 + 0.0193250304602198 w_k1_FDX_s001
                                 + 0.0294883456565068 w_k1_FDX_s002
                                 + 0.0163594528234521 w_k1_FDX_s003
                                 + 0.0263931177771696 w_k1_FDX_s004
                                 + 0.0170371537010018 w_k1_FDX_s005
                                 + 0.0311520243686634 w_k1_FDX_s006
                                 + 0.057318344640433 w_k1_FDX_s007
                                 + 0.0104766725707021 w_k1_FDX_s010
                                 + 0.0186231342819108 w_k1_FDX_s011
                                 + 0.0244725144931572 w_k1_FDX_s012
                                 + 0.0362309292751463 w_k1_FDX_s013
                                 + 0.0116080633461676 w_k1_FDX_s014
                                 + 0.0239512633652085 w_k1_FDX_s015
                                 + 0.0373356556428152 w_k1_FDX_s016
                                 + 0.0336146443847236 w_k1_FDX_s017
                                 + 0.0802802017081892 w_k1_FDX_s100
                                 + 0.0322533676993587 w_k1_FDX_s101
                                 + 0.0393991719610534 w_k1_FDX_s102
                                 + 0.0415117883977257 w_k1_FDX_s103
                                 + 0.0361891007895109 w_k1_FDX_s104
                                 + 0.0407801912425551 w_k1_FDX_s105
                                 + 0.0160695553391396 w_k1_FDX_s106
                                 + 0.0195081727671149 w_k1_FDX_s107
                                 + 0.0466598944048132 w_k1_FDX_s110
                                 + 0.0192577238079419 w_k1_FDX_s111
                                 + 0.0238861239039787 w_k1_FDX_s112
                                 + 0.0408230912463884 w_k1_FDX_s113
                                 + 0.0230268131895139 w_k1_FDX_s114
                                 + 0.0145795744049268 w_k1_FDX_s115
                                 + 0.0200064256568775 w_k1_FDX_s116
                                 + 0.0679781375861115 w_k1_FDX_s117
                                 + 0.057507486604409 w_k1_IBM_s000
                                 + 0.0197282430822143 w_k1_IBM_s001
                                 + 0.0312057771843738 w_k1_IBM_s002
                                 + 0.0175358468260807 w_k1_IBM_s003
                                 + 0.0280616175121251 w_k1_IBM_s004
                                 + 0.0177283712551915 w_k1_IBM_s005
                                 + 0.0306545329034914 w_k1_IBM_s006
                                 + 0.0550371049857075 w_k1_IBM_s007
                                 + 0.00970147941081588 w_k1_IBM_s010
                                 + 0.0185744421769916 w_k1_IBM_s011
                                 + 0.0249322756180314 w_k1_IBM_s012
                                 + 0.0372043548067799 w_k1_IBM_s013
                                 + 0.0119391366192803 w_k1_IBM_s014
                                 + 0.026301566900781 w_k1_IBM_s015
                                 + 0.0383694666049768 w_k1_IBM_s016
                                 + 0.033164328096446 w_k1_IBM_s017
                                 + 0.0840723271378982 w_k1_IBM_s100
                                 + 0.0317858330205144 w_k1_IBM_s101
                                 + 0.0405318963693627 w_k1_IBM_s102
                                 + 0.0419307341901717 w_k1_IBM_s103
                                 + 0.037200188651563 w_k1_IBM_s104
                                 + 0.04049236160301 w_k1_IBM_s105
                                 + 0.0175567982391592 w_k1_IBM_s106
                                 + 0.0201353922233084 w_k1_IBM_s107
                                 + 0.0466530816782507 w_k1_IBM_s110
                                 + 0.0203485370395304 w_k1_IBM_s111
                                 + 0.0232735169489437 w_k1_IBM_s112
                                 + 0.0408502504768873 w_k1_IBM_s113
                                 + 0.0233270024351816 w_k1_IBM_s114
                                 + 0.0137969079134949 w_k1_IBM_s115
                                 + 0.020922043727861 w_k1_IBM_s116
                                 + 0.0718745828697559 w_k1_IBM_s117
                                 + 0.0572822384111357 w_k1_INTC_s000
                                 + 0.021255900321986 w_k1_INTC_s001
                                 + 0.031871857209129 w_k1_INTC_s002
                                 + 0.0191599233697598 w_k1_INTC_s003
                                 + 0.024789785619771 w_k1_INTC_s004
                                 + 0.0174092455015647 w_k1_INTC_s005
                                 + 0.0268259735059243 w_k1_INTC_s006
                                 + 0.0576586430922926 w_k1_INTC_s007
                                 + 0.0101402143849292 w_k1_INTC_s010
                                 + 0.0161317614167785 w_k1_INTC_s011
                                 + 0.026367710483291 w_k1_INTC_s012
                                 + 0.0384312122934144 w_k1_INTC_s013
                                 + 0.0104109797426662 w_k1_INTC_s014
                                 + 0.0253207882932169 w_k1_INTC_s015
                                 + 0.0391119111508357 w_k1_INTC_s016
                                 + 0.0343440356734866 w_k1_INTC_s017
                                 + 0.0850073013297876 w_k1_INTC_s100
                                 + 0.0314296657121824 w_k1_INTC_s101
                                 + 0.0340557855687654 w_k1_INTC_s102
                                 + 0.0447539002551418 w_k1_INTC_s103
                                 + 0.0359509981900294 w_k1_INTC_s104
                                 + 0.0447120037546698 w_k1_INTC_s105
                                 + 0.0168325314266793 w_k1_INTC_s106
                                 + 0.0187751277392525 w_k1_INTC_s107
                                 + 0.0411455003902917 w_k1_INTC_s110
                                 + 0.0191366304495768 w_k1_INTC_s111
                                 + 0.0242713736477453 w_k1_INTC_s112
                                 + 0.0384952832918585 w_k1_INTC_s113
                                 + 0.0216629899224018 w_k1_INTC_s114
                                 + 0.0143098337878768 w_k1_INTC_s115
                                 + 0.0180622132788763 w_k1_INTC_s116
                                 + 0.0647599125881634 w_k1_INTC_s117
                                 + 0.057130906070583 w_k1_KO_s000
                                 + 0.0217360020381125 w_k1_KO_s001
                                 + 0.030258194147322 w_k1_KO_s002
                                 + 0.0165705366759293 w_k1_KO_s003
                                 + 0.0243973413668218 w_k1_KO_s004
                                 + 0.0159238970232676 w_k1_KO_s005
                                 + 0.0313286749290416 w_k1_KO_s006
                                 + 0.0551615009747009 w_k1_KO_s007
                                 + 0.010675801008089 w_k1_KO_s010
                                 + 0.0183155458969144 w_k1_KO_s011
                                 + 0.0233981910136118 w_k1_KO_s012
                                 + 0.0368208949568009 w_k1_KO_s013
                                 + 0.0123366112340148 w_k1_KO_s014
                                 + 0.0253294956810759 w_k1_KO_s015
                                 + 0.0354653347047848 w_k1_KO_s016
                                 + 0.0358982529814021 w_k1_KO_s017
                                 + 0.0913594631788232 w_k1_KO_s100
                                 + 0.0330687298765595 w_k1_KO_s101
                                 + 0.0396820488931037 w_k1_KO_s102
                                 + 0.0450837427177798 w_k1_KO_s103
                                 + 0.0380523831533049 w_k1_KO_s104
                                 + 0.0402824443055862 w_k1_KO_s105
                                 + 0.015417043449781 w_k1_KO_s106
                                 + 0.0209607056962803 w_k1_KO_s107
                                 + 0.0437772492583081 w_k1_KO_s110
                                 + 0.019587287322389 w_k1_KO_s111
                                 + 0.0206792621103159 w_k1_KO_s112
                                 + 0.0364860837076804 w_k1_KO_s113
                                 + 0.0214296355588937 w_k1_KO_s114
                                 + 0.0127893803951566 w_k1_KO_s115
                                 + 0.0202648000714094 w_k1_KO_s116
                                 + 0.0725511570135711 w_k1_KO_s117
                                 + 0.0557668109798267 w_k1_MAT_s000
                                 + 0.0304731911342557 w_k1_MAT_s001
                                 + 0.0231784829554922 w_k1_MAT_s002
                                 + 0.0173965282451357 w_k1_MAT_s003
                                 + 0.028702701635348 w_k1_MAT_s004
                                 + 0.0194705126141157 w_k1_MAT_s005
                                 + 0.0352466876858093 w_k1_MAT_s006
                                 + 0.0532833721274852 w_k1_MAT_s007
                                 + 0.0126435901205831 w_k1_MAT_s010
                                 + 0.017708562940699 w_k1_MAT_s011
                                 + 0.0198308794878928 w_k1_MAT_s012
                                 + 0.0495274893156578 w_k1_MAT_s013
                                 + 0.0095821529550332 w_k1_MAT_s014
                                 + 0.0258951519297225 w_k1_MAT_s015
                                 + 0.0320753122418249 w_k1_MAT_s016
                                 + 0.0461072006442968 w_k1_MAT_s017
                                 + 0.0918822649569481 w_k1_MAT_s100
                                 + 0.0353439065279383 w_k1_MAT_s101
                                 + 0.0459379791132461 w_k1_MAT_s102
                                 + 0.050288514829466 w_k1_MAT_s103
                                 + 0.0322113575748365 w_k1_MAT_s104
                                 + 0.0454026812314891 w_k1_MAT_s105
                                 + 0.0193339752081485 w_k1_MAT_s106
                                 + 0.025001166017376 w_k1_MAT_s107
                                 + 0.05095096212703 w_k1_MAT_s110
                                 + 0.0194323656948547 w_k1_MAT_s111
                                 + 0.0232415459611819 w_k1_MAT_s112
                                 + 0.0491222332922893 w_k1_MAT_s113
                                 + 0.0226245561765192 w_k1_MAT_s114
                                 + 0.019395569075346 w_k1_MAT_s115
                                 + 0.0221353546972487 w_k1_MAT_s116
                                 + 0.0719979132574007 w_k1_MAT_s117
                                 + 0.0576587506869629 w_k1_MSFT_s000
                                 + 0.0208647504707945 w_k1_MSFT_s001
                                 + 0.0282267276333036 w_k1_MSFT_s002
                                 + 0.0171204958422711 w_k1_MSFT_s003
                                 + 0.0252931138930081 w_k1_MSFT_s004
                                 + 0.016949701174642 w_k1_MSFT_s005
                                 + 0.0308237720058179 w_k1_MSFT_s006
                                 + 0.0625104495191409 w_k1_MSFT_s007
                                 + 0.00987581546577546 w_k1_MSFT_s010
                                 + 0.0165726160992388 w_k1_MSFT_s011
                                 + 0.0243664374483995 w_k1_MSFT_s012
                                 + 0.0356677456941136 w_k1_MSFT_s013
                                 + 0.0107574628153271 w_k1_MSFT_s014
                                 + 0.02517123548676 w_k1_MSFT_s015
                                 + 0.0403275381038506 w_k1_MSFT_s016
                                 + 0.0311980976201443 w_k1_MSFT_s017
                                 + 0.0808069206605953 w_k1_MSFT_s100
                                 + 0.0334635097632114 w_k1_MSFT_s101
                                 + 0.0367028673694448 w_k1_MSFT_s102
                                 + 0.0444483027587387 w_k1_MSFT_s103
                                 + 0.0371944839417586 w_k1_MSFT_s104
                                 + 0.0420764226944776 w_k1_MSFT_s105
                                 + 0.0160579927260207 w_k1_MSFT_s106
                                 + 0.0180345651219014 w_k1_MSFT_s107
                                 + 0.045612608797567 w_k1_MSFT_s110
                                 + 0.0196642574331335 w_k1_MSFT_s111
                                 + 0.0243387921165421 w_k1_MSFT_s112
                                 + 0.0379467921777051 w_k1_MSFT_s113
                                 + 0.0230209793867002 w_k1_MSFT_s114
                                 + 0.0139935062866611 w_k1_MSFT_s115
                                 + 0.0193165627450853 w_k1_MSFT_s116
                                 + 0.0689911814416457 w_k1_MSFT_s117
                                 + 0.0599184137629808 w_k1_PG_s000
                                 + 0.0207263173084778 w_k1_PG_s001
                                 + 0.0293878069493942 w_k1_PG_s002
                                 + 0.0162036907817549 w_k1_PG_s003
                                 + 0.0255242304369349 w_k1_PG_s004
                                 + 0.0171998239682941 w_k1_PG_s005
                                 + 0.0292189110958352 w_k1_PG_s006
                                 + 0.0564558377584578 w_k1_PG_s007
                                 + 0.00992134961629887 w_k1_PG_s010
                                 + 0.0178354047322506 w_k1_PG_s011
                                 + 0.0238156072204298 w_k1_PG_s012
                                 + 0.0376837862726391 w_k1_PG_s013
                                 + 0.0112343628686557 w_k1_PG_s014
                                 + 0.0253081679005203 w_k1_PG_s015
                                 + 0.0361088310613033 w_k1_PG_s016
                                 + 0.034748075022225 w_k1_PG_s017
                                 + 0.0893210435571696 w_k1_PG_s100
                                 + 0.0317121376771788 w_k1_PG_s101
                                 + 0.037108592397318 w_k1_PG_s102
                                 + 0.0410098960036789 w_k1_PG_s103
                                 + 0.0362187195894477 w_k1_PG_s104
                                 + 0.0392532779611913 w_k1_PG_s105
                                 + 0.0156863774846798 w_k1_PG_s106
                                 + 0.0203770810395563 w_k1_PG_s107
                                 + 0.0458436709515819 w_k1_PG_s110
                                 + 0.0200683874008644 w_k1_PG_s111
                                 + 0.0219813581114908 w_k1_PG_s112
                                 + 0.0381219675552876 w_k1_PG_s113
                                 + 0.0222687864194402 w_k1_PG_s114
                                 + 0.0143935797123778 w_k1_PG_s115
                                 + 0.0199778726293295 w_k1_PG_s116
                                 + 0.0705926725026934 w_k1_PG_s117 >= 1.06
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
