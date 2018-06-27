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
 w_k1_ADM_s0_return_constraint:  1.16127 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.95448 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.20467 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.987229 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.13624 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.987358 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.00531 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.775939 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.06075 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.15938 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.07972 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.802072 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.21033 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.721522 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.05558 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.05054 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.795123 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.66384 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.22123 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.784599 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.800577 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.935901 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.953521 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.702933 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.17713 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.24043 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.14731 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.912109 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.809763 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.42196 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.14781 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.08301 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.906243 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.829205 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.30544 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.613075 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.23448 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.805803 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.22842 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.873786 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.17915 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.944769 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.32764 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.01636 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.943642 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.659529 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.02928 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.61616 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.32421 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.753509 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.4745 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.938608 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.27248 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.00463 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.773388 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.85113 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.40379 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.826096 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.2899 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.02656 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.17409 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.759924 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.11108 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.17028 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.1922 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.916788 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.945433 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.73683 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.03472 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.01123 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.998153 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.84383 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.52339 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.540652 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.06713 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.677701 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.36613 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.881257 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.24546 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.941882 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.40545 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.930856 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.801516 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.55336 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.06895 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.91005 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.34732 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.590634 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.27034 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.860907 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.37859 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.96457 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.596573 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 2.02286 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.39901 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.825088 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.25204 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.988142 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.04453 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.690965 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.911661 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.2569 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.29799 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.732221 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.864443 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.7319 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.944813 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.25029 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.92706 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.809157 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.79281 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.513861 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.29898 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.630373 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.20076 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.928108 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.19548 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.930093 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.19545 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.928567 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.967653 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.70393 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.05392 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.52202 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.26883 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.798275 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.15307 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.924746 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.26328 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.02438 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.758617 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.57212 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.22812 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.765152 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.00168 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.9875 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.903549 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.664648 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.02236 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.24807 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.13977 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.874215 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.85706 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.42126 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.19808 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.21685 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.972217 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.90241 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.46595 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.63185 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.19425 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.849865 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.19342 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.944963 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.912315 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.16313 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.40208 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.886538 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.693539 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.803429 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.929834 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.49564 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.10299 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.600739 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.78456 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.24217 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.830312 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.738628 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.615403 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.47675 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.24828 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.16748 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.65504 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.0048 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.57405 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.36723 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.892526 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.717347 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.19405 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.663725 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.13583 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.31182 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.674876 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.924654 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.55672 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.958752 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.31982 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.853911 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.23045 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.584599 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.20965 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.929526 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.00194 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.929036 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.16494 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.92275 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.01372 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.812079 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.15712 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.52032 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.30639 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.863196 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.33297 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.958734 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.11144 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.970597 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.701588 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.45189 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.26083 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.788494 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.16523 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.974249 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.906397 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.742409 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.888895 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.972202 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.29073 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.976105 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.00638 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.36574 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.11696 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.30887 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.04361 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.973546 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.49831 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.771825 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.25266 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.864081 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.24861 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.918055 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.00271 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.03252 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.34618 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.910864 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.755059 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.733167 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.10962 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.71128 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.27094 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.607317 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.45086 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.07888 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.05263 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.836219 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.65737 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.51047 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.29166 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.873822 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.34071 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.955076 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.27102 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.941346 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.913439 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.993758 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.27774 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.754115 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.867734 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.36268 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.751702 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.0768 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.27501 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.91755 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.41777 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.735132 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.22294 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.635763 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.20587 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.965354 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.09541 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.13314 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.65349 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.919478 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.546018 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.683074 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.820409 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.41985 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.1081 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.430069 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.6614 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.28844 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.892887 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.931066 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.653069 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.48271 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.19056 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.00409 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.81294 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.81053 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.49276 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.05596 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.749969 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.666058 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.0794 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.548749 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.04133 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.20259 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.617964 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.7476 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.42082 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.919641 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.1635 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.496418 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.06105 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.429021 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.19036 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.984786 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.11557 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.948514 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.08111 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.917139 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.88768 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.78585 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.13412 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.4686 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.29909 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.831672 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.07362 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.890036 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.10624 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.01889 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.783317 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.39578 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.12617 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.821658 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.06493 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.07006 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.99696 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.716502 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.02278 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.24194 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.28771 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.869299 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.896719 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.37757 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.97688 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.10197 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.920263 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.867333 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.32244 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.638915 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.1824 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.792017 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.10897 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.968064 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.12397 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.04411 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.18116 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.931777 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.830393 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.810186 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.983093 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.27978 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.05372 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.809389 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.10103 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.94545 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.995576 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.997614 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.753991 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.38176 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.0484 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.833092 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.08354 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.97971 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.05586 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.84445 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.986471 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.05962 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.13163 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.806745 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.913305 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.22745 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.913096 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.04263 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.07515 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.894538 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.28099 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.74476 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.08899 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.835814 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.10000004546417 z_k1_s0000
                                 - 0.132999913992363 z_k1_s0001
                                 - 0.434000647764441 z_k1_s0002
                                 - 0.177000053696583 z_k1_s0003
                                 - 0.146999929807346 z_k1_s0004
                                 - 0.233999296836254 z_k1_s0005
                                 - 0.194000601650427 z_k1_s0006
                                 - 0.155999951224119 z_k1_s0007
                                 - 0.0989999299203543 z_k1_s0010
                                 - 0.0929999235039797 z_k1_s0011
                                 - 0.0669999347996512 z_k1_s0012
                                 - 0.196999291821869 z_k1_s0013
                                 - 0.246000145771476 z_k1_s0014
                                 - 0.16899980927825 z_k1_s0015
                                 - 0.248000108810306 z_k1_s0016
                                 - 0.0539999404474869 z_k1_s0017
                                 - 0.316997925740457 z_k1_s0020
                                 - 0.247999359749286 z_k1_s0021
                                 - 0.310998155410551 z_k1_s0022
                                 - 0.547998137421217 z_k1_s0023
                                 - 0.282999327804093 z_k1_s0024
                                 - 0.442000182930792 z_k1_s0025
                                 - 0.618999443649273 z_k1_s0026
                                 - 0.249999383792359 z_k1_s0027
                                 - 0.0720000296114128 z_k1_s0030
                                 - 0.226000253113574 z_k1_s0031
                                 - 0.0579999918203103 z_k1_s0032
                                 - 0.101999561449584 z_k1_s0033
                                 - 0.0710000131834793 z_k1_s0034
                                 - 0.204000083899003 z_k1_s0035
                                 - 0.0459999868850748 z_k1_s0036
                                 - 0.181999914684432 z_k1_s0037
                                 - 0.146000255281297 z_k1_s0040
                                 - 0.172000766496333 z_k1_s0041
                                 - 0.108000177660124 z_k1_s0042
                                 - 0.3330004797853 z_k1_s0043
                                 - 0.228999794926786 z_k1_s0044
                                 - 0.160000112088891 z_k1_s0045
                                 - 0.136000163275639 z_k1_s0046
                                 - 0.0760001552423456 z_k1_s0047
                                 - 0.358000652182052 z_k1_s0050
                                 - 0.164000377513789 z_k1_s0051
                                 - 0.21600066911621 z_k1_s0052
                                 - 0.134000304158816 z_k1_s0053
                                 - 0.136000356035947 z_k1_s0054
                                 - 0.199000844862342 z_k1_s0055
                                 - 0.163000439675473 z_k1_s0056
                                 - 0.392000300689788 z_k1_s0057
                                 - 0.183000207164391 z_k1_s0060
                                 - 0.0539999833272041 z_k1_s0061
                                 - 0.0750000032100034 z_k1_s0062
                                 - 0.0899999722120069 z_k1_s0063
                                 - 0.12599996109681 z_k1_s0064
                                 - 0.122999698356432 z_k1_s0065
                                 - 0.0930002349523843 z_k1_s0066
                                 - 0.0470000294329331 z_k1_s0067
                                 - 0.103999868105034 z_k1_s0070
                                 - 0.296000281529007 z_k1_s0071
                                 - 0.150999637537301 z_k1_s0072
                                 - 0.126999883358985 z_k1_s0073
                                 - 0.282999895516311 z_k1_s0074
                                 - 0.249999131022311 z_k1_s0075
                                 - 0.12599986443492 z_k1_s0076
                                 - 0.0629999322174598 z_k1_s0077
                                 - 0.0579999668322285 z_k1_s0100
                                 - 0.0579999668322285 z_k1_s0101
                                 - 0.141999946830631 z_k1_s0102
                                 - 0.15899977824696 z_k1_s0103
                                 - 0.251000220911916 z_k1_s0104
                                 - 0.27499998291144 z_k1_s0105
                                 - 0.232000409328959 z_k1_s0106
                                 - 0.179999569996544 z_k1_s0107
                                 - 0.369000962602803 z_k1_s0110
                                 - 0.495000779102067 z_k1_s0111
                                 - 0.530998226676927 z_k1_s0112
                                 - 0.49800027473373 z_k1_s0113
                                 - 0.54400070774067 z_k1_s0114
                                 - 0.315999489791878 z_k1_s0115
                                 - 0.451999841726791 z_k1_s0116
                                 - 0.294999520375334 z_k1_s0117
                                 - 0.263000179283232 z_k1_s0120
                                 - 0.604001307651914 z_k1_s0121
                                 - 0.46400006302671 z_k1_s0122
                                 - 0.833999517622193 z_k1_s0123
                                 - 0.527000523688059 z_k1_s0124
                                 - 0.696998768361778 z_k1_s0125
                                 - 0.634001290300194 z_k1_s0126
                                 - 0.947999451685658 z_k1_s0127
                                 - 0.185000749496325 z_k1_s0130
                                 - 0.125000119997246 z_k1_s0131
                                 - 0.15400073779702 z_k1_s0132
                                 - 0.120000054997314 z_k1_s0133
                                 - 0.192000509395998 z_k1_s0134
                                 - 0.217000081895131 z_k1_s0135
                                 - 0.208999496294903 z_k1_s0136
                                 - 0.303000357093371 z_k1_s0137
                                 - 0.772999747357199 z_k1_s0140
                                 - 0.155999901240026 z_k1_s0141
                                 - 0.509999215977214 z_k1_s0142
                                 - 0.163999865431836 z_k1_s0143
                                 - 0.154999935691036 z_k1_s0144
                                 - 0.154999935691036 z_k1_s0145
                                 - 0.235999782958018 z_k1_s0146
                                 - 0.249001013694128 z_k1_s0147
                                 - 0.251000120194623 z_k1_s0150
                                 - 0.295999850006305 z_k1_s0151
                                 - 0.133000118932628 z_k1_s0152
                                 - 0.0729999598502025 z_k1_s0153
                                 - 0.101999958840014 z_k1_s0154
                                 - 0.16899943538181 z_k1_s0155
                                 - 0.35300085803491 z_k1_s0156
                                 - 0.181000713598733 z_k1_s0157
                                 - 0.497998494940544 z_k1_s0160
                                 - 0.265000195249364 z_k1_s0161
                                 - 0.214999637884559 z_k1_s0162
                                 - 0.376999236719246 z_k1_s0163
                                 - 0.276999640586389 z_k1_s0164
                                 - 0.275999897724519 z_k1_s0165
                                 - 0.210999654039245 z_k1_s0166
                                 - 0.412000360863039 z_k1_s0167
                                 - 0.323998770136065 z_k1_s0170
                                 - 0.363998222684741 z_k1_s0171
                                 - 0.142999706732254 z_k1_s0172
                                 - 0.232999337714564 z_k1_s0173
                                 - 0.452999284724704 z_k1_s0174
                                 - 0.173999467332004 z_k1_s0175
                                 - 0.19399944010571 z_k1_s0176
                                 - 0.579998224440013 z_k1_s0177
                                 - 0.138999219999165 z_k1_s1000
                                 - 0.376997771369603 z_k1_s1001
                                 - 0.417998603692798 z_k1_s1002
                                 - 0.110999364379262 z_k1_s1003
                                 - 0.182998898230982 z_k1_s1004
                                 - 0.491996703954052 z_k1_s1005
                                 - 0.114999375381925 z_k1_s1006
                                 - 0.378997001975351 z_k1_s1007
                                 - 0.102999864214646 z_k1_s1010
                                 - 0.180999527631981 z_k1_s1011
                                 - 0.118999746941365 z_k1_s1012
                                 - 0.102999864214646 z_k1_s1013
                                 - 0.0969998611672105 z_k1_s1014
                                 - 0.182999654047568 z_k1_s1015
                                 - 0.178999401216394 z_k1_s1016
                                 - 0.289000084084921 z_k1_s1017
                                 - 0.611000675932861 z_k1_s1020
                                 - 0.221999706304458 z_k1_s1021
                                 - 0.600999907974454 z_k1_s1022
                                 - 0.299999554289886 z_k1_s1023
                                 - 0.886000116823832 z_k1_s1024
                                 - 0.292999378018422 z_k1_s1025
                                 - 0.317999491417337 z_k1_s1026
                                 - 0.381998264260996 z_k1_s1027
                                 - 0.194998906800323 z_k1_s1030
                                 - 0.179999139114909 z_k1_s1031
                                 - 0.143999268472151 z_k1_s1032
                                 - 0.165999213209267 z_k1_s1033
                                 - 0.384999032899546 z_k1_s1034
                                 - 0.291998760458921 z_k1_s1035
                                 - 0.162999045573302 z_k1_s1036
                                 - 0.615996739848216 z_k1_s1037
                                 - 0.0629998585842689 z_k1_s1040
                                 - 0.300999203547292 z_k1_s1041
                                 - 0.242999609853313 z_k1_s1042
                                 - 0.161999739902209 z_k1_s1043
                                 - 0.082999830947496 z_k1_s1044
                                 - 0.0679998818750182 z_k1_s1045
                                 - 0.22199977779166 z_k1_s1046
                                 - 0.0659999208786266 z_k1_s1047
                                 - 0.289999809076444 z_k1_s1050
                                 - 0.368998175733299 z_k1_s1051
                                 - 0.421000675820387 z_k1_s1052
                                 - 0.52499960200051 z_k1_s1053
                                 - 0.451000132449497 z_k1_s1054
                                 - 0.315999540621474 z_k1_s1055
                                 - 0.333000649976583 z_k1_s1056
                                 - 0.331998440957793 z_k1_s1057
                                 - 0.469997032825513 z_k1_s1060
                                 - 0.576996526765902 z_k1_s1061
                                 - 0.272998938893636 z_k1_s1062
                                 - 0.596996939649746 z_k1_s1063
                                 - 0.339998693061608 z_k1_s1064
                                 - 0.610999038660552 z_k1_s1065
                                 - 0.35699850101524 z_k1_s1066
                                 - 0.394998199499273 z_k1_s1067
                                 - 0.116999730235208 z_k1_s1070
                                 - 0.458999927498068 z_k1_s1071
                                 - 0.0929997670587852 z_k1_s1072
                                 - 0.211000548674641 z_k1_s1073
                                 - 0.339000111615953 z_k1_s1074
                                 - 0.152999674999843 z_k1_s1075
                                 - 0.29499969779329 z_k1_s1076
                                 - 0.137999788264432 z_k1_s1077
                                 - 0.558001137689259 z_k1_s1100
                                 - 0.313000213243894 z_k1_s1101
                                 - 0.820000168932481 z_k1_s1102
                                 - 0.395998593970731 z_k1_s1103
                                 - 0.75399913260271 z_k1_s1104
                                 - 0.52400187449039 z_k1_s1105
                                 - 0.20500004223312 z_k1_s1106
                                 - 0.242000114938678 z_k1_s1107
                                 - 0.571001081479995 z_k1_s1110
                                 - 1.02399766653858 z_k1_s1111
                                 - 0.636001088719073 z_k1_s1112
                                 - 0.398000524676826 z_k1_s1113
                                 - 1.62699992958705 z_k1_s1114
                                 - 1.35000129404387 z_k1_s1115
                                 - 0.77900244276679 z_k1_s1116
                                 - 1.04899824116974 z_k1_s1117
                                 - 0.422001288194187 z_k1_s1120
                                 - 1.10600054009799 z_k1_s1121
                                 - 0.542002002926049 z_k1_s1122
                                 - 0.949006660229028 z_k1_s1123
                                 - 1.42100804860243 z_k1_s1124
                                 - 1.83901018252543 z_k1_s1125
                                 - 0.680002469141376 z_k1_s1126
                                 - 0.946002749133851 z_k1_s1127
                                 - 1.86600007211166 z_k1_s1130
                                 - 0.55199930608632 z_k1_s1131
                                 - 1.10600276581755 z_k1_s1132
                                 - 1.84399581718546 z_k1_s1133
                                 - 1.55499642542921 z_k1_s1134
                                 - 1.57000031778721 z_k1_s1135
                                 - 1.11699398928912 z_k1_s1136
                                 - 1.29399848394574 z_k1_s1137
                                 - 0.745002524297113 z_k1_s1140
                                 - 0.298000007118115 z_k1_s1141
                                 - 0.316000199320692 z_k1_s1142
                                 - 0.731001205105369 z_k1_s1143
                                 - 0.502002185413978 z_k1_s1144
                                 - 0.497000210372975 z_k1_s1145
                                 - 0.858998227276081 z_k1_s1146
                                 - 1.06500259922937 z_k1_s1147
                                 - 0.228001158891011 z_k1_s1150
                                 - 0.668001391386141 z_k1_s1151
                                 - 0.56800145191056 z_k1_s1152
                                 - 0.970006396149138 z_k1_s1153
                                 - 0.655002596380486 z_k1_s1154
                                 - 0.959002811428312 z_k1_s1155
                                 - 0.346001885556311 z_k1_s1156
                                 - 0.594004029947581 z_k1_s1157
                                 - 0.243000225633559 z_k1_s1160
                                 - 1.120999027489 z_k1_s1161
                                 - 0.292000125082184 z_k1_s1162
                                 - 0.789999586009413 z_k1_s1163
                                 - 0.246000400548927 z_k1_s1164
                                 - 0.585999521773933 z_k1_s1165
                                 - 0.572002085504938 z_k1_s1166
                                 - 0.375000316903616 z_k1_s1167
                                 - 0.181000197169624 z_k1_s1170
                                 - 0.51700122261231 z_k1_s1171
                                 - 0.680001163775852 z_k1_s1172
                                 - 0.301000402541833 z_k1_s1173
                                 - 0.739999014459454 z_k1_s1174
                                 - 0.874002171395008 z_k1_s1175
                                 - 0.387000324524999 z_k1_s1176
                                 - 0.823998707819499 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.05008 w_k1_ADM_s000
                                 + 1.28325 w_k1_AXP_s000
                                 + 1.35357 w_k1_DIS_s000 + 1.1887 w_k1_FDX_s000
                                 + 1.06315 w_k1_IBM_s000
                                 + 1.14892 w_k1_INTC_s000
                                 + 1.30846 w_k1_KO_s000 + 1.20833 w_k1_MAT_s000
                                 + 1.30908 w_k1_MSFT_s000
                                 + 1.17072 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.826947 w_k1_ADM_s000
                                 + 1.08948 w_k1_AXP_s000
                                 + 0.911726 w_k1_DIS_s000
                                 + 1.04583 w_k1_FDX_s000
                                 + 0.964627 w_k1_IBM_s000
                                 + 1.0449 w_k1_INTC_s000
                                 + 0.992106 w_k1_KO_s000
                                 + 1.35125 w_k1_MAT_s000
                                 + 1.1482 w_k1_MSFT_s000 + 1.02069 w_k1_PG_s000
                                 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.79044 w_k1_ADM_s000
                                 + 0.737958 w_k1_AXP_s000
                                 + 0.745323 w_k1_DIS_s000
                                 + 0.795204 w_k1_FDX_s000
                                 + 0.922403 w_k1_IBM_s000
                                 + 0.730076 w_k1_INTC_s000
                                 + 0.861519 w_k1_KO_s000
                                 + 0.953668 w_k1_MAT_s000
                                 + 0.941865 w_k1_MSFT_s000
                                 + 0.923411 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.769071 w_k1_ADM_s000
                                 + 0.69744 w_k1_AXP_s000
                                 + 1.10626 w_k1_DIS_s000
                                 + 0.912897 w_k1_FDX_s000
                                 + 0.911179 w_k1_IBM_s000
                                 + 0.945606 w_k1_INTC_s000
                                 + 1.11689 w_k1_KO_s000
                                 + 0.976576 w_k1_MAT_s000
                                 + 1.14524 w_k1_MSFT_s000
                                 + 1.00506 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.774769 w_k1_ADM_s000
                                 + 0.769957 w_k1_AXP_s000
                                 + 0.964223 w_k1_DIS_s000
                                 + 1.02998 w_k1_FDX_s000
                                 + 0.584113 w_k1_IBM_s000
                                 + 1.1098 w_k1_INTC_s000
                                 + 0.799684 w_k1_KO_s000
                                 + 0.649319 w_k1_MAT_s000
                                 + 1.09857 w_k1_MSFT_s000
                                 + 0.924234 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.056 w_k1_ADM_s000
                                 + 0.91009 w_k1_AXP_s000 + 1.2743 w_k1_DIS_s000
                                 + 1.17786 w_k1_FDX_s000
                                 + 0.723637 w_k1_IBM_s000
                                 + 1.18264 w_k1_INTC_s000
                                 + 0.940165 w_k1_KO_s000
                                 + 0.704614 w_k1_MAT_s000
                                 + 1.29668 w_k1_MSFT_s000
                                 + 1.02803 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.948268 w_k1_ADM_s000
                                 + 1.3818 w_k1_AXP_s000 + 1.46821 w_k1_DIS_s000
                                 + 1.25366 w_k1_FDX_s000
                                 + 1.38206 w_k1_IBM_s000
                                 + 1.17971 w_k1_INTC_s000
                                 + 1.38604 w_k1_KO_s000 + 2.1106 w_k1_MAT_s000
                                 + 1.16081 w_k1_MSFT_s000
                                 + 1.22041 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.955504 w_k1_ADM_s000
                                 + 1.10219 w_k1_AXP_s000
                                 + 1.04164 w_k1_DIS_s000
                                 + 0.999586 w_k1_FDX_s000
                                 + 1.35108 w_k1_IBM_s000
                                 + 1.12364 w_k1_INTC_s000
                                 + 1.30843 w_k1_KO_s000 + 1.3181 w_k1_MAT_s000
                                 + 1.12354 w_k1_MSFT_s000
                                 + 1.15397 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.630047 w_k1_ADM_s001
                                 + 0.729761 w_k1_AXP_s001
                                 + 0.501023 w_k1_DIS_s001
                                 + 0.679949 w_k1_FDX_s001
                                 + 0.597564 w_k1_IBM_s001
                                 + 0.712278 w_k1_INTC_s001
                                 + 0.567407 w_k1_KO_s001
                                 + 0.601537 w_k1_MAT_s001
                                 + 0.617734 w_k1_MSFT_s001
                                 + 0.744753 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.77327 w_k1_ADM_s001
                                 + 0.915599 w_k1_AXP_s001
                                 + 0.73182 w_k1_DIS_s001
                                 + 0.769571 w_k1_FDX_s001
                                 + 1.00892 w_k1_IBM_s001
                                 + 0.712737 w_k1_INTC_s001
                                 + 0.777072 w_k1_KO_s001 + 1.1772 w_k1_MAT_s001
                                 + 0.699834 w_k1_MSFT_s001
                                 + 0.732947 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.701413 w_k1_ADM_s001
                                 + 1.06269 w_k1_AXP_s001
                                 + 0.880487 w_k1_DIS_s001
                                 + 0.910219 w_k1_FDX_s001
                                 + 1.47606 w_k1_IBM_s001
                                 + 1.10992 w_k1_INTC_s001
                                 + 1.11686 w_k1_KO_s001 + 1.6172 w_k1_MAT_s001
                                 + 0.868279 w_k1_MSFT_s001
                                 + 0.969579 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.806721 w_k1_ADM_s001
                                 + 1.13849 w_k1_AXP_s001
                                 + 1.45863 w_k1_DIS_s001
                                 + 1.13659 w_k1_FDX_s001
                                 + 1.25935 w_k1_IBM_s001
                                 + 1.09987 w_k1_INTC_s001 + 1.3341 w_k1_KO_s001
                                 + 1.33887 w_k1_MAT_s001
                                 + 1.24182 w_k1_MSFT_s001
                                 + 1.10288 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.899085 w_k1_ADM_s001
                                 + 0.8289 w_k1_AXP_s001
                                 + 0.872418 w_k1_DIS_s001
                                 + 0.886707 w_k1_FDX_s001
                                 + 0.809007 w_k1_IBM_s001
                                 + 0.99453 w_k1_INTC_s001
                                 + 0.878619 w_k1_KO_s001
                                 + 0.680299 w_k1_MAT_s001
                                 + 0.848203 w_k1_MSFT_s001
                                 + 0.819444 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.0166 w_k1_ADM_s001
                                 + 1.17578 w_k1_AXP_s001
                                 + 1.12945 w_k1_DIS_s001
                                 + 1.06459 w_k1_FDX_s001
                                 + 0.907482 w_k1_IBM_s001
                                 + 0.954969 w_k1_INTC_s001
                                 + 1.02037 w_k1_KO_s001 + 1.0602 w_k1_MAT_s001
                                 + 0.953165 w_k1_MSFT_s001
                                 + 0.956754 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.922128 w_k1_ADM_s001
                                 + 1.30837 w_k1_AXP_s001
                                 + 1.17458 w_k1_DIS_s001
                                 + 1.04568 w_k1_FDX_s001
                                 + 1.30723 w_k1_IBM_s001
                                 + 0.816751 w_k1_INTC_s001
                                 + 1.09928 w_k1_KO_s001 + 1.46092 w_k1_MAT_s001
                                 + 1.01431 w_k1_MSFT_s001
                                 + 1.08022 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.32339 w_k1_ADM_s001
                                 + 1.77179 w_k1_AXP_s001
                                 + 1.74823 w_k1_DIS_s001
                                 + 1.54071 w_k1_FDX_s001
                                 + 1.41732 w_k1_IBM_s001
                                 + 1.40639 w_k1_INTC_s001
                                 + 1.25759 w_k1_KO_s001 + 1.41943 w_k1_MAT_s001
                                 + 1.21573 w_k1_MSFT_s001
                                 + 1.35799 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.726814 w_k1_ADM_s002
                                 + 0.612162 w_k1_AXP_s002
                                 + 0.588278 w_k1_DIS_s002
                                 + 0.696281 w_k1_FDX_s002
                                 + 0.873171 w_k1_IBM_s002
                                 + 0.735724 w_k1_INTC_s002
                                 + 0.708971 w_k1_KO_s002
                                 + 0.834516 w_k1_MAT_s002
                                 + 0.767362 w_k1_MSFT_s002
                                 + 0.835122 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.03684 w_k1_ADM_s002
                                 + 1.08069 w_k1_AXP_s002 + 1.6743 w_k1_DIS_s002
                                 + 1.32928 w_k1_FDX_s002
                                 + 1.05001 w_k1_IBM_s002
                                 + 1.17941 w_k1_INTC_s002 + 1.1104 w_k1_KO_s002
                                 + 1.026 w_k1_MAT_s002 + 1.28402 w_k1_MSFT_s002
                                 + 1.38567 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.773395 w_k1_ADM_s002
                                 + 0.75709 w_k1_AXP_s002
                                 + 0.962961 w_k1_DIS_s002
                                 + 0.952403 w_k1_FDX_s002
                                 + 0.964926 w_k1_IBM_s002
                                 + 1.00228 w_k1_INTC_s002
                                 + 0.877652 w_k1_KO_s002
                                 + 1.00381 w_k1_MAT_s002
                                 + 0.932046 w_k1_MSFT_s002
                                 + 1.04763 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.878118 w_k1_ADM_s002
                                 + 0.980208 w_k1_AXP_s002
                                 + 0.736768 w_k1_DIS_s002
                                 + 0.915306 w_k1_FDX_s002
                                 + 1.12981 w_k1_IBM_s002
                                 + 0.882549 w_k1_INTC_s002
                                 + 0.80526 w_k1_KO_s002 + 1.19976 w_k1_MAT_s002
                                 + 0.806214 w_k1_MSFT_s002
                                 + 0.900013 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.781791 w_k1_ADM_s002
                                 + 1.07242 w_k1_AXP_s002
                                 + 0.858894 w_k1_DIS_s002
                                 + 1.03096 w_k1_FDX_s002
                                 + 0.662624 w_k1_IBM_s002
                                 + 0.92842 w_k1_INTC_s002
                                 + 0.670759 w_k1_KO_s002
                                 + 0.761321 w_k1_MAT_s002
                                 + 0.858352 w_k1_MSFT_s002
                                 + 0.881732 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.02124 w_k1_ADM_s002
                                 + 0.93913 w_k1_AXP_s002
                                 + 1.17923 w_k1_DIS_s002 + 1.0372 w_k1_FDX_s002
                                 + 1.08594 w_k1_IBM_s002
                                 + 1.02089 w_k1_INTC_s002
                                 + 1.08237 w_k1_KO_s002 + 1.36452 w_k1_MAT_s002
                                 + 0.958588 w_k1_MSFT_s002
                                 + 1.10056 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.22341 w_k1_ADM_s002
                                 + 1.3688 w_k1_AXP_s002 + 1.22046 w_k1_DIS_s002
                                 + 1.20199 w_k1_FDX_s002
                                 + 1.27678 w_k1_IBM_s002
                                 + 1.04298 w_k1_INTC_s002
                                 + 1.12872 w_k1_KO_s002 + 1.83515 w_k1_MAT_s002
                                 + 1.04308 w_k1_MSFT_s002
                                 + 1.01512 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.69926 w_k1_ADM_s002
                                 + 0.716441 w_k1_AXP_s002
                                 + 0.851878 w_k1_DIS_s002
                                 + 0.827573 w_k1_FDX_s002
                                 + 0.885061 w_k1_IBM_s002
                                 + 0.905051 w_k1_INTC_s002
                                 + 0.807427 w_k1_KO_s002
                                 + 0.775937 w_k1_MAT_s002
                                 + 0.885357 w_k1_MSFT_s002
                                 + 0.871766 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.920626 w_k1_ADM_s003
                                 + 1.2156 w_k1_AXP_s003 + 1.71133 w_k1_DIS_s003
                                 + 1.09271 w_k1_FDX_s003
                                 + 1.57083 w_k1_IBM_s003
                                 + 1.0357 w_k1_INTC_s003 + 1.22015 w_k1_KO_s003
                                 + 1.76668 w_k1_MAT_s003
                                 + 1.19867 w_k1_MSFT_s003
                                 + 1.03302 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.25691 w_k1_ADM_s003
                                 + 0.882257 w_k1_AXP_s003
                                 + 0.758695 w_k1_DIS_s003
                                 + 0.799434 w_k1_FDX_s003
                                 + 0.842009 w_k1_IBM_s003
                                 + 0.800342 w_k1_INTC_s003
                                 + 0.80058 w_k1_KO_s003 + 0.75877 w_k1_MAT_s003
                                 + 0.842741 w_k1_MSFT_s003
                                 + 0.89958 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.22975 w_k1_ADM_s003
                                 + 1.07885 w_k1_AXP_s003
                                 + 0.785626 w_k1_DIS_s003
                                 + 0.82524 w_k1_FDX_s003
                                 + 1.60784 w_k1_IBM_s003
                                 + 0.980344 w_k1_INTC_s003
                                 + 1.16754 w_k1_KO_s003 + 2.17633 w_k1_MAT_s003
                                 + 0.920641 w_k1_MSFT_s003
                                 + 0.962637 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.09894 w_k1_ADM_s003
                                 + 1.05079 w_k1_AXP_s003
                                 + 0.979721 w_k1_DIS_s003
                                 + 0.999614 w_k1_FDX_s003
                                 + 1.36338 w_k1_IBM_s003
                                 + 0.992755 w_k1_INTC_s003
                                 + 1.1695 w_k1_KO_s003 + 1.49497 w_k1_MAT_s003
                                 + 1.0056 w_k1_MSFT_s003 + 1.11106 w_k1_PG_s003
                                 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.24611 w_k1_ADM_s003
                                 + 0.985991 w_k1_AXP_s003
                                 + 0.920058 w_k1_DIS_s003
                                 + 1.1432 w_k1_FDX_s003
                                 + 0.663507 w_k1_IBM_s003
                                 + 1.11233 w_k1_INTC_s003
                                 + 0.706603 w_k1_KO_s003
                                 + 0.472582 w_k1_MAT_s003
                                 + 1.04929 w_k1_MSFT_s003
                                 + 0.927501 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.22662 w_k1_ADM_s003
                                 + 1.16973 w_k1_AXP_s003
                                 + 0.936338 w_k1_DIS_s003
                                 + 1.03942 w_k1_FDX_s003
                                 + 0.801062 w_k1_IBM_s003
                                 + 0.974559 w_k1_INTC_s003
                                 + 0.793005 w_k1_KO_s003
                                 + 0.994676 w_k1_MAT_s003
                                 + 1.04263 w_k1_MSFT_s003
                                 + 1.05322 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.34287 w_k1_ADM_s003
                                 + 1.53257 w_k1_AXP_s003
                                 + 1.24888 w_k1_DIS_s003
                                 + 1.06888 w_k1_FDX_s003
                                 + 1.90521 w_k1_IBM_s003
                                 + 1.08954 w_k1_INTC_s003
                                 + 1.39055 w_k1_KO_s003 + 2.50973 w_k1_MAT_s003
                                 + 1.05503 w_k1_MSFT_s003
                                 + 1.22146 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.3004 w_k1_ADM_s003
                                 + 1.4181 w_k1_AXP_s003 + 1.42581 w_k1_DIS_s003
                                 + 1.35059 w_k1_FDX_s003
                                 + 1.00323 w_k1_IBM_s003 + 1.323 w_k1_INTC_s003
                                 + 1.12082 w_k1_KO_s003 + 1.15346 w_k1_MAT_s003
                                 + 1.28673 w_k1_MSFT_s003
                                 + 1.14283 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.25567 w_k1_ADM_s004
                                 + 1.50109 w_k1_AXP_s004
                                 + 1.52989 w_k1_DIS_s004
                                 + 1.35515 w_k1_FDX_s004
                                 + 0.996648 w_k1_IBM_s004
                                 + 0.96386 w_k1_INTC_s004
                                 + 1.01471 w_k1_KO_s004 + 1.32334 w_k1_MAT_s004
                                 + 1.13586 w_k1_MSFT_s004
                                 + 1.15924 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.875005 w_k1_ADM_s004
                                 + 0.805484 w_k1_AXP_s004
                                 + 1.01335 w_k1_DIS_s004
                                 + 0.811811 w_k1_FDX_s004
                                 + 1.35402 w_k1_IBM_s004
                                 + 0.910473 w_k1_INTC_s004
                                 + 1.05333 w_k1_KO_s004 + 1.09127 w_k1_MAT_s004
                                 + 0.927134 w_k1_MSFT_s004
                                 + 1.20931 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.981832 w_k1_ADM_s004
                                 + 1.47042 w_k1_AXP_s004
                                 + 1.48918 w_k1_DIS_s004
                                 + 1.17152 w_k1_FDX_s004 + 1.507 w_k1_IBM_s004
                                 + 1.31612 w_k1_INTC_s004
                                 + 1.47084 w_k1_KO_s004 + 1.56026 w_k1_MAT_s004
                                 + 1.18767 w_k1_MSFT_s004
                                 + 1.21021 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.901667 w_k1_ADM_s004
                                 + 1.05652 w_k1_AXP_s004
                                 + 0.834934 w_k1_DIS_s004
                                 + 0.913911 w_k1_FDX_s004
                                 + 0.813692 w_k1_IBM_s004
                                 + 0.840269 w_k1_INTC_s004
                                 + 0.755866 w_k1_KO_s004
                                 + 0.912769 w_k1_MAT_s004
                                 + 0.808945 w_k1_MSFT_s004
                                 + 0.890378 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.929062 w_k1_ADM_s004
                                 + 0.997223 w_k1_AXP_s004
                                 + 1.22191 w_k1_DIS_s004
                                 + 1.08989 w_k1_FDX_s004
                                 + 0.93076 w_k1_IBM_s004
                                 + 1.06153 w_k1_INTC_s004
                                 + 1.03623 w_k1_KO_s004
                                 + 0.658564 w_k1_MAT_s004
                                 + 0.92276 w_k1_MSFT_s004
                                 + 1.01528 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.01469 w_k1_ADM_s004
                                 + 1.11139 w_k1_AXP_s004
                                 + 0.852573 w_k1_DIS_s004
                                 + 0.923764 w_k1_FDX_s004
                                 + 1.21337 w_k1_IBM_s004
                                 + 0.792426 w_k1_INTC_s004
                                 + 1.00817 w_k1_KO_s004 + 1.5627 w_k1_MAT_s004
                                 + 0.836303 w_k1_MSFT_s004
                                 + 1.02688 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.13339 w_k1_ADM_s004
                                 + 0.899571 w_k1_AXP_s004
                                 + 0.850288 w_k1_DIS_s004
                                 + 1.07165 w_k1_FDX_s004
                                 + 0.630481 w_k1_IBM_s004
                                 + 0.812097 w_k1_INTC_s004
                                 + 0.690554 w_k1_KO_s004
                                 + 0.470457 w_k1_MAT_s004
                                 + 0.806057 w_k1_MSFT_s004
                                 + 0.953554 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.12583 w_k1_ADM_s004
                                 + 1.21958 w_k1_AXP_s004 + 1.271 w_k1_DIS_s004
                                 + 1.0152 w_k1_FDX_s004 + 1.36614 w_k1_IBM_s004
                                 + 1.14678 w_k1_INTC_s004
                                 + 1.29394 w_k1_KO_s004
                                 + 0.867119 w_k1_MAT_s004
                                 + 0.930567 w_k1_MSFT_s004
                                 + 1.07215 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.888189 w_k1_ADM_s005
                                 + 1.12323 w_k1_AXP_s005
                                 + 1.04722 w_k1_DIS_s005
                                 + 0.995511 w_k1_FDX_s005
                                 + 1.22192 w_k1_IBM_s005
                                 + 0.97898 w_k1_INTC_s005 + 1.118 w_k1_KO_s005
                                 + 1.55654 w_k1_MAT_s005
                                 + 0.881791 w_k1_MSFT_s005
                                 + 1.02738 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.809506 w_k1_ADM_s005
                                 + 0.653839 w_k1_AXP_s005
                                 + 0.667425 w_k1_DIS_s005
                                 + 0.866091 w_k1_FDX_s005
                                 + 0.610075 w_k1_IBM_s005
                                 + 0.876454 w_k1_INTC_s005
                                 + 0.779076 w_k1_KO_s005
                                 + 0.656104 w_k1_MAT_s005
                                 + 0.84721 w_k1_MSFT_s005
                                 + 0.801169 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.20634 w_k1_ADM_s005
                                 + 1.49699 w_k1_AXP_s005
                                 + 1.38038 w_k1_DIS_s005
                                 + 1.30965 w_k1_FDX_s005
                                 + 1.55189 w_k1_IBM_s005
                                 + 1.26359 w_k1_INTC_s005
                                 + 1.57765 w_k1_KO_s005 + 2.89396 w_k1_MAT_s005
                                 + 1.1382 w_k1_MSFT_s005 + 1.24955 w_k1_PG_s005
                                 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.23329 w_k1_ADM_s005
                                 + 1.16338 w_k1_AXP_s005
                                 + 1.77976 w_k1_DIS_s005 + 1.3947 w_k1_FDX_s005
                                 + 1.10525 w_k1_IBM_s005
                                 + 1.19619 w_k1_INTC_s005
                                 + 1.27293 w_k1_KO_s005 + 1.32985 w_k1_MAT_s005
                                 + 1.28501 w_k1_MSFT_s005
                                 + 1.05488 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.17958 w_k1_ADM_s005
                                 + 0.95321 w_k1_AXP_s005
                                 + 0.930977 w_k1_DIS_s005
                                 + 1.13523 w_k1_FDX_s005
                                 + 0.78388 w_k1_IBM_s005
                                 + 0.958546 w_k1_INTC_s005
                                 + 0.95975 w_k1_KO_s005
                                 + 0.980742 w_k1_MAT_s005
                                 + 0.913651 w_k1_MSFT_s005
                                 + 0.82362 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.01902 w_k1_ADM_s005
                                 + 0.756413 w_k1_AXP_s005
                                 + 1.10034 w_k1_DIS_s005
                                 + 0.991575 w_k1_FDX_s005
                                 + 0.763992 w_k1_IBM_s005
                                 + 0.866075 w_k1_INTC_s005
                                 + 1.09842 w_k1_KO_s005
                                 + 0.679078 w_k1_MAT_s005
                                 + 0.958363 w_k1_MSFT_s005
                                 + 1.02235 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.94836 w_k1_ADM_s005
                                 + 0.886184 w_k1_AXP_s005
                                 + 1.23733 w_k1_DIS_s005
                                 + 1.12127 w_k1_FDX_s005
                                 + 1.15074 w_k1_IBM_s005
                                 + 1.19355 w_k1_INTC_s005
                                 + 1.25091 w_k1_KO_s005 + 1.07711 w_k1_MAT_s005
                                 + 1.14454 w_k1_MSFT_s005
                                 + 1.07349 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.754458 w_k1_ADM_s005
                                 + 0.795475 w_k1_AXP_s005
                                 + 0.746943 w_k1_DIS_s005
                                 + 0.860916 w_k1_FDX_s005
                                 + 0.988893 w_k1_IBM_s005
                                 + 0.907791 w_k1_INTC_s005
                                 + 0.927367 w_k1_KO_s005
                                 + 1.67857 w_k1_MAT_s005
                                 + 0.874542 w_k1_MSFT_s005
                                 + 0.918279 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.941029 w_k1_ADM_s006
                                 + 0.951585 w_k1_AXP_s006
                                 + 1.18925 w_k1_DIS_s006
                                 + 1.07037 w_k1_FDX_s006
                                 + 0.932003 w_k1_IBM_s006
                                 + 0.979738 w_k1_INTC_s006
                                 + 0.945229 w_k1_KO_s006
                                 + 0.816864 w_k1_MAT_s006
                                 + 1.10905 w_k1_MSFT_s006
                                 + 1.02708 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.842184 w_k1_ADM_s006
                                 + 1.3076 w_k1_AXP_s006
                                 + 0.937437 w_k1_DIS_s006
                                 + 0.99461 w_k1_FDX_s006
                                 + 1.00551 w_k1_IBM_s006
                                 + 1.04371 w_k1_INTC_s006
                                 + 0.960937 w_k1_KO_s006
                                 + 1.20192 w_k1_MAT_s006
                                 + 1.03203 w_k1_MSFT_s006
                                 + 0.98832 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.877839 w_k1_ADM_s006
                                 + 0.914464 w_k1_AXP_s006
                                 + 0.872085 w_k1_DIS_s006
                                 + 0.877524 w_k1_FDX_s006
                                 + 1.1102 w_k1_IBM_s006
                                 + 0.802653 w_k1_INTC_s006
                                 + 0.963879 w_k1_KO_s006
                                 + 1.57677 w_k1_MAT_s006
                                 + 1.0561 w_k1_MSFT_s006
                                 + 0.937164 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.217 w_k1_ADM_s006
                                 + 1.45526 w_k1_AXP_s006
                                 + 1.25716 w_k1_DIS_s006
                                 + 1.19869 w_k1_FDX_s006
                                 + 1.09266 w_k1_IBM_s006
                                 + 1.18334 w_k1_INTC_s006
                                 + 1.06815 w_k1_KO_s006 + 1.06024 w_k1_MAT_s006
                                 + 1.1784 w_k1_MSFT_s006 + 1.15052 w_k1_PG_s006
                                 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.06559 w_k1_ADM_s006
                                 + 1.50862 w_k1_AXP_s006
                                 + 1.57543 w_k1_DIS_s006 + 1.1444 w_k1_FDX_s006
                                 + 1.40969 w_k1_IBM_s006
                                 + 0.871426 w_k1_INTC_s006
                                 + 1.34953 w_k1_KO_s006 + 1.97418 w_k1_MAT_s006
                                 + 1.16366 w_k1_MSFT_s006
                                 + 1.35756 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.832913 w_k1_ADM_s006
                                 + 1.02563 w_k1_AXP_s006
                                 + 0.815136 w_k1_DIS_s006
                                 + 0.969023 w_k1_FDX_s006
                                 + 0.79492 w_k1_IBM_s006
                                 + 0.804184 w_k1_INTC_s006
                                 + 0.872476 w_k1_KO_s006
                                 + 1.06015 w_k1_MAT_s006
                                 + 0.92448 w_k1_MSFT_s006
                                 + 1.00329 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.6895 w_k1_ADM_s006
                                 + 0.723033 w_k1_AXP_s006
                                 + 0.646417 w_k1_DIS_s006
                                 + 0.753446 w_k1_FDX_s006
                                 + 0.632588 w_k1_IBM_s006
                                 + 0.662005 w_k1_INTC_s006
                                 + 0.691099 w_k1_KO_s006
                                 + 0.805688 w_k1_MAT_s006
                                 + 0.848691 w_k1_MSFT_s006
                                 + 0.881854 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.748051 w_k1_ADM_s006
                                 + 0.85477 w_k1_AXP_s006
                                 + 0.675512 w_k1_DIS_s006
                                 + 0.693347 w_k1_FDX_s006
                                 + 0.949468 w_k1_IBM_s006
                                 + 0.725986 w_k1_INTC_s006
                                 + 0.731274 w_k1_KO_s006
                                 + 0.728212 w_k1_MAT_s006
                                 + 0.669844 w_k1_MSFT_s006
                                 + 0.846117 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.39689 w_k1_ADM_s007
                                 + 1.30447 w_k1_AXP_s007
                                 + 1.04253 w_k1_DIS_s007
                                 + 1.07827 w_k1_FDX_s007
                                 + 1.58647 w_k1_IBM_s007
                                 + 1.1951 w_k1_INTC_s007 + 1.2051 w_k1_KO_s007
                                 + 1.73312 w_k1_MAT_s007
                                 + 1.18559 w_k1_MSFT_s007 + 1.1177 w_k1_PG_s007
                                 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.50723 w_k1_ADM_s007
                                 + 1.01747 w_k1_AXP_s007
                                 + 1.11923 w_k1_DIS_s007
                                 + 0.992543 w_k1_FDX_s007
                                 + 1.15276 w_k1_IBM_s007
                                 + 1.16116 w_k1_INTC_s007
                                 + 1.20488 w_k1_KO_s007
                                 + 0.727702 w_k1_MAT_s007
                                 + 1.02796 w_k1_MSFT_s007
                                 + 1.16708 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.12348 w_k1_ADM_s007
                                 + 1.01888 w_k1_AXP_s007
                                 + 1.03093 w_k1_DIS_s007
                                 + 0.879603 w_k1_FDX_s007
                                 + 0.968577 w_k1_IBM_s007
                                 + 0.892409 w_k1_INTC_s007
                                 + 0.873482 w_k1_KO_s007
                                 + 0.838999 w_k1_MAT_s007
                                 + 0.863637 w_k1_MSFT_s007
                                 + 0.900613 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.24391 w_k1_ADM_s007
                                 + 1.20208 w_k1_AXP_s007
                                 + 1.61536 w_k1_DIS_s007
                                 + 1.14403 w_k1_FDX_s007
                                 + 1.43941 w_k1_IBM_s007
                                 + 1.07791 w_k1_INTC_s007
                                 + 1.22144 w_k1_KO_s007 + 1.57142 w_k1_MAT_s007
                                 + 1.22155 w_k1_MSFT_s007
                                 + 1.13635 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.34629 w_k1_ADM_s007
                                 + 0.985998 w_k1_AXP_s007
                                 + 0.912757 w_k1_DIS_s007
                                 + 0.888471 w_k1_FDX_s007
                                 + 0.805687 w_k1_IBM_s007
                                 + 0.892464 w_k1_INTC_s007
                                 + 0.826957 w_k1_KO_s007
                                 + 0.760032 w_k1_MAT_s007
                                 + 0.917329 w_k1_MSFT_s007
                                 + 1.00967 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.0323 w_k1_ADM_s007
                                 + 0.607856 w_k1_AXP_s007
                                 + 0.629007 w_k1_DIS_s007
                                 + 0.741443 w_k1_FDX_s007
                                 + 0.766792 w_k1_IBM_s007
                                 + 0.95237 w_k1_INTC_s007
                                 + 0.741689 w_k1_KO_s007
                                 + 0.529263 w_k1_MAT_s007
                                 + 0.916801 w_k1_MSFT_s007
                                 + 0.81899 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.85723 w_k1_ADM_s007
                                 + 1.45567 w_k1_AXP_s007
                                 + 1.46069 w_k1_DIS_s007
                                 + 1.27876 w_k1_FDX_s007
                                 + 1.02748 w_k1_IBM_s007
                                 + 1.08609 w_k1_INTC_s007
                                 + 1.16233 w_k1_KO_s007
                                 + 0.986154 w_k1_MAT_s007
                                 + 1.28979 w_k1_MSFT_s007
                                 + 1.32713 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.20495 w_k1_ADM_s007
                                 + 1.47097 w_k1_AXP_s007
                                 + 1.01216 w_k1_DIS_s007 + 1.0581 w_k1_FDX_s007
                                 + 0.936859 w_k1_IBM_s007
                                 + 1.26095 w_k1_INTC_s007
                                 + 0.973841 w_k1_KO_s007
                                 + 1.17356 w_k1_MAT_s007
                                 + 1.19378 w_k1_MSFT_s007
                                 + 1.07399 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.796824 w_k1_ADM_s010
                                 + 0.636338 w_k1_AXP_s010
                                 + 0.649831 w_k1_DIS_s010
                                 + 0.667029 w_k1_FDX_s010
                                 + 1.05881 w_k1_IBM_s010
                                 + 0.71158 w_k1_INTC_s010
                                 + 0.871316 w_k1_KO_s010
                                 + 1.19321 w_k1_MAT_s010
                                 + 0.811117 w_k1_MSFT_s010
                                 + 0.953593 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.749228 w_k1_ADM_s010
                                 + 0.531237 w_k1_AXP_s010
                                 + 0.551162 w_k1_DIS_s010
                                 + 0.687309 w_k1_FDX_s010
                                 + 0.682009 w_k1_IBM_s010
                                 + 0.768112 w_k1_INTC_s010
                                 + 0.701375 w_k1_KO_s010
                                 + 0.647622 w_k1_MAT_s010
                                 + 0.852148 w_k1_MSFT_s010
                                 + 0.930823 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.27717 w_k1_ADM_s010
                                 + 1.24149 w_k1_AXP_s010
                                 + 1.21315 w_k1_DIS_s010
                                 + 1.08352 w_k1_FDX_s010
                                 + 1.41483 w_k1_IBM_s010
                                 + 1.13597 w_k1_INTC_s010
                                 + 1.33943 w_k1_KO_s010 + 1.50851 w_k1_MAT_s010
                                 + 1.19074 w_k1_MSFT_s010
                                 + 1.17689 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.24031 w_k1_ADM_s010
                                 + 1.28319 w_k1_AXP_s010
                                 + 1.68599 w_k1_DIS_s010
                                 + 1.28322 w_k1_FDX_s010
                                 + 1.14816 w_k1_IBM_s010
                                 + 1.24382 w_k1_INTC_s010
                                 + 1.33184 w_k1_KO_s010 + 1.25603 w_k1_MAT_s010
                                 + 1.51669 w_k1_MSFT_s010
                                 + 1.29998 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.876292 w_k1_ADM_s010
                                 + 0.709387 w_k1_AXP_s010
                                 + 0.810688 w_k1_DIS_s010
                                 + 0.808946 w_k1_FDX_s010
                                 + 0.839545 w_k1_IBM_s010
                                 + 0.975158 w_k1_INTC_s010
                                 + 0.84944 w_k1_KO_s010
                                 + 0.697054 w_k1_MAT_s010
                                 + 0.960269 w_k1_MSFT_s010
                                 + 0.868697 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.12732 w_k1_ADM_s010
                                 + 0.983355 w_k1_AXP_s010
                                 + 0.878592 w_k1_DIS_s010
                                 + 1.01967 w_k1_FDX_s010
                                 + 0.805848 w_k1_IBM_s010
                                 + 1.09132 w_k1_INTC_s010
                                 + 0.909005 w_k1_KO_s010
                                 + 1.32331 w_k1_MAT_s010
                                 + 1.09285 w_k1_MSFT_s010
                                 + 0.996516 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.02782 w_k1_ADM_s010
                                 + 0.890626 w_k1_AXP_s010
                                 + 1.02075 w_k1_DIS_s010
                                 + 0.901957 w_k1_FDX_s010
                                 + 1.20598 w_k1_IBM_s010
                                 + 1.07113 w_k1_INTC_s010
                                 + 1.14669 w_k1_KO_s010 + 1.27671 w_k1_MAT_s010
                                 + 1.09017 w_k1_MSFT_s010
                                 + 1.13262 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.00213 w_k1_ADM_s010
                                 + 1.09412 w_k1_AXP_s010
                                 + 1.15646 w_k1_DIS_s010
                                 + 1.19359 w_k1_FDX_s010
                                 + 0.898912 w_k1_IBM_s010
                                 + 1.27403 w_k1_INTC_s010
                                 + 0.96007 w_k1_KO_s010
                                 + 0.998613 w_k1_MAT_s010
                                 + 1.20062 w_k1_MSFT_s010
                                 + 1.11946 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.02865 w_k1_ADM_s011
                                 + 1.04504 w_k1_AXP_s011
                                 + 0.822073 w_k1_DIS_s011
                                 + 1.01447 w_k1_FDX_s011
                                 + 0.73778 w_k1_IBM_s011
                                 + 1.09943 w_k1_INTC_s011
                                 + 0.920862 w_k1_KO_s011
                                 + 0.785272 w_k1_MAT_s011
                                 + 0.877318 w_k1_MSFT_s011
                                 + 0.841643 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.760509 w_k1_ADM_s011
                                 + 0.984158 w_k1_AXP_s011
                                 + 0.828864 w_k1_DIS_s011
                                 + 0.863554 w_k1_FDX_s011
                                 + 1.25148 w_k1_IBM_s011
                                 + 0.939051 w_k1_INTC_s011
                                 + 1.0721 w_k1_KO_s011 + 1.08297 w_k1_MAT_s011
                                 + 0.949888 w_k1_MSFT_s011
                                 + 0.966323 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.09887 w_k1_ADM_s011
                                 + 1.6317 w_k1_AXP_s011 + 1.40946 w_k1_DIS_s011
                                 + 1.23332 w_k1_FDX_s011
                                 + 1.39047 w_k1_IBM_s011
                                 + 1.0984 w_k1_INTC_s011 + 1.38973 w_k1_KO_s011
                                 + 1.60971 w_k1_MAT_s011
                                 + 0.965213 w_k1_MSFT_s011
                                 + 1.12143 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.734928 w_k1_ADM_s011
                                 + 0.794521 w_k1_AXP_s011
                                 + 0.572431 w_k1_DIS_s011
                                 + 0.731171 w_k1_FDX_s011
                                 + 0.930507 w_k1_IBM_s011
                                 + 0.711022 w_k1_INTC_s011
                                 + 0.807527 w_k1_KO_s011
                                 + 0.786263 w_k1_MAT_s011
                                 + 0.710867 w_k1_MSFT_s011
                                 + 0.795163 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.06663 w_k1_ADM_s011
                                 + 1.14897 w_k1_AXP_s011
                                 + 0.991997 w_k1_DIS_s011
                                 + 1.08952 w_k1_FDX_s011
                                 + 0.867653 w_k1_IBM_s011
                                 + 0.928141 w_k1_INTC_s011
                                 + 0.955752 w_k1_KO_s011
                                 + 0.822592 w_k1_MAT_s011
                                 + 0.969795 w_k1_MSFT_s011
                                 + 0.906735 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.789647 w_k1_ADM_s011
                                 + 0.668436 w_k1_AXP_s011
                                 + 0.813626 w_k1_DIS_s011
                                 + 0.929384 w_k1_FDX_s011
                                 + 0.660136 w_k1_IBM_s011
                                 + 0.944235 w_k1_INTC_s011
                                 + 0.89866 w_k1_KO_s011
                                 + 0.492546 w_k1_MAT_s011
                                 + 0.967378 w_k1_MSFT_s011
                                 + 0.792037 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.795624 w_k1_ADM_s011
                                 + 0.862851 w_k1_AXP_s011
                                 + 1.17345 w_k1_DIS_s011
                                 + 1.02066 w_k1_FDX_s011
                                 + 1.08109 w_k1_IBM_s011
                                 + 1.09192 w_k1_INTC_s011 + 1.123 w_k1_KO_s011
                                 + 0.837753 w_k1_MAT_s011
                                 + 1.08876 w_k1_MSFT_s011
                                 + 0.940005 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.18006 w_k1_ADM_s011
                                 + 1.14415 w_k1_AXP_s011
                                 + 1.32086 w_k1_DIS_s011
                                 + 1.29712 w_k1_FDX_s011
                                 + 0.81302 w_k1_IBM_s011
                                 + 1.11225 w_k1_INTC_s011
                                 + 1.08993 w_k1_KO_s011
                                 + 0.810348 w_k1_MAT_s011
                                 + 1.24986 w_k1_MSFT_s011
                                 + 1.00244 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.978973 w_k1_ADM_s012
                                 + 0.892722 w_k1_AXP_s012
                                 + 0.755883 w_k1_DIS_s012
                                 + 0.789554 w_k1_FDX_s012
                                 + 1.29532 w_k1_IBM_s012
                                 + 0.786438 w_k1_INTC_s012
                                 + 0.855099 w_k1_KO_s012
                                 + 1.30649 w_k1_MAT_s012
                                 + 0.666292 w_k1_MSFT_s012
                                 + 1.00654 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.982134 w_k1_ADM_s012
                                 + 1.09713 w_k1_AXP_s012
                                 + 1.42243 w_k1_DIS_s012
                                 + 1.20315 w_k1_FDX_s012
                                 + 1.19558 w_k1_IBM_s012
                                 + 1.24264 w_k1_INTC_s012
                                 + 1.08224 w_k1_KO_s012 + 1.04358 w_k1_MAT_s012
                                 + 1.1265 w_k1_MSFT_s012 + 1.11398 w_k1_PG_s012
                                 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.843924 w_k1_ADM_s012
                                 + 0.985046 w_k1_AXP_s012
                                 + 1.29285 w_k1_DIS_s012
                                 + 1.01348 w_k1_FDX_s012
                                 + 1.73157 w_k1_IBM_s012
                                 + 1.29137 w_k1_INTC_s012
                                 + 1.40415 w_k1_KO_s012 + 1.33265 w_k1_MAT_s012
                                 + 1.07582 w_k1_MSFT_s012
                                 + 1.05932 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.810715 w_k1_ADM_s012
                                 + 0.551786 w_k1_AXP_s012
                                 + 0.660638 w_k1_DIS_s012
                                 + 0.808422 w_k1_FDX_s012
                                 + 0.744074 w_k1_IBM_s012
                                 + 0.919503 w_k1_INTC_s012
                                 + 0.775507 w_k1_KO_s012
                                 + 0.538672 w_k1_MAT_s012
                                 + 0.774745 w_k1_MSFT_s012
                                 + 0.858484 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.1736 w_k1_ADM_s012
                                 + 1.41011 w_k1_AXP_s012
                                 + 1.67174 w_k1_DIS_s012
                                 + 1.37847 w_k1_FDX_s012
                                 + 1.34229 w_k1_IBM_s012
                                 + 1.64544 w_k1_INTC_s012
                                 + 1.31226 w_k1_KO_s012 + 1.5984 w_k1_MAT_s012
                                 + 1.41399 w_k1_MSFT_s012 + 1.3648 w_k1_PG_s012
                                 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.04607 w_k1_ADM_s012
                                 + 1.06453 w_k1_AXP_s012 + 1.2 w_k1_DIS_s012
                                 + 1.25639 w_k1_FDX_s012
                                 + 0.776123 w_k1_IBM_s012
                                 + 1.24456 w_k1_INTC_s012
                                 + 0.895254 w_k1_KO_s012
                                 + 0.658748 w_k1_MAT_s012
                                 + 1.11785 w_k1_MSFT_s012
                                 + 1.14775 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.723385 w_k1_ADM_s012
                                 + 0.801601 w_k1_AXP_s012
                                 + 0.971285 w_k1_DIS_s012
                                 + 0.863442 w_k1_FDX_s012
                                 + 1.19705 w_k1_IBM_s012
                                 + 0.955134 w_k1_INTC_s012
                                 + 1.03736 w_k1_KO_s012 + 1.07329 w_k1_MAT_s012
                                 + 0.928347 w_k1_MSFT_s012
                                 + 0.962257 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.902125 w_k1_ADM_s012
                                 + 0.823303 w_k1_AXP_s012
                                 + 1.01188 w_k1_DIS_s012
                                 + 0.983327 w_k1_FDX_s012
                                 + 0.792553 w_k1_IBM_s012
                                 + 1.07556 w_k1_INTC_s012
                                 + 0.954522 w_k1_KO_s012
                                 + 0.824152 w_k1_MAT_s012
                                 + 1.02497 w_k1_MSFT_s012
                                 + 1.00554 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.662475 w_k1_ADM_s013
                                 + 0.756198 w_k1_AXP_s013
                                 + 0.929618 w_k1_DIS_s013
                                 + 0.819621 w_k1_FDX_s013
                                 + 0.996933 w_k1_IBM_s013
                                 + 0.888302 w_k1_INTC_s013
                                 + 0.999322 w_k1_KO_s013
                                 + 1.12022 w_k1_MAT_s013
                                 + 0.883675 w_k1_MSFT_s013
                                 + 0.756355 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.924774 w_k1_ADM_s013
                                 + 0.968683 w_k1_AXP_s013
                                 + 1.23966 w_k1_DIS_s013
                                 + 1.16352 w_k1_FDX_s013
                                 + 0.745411 w_k1_IBM_s013
                                 + 0.858935 w_k1_INTC_s013
                                 + 0.812243 w_k1_KO_s013
                                 + 0.812517 w_k1_MAT_s013
                                 + 0.897706 w_k1_MSFT_s013
                                 + 0.939482 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.923439 w_k1_ADM_s013
                                 + 0.915851 w_k1_AXP_s013
                                 + 0.814162 w_k1_DIS_s013
                                 + 1.05904 w_k1_FDX_s013
                                 + 0.609542 w_k1_IBM_s013
                                 + 1.12603 w_k1_INTC_s013
                                 + 0.686747 w_k1_KO_s013
                                 + 0.775985 w_k1_MAT_s013
                                 + 0.967161 w_k1_MSFT_s013
                                 + 0.876711 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.12169 w_k1_ADM_s013
                                 + 1.26806 w_k1_AXP_s013
                                 + 1.70684 w_k1_DIS_s013
                                 + 1.31632 w_k1_FDX_s013
                                 + 1.11039 w_k1_IBM_s013
                                 + 1.30572 w_k1_INTC_s013
                                 + 1.42229 w_k1_KO_s013 + 1.65544 w_k1_MAT_s013
                                 + 1.57289 w_k1_MSFT_s013
                                 + 1.27017 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.604121 w_k1_ADM_s013
                                 + 0.655893 w_k1_AXP_s013
                                 + 0.637936 w_k1_DIS_s013
                                 + 0.716423 w_k1_FDX_s013
                                 + 0.701171 w_k1_IBM_s013
                                 + 0.666139 w_k1_INTC_s013
                                 + 0.771043 w_k1_KO_s013
                                 + 1.09431 w_k1_MAT_s013
                                 + 0.830285 w_k1_MSFT_s013
                                 + 0.700865 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.798349 w_k1_ADM_s013
                                 + 1.40096 w_k1_AXP_s013
                                 + 1.31631 w_k1_DIS_s013
                                 + 1.06702 w_k1_FDX_s013
                                 + 1.74519 w_k1_IBM_s013
                                 + 1.08007 w_k1_INTC_s013
                                 + 1.25565 w_k1_KO_s013 + 1.87962 w_k1_MAT_s013
                                 + 1.06525 w_k1_MSFT_s013
                                 + 1.13113 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.08875 w_k1_ADM_s013
                                 + 1.139 w_k1_AXP_s013 + 1.18761 w_k1_DIS_s013
                                 + 1.08007 w_k1_FDX_s013
                                 + 1.07001 w_k1_IBM_s013
                                 + 0.775477 w_k1_INTC_s013
                                 + 0.971259 w_k1_KO_s013 + 1.5132 w_k1_MAT_s013
                                 + 1.05471 w_k1_MSFT_s013
                                 + 0.938483 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.947629 w_k1_ADM_s013
                                 + 1.06666 w_k1_AXP_s013
                                 + 0.793399 w_k1_DIS_s013
                                 + 0.906778 w_k1_FDX_s013
                                 + 0.874112 w_k1_IBM_s013
                                 + 0.780023 w_k1_INTC_s013
                                 + 0.946305 w_k1_KO_s013
                                 + 1.28631 w_k1_MAT_s013
                                 + 0.997833 w_k1_MSFT_s013
                                 + 0.984334 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.971793 w_k1_ADM_s014
                                 + 0.919695 w_k1_AXP_s014
                                 + 0.832243 w_k1_DIS_s014
                                 + 0.915981 w_k1_FDX_s014
                                 + 0.770288 w_k1_IBM_s014
                                 + 0.687127 w_k1_INTC_s014
                                 + 0.726631 w_k1_KO_s014 + 1.0901 w_k1_MAT_s014
                                 + 0.910246 w_k1_MSFT_s014
                                 + 0.98322 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.19326 w_k1_ADM_s014
                                 + 1.06843 w_k1_AXP_s014
                                 + 1.16487 w_k1_DIS_s014
                                 + 0.996359 w_k1_FDX_s014
                                 + 1.53235 w_k1_IBM_s014
                                 + 0.867188 w_k1_INTC_s014
                                 + 1.22595 w_k1_KO_s014 + 1.89358 w_k1_MAT_s014
                                 + 1.03905 w_k1_MSFT_s014
                                 + 1.13679 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.06958 w_k1_ADM_s014
                                 + 1.12411 w_k1_AXP_s014
                                 + 1.14286 w_k1_DIS_s014
                                 + 1.05708 w_k1_FDX_s014
                                 + 1.06996 w_k1_IBM_s014
                                 + 1.02586 w_k1_INTC_s014
                                 + 1.05618 w_k1_KO_s014 + 1.40309 w_k1_MAT_s014
                                 + 1.1656 w_k1_MSFT_s014 + 1.05551 w_k1_PG_s014
                                 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.799511 w_k1_ADM_s014
                                 + 1.07026 w_k1_AXP_s014
                                 + 0.711327 w_k1_DIS_s014
                                 + 0.835934 w_k1_FDX_s014
                                 + 1.20625 w_k1_IBM_s014
                                 + 0.937197 w_k1_INTC_s014
                                 + 0.956268 w_k1_KO_s014
                                 + 1.34428 w_k1_MAT_s014
                                 + 0.865952 w_k1_MSFT_s014
                                 + 0.875096 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.13861 w_k1_ADM_s014
                                 + 1.40841 w_k1_AXP_s014
                                 + 1.54233 w_k1_DIS_s014
                                 + 1.23855 w_k1_FDX_s014
                                 + 1.65994 w_k1_IBM_s014
                                 + 1.05557 w_k1_INTC_s014 + 1.3404 w_k1_KO_s014
                                 + 1.79922 w_k1_MAT_s014
                                 + 1.31568 w_k1_MSFT_s014
                                 + 1.29578 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.20281 w_k1_ADM_s014
                                 + 1.33745 w_k1_AXP_s014
                                 + 1.57212 w_k1_DIS_s014
                                 + 1.45612 w_k1_FDX_s014
                                 + 1.09837 w_k1_IBM_s014
                                 + 0.969324 w_k1_INTC_s014
                                 + 1.37659 w_k1_KO_s014 + 1.60161 w_k1_MAT_s014
                                 + 1.56568 w_k1_MSFT_s014
                                 + 1.22605 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.92171 w_k1_ADM_s014
                                 + 0.949637 w_k1_AXP_s014
                                 + 1.02605 w_k1_DIS_s014
                                 + 0.841071 w_k1_FDX_s014
                                 + 1.12224 w_k1_IBM_s014
                                 + 0.929041 w_k1_INTC_s014
                                 + 1.05933 w_k1_KO_s014 + 1.24547 w_k1_MAT_s014
                                 + 1.04145 w_k1_MSFT_s014
                                 + 1.10234 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.645042 w_k1_ADM_s014
                                 + 0.648421 w_k1_AXP_s014
                                 + 0.647791 w_k1_DIS_s014
                                 + 0.729169 w_k1_FDX_s014
                                 + 0.88538 w_k1_IBM_s014
                                 + 0.724884 w_k1_INTC_s014
                                 + 0.782968 w_k1_KO_s014
                                 + 0.963362 w_k1_MAT_s014
                                 + 0.861062 w_k1_MSFT_s014
                                 + 0.843931 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.745914 w_k1_ADM_s015
                                 + 0.834742 w_k1_AXP_s015
                                 + 0.732619 w_k1_DIS_s015
                                 + 0.841706 w_k1_FDX_s015
                                 + 1.13074 w_k1_IBM_s015
                                 + 1.02086 w_k1_INTC_s015
                                 + 0.994333 w_k1_KO_s015
                                 + 1.14926 w_k1_MAT_s015
                                 + 0.810092 w_k1_MSFT_s015
                                 + 0.906313 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.972726 w_k1_ADM_s015
                                 + 0.828256 w_k1_AXP_s015
                                 + 0.683095 w_k1_DIS_s015
                                 + 0.90083 w_k1_FDX_s015
                                 + 0.850719 w_k1_IBM_s015
                                 + 0.919195 w_k1_INTC_s015
                                 + 0.810199 w_k1_KO_s015
                                 + 0.860755 w_k1_MAT_s015
                                 + 0.867966 w_k1_MSFT_s015
                                 + 0.959433 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.978985 w_k1_ADM_s015
                                 + 0.68811 w_k1_AXP_s015
                                 + 0.50159 w_k1_DIS_s015
                                 + 0.743179 w_k1_FDX_s015
                                 + 0.69262 w_k1_IBM_s015
                                 + 0.788968 w_k1_INTC_s015
                                 + 0.830194 w_k1_KO_s015
                                 + 0.859879 w_k1_MAT_s015
                                 + 0.712806 w_k1_MSFT_s015
                                 + 0.848093 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.875774 w_k1_ADM_s015
                                 + 0.897634 w_k1_AXP_s015
                                 + 0.706826 w_k1_DIS_s015
                                 + 0.995682 w_k1_FDX_s015
                                 + 0.639269 w_k1_IBM_s015
                                 + 1.04267 w_k1_INTC_s015
                                 + 0.82889 w_k1_KO_s015
                                 + 0.511244 w_k1_MAT_s015
                                 + 0.953313 w_k1_MSFT_s015
                                 + 0.980526 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.38771 w_k1_ADM_s015
                                 + 1.38921 w_k1_AXP_s015
                                 + 1.33676 w_k1_DIS_s015
                                 + 1.37087 w_k1_FDX_s015
                                 + 1.02894 w_k1_IBM_s015
                                 + 1.20049 w_k1_INTC_s015 + 1.2978 w_k1_KO_s015
                                 + 1.05802 w_k1_MAT_s015
                                 + 1.16684 w_k1_MSFT_s015
                                 + 1.23737 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.39817 w_k1_ADM_s015
                                 + 1.61487 w_k1_AXP_s015
                                 + 1.41935 w_k1_DIS_s015
                                 + 1.26843 w_k1_FDX_s015
                                 + 1.50325 w_k1_IBM_s015
                                 + 1.41039 w_k1_INTC_s015
                                 + 1.38477 w_k1_KO_s015 + 1.78371 w_k1_MAT_s015
                                 + 1.21578 w_k1_MSFT_s015
                                 + 1.20682 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.3167 w_k1_ADM_s015
                                 + 1.19277 w_k1_AXP_s015
                                 + 1.10087 w_k1_DIS_s015
                                 + 1.14587 w_k1_FDX_s015 + 1.1334 w_k1_IBM_s015
                                 + 0.972317 w_k1_INTC_s015
                                 + 1.17348 w_k1_KO_s015 + 1.37509 w_k1_MAT_s015
                                 + 1.05609 w_k1_MSFT_s015
                                 + 1.07478 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.17475 w_k1_ADM_s015
                                 + 1.14152 w_k1_AXP_s015
                                 + 0.972168 w_k1_DIS_s015
                                 + 1.08832 w_k1_FDX_s015
                                 + 0.803702 w_k1_IBM_s015
                                 + 1.38069 w_k1_INTC_s015
                                 + 0.968487 w_k1_KO_s015
                                 + 0.82003 w_k1_MAT_s015
                                 + 1.05834 w_k1_MSFT_s015
                                 + 1.04803 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.1987 w_k1_ADM_s016
                                 + 0.780519 w_k1_AXP_s016
                                 + 0.605815 w_k1_DIS_s016
                                 + 0.814848 w_k1_FDX_s016
                                 + 0.639568 w_k1_IBM_s016
                                 + 0.742085 w_k1_INTC_s016
                                 + 0.663595 w_k1_KO_s016
                                 + 0.733417 w_k1_MAT_s016
                                 + 0.777341 w_k1_MSFT_s016
                                 + 0.801432 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.97341 w_k1_ADM_s016
                                 + 1.52868 w_k1_AXP_s016
                                 + 1.54988 w_k1_DIS_s016 + 1.4357 w_k1_FDX_s016
                                 + 1.10215 w_k1_IBM_s016
                                 + 1.17971 w_k1_INTC_s016
                                 + 1.25924 w_k1_KO_s016 + 1.02247 w_k1_MAT_s016
                                 + 1.29677 w_k1_MSFT_s016
                                 + 1.16496 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.47994 w_k1_ADM_s016
                                 + 0.867295 w_k1_AXP_s016
                                 + 1.03745 w_k1_DIS_s016
                                 + 1.03308 w_k1_FDX_s016
                                 + 1.03978 w_k1_IBM_s016
                                 + 0.784279 w_k1_INTC_s016
                                 + 0.998028 w_k1_KO_s016
                                 + 0.813267 w_k1_MAT_s016
                                 + 0.963377 w_k1_MSFT_s016
                                 + 0.965242 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.72795 w_k1_ADM_s016
                                 + 1.45848 w_k1_AXP_s016
                                 + 1.33526 w_k1_DIS_s016
                                 + 1.24695 w_k1_FDX_s016
                                 + 1.66729 w_k1_IBM_s016
                                 + 1.09806 w_k1_INTC_s016
                                 + 1.30443 w_k1_KO_s016 + 1.65329 w_k1_MAT_s016
                                 + 1.16954 w_k1_MSFT_s016
                                 + 1.28141 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.18735 w_k1_ADM_s016
                                 + 1.16622 w_k1_AXP_s016
                                 + 0.845777 w_k1_DIS_s016
                                 + 1.04285 w_k1_FDX_s016
                                 + 0.886715 w_k1_IBM_s016
                                 + 0.910327 w_k1_INTC_s016
                                 + 0.915377 w_k1_KO_s016
                                 + 1.07815 w_k1_MAT_s016
                                 + 0.965155 w_k1_MSFT_s016
                                 + 0.958212 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.13461 w_k1_ADM_s016
                                 + 1.01219 w_k1_AXP_s016
                                 + 0.802025 w_k1_DIS_s016
                                 + 0.925248 w_k1_FDX_s016
                                 + 1.13152 w_k1_IBM_s016
                                 + 0.827756 w_k1_INTC_s016
                                 + 1.03195 w_k1_KO_s016 + 1.40089 w_k1_MAT_s016
                                 + 0.962245 w_k1_MSFT_s016
                                 + 1.11194 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.04121 w_k1_ADM_s016
                                 + 0.880029 w_k1_AXP_s016
                                 + 0.927662 w_k1_DIS_s016
                                 + 0.875921 w_k1_FDX_s016
                                 + 1.37543 w_k1_IBM_s016
                                 + 0.688709 w_k1_INTC_s016
                                 + 1.06254 w_k1_KO_s016 + 1.28644 w_k1_MAT_s016
                                 + 1.00702 w_k1_MSFT_s016
                                 + 0.900119 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.17267 w_k1_ADM_s016
                                 + 0.879037 w_k1_AXP_s016
                                 + 0.909633 w_k1_DIS_s016
                                 + 1.03256 w_k1_FDX_s016
                                 + 0.747452 w_k1_IBM_s016
                                 + 0.928599 w_k1_INTC_s016
                                 + 0.826908 w_k1_KO_s016
                                 + 0.681876 w_k1_MAT_s016
                                 + 0.95737 w_k1_MSFT_s016
                                 + 0.889469 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.07551 w_k1_ADM_s017
                                 + 0.810394 w_k1_AXP_s017
                                 + 0.886814 w_k1_DIS_s017
                                 + 0.947753 w_k1_FDX_s017
                                 + 0.834643 w_k1_IBM_s017
                                 + 0.671682 w_k1_INTC_s017
                                 + 0.859569 w_k1_KO_s017 + 1.0051 w_k1_MAT_s017
                                 + 0.774571 w_k1_MSFT_s017
                                 + 0.878814 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.15452 w_k1_ADM_s017
                                 + 1.20248 w_k1_AXP_s017
                                 + 1.10318 w_k1_DIS_s017
                                 + 1.04144 w_k1_FDX_s017
                                 + 1.40235 w_k1_IBM_s017
                                 + 0.937933 w_k1_INTC_s017
                                 + 1.34668 w_k1_KO_s017 + 2.14331 w_k1_MAT_s017
                                 + 1.01401 w_k1_MSFT_s017 + 1.0801 w_k1_PG_s017
                                 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.27147 w_k1_ADM_s017
                                 + 1.42541 w_k1_AXP_s017
                                 + 1.82842 w_k1_DIS_s017
                                 + 1.55825 w_k1_FDX_s017
                                 + 1.35235 w_k1_IBM_s017
                                 + 1.70032 w_k1_INTC_s017
                                 + 1.30464 w_k1_KO_s017 + 2.01858 w_k1_MAT_s017
                                 + 1.28765 w_k1_MSFT_s017
                                 + 1.14928 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.960884 w_k1_ADM_s017
                                 + 1.15981 w_k1_AXP_s017
                                 + 1.30254 w_k1_DIS_s017
                                 + 1.17862 w_k1_FDX_s017 + 1.0757 w_k1_IBM_s017
                                 + 1.301 w_k1_INTC_s017 + 1.17868 w_k1_KO_s017
                                 + 1.55001 w_k1_MAT_s017
                                 + 1.11971 w_k1_MSFT_s017
                                 + 1.09893 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.936185 w_k1_ADM_s017
                                 + 0.838064 w_k1_AXP_s017
                                 + 1.09341 w_k1_DIS_s017
                                 + 0.937638 w_k1_FDX_s017
                                 + 1.10548 w_k1_IBM_s017
                                 + 1.00911 w_k1_INTC_s017
                                 + 1.04197 w_k1_KO_s017 + 1.1064 w_k1_MAT_s017
                                 + 1.02582 w_k1_MSFT_s017
                                 + 1.05241 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.18112 w_k1_ADM_s017
                                 + 0.929909 w_k1_AXP_s017
                                 + 1.08972 w_k1_DIS_s017 + 1.1059 w_k1_FDX_s017
                                 + 0.802375 w_k1_IBM_s017
                                 + 1.01076 w_k1_INTC_s017
                                 + 0.942193 w_k1_KO_s017
                                 + 1.15136 w_k1_MAT_s017
                                 + 1.05316 w_k1_MSFT_s017
                                 + 0.978359 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.12774 w_k1_ADM_s017
                                 + 1.04248 w_k1_AXP_s017
                                 + 1.52449 w_k1_DIS_s017 + 1.2521 w_k1_FDX_s017
                                 + 0.87474 w_k1_IBM_s017
                                 + 1.35279 w_k1_INTC_s017
                                 + 1.34026 w_k1_KO_s017 + 1.07518 w_k1_MAT_s017
                                 + 1.1478 w_k1_MSFT_s017 + 1.09487 w_k1_PG_s017
                                 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.778073 w_k1_ADM_s017
                                 + 0.762754 w_k1_AXP_s017
                                 + 0.718662 w_k1_DIS_s017
                                 + 0.728087 w_k1_FDX_s017
                                 + 0.797229 w_k1_IBM_s017
                                 + 0.775983 w_k1_INTC_s017
                                 + 0.855208 w_k1_KO_s017 + 1.0311 w_k1_MAT_s017
                                 + 0.72113 w_k1_MSFT_s017
                                 + 0.813541 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.14188 w_k1_ADM_s100
                                 + 1.51909 w_k1_AXP_s100 + 1.255 w_k1_DIS_s100
                                 + 1.43419 w_k1_FDX_s100
                                 + 0.977812 w_k1_IBM_s100
                                 + 1.3775 w_k1_INTC_s100
                                 + 0.960449 w_k1_KO_s100
                                 + 1.28972 w_k1_MAT_s100
                                 + 1.13907 w_k1_MSFT_s100
                                 + 1.08923 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.09641 w_k1_ADM_s100
                                 + 1.3458 w_k1_AXP_s100 + 1.45385 w_k1_DIS_s100
                                 + 1.226 w_k1_FDX_s100 + 1.19137 w_k1_IBM_s100
                                 + 1.32264 w_k1_INTC_s100
                                 + 1.22411 w_k1_KO_s100 + 1.18895 w_k1_MAT_s100
                                 + 1.17418 w_k1_MSFT_s100
                                 + 1.07699 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.752398 w_k1_ADM_s100
                                 + 0.759167 w_k1_AXP_s100
                                 + 0.761895 w_k1_DIS_s100
                                 + 0.841247 w_k1_FDX_s100
                                 + 0.871306 w_k1_IBM_s100
                                 + 0.91217 w_k1_INTC_s100
                                 + 0.735003 w_k1_KO_s100
                                 + 0.913067 w_k1_MAT_s100
                                 + 0.951421 w_k1_MSFT_s100
                                 + 0.888665 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.36793 w_k1_ADM_s100
                                 + 0.931347 w_k1_AXP_s100
                                 + 1.32615 w_k1_DIS_s100
                                 + 1.11947 w_k1_FDX_s100
                                 + 1.19481 w_k1_IBM_s100
                                 + 1.27238 w_k1_INTC_s100
                                 + 1.21536 w_k1_KO_s100 + 1.32061 w_k1_MAT_s100
                                 + 1.20225 w_k1_MSFT_s100
                                 + 1.06247 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.17331 w_k1_ADM_s100
                                 + 1.30193 w_k1_AXP_s100
                                 + 1.49394 w_k1_DIS_s100
                                 + 1.19277 w_k1_FDX_s100
                                 + 1.68406 w_k1_IBM_s100
                                 + 1.24933 w_k1_INTC_s100
                                 + 1.39906 w_k1_KO_s100 + 1.71382 w_k1_MAT_s100
                                 + 1.18568 w_k1_MSFT_s100
                                 + 1.15327 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.02828 w_k1_ADM_s100
                                 + 1.15626 w_k1_AXP_s100
                                 + 0.931166 w_k1_DIS_s100
                                 + 1.1977 w_k1_FDX_s100
                                 + 0.774571 w_k1_IBM_s100
                                 + 1.16174 w_k1_INTC_s100
                                 + 0.826283 w_k1_KO_s100
                                 + 0.91212 w_k1_MAT_s100
                                 + 0.965528 w_k1_MSFT_s100
                                 + 0.928289 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.803503 w_k1_ADM_s100
                                 + 0.532223 w_k1_AXP_s100
                                 + 0.642576 w_k1_DIS_s100
                                 + 0.779145 w_k1_FDX_s100
                                 + 0.579544 w_k1_IBM_s100
                                 + 0.838691 w_k1_INTC_s100
                                 + 0.645442 w_k1_KO_s100
                                 + 0.499843 w_k1_MAT_s100
                                 + 0.853794 w_k1_MSFT_s100
                                 + 0.768186 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.968146 w_k1_ADM_s100
                                 + 0.926349 w_k1_AXP_s100
                                 + 0.79383 w_k1_DIS_s100
                                 + 0.963985 w_k1_FDX_s100
                                 + 1.18124 w_k1_IBM_s100
                                 + 0.843107 w_k1_INTC_s100
                                 + 0.857722 w_k1_KO_s100
                                 + 1.10318 w_k1_MAT_s100
                                 + 0.784028 w_k1_MSFT_s100
                                 + 0.889369 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.979627 w_k1_ADM_s101
                                 + 1.25227 w_k1_AXP_s101
                                 + 1.20517 w_k1_DIS_s101
                                 + 1.28302 w_k1_FDX_s101
                                 + 0.897748 w_k1_IBM_s101
                                 + 1.18269 w_k1_INTC_s101
                                 + 1.05546 w_k1_KO_s101
                                 + 0.909442 w_k1_MAT_s101
                                 + 1.23928 w_k1_MSFT_s101
                                 + 1.02409 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.08807 w_k1_ADM_s101
                                 + 0.828305 w_k1_AXP_s101
                                 + 0.71218 w_k1_DIS_s101
                                 + 0.816848 w_k1_FDX_s101
                                 + 0.592832 w_k1_IBM_s101
                                 + 0.793089 w_k1_INTC_s101
                                 + 0.740796 w_k1_KO_s101
                                 + 0.595281 w_k1_MAT_s101
                                 + 0.863453 w_k1_MSFT_s101
                                 + 0.853986 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.797964 w_k1_ADM_s101
                                 + 0.601844 w_k1_AXP_s101
                                 + 0.764676 w_k1_DIS_s101
                                 + 0.813111 w_k1_FDX_s101
                                 + 0.918478 w_k1_IBM_s101
                                 + 0.859548 w_k1_INTC_s101
                                 + 0.978046 w_k1_KO_s101
                                 + 0.788173 w_k1_MAT_s101
                                 + 0.944162 w_k1_MSFT_s101
                                 + 0.915158 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.964785 w_k1_ADM_s101
                                 + 0.740041 w_k1_AXP_s101
                                 + 0.642481 w_k1_DIS_s101
                                 + 0.753157 w_k1_FDX_s101
                                 + 0.72398 w_k1_IBM_s101
                                 + 0.787425 w_k1_INTC_s101
                                 + 0.805236 w_k1_KO_s101
                                 + 0.620187 w_k1_MAT_s101
                                 + 0.90542 w_k1_MSFT_s101
                                 + 0.834689 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.26426 w_k1_ADM_s101
                                 + 1.13552 w_k1_AXP_s101
                                 + 1.35859 w_k1_DIS_s101
                                 + 1.18759 w_k1_FDX_s101
                                 + 1.96697 w_k1_IBM_s101
                                 + 1.15479 w_k1_INTC_s101 + 1.738 w_k1_KO_s101
                                 + 1.99298 w_k1_MAT_s101
                                 + 1.43875 w_k1_MSFT_s101
                                 + 1.28232 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.12195 w_k1_ADM_s101
                                 + 1.03223 w_k1_AXP_s101
                                 + 0.977745 w_k1_DIS_s101
                                 + 0.953929 w_k1_FDX_s101
                                 + 0.842022 w_k1_IBM_s101
                                 + 0.901901 w_k1_INTC_s101
                                 + 0.973768 w_k1_KO_s101
                                 + 0.794648 w_k1_MAT_s101
                                 + 1.09987 w_k1_MSFT_s101
                                 + 0.97395 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.807276 w_k1_ADM_s101
                                 + 0.92143 w_k1_AXP_s101
                                 + 0.768428 w_k1_DIS_s101
                                 + 0.890878 w_k1_FDX_s101
                                 + 1.02092 w_k1_IBM_s101
                                 + 0.88092 w_k1_INTC_s101
                                 + 0.951964 w_k1_KO_s101 + 1.0852 w_k1_MAT_s101
                                 + 0.945769 w_k1_MSFT_s101
                                 + 1.05456 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.09196 w_k1_ADM_s101
                                 + 1.05927 w_k1_AXP_s101
                                 + 1.13282 w_k1_DIS_s101
                                 + 1.05469 w_k1_FDX_s101
                                 + 1.29447 w_k1_IBM_s101
                                 + 0.999891 w_k1_INTC_s101
                                 + 1.26117 w_k1_KO_s101 + 1.34315 w_k1_MAT_s101
                                 + 1.22984 w_k1_MSFT_s101
                                 + 1.24625 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.05106 w_k1_ADM_s102
                                 + 1.094 w_k1_AXP_s102 + 1.07966 w_k1_DIS_s102
                                 + 1.05455 w_k1_FDX_s102
                                 + 0.944992 w_k1_IBM_s102
                                 + 1.16938 w_k1_INTC_s102
                                 + 0.945026 w_k1_KO_s102
                                 + 0.987145 w_k1_MAT_s102
                                 + 1.08147 w_k1_MSFT_s102
                                 + 0.987419 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.747332 w_k1_ADM_s102
                                 + 0.892907 w_k1_AXP_s102
                                 + 0.683462 w_k1_DIS_s102
                                 + 0.810154 w_k1_FDX_s102
                                 + 0.696016 w_k1_IBM_s102
                                 + 0.927543 w_k1_INTC_s102
                                 + 0.752138 w_k1_KO_s102
                                 + 0.856126 w_k1_MAT_s102
                                 + 0.707423 w_k1_MSFT_s102
                                 + 0.792664 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.87249 w_k1_ADM_s102
                                 + 0.718155 w_k1_AXP_s102
                                 + 0.836454 w_k1_DIS_s102
                                 + 0.808995 w_k1_FDX_s102
                                 + 1.05943 w_k1_IBM_s102
                                 + 0.986439 w_k1_INTC_s102
                                 + 0.917775 w_k1_KO_s102
                                 + 0.976812 w_k1_MAT_s102
                                 + 0.859808 w_k1_MSFT_s102
                                 + 0.940057 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.861173 w_k1_ADM_s102
                                 + 0.801161 w_k1_AXP_s102
                                 + 0.968087 w_k1_DIS_s102
                                 + 0.93807 w_k1_FDX_s102
                                 + 0.744231 w_k1_IBM_s102
                                 + 1.11382 w_k1_INTC_s102
                                 + 0.880562 w_k1_KO_s102
                                 + 0.974846 w_k1_MAT_s102
                                 + 0.91813 w_k1_MSFT_s102
                                 + 0.950801 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.13671 w_k1_ADM_s102
                                 + 1.23123 w_k1_AXP_s102
                                 + 1.31897 w_k1_DIS_s102
                                 + 1.17715 w_k1_FDX_s102
                                 + 1.23942 w_k1_IBM_s102
                                 + 1.33778 w_k1_INTC_s102
                                 + 1.19325 w_k1_KO_s102 + 1.48696 w_k1_MAT_s102
                                 + 1.10505 w_k1_MSFT_s102
                                 + 1.11035 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.654946 w_k1_ADM_s102
                                 + 0.505625 w_k1_AXP_s102
                                 + 0.665111 w_k1_DIS_s102
                                 + 0.701952 w_k1_FDX_s102
                                 + 0.862344 w_k1_IBM_s102
                                 + 0.898662 w_k1_INTC_s102
                                 + 0.857418 w_k1_KO_s102
                                 + 0.798367 w_k1_MAT_s102
                                 + 0.714069 w_k1_MSFT_s102
                                 + 0.799339 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.17931 w_k1_ADM_s102
                                 + 1.12368 w_k1_AXP_s102
                                 + 1.57617 w_k1_DIS_s102
                                 + 1.10487 w_k1_FDX_s102
                                 + 1.59903 w_k1_IBM_s102
                                 + 1.08154 w_k1_INTC_s102
                                 + 1.42009 w_k1_KO_s102 + 2.15578 w_k1_MAT_s102
                                 + 1.23421 w_k1_MSFT_s102
                                 + 1.21081 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.955078 w_k1_ADM_s102
                                 + 0.987245 w_k1_AXP_s102
                                 + 1.28059 w_k1_DIS_s102
                                 + 1.02972 w_k1_FDX_s102
                                 + 1.08392 w_k1_IBM_s102
                                 + 1.32167 w_k1_INTC_s102
                                 + 1.24986 w_k1_KO_s102 + 1.13522 w_k1_MAT_s102
                                 + 1.10298 w_k1_MSFT_s102
                                 + 1.10269 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.571609 w_k1_ADM_s103
                                 + 0.65657 w_k1_AXP_s103
                                 + 0.597795 w_k1_DIS_s103
                                 + 0.643218 w_k1_FDX_s103
                                 + 0.969229 w_k1_IBM_s103
                                 + 1.00636 w_k1_INTC_s103
                                 + 0.754115 w_k1_KO_s103
                                 + 0.898055 w_k1_MAT_s103
                                 + 0.731018 w_k1_MSFT_s103
                                 + 0.735516 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.60474 w_k1_ADM_s103
                                 + 1.45392 w_k1_AXP_s103
                                 + 1.73806 w_k1_DIS_s103
                                 + 1.48194 w_k1_FDX_s103
                                 + 0.977394 w_k1_IBM_s103
                                 + 1.65976 w_k1_INTC_s103
                                 + 1.17186 w_k1_KO_s103 + 1.13178 w_k1_MAT_s103
                                 + 1.50521 w_k1_MSFT_s103
                                 + 1.19634 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.01144 w_k1_ADM_s103
                                 + 1.05386 w_k1_AXP_s103
                                 + 1.31456 w_k1_DIS_s103 + 1.1027 w_k1_FDX_s103
                                 + 1.25497 w_k1_IBM_s103
                                 + 1.41841 w_k1_INTC_s103
                                 + 1.18052 w_k1_KO_s103 + 1.31612 w_k1_MAT_s103
                                 + 1.16744 w_k1_MSFT_s103
                                 + 1.06971 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.796325 w_k1_ADM_s103
                                 + 1.17662 w_k1_AXP_s103
                                 + 0.99866 w_k1_DIS_s103
                                 + 0.919473 w_k1_FDX_s103
                                 + 1.4481 w_k1_IBM_s103
                                 + 1.15087 w_k1_INTC_s103
                                 + 1.05604 w_k1_KO_s103 + 1.5382 w_k1_MAT_s103
                                 + 0.945641 w_k1_MSFT_s103
                                 + 1.13807 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.768617 w_k1_ADM_s103
                                 + 0.651618 w_k1_AXP_s103
                                 + 0.822579 w_k1_DIS_s103
                                 + 0.882337 w_k1_FDX_s103
                                 + 0.664745 w_k1_IBM_s103
                                 + 0.940302 w_k1_INTC_s103
                                 + 0.75455 w_k1_KO_s103
                                 + 0.789018 w_k1_MAT_s103
                                 + 0.843628 w_k1_MSFT_s103
                                 + 0.808369 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.10933 w_k1_ADM_s103
                                 + 1.5124 w_k1_AXP_s103 + 1.26868 w_k1_DIS_s103
                                 + 1.20734 w_k1_FDX_s103
                                 + 1.31879 w_k1_IBM_s103
                                 + 1.46107 w_k1_INTC_s103
                                 + 1.06822 w_k1_KO_s103 + 1.74981 w_k1_MAT_s103
                                 + 1.11102 w_k1_MSFT_s103
                                 + 1.03047 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.05263 w_k1_ADM_s103
                                 + 1.10361 w_k1_AXP_s103
                                 + 1.16983 w_k1_DIS_s103
                                 + 1.25336 w_k1_FDX_s103
                                 + 0.745514 w_k1_IBM_s103
                                 + 1.34205 w_k1_INTC_s103
                                 + 0.906762 w_k1_KO_s103
                                 + 0.983052 w_k1_MAT_s103
                                 + 1.16487 w_k1_MSFT_s103
                                 + 1.04416 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.926513 w_k1_ADM_s103
                                 + 1.10786 w_k1_AXP_s103
                                 + 0.872263 w_k1_DIS_s103
                                 + 0.966217 w_k1_FDX_s103
                                 + 0.921171 w_k1_IBM_s103
                                 + 1.11247 w_k1_INTC_s103
                                 + 0.868721 w_k1_KO_s103 + 1.2277 w_k1_MAT_s103
                                 + 0.958219 w_k1_MSFT_s103
                                 + 0.93216 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.06759 w_k1_ADM_s104
                                 + 0.958854 w_k1_AXP_s104
                                 + 1.05138 w_k1_DIS_s104
                                 + 0.998552 w_k1_FDX_s104
                                 + 1.2909 w_k1_IBM_s104
                                 + 1.02515 w_k1_INTC_s104
                                 + 0.957714 w_k1_KO_s104
                                 + 0.736818 w_k1_MAT_s104
                                 + 1.0562 w_k1_MSFT_s104
                                 + 0.975748 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.950112 w_k1_ADM_s104
                                 + 0.686757 w_k1_AXP_s104
                                 + 0.778408 w_k1_DIS_s104
                                 + 0.872689 w_k1_FDX_s104
                                 + 0.75455 w_k1_IBM_s104
                                 + 0.817506 w_k1_INTC_s104
                                 + 0.799775 w_k1_KO_s104
                                 + 0.481891 w_k1_MAT_s104
                                 + 0.867567 w_k1_MSFT_s104
                                 + 0.818433 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.859533 w_k1_ADM_s104
                                 + 1.02888 w_k1_AXP_s104
                                 + 0.915247 w_k1_DIS_s104
                                 + 0.881212 w_k1_FDX_s104
                                 + 1.06318 w_k1_IBM_s104
                                 + 0.780559 w_k1_INTC_s104
                                 + 0.858959 w_k1_KO_s104
                                 + 1.12212 w_k1_MAT_s104
                                 + 0.733857 w_k1_MSFT_s104
                                 + 0.907963 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.24511 w_k1_ADM_s104
                                 + 1.14111 w_k1_AXP_s104
                                 + 1.35373 w_k1_DIS_s104
                                 + 1.21169 w_k1_FDX_s104
                                 + 1.00837 w_k1_IBM_s104
                                 + 0.907151 w_k1_INTC_s104
                                 + 1.08697 w_k1_KO_s104
                                 + 0.923731 w_k1_MAT_s104
                                 + 1.11933 w_k1_MSFT_s104
                                 + 1.10844 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.31184 w_k1_ADM_s104
                                 + 1.54002 w_k1_AXP_s104
                                 + 1.36947 w_k1_DIS_s104
                                 + 1.23917 w_k1_FDX_s104
                                 + 1.41621 w_k1_IBM_s104
                                 + 0.938973 w_k1_INTC_s104
                                 + 1.1001 w_k1_KO_s104 + 1.31867 w_k1_MAT_s104
                                 + 0.90003 w_k1_MSFT_s104
                                 + 1.08515 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.13993 w_k1_ADM_s104
                                 + 1.30262 w_k1_AXP_s104 + 1.23 w_k1_DIS_s104
                                 + 1.02862 w_k1_FDX_s104
                                 + 1.13642 w_k1_IBM_s104
                                 + 0.772438 w_k1_INTC_s104
                                 + 0.911121 w_k1_KO_s104
                                 + 1.00873 w_k1_MAT_s104
                                 + 0.878396 w_k1_MSFT_s104
                                 + 1.06863 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.02907 w_k1_ADM_s104
                                 + 1.12768 w_k1_AXP_s104
                                 + 1.10409 w_k1_DIS_s104
                                 + 1.14319 w_k1_FDX_s104
                                 + 0.884797 w_k1_IBM_s104
                                 + 0.895905 w_k1_INTC_s104
                                 + 0.806176 w_k1_KO_s104
                                 + 1.03958 w_k1_MAT_s104
                                 + 1.11737 w_k1_MSFT_s104
                                 + 1.00334 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.890179 w_k1_ADM_s104
                                 + 1.1324 w_k1_AXP_s104 + 1.48723 w_k1_DIS_s104
                                 + 1.05886 w_k1_FDX_s104
                                 + 1.73758 w_k1_IBM_s104
                                 + 1.01864 w_k1_INTC_s104
                                 + 1.58036 w_k1_KO_s104 + 1.49461 w_k1_MAT_s104
                                 + 1.28447 w_k1_MSFT_s104
                                 + 1.17096 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.54278 w_k1_ADM_s105
                                 + 1.43817 w_k1_AXP_s105
                                 + 1.47048 w_k1_DIS_s105
                                 + 1.50844 w_k1_FDX_s105
                                 + 1.07411 w_k1_IBM_s105
                                 + 1.22008 w_k1_INTC_s105
                                 + 1.14596 w_k1_KO_s105 + 1.28437 w_k1_MAT_s105
                                 + 1.38003 w_k1_MSFT_s105
                                 + 1.30873 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.06586 w_k1_ADM_s105
                                 + 1.25398 w_k1_AXP_s105
                                 + 1.42829 w_k1_DIS_s105
                                 + 1.23006 w_k1_FDX_s105
                                 + 1.33794 w_k1_IBM_s105
                                 + 1.09418 w_k1_INTC_s105
                                 + 1.23128 w_k1_KO_s105 + 1.57357 w_k1_MAT_s105
                                 + 1.16357 w_k1_MSFT_s105
                                 + 1.20317 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.1816 w_k1_ADM_s105
                                 + 1.08337 w_k1_AXP_s105
                                 + 1.00511 w_k1_DIS_s105
                                 + 0.983789 w_k1_FDX_s105
                                 + 1.38583 w_k1_IBM_s105
                                 + 0.850322 w_k1_INTC_s105
                                 + 1.09848 w_k1_KO_s105 + 1.4915 w_k1_MAT_s105
                                 + 1.08693 w_k1_MSFT_s105
                                 + 1.07476 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.897303 w_k1_ADM_s105
                                 + 0.790421 w_k1_AXP_s105
                                 + 0.78121 w_k1_DIS_s105
                                 + 0.847925 w_k1_FDX_s105
                                 + 0.764771 w_k1_IBM_s105
                                 + 0.860356 w_k1_INTC_s105
                                 + 0.747927 w_k1_KO_s105
                                 + 0.925479 w_k1_MAT_s105
                                 + 0.901344 w_k1_MSFT_s105
                                 + 0.989083 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.29051 w_k1_ADM_s105
                                 + 1.22333 w_k1_AXP_s105
                                 + 0.926344 w_k1_DIS_s105
                                 + 1.1996 w_k1_FDX_s105
                                 + 0.804661 w_k1_IBM_s105
                                 + 0.962954 w_k1_INTC_s105
                                 + 0.914964 w_k1_KO_s105
                                 + 1.16478 w_k1_MAT_s105
                                 + 1.08276 w_k1_MSFT_s105 + 1.1057 w_k1_PG_s105
                                 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.920438 w_k1_ADM_s105
                                 + 0.847834 w_k1_AXP_s105
                                 + 1.01099 w_k1_DIS_s105
                                 + 0.978353 w_k1_FDX_s105
                                 + 1.09596 w_k1_IBM_s105
                                 + 0.967836 w_k1_INTC_s105
                                 + 1.0211 w_k1_KO_s105 + 1.32348 w_k1_MAT_s105
                                 + 1.00531 w_k1_MSFT_s105 + 1.0227 w_k1_PG_s105
                                 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.799788 w_k1_ADM_s105
                                 + 0.701552 w_k1_AXP_s105
                                 + 0.554197 w_k1_DIS_s105
                                 + 0.684457 w_k1_FDX_s105
                                 + 0.897958 w_k1_IBM_s105
                                 + 0.703236 w_k1_INTC_s105
                                 + 0.719825 w_k1_KO_s105
                                 + 0.546381 w_k1_MAT_s105
                                 + 0.730983 w_k1_MSFT_s105
                                 + 0.86635 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.624424 w_k1_ADM_s105
                                 + 0.751664 w_k1_AXP_s105
                                 + 0.730596 w_k1_DIS_s105
                                 + 0.740481 w_k1_FDX_s105
                                 + 1.27907 w_k1_IBM_s105
                                 + 0.726153 w_k1_INTC_s105
                                 + 0.949564 w_k1_KO_s105
                                 + 1.29262 w_k1_MAT_s105
                                 + 0.752576 w_k1_MSFT_s105
                                 + 0.99686 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.50448 w_k1_ADM_s106
                                 + 1.33416 w_k1_AXP_s106
                                 + 1.33374 w_k1_DIS_s106
                                 + 1.21479 w_k1_FDX_s106
                                 + 0.880155 w_k1_IBM_s106
                                 + 1.16727 w_k1_INTC_s106 + 1.2601 w_k1_KO_s106
                                 + 0.923398 w_k1_MAT_s106
                                 + 1.19828 w_k1_MSFT_s106
                                 + 1.09873 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.12585 w_k1_ADM_s106
                                 + 0.802282 w_k1_AXP_s106
                                 + 0.878292 w_k1_DIS_s106
                                 + 0.884961 w_k1_FDX_s106
                                 + 1.06499 w_k1_IBM_s106
                                 + 0.744866 w_k1_INTC_s106
                                 + 1.04074 w_k1_KO_s106
                                 + 0.767393 w_k1_MAT_s106
                                 + 0.962067 w_k1_MSFT_s106
                                 + 0.973868 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.935781 w_k1_ADM_s106
                                 + 0.712691 w_k1_AXP_s106
                                 + 0.921704 w_k1_DIS_s106
                                 + 0.923359 w_k1_FDX_s106
                                 + 0.721314 w_k1_IBM_s106
                                 + 0.75203 w_k1_INTC_s106
                                 + 0.998368 w_k1_KO_s106
                                 + 0.685473 w_k1_MAT_s106
                                 + 1.08314 w_k1_MSFT_s106 + 1.0355 w_k1_PG_s106
                                 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.13829 w_k1_ADM_s106
                                 + 1.17291 w_k1_AXP_s106
                                 + 0.987326 w_k1_DIS_s106
                                 + 1.07022 w_k1_FDX_s106
                                 + 1.07765 w_k1_IBM_s106
                                 + 0.829688 w_k1_INTC_s106
                                 + 1.15682 w_k1_KO_s106 + 1.05883 w_k1_MAT_s106
                                 + 0.918308 w_k1_MSFT_s106
                                 + 0.984041 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.41561 w_k1_ADM_s106
                                 + 1.39846 w_k1_AXP_s106
                                 + 1.34913 w_k1_DIS_s106
                                 + 1.14633 w_k1_FDX_s106
                                 + 1.53082 w_k1_IBM_s106
                                 + 1.10838 w_k1_INTC_s106
                                 + 1.55177 w_k1_KO_s106 + 1.86307 w_k1_MAT_s106
                                 + 1.18258 w_k1_MSFT_s106
                                 + 1.27466 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.05201 w_k1_ADM_s106
                                 + 0.966994 w_k1_AXP_s106
                                 + 0.877457 w_k1_DIS_s106
                                 + 0.963487 w_k1_FDX_s106
                                 + 0.818924 w_k1_IBM_s106
                                 + 1.13657 w_k1_INTC_s106 + 1.0757 w_k1_KO_s106
                                 + 0.846461 w_k1_MAT_s106
                                 + 1.06516 w_k1_MSFT_s106
                                 + 0.911686 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.74199 w_k1_ADM_s106
                                 + 1.68596 w_k1_AXP_s106
                                 + 1.57298 w_k1_DIS_s106
                                 + 1.33545 w_k1_FDX_s106
                                 + 1.18355 w_k1_IBM_s106
                                 + 1.18615 w_k1_INTC_s106
                                 + 1.57931 w_k1_KO_s106 + 1.16272 w_k1_MAT_s106
                                 + 1.32641 w_k1_MSFT_s106
                                 + 1.31274 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.845156 w_k1_ADM_s106
                                 + 0.772463 w_k1_AXP_s106
                                 + 0.652906 w_k1_DIS_s106
                                 + 0.754318 w_k1_FDX_s106
                                 + 0.671025 w_k1_IBM_s106
                                 + 0.828096 w_k1_INTC_s106
                                 + 0.816212 w_k1_KO_s106
                                 + 0.584868 w_k1_MAT_s106
                                 + 0.776351 w_k1_MSFT_s106
                                 + 0.76335 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.708874 w_k1_ADM_s107
                                 + 1.05905 w_k1_AXP_s107
                                 + 1.27072 w_k1_DIS_s107
                                 + 0.961209 w_k1_FDX_s107
                                 + 1.84109 w_k1_IBM_s107
                                 + 1.09002 w_k1_INTC_s107
                                 + 1.41162 w_k1_KO_s107 + 1.75145 w_k1_MAT_s107
                                 + 1.13937 w_k1_MSFT_s107
                                 + 1.06462 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.912058 w_k1_ADM_s107
                                 + 0.90085 w_k1_AXP_s107
                                 + 1.12965 w_k1_DIS_s107
                                 + 1.09688 w_k1_FDX_s107
                                 + 0.718797 w_k1_IBM_s107
                                 + 1.17024 w_k1_INTC_s107
                                 + 0.896839 w_k1_KO_s107
                                 + 0.587608 w_k1_MAT_s107
                                 + 1.10079 w_k1_MSFT_s107 + 1.0095 w_k1_PG_s107
                                 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.788402 w_k1_ADM_s107
                                 + 0.705722 w_k1_AXP_s107
                                 + 0.902073 w_k1_DIS_s107
                                 + 0.8462 w_k1_FDX_s107 + 1.21537 w_k1_IBM_s107
                                 + 0.889442 w_k1_INTC_s107
                                 + 1.02384 w_k1_KO_s107
                                 + 0.886687 w_k1_MAT_s107
                                 + 0.945041 w_k1_MSFT_s107
                                 + 0.941913 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.767158 w_k1_ADM_s107
                                 + 0.707733 w_k1_AXP_s107
                                 + 0.733602 w_k1_DIS_s107
                                 + 0.842625 w_k1_FDX_s107
                                 + 0.898406 w_k1_IBM_s107
                                 + 0.852489 w_k1_INTC_s107
                                 + 0.849585 w_k1_KO_s107
                                 + 0.94827 w_k1_MAT_s107
                                 + 0.869951 w_k1_MSFT_s107
                                 + 0.859977 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.44426 w_k1_ADM_s107
                                 + 1.36326 w_k1_AXP_s107
                                 + 1.37196 w_k1_DIS_s107
                                 + 1.38907 w_k1_FDX_s107
                                 + 1.11349 w_k1_IBM_s107
                                 + 1.01964 w_k1_INTC_s107
                                 + 1.06235 w_k1_KO_s107 + 1.21007 w_k1_MAT_s107
                                 + 1.05125 w_k1_MSFT_s107
                                 + 1.12401 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.568908 w_k1_ADM_s107
                                 + 0.630828 w_k1_AXP_s107
                                 + 0.650209 w_k1_DIS_s107
                                 + 0.735938 w_k1_FDX_s107
                                 + 0.665116 w_k1_IBM_s107
                                 + 0.763973 w_k1_INTC_s107
                                 + 0.715711 w_k1_KO_s107
                                 + 0.674085 w_k1_MAT_s107
                                 + 0.752549 w_k1_MSFT_s107
                                 + 0.793863 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.907078 w_k1_ADM_s107
                                 + 1.03982 w_k1_AXP_s107
                                 + 0.99738 w_k1_DIS_s107
                                 + 0.927887 w_k1_FDX_s107
                                 + 1.11934 w_k1_IBM_s107
                                 + 0.841619 w_k1_INTC_s107
                                 + 1.0341 w_k1_KO_s107 + 1.41502 w_k1_MAT_s107
                                 + 0.929915 w_k1_MSFT_s107
                                 + 0.981596 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.974101 w_k1_ADM_s107
                                 + 0.967115 w_k1_AXP_s107
                                 + 1.44335 w_k1_DIS_s107
                                 + 1.07096 w_k1_FDX_s107
                                 + 1.43856 w_k1_IBM_s107
                                 + 0.82135 w_k1_INTC_s107
                                 + 1.33733 w_k1_KO_s107 + 1.44611 w_k1_MAT_s107
                                 + 1.02947 w_k1_MSFT_s107
                                 + 1.10131 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.975266 w_k1_ADM_s110
                                 + 1.17145 w_k1_AXP_s110
                                 + 0.956926 w_k1_DIS_s110
                                 + 0.913514 w_k1_FDX_s110
                                 + 0.976318 w_k1_IBM_s110
                                 + 0.959597 w_k1_INTC_s110
                                 + 1.0598 w_k1_KO_s110 + 1.03788 w_k1_MAT_s110
                                 + 0.914663 w_k1_MSFT_s110
                                 + 1.07678 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.36959 w_k1_ADM_s110
                                 + 1.57721 w_k1_AXP_s110
                                 + 1.37397 w_k1_DIS_s110
                                 + 1.37071 w_k1_FDX_s110
                                 + 0.927866 w_k1_IBM_s110
                                 + 1.16123 w_k1_INTC_s110
                                 + 1.21243 w_k1_KO_s110 + 1.09608 w_k1_MAT_s110
                                 + 1.3211 w_k1_MSFT_s110 + 1.239 w_k1_PG_s110
                                 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.00131 w_k1_ADM_s110
                                 + 1.30656 w_k1_AXP_s110
                                 + 1.17912 w_k1_DIS_s110
                                 + 1.08758 w_k1_FDX_s110
                                 + 1.32754 w_k1_IBM_s110
                                 + 1.14956 w_k1_INTC_s110
                                 + 1.41275 w_k1_KO_s110 + 1.5508 w_k1_MAT_s110
                                 + 1.32269 w_k1_MSFT_s110
                                 + 1.20417 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.766957 w_k1_ADM_s110
                                 + 0.810163 w_k1_AXP_s110
                                 + 0.923207 w_k1_DIS_s110
                                 + 1.01807 w_k1_FDX_s110
                                 + 0.664391 w_k1_IBM_s110
                                 + 0.958191 w_k1_INTC_s110
                                 + 0.772191 w_k1_KO_s110
                                 + 0.577403 w_k1_MAT_s110
                                 + 1.03331 w_k1_MSFT_s110
                                 + 1.02888 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.789012 w_k1_ADM_s110
                                 + 0.939911 w_k1_AXP_s110
                                 + 0.853995 w_k1_DIS_s110
                                 + 0.855679 w_k1_FDX_s110
                                 + 1.04068 w_k1_IBM_s110
                                 + 0.941201 w_k1_INTC_s110
                                 + 1.1612 w_k1_KO_s110 + 1.22417 w_k1_MAT_s110
                                 + 1.00227 w_k1_MSFT_s110
                                 + 1.02343 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.18051 w_k1_ADM_s110
                                 + 1.2572 w_k1_AXP_s110 + 1.64032 w_k1_DIS_s110
                                 + 1.32087 w_k1_FDX_s110
                                 + 1.08067 w_k1_IBM_s110
                                 + 1.14112 w_k1_INTC_s110
                                 + 1.41676 w_k1_KO_s110 + 1.41484 w_k1_MAT_s110
                                 + 1.364 w_k1_MSFT_s110 + 1.21541 w_k1_PG_s110
                                 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.750491 w_k1_ADM_s110
                                 + 0.646132 w_k1_AXP_s110
                                 + 0.600515 w_k1_DIS_s110
                                 + 0.792142 w_k1_FDX_s110
                                 + 0.555617 w_k1_IBM_s110
                                 + 0.710509 w_k1_INTC_s110
                                 + 0.760423 w_k1_KO_s110
                                 + 0.65592 w_k1_MAT_s110
                                 + 0.860462 w_k1_MSFT_s110
                                 + 0.788866 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.725557 w_k1_ADM_s110
                                 + 0.748314 w_k1_AXP_s110
                                 + 0.736009 w_k1_DIS_s110
                                 + 0.693735 w_k1_FDX_s110
                                 + 0.934745 w_k1_IBM_s110
                                 + 0.869927 w_k1_INTC_s110
                                 + 1.11275 w_k1_KO_s110 + 1.10352 w_k1_MAT_s110
                                 + 0.865111 w_k1_MSFT_s110
                                 + 0.983715 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.954778 w_k1_ADM_s111
                                 + 0.813451 w_k1_AXP_s111
                                 + 1.15235 w_k1_DIS_s111
                                 + 1.00917 w_k1_FDX_s111
                                 + 0.795001 w_k1_IBM_s111
                                 + 0.97683 w_k1_INTC_s111
                                 + 0.989174 w_k1_KO_s111
                                 + 0.969727 w_k1_MAT_s111
                                 + 1.31214 w_k1_MSFT_s111
                                 + 0.925034 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.978804 w_k1_ADM_s111
                                 + 1.14913 w_k1_AXP_s111
                                 + 0.842803 w_k1_DIS_s111
                                 + 0.950097 w_k1_FDX_s111
                                 + 0.971762 w_k1_IBM_s111
                                 + 0.954449 w_k1_INTC_s111
                                 + 0.871618 w_k1_KO_s111
                                 + 1.01759 w_k1_MAT_s111
                                 + 1.00104 w_k1_MSFT_s111
                                 + 1.03618 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.47903 w_k1_ADM_s111
                                 + 1.52958 w_k1_AXP_s111
                                 + 1.41161 w_k1_DIS_s111 + 1.3933 w_k1_FDX_s111
                                 + 0.916033 w_k1_IBM_s111
                                 + 1.15599 w_k1_INTC_s111
                                 + 1.10932 w_k1_KO_s111 + 1.14911 w_k1_MAT_s111
                                 + 1.28249 w_k1_MSFT_s111
                                 + 1.12701 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.44137 w_k1_ADM_s111
                                 + 1.03343 w_k1_AXP_s111
                                 + 1.73567 w_k1_DIS_s111
                                 + 1.41278 w_k1_FDX_s111
                                 + 0.927846 w_k1_IBM_s111
                                 + 1.10933 w_k1_INTC_s111
                                 + 1.27125 w_k1_KO_s111
                                 + 0.805583 w_k1_MAT_s111
                                 + 1.41319 w_k1_MSFT_s111
                                 + 1.17666 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.01471 w_k1_ADM_s111
                                 + 0.901123 w_k1_AXP_s111
                                 + 0.962498 w_k1_DIS_s111
                                 + 0.95594 w_k1_FDX_s111
                                 + 0.903771 w_k1_IBM_s111
                                 + 0.905828 w_k1_INTC_s111
                                 + 0.977706 w_k1_KO_s111
                                 + 0.955529 w_k1_MAT_s111
                                 + 1.04175 w_k1_MSFT_s111 + 1.0788 w_k1_PG_s111
                                 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.13804 w_k1_ADM_s111
                                 + 1.14953 w_k1_AXP_s111
                                 + 1.35424 w_k1_DIS_s111
                                 + 1.18575 w_k1_FDX_s111
                                 + 1.22408 w_k1_IBM_s111
                                 + 1.03759 w_k1_INTC_s111
                                 + 1.27053 w_k1_KO_s111 + 1.23605 w_k1_MAT_s111
                                 + 1.19889 w_k1_MSFT_s111
                                 + 1.15612 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.796314 w_k1_ADM_s111
                                 + 0.71418 w_k1_AXP_s111
                                 + 0.724393 w_k1_DIS_s111
                                 + 0.768887 w_k1_FDX_s111
                                 + 0.646923 w_k1_IBM_s111
                                 + 0.784026 w_k1_INTC_s111
                                 + 0.683956 w_k1_KO_s111
                                 + 0.530662 w_k1_MAT_s111
                                 + 0.834221 w_k1_MSFT_s111
                                 + 0.882024 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.818316 w_k1_ADM_s111
                                 + 0.895037 w_k1_AXP_s111
                                 + 0.943485 w_k1_DIS_s111
                                 + 0.785042 w_k1_FDX_s111
                                 + 1.15728 w_k1_IBM_s111
                                 + 0.679101 w_k1_INTC_s111
                                 + 1.02552 w_k1_KO_s111 + 1.18003 w_k1_MAT_s111
                                 + 0.915773 w_k1_MSFT_s111
                                 + 0.90524 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.841722 w_k1_ADM_s112
                                 + 1.0751 w_k1_AXP_s112 + 1.04832 w_k1_DIS_s112
                                 + 0.878367 w_k1_FDX_s112
                                 + 1.68747 w_k1_IBM_s112
                                 + 1.03967 w_k1_INTC_s112
                                 + 1.20237 w_k1_KO_s112 + 1.55225 w_k1_MAT_s112
                                 + 0.94668 w_k1_MSFT_s112
                                 + 0.976754 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.24101 w_k1_ADM_s112
                                 + 1.4841 w_k1_AXP_s112 + 1.47068 w_k1_DIS_s112
                                 + 1.21743 w_k1_FDX_s112
                                 + 1.11832 w_k1_IBM_s112
                                 + 1.07162 w_k1_INTC_s112
                                 + 1.00249 w_k1_KO_s112 + 1.53787 w_k1_MAT_s112
                                 + 1.1679 w_k1_MSFT_s112 + 1.15677 w_k1_PG_s112
                                 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.77334 w_k1_ADM_s112
                                 + 0.695789 w_k1_AXP_s112
                                 + 0.608682 w_k1_DIS_s112
                                 + 0.710997 w_k1_FDX_s112
                                 + 0.912713 w_k1_IBM_s112
                                 + 0.754634 w_k1_INTC_s112
                                 + 0.794162 w_k1_KO_s112
                                 + 0.89409 w_k1_MAT_s112
                                 + 0.697096 w_k1_MSFT_s112
                                 + 0.863818 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.07579 w_k1_ADM_s112
                                 + 1.18293 w_k1_AXP_s112 + 1.4771 w_k1_DIS_s112
                                 + 1.22138 w_k1_FDX_s112
                                 + 0.905774 w_k1_IBM_s112
                                 + 1.0021 w_k1_INTC_s112 + 1.03185 w_k1_KO_s112
                                 + 0.793627 w_k1_MAT_s112
                                 + 1.12419 w_k1_MSFT_s112
                                 + 1.06446 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.910134 w_k1_ADM_s112
                                 + 0.894771 w_k1_AXP_s112
                                 + 1.10473 w_k1_DIS_s112
                                 + 0.926307 w_k1_FDX_s112
                                 + 1.05342 w_k1_IBM_s112
                                 + 1.08035 w_k1_INTC_s112
                                 + 0.937058 w_k1_KO_s112
                                 + 1.02302 w_k1_MAT_s112
                                 + 1.09837 w_k1_MSFT_s112
                                 + 0.881391 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.927911 w_k1_ADM_s112
                                 + 0.795506 w_k1_AXP_s112
                                 + 0.815255 w_k1_DIS_s112
                                 + 0.834664 w_k1_FDX_s112
                                 + 0.687662 w_k1_IBM_s112
                                 + 0.926671 w_k1_INTC_s112
                                 + 0.749369 w_k1_KO_s112
                                 + 0.517858 w_k1_MAT_s112
                                 + 0.939236 w_k1_MSFT_s112
                                 + 0.8067 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.5167 w_k1_ADM_s112
                                 + 1.33698 w_k1_AXP_s112
                                 + 1.78861 w_k1_DIS_s112
                                 + 1.30582 w_k1_FDX_s112
                                 + 1.58157 w_k1_IBM_s112
                                 + 1.5798 w_k1_INTC_s112 + 1.43282 w_k1_KO_s112
                                 + 1.42198 w_k1_MAT_s112
                                 + 1.46528 w_k1_MSFT_s112
                                 + 1.22603 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.863341 w_k1_ADM_s112
                                 + 0.903778 w_k1_AXP_s112
                                 + 0.839419 w_k1_DIS_s112
                                 + 0.773057 w_k1_FDX_s112 + 1.109 w_k1_IBM_s112
                                 + 0.71825 w_k1_INTC_s112
                                 + 0.860397 w_k1_KO_s112 + 1.16 w_k1_MAT_s112
                                 + 0.748022 w_k1_MSFT_s112
                                 + 0.918048 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.06992 w_k1_ADM_s113
                                 + 0.757045 w_k1_AXP_s113
                                 + 0.95982 w_k1_DIS_s113
                                 + 0.932916 w_k1_FDX_s113
                                 + 0.791111 w_k1_IBM_s113
                                 + 0.836255 w_k1_INTC_s113
                                 + 0.901476 w_k1_KO_s113
                                 + 0.59151 w_k1_MAT_s113
                                 + 0.983676 w_k1_MSFT_s113
                                 + 0.973177 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.08795 w_k1_ADM_s113
                                 + 1.2434 w_k1_AXP_s113 + 1.32793 w_k1_DIS_s113
                                 + 1.05477 w_k1_FDX_s113
                                 + 1.71227 w_k1_IBM_s113
                                 + 1.09537 w_k1_INTC_s113
                                 + 1.24654 w_k1_KO_s113 + 1.48682 w_k1_MAT_s113
                                 + 1.07372 w_k1_MSFT_s113 + 1.2005 w_k1_PG_s113
                                 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.46061 w_k1_ADM_s113
                                 + 1.27709 w_k1_AXP_s113
                                 + 0.985791 w_k1_DIS_s113
                                 + 1.19497 w_k1_FDX_s113
                                 + 0.830122 w_k1_IBM_s113
                                 + 0.983118 w_k1_INTC_s113
                                 + 0.931129 w_k1_KO_s113
                                 + 0.810364 w_k1_MAT_s113
                                 + 1.06163 w_k1_MSFT_s113
                                 + 1.13828 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.999778 w_k1_ADM_s113
                                 + 0.808953 w_k1_AXP_s113
                                 + 0.71007 w_k1_DIS_s113
                                 + 0.848186 w_k1_FDX_s113
                                 + 0.751212 w_k1_IBM_s113
                                 + 0.767496 w_k1_INTC_s113
                                 + 0.75305 w_k1_KO_s113
                                 + 0.647259 w_k1_MAT_s113
                                 + 0.754454 w_k1_MSFT_s113
                                 + 0.844492 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.18566 w_k1_ADM_s113
                                 + 1.4132 w_k1_AXP_s113 + 1.3632 w_k1_DIS_s113
                                 + 1.23085 w_k1_FDX_s113
                                 + 1.08932 w_k1_IBM_s113
                                 + 1.1278 w_k1_INTC_s113
                                 + 0.949055 w_k1_KO_s113
                                 + 1.31005 w_k1_MAT_s113
                                 + 1.2733 w_k1_MSFT_s113 + 1.12257 w_k1_PG_s113
                                 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.82617 w_k1_ADM_s113
                                 + 0.910163 w_k1_AXP_s113
                                 + 0.961281 w_k1_DIS_s113
                                 + 0.794082 w_k1_FDX_s113
                                 + 1.31889 w_k1_IBM_s113
                                 + 0.97686 w_k1_INTC_s113
                                 + 1.11751 w_k1_KO_s113 + 1.00129 w_k1_MAT_s113
                                 + 0.867425 w_k1_MSFT_s113
                                 + 0.973829 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.994069 w_k1_ADM_s113
                                 + 1.20054 w_k1_AXP_s113
                                 + 1.16585 w_k1_DIS_s113
                                 + 1.05316 w_k1_FDX_s113
                                 + 1.33168 w_k1_IBM_s113
                                 + 1.08718 w_k1_INTC_s113
                                 + 1.14673 w_k1_KO_s113 + 1.37499 w_k1_MAT_s113
                                 + 1.05038 w_k1_MSFT_s113
                                 + 1.12005 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.20354 w_k1_ADM_s113
                                 + 0.990206 w_k1_AXP_s113
                                 + 1.35741 w_k1_DIS_s113
                                 + 1.14688 w_k1_FDX_s113
                                 + 0.941168 w_k1_IBM_s113
                                 + 1.14435 w_k1_INTC_s113
                                 + 1.14191 w_k1_KO_s113
                                 + 0.927228 w_k1_MAT_s113
                                 + 1.25577 w_k1_MSFT_s113 + 1.0596 w_k1_PG_s113
                                 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.96361 w_k1_ADM_s114
                                 + 0.914228 w_k1_AXP_s114
                                 + 0.894044 w_k1_DIS_s114
                                 + 0.91383 w_k1_FDX_s114
                                 + 1.18472 w_k1_IBM_s114
                                 + 0.804385 w_k1_INTC_s114
                                 + 1.23759 w_k1_KO_s114 + 1.16801 w_k1_MAT_s114
                                 + 0.882282 w_k1_MSFT_s114
                                 + 0.903229 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.09481 w_k1_ADM_s114
                                 + 1.0247 w_k1_AXP_s114 + 1.27092 w_k1_DIS_s114
                                 + 1.34609 w_k1_FDX_s114
                                 + 0.676472 w_k1_IBM_s114
                                 + 1.05743 w_k1_INTC_s114
                                 + 0.928726 w_k1_KO_s114
                                 + 0.657769 w_k1_MAT_s114
                                 + 1.21258 w_k1_MSFT_s114 + 1.0013 w_k1_PG_s114
                                 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.26856 w_k1_ADM_s114
                                 + 1.78215 w_k1_AXP_s114
                                 + 1.94382 w_k1_DIS_s114
                                 + 1.48792 w_k1_FDX_s114
                                 + 1.19803 w_k1_IBM_s114
                                 + 1.3341 w_k1_INTC_s114 + 1.28757 w_k1_KO_s114
                                 + 1.33942 w_k1_MAT_s114
                                 + 1.35198 w_k1_MSFT_s114
                                 + 1.27471 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.744205 w_k1_ADM_s114
                                 + 0.597302 w_k1_AXP_s114
                                 + 0.661829 w_k1_DIS_s114
                                 + 0.777879 w_k1_FDX_s114
                                 + 0.81519 w_k1_IBM_s114
                                 + 0.712331 w_k1_INTC_s114
                                 + 0.757047 w_k1_KO_s114
                                 + 0.819974 w_k1_MAT_s114
                                 + 0.807066 w_k1_MSFT_s114
                                 + 0.805483 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.998786 w_k1_ADM_s114
                                 + 1.07158 w_k1_AXP_s114
                                 + 1.32958 w_k1_DIS_s114
                                 + 1.26684 w_k1_FDX_s114
                                 + 0.983996 w_k1_IBM_s114
                                 + 1.24486 w_k1_INTC_s114
                                 + 1.27322 w_k1_KO_s114 + 1.40432 w_k1_MAT_s114
                                 + 1.2623 w_k1_MSFT_s114 + 1.12924 w_k1_PG_s114
                                 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.32813 w_k1_ADM_s114
                                 + 1.09371 w_k1_AXP_s114
                                 + 0.997334 w_k1_DIS_s114
                                 + 1.15999 w_k1_FDX_s114
                                 + 0.740408 w_k1_IBM_s114
                                 + 1.02018 w_k1_INTC_s114
                                 + 0.90119 w_k1_KO_s114
                                 + 0.898838 w_k1_MAT_s114
                                 + 1.15312 w_k1_MSFT_s114
                                 + 0.957978 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.866168 w_k1_ADM_s114
                                 + 0.82091 w_k1_AXP_s114
                                 + 0.805893 w_k1_DIS_s114
                                 + 0.955197 w_k1_FDX_s114
                                 + 0.78296 w_k1_IBM_s114
                                 + 0.916269 w_k1_INTC_s114
                                 + 0.84081 w_k1_KO_s114
                                 + 0.863942 w_k1_MAT_s114
                                 + 0.911645 w_k1_MSFT_s114
                                 + 0.939517 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.973455 w_k1_ADM_s114
                                 + 1.07312 w_k1_AXP_s114
                                 + 1.30628 w_k1_DIS_s114
                                 + 1.02669 w_k1_FDX_s114
                                 + 1.50995 w_k1_IBM_s114
                                 + 1.0038 w_k1_INTC_s114 + 1.52108 w_k1_KO_s114
                                 + 1.87183 w_k1_MAT_s114
                                 + 1.25044 w_k1_MSFT_s114
                                 + 1.18778 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.01945 w_k1_ADM_s115
                                 + 0.89169 w_k1_AXP_s115
                                 + 1.32113 w_k1_DIS_s115
                                 + 1.14245 w_k1_FDX_s115
                                 + 0.63802 w_k1_IBM_s115
                                 + 1.22531 w_k1_INTC_s115
                                 + 0.922529 w_k1_KO_s115
                                 + 0.781123 w_k1_MAT_s115
                                 + 1.22164 w_k1_MSFT_s115
                                 + 1.01709 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.988287 w_k1_ADM_s115
                                 + 1.07335 w_k1_AXP_s115
                                 + 0.89746 w_k1_DIS_s115
                                 + 0.942759 w_k1_FDX_s115
                                 + 0.89685 w_k1_IBM_s115
                                 + 0.91401 w_k1_INTC_s115
                                 + 0.872952 w_k1_KO_s115
                                 + 1.46131 w_k1_MAT_s115
                                 + 1.02701 w_k1_MSFT_s115
                                 + 0.887614 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.800145 w_k1_ADM_s115
                                 + 0.639002 w_k1_AXP_s115
                                 + 0.613894 w_k1_DIS_s115
                                 + 0.788393 w_k1_FDX_s115
                                 + 0.738345 w_k1_IBM_s115
                                 + 0.835051 w_k1_INTC_s115
                                 + 0.677197 w_k1_KO_s115
                                 + 0.729397 w_k1_MAT_s115
                                 + 0.788361 w_k1_MSFT_s115
                                 + 0.828612 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.15752 w_k1_ADM_s115
                                 + 1.11196 w_k1_AXP_s115
                                 + 0.935204 w_k1_DIS_s115
                                 + 0.894489 w_k1_FDX_s115 + 1.476 w_k1_IBM_s115
                                 + 0.981499 w_k1_INTC_s115
                                 + 1.14474 w_k1_KO_s115 + 1.53317 w_k1_MAT_s115
                                 + 0.90973 w_k1_MSFT_s115
                                 + 1.01759 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.07828 w_k1_ADM_s115
                                 + 1.02922 w_k1_AXP_s115
                                 + 1.09552 w_k1_DIS_s115
                                 + 1.06472 w_k1_FDX_s115
                                 + 0.800133 w_k1_IBM_s115
                                 + 1.0294 w_k1_INTC_s115
                                 + 0.970169 w_k1_KO_s115
                                 + 1.22674 w_k1_MAT_s115
                                 + 1.09321 w_k1_MSFT_s115
                                 + 1.05275 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.26518 w_k1_ADM_s115
                                 + 1.34628 w_k1_AXP_s115
                                 + 1.66404 w_k1_DIS_s115
                                 + 1.36994 w_k1_FDX_s115
                                 + 1.34901 w_k1_IBM_s115
                                 + 1.51056 w_k1_INTC_s115
                                 + 1.22774 w_k1_KO_s115 + 1.43857 w_k1_MAT_s115
                                 + 1.41894 w_k1_MSFT_s115
                                 + 1.17645 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.07688 w_k1_ADM_s115
                                 + 0.701143 w_k1_AXP_s115
                                 + 0.688037 w_k1_DIS_s115
                                 + 0.893395 w_k1_FDX_s115
                                 + 0.560226 w_k1_IBM_s115
                                 + 0.810639 w_k1_INTC_s115
                                 + 0.646502 w_k1_KO_s115
                                 + 0.497941 w_k1_MAT_s115
                                 + 0.872228 w_k1_MSFT_s115
                                 + 0.869333 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.25767 w_k1_ADM_s115
                                 + 1.21177 w_k1_AXP_s115
                                 + 1.20774 w_k1_DIS_s115
                                 + 1.06034 w_k1_FDX_s115
                                 + 1.05417 w_k1_IBM_s115
                                 + 0.996349 w_k1_INTC_s115
                                 + 1.08148 w_k1_KO_s115 + 1.69494 w_k1_MAT_s115
                                 + 1.09898 w_k1_MSFT_s115
                                 + 1.09702 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.741131 w_k1_ADM_s116
                                 + 0.718111 w_k1_AXP_s116
                                 + 0.57617 w_k1_DIS_s116
                                 + 0.708093 w_k1_FDX_s116
                                 + 0.69481 w_k1_IBM_s116
                                 + 0.739914 w_k1_INTC_s116
                                 + 0.665181 w_k1_KO_s116
                                 + 0.770934 w_k1_MAT_s116
                                 + 0.695205 w_k1_MSFT_s116
                                 + 0.859891 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.979773 w_k1_ADM_s116
                                 + 1.33625 w_k1_AXP_s116
                                 + 1.18808 w_k1_DIS_s116 + 1.1614 w_k1_FDX_s116
                                 + 1.09418 w_k1_IBM_s116
                                 + 1.19022 w_k1_INTC_s116
                                 + 1.17639 w_k1_KO_s116 + 1.42276 w_k1_MAT_s116
                                 + 1.05904 w_k1_MSFT_s116
                                 + 1.14562 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.25356 w_k1_ADM_s116
                                 + 1.7059 w_k1_AXP_s116 + 1.15155 w_k1_DIS_s116
                                 + 1.31409 w_k1_FDX_s116
                                 + 0.921874 w_k1_IBM_s116
                                 + 1.09328 w_k1_INTC_s116
                                 + 0.929741 w_k1_KO_s116
                                 + 0.862848 w_k1_MAT_s116
                                 + 1.09326 w_k1_MSFT_s116
                                 + 1.18614 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.825208 w_k1_ADM_s116
                                 + 0.76598 w_k1_AXP_s116
                                 + 0.795371 w_k1_DIS_s116
                                 + 0.893017 w_k1_FDX_s116
                                 + 0.959541 w_k1_IBM_s116
                                 + 0.917186 w_k1_INTC_s116
                                 + 0.919804 w_k1_KO_s116
                                 + 0.711012 w_k1_MAT_s116
                                 + 0.885115 w_k1_MSFT_s116
                                 + 0.891351 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.856394 w_k1_ADM_s116
                                 + 0.845388 w_k1_AXP_s116
                                 + 0.938031 w_k1_DIS_s116
                                 + 0.936934 w_k1_FDX_s116
                                 + 0.682663 w_k1_IBM_s116
                                 + 0.829961 w_k1_INTC_s116
                                 + 0.898972 w_k1_KO_s116
                                 + 0.576233 w_k1_MAT_s116
                                 + 0.93617 w_k1_MSFT_s116
                                 + 1.00518 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.95237 w_k1_ADM_s116
                                 + 1.23225 w_k1_AXP_s116
                                 + 0.791796 w_k1_DIS_s116
                                 + 0.988057 w_k1_FDX_s116
                                 + 0.873825 w_k1_IBM_s116
                                 + 0.931164 w_k1_INTC_s116
                                 + 0.964979 w_k1_KO_s116
                                 + 1.21552 w_k1_MAT_s116
                                 + 0.796569 w_k1_MSFT_s116
                                 + 1.03175 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.799484 w_k1_ADM_s116
                                 + 0.938451 w_k1_AXP_s116
                                 + 0.905857 w_k1_DIS_s116
                                 + 0.837409 w_k1_FDX_s116
                                 + 1.23212 w_k1_IBM_s116
                                 + 0.865215 w_k1_INTC_s116
                                 + 1.01868 w_k1_KO_s116 + 1.2651 w_k1_MAT_s116
                                 + 0.803568 w_k1_MSFT_s116
                                 + 1.04689 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.902335 w_k1_ADM_s116
                                 + 1.44733 w_k1_AXP_s116
                                 + 1.48171 w_k1_DIS_s116
                                 + 1.19005 w_k1_FDX_s116
                                 + 1.45931 w_k1_IBM_s116
                                 + 0.968308 w_k1_INTC_s116
                                 + 1.55584 w_k1_KO_s116 + 1.84506 w_k1_MAT_s116
                                 + 1.05435 w_k1_MSFT_s116
                                 + 1.25533 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.848513 w_k1_ADM_s117
                                 + 0.606003 w_k1_AXP_s117
                                 + 0.880395 w_k1_DIS_s117
                                 + 0.766137 w_k1_FDX_s117
                                 + 1.05688 w_k1_IBM_s117
                                 + 0.836316 w_k1_INTC_s117
                                 + 1.03258 w_k1_KO_s117
                                 + 0.746585 w_k1_MAT_s117
                                 + 0.794683 w_k1_MSFT_s117
                                 + 0.849087 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.6442 w_k1_ADM_s117
                                 + 1.34329 w_k1_AXP_s117
                                 + 1.75868 w_k1_DIS_s117
                                 + 1.39909 w_k1_FDX_s117
                                 + 1.34232 w_k1_IBM_s117
                                 + 1.21768 w_k1_INTC_s117
                                 + 1.61644 w_k1_KO_s117 + 1.96549 w_k1_MAT_s117
                                 + 1.43415 w_k1_MSFT_s117
                                 + 1.28428 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.25524 w_k1_ADM_s117
                                 + 0.968078 w_k1_AXP_s117
                                 + 1.03185 w_k1_DIS_s117
                                 + 1.08757 w_k1_FDX_s117
                                 + 0.797483 w_k1_IBM_s117
                                 + 1.00472 w_k1_INTC_s117
                                 + 1.00571 w_k1_KO_s117 + 1.03453 w_k1_MAT_s117
                                 + 1.12125 w_k1_MSFT_s117
                                 + 0.921902 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.999532 w_k1_ADM_s117
                                 + 1.21345 w_k1_AXP_s117
                                 + 1.10782 w_k1_DIS_s117
                                 + 0.976163 w_k1_FDX_s117
                                 + 1.70695 w_k1_IBM_s117
                                 + 1.11968 w_k1_INTC_s117
                                 + 1.27033 w_k1_KO_s117 + 1.91464 w_k1_MAT_s117
                                 + 0.979725 w_k1_MSFT_s117
                                 + 1.00493 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.915654 w_k1_ADM_s117
                                 + 0.766881 w_k1_AXP_s117
                                 + 0.783154 w_k1_DIS_s117
                                 + 0.909498 w_k1_FDX_s117
                                 + 0.708304 w_k1_IBM_s117
                                 + 0.936076 w_k1_INTC_s117
                                 + 0.908952 w_k1_KO_s117
                                 + 0.962155 w_k1_MAT_s117
                                 + 0.925352 w_k1_MSFT_s117
                                 + 0.824354 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.22898 w_k1_ADM_s117
                                 + 1.26581 w_k1_AXP_s117
                                 + 1.18817 w_k1_DIS_s117 + 1.188 w_k1_FDX_s117
                                 + 0.988703 w_k1_IBM_s117
                                 + 1.07326 w_k1_INTC_s117
                                 + 1.17238 w_k1_KO_s117 + 1.19916 w_k1_MAT_s117
                                 + 1.00733 w_k1_MSFT_s117
                                 + 1.08823 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.874946 w_k1_ADM_s117
                                 + 0.664907 w_k1_AXP_s117
                                 + 0.617257 w_k1_DIS_s117
                                 + 0.698862 w_k1_FDX_s117
                                 + 0.827828 w_k1_IBM_s117
                                 + 0.774921 w_k1_INTC_s117
                                 + 0.839168 w_k1_KO_s117
                                 + 0.802606 w_k1_MAT_s117
                                 + 0.698698 w_k1_MSFT_s117
                                 + 0.828053 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.11481 w_k1_ADM_s117
                                 + 0.94729 w_k1_AXP_s117
                                 + 1.05294 w_k1_DIS_s117
                                 + 0.947554 w_k1_FDX_s117
                                 + 1.21156 w_k1_IBM_s117
                                 + 1.00043 w_k1_INTC_s117
                                 + 1.17925 w_k1_KO_s117 + 1.13699 w_k1_MAT_s117
                                 + 0.982359 w_k1_MSFT_s117
                                 + 0.955209 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0138818639392896 w_k1_ADM_s000
                                 + 0.0103333782335896 w_k1_ADM_s001
                                 + 0.028140049061665 w_k1_ADM_s002
                                 + 0.0117091306618671 w_k1_ADM_s003
                                 + 0.0135493324631737 w_k1_ADM_s004
                                 + 0.0169010301389123 w_k1_ADM_s005
                                 + 0.00729048745225321 w_k1_ADM_s006
                                 + 0.0186803490661457 w_k1_ADM_s007
                                 + 0.0141703814788826 w_k1_ADM_s010
                                 + 0.0324303233505477 w_k1_ADM_s011
                                 + 0.0457983534787792 w_k1_ADM_s012
                                 + 0.0131888184659422 w_k1_ADM_s013
                                 + 0.0235500826472458 w_k1_ADM_s014
                                 + 0.0172455096354754 w_k1_ADM_s015
                                 + 0.034344104065351 w_k1_ADM_s016
                                 + 0.0247407959135711 w_k1_ADM_s017
                                 + 0.0221835681239377 w_k1_ADM_s100
                                 + 0.0128019980084784 w_k1_ADM_s101
                                 + 0.0352970586272075 w_k1_ADM_s102
                                 + 0.0204029735156169 w_k1_ADM_s103
                                 + 0.0123741731258822 w_k1_ADM_s104
                                 + 0.0315576030189533 w_k1_ADM_s105
                                 + 0.0437154110770123 w_k1_ADM_s106
                                 + 0.0171542509184399 w_k1_ADM_s107
                                 + 0.0364060914295407 w_k1_ADM_s110
                                 + 0.0772783415622897 w_k1_ADM_s111
                                 + 0.0801567742368978 w_k1_ADM_s112
                                 + 0.118645758497325 w_k1_ADM_s113
                                 + 0.0493126776084626 w_k1_ADM_s114
                                 + 0.0550914913337988 w_k1_ADM_s115
                                 + 0.0386081729341428 w_k1_ADM_s116
                                 + 0.051669792572846 w_k1_ADM_s117
                                 + 0.0148310263472405 w_k1_AXP_s000
                                 + 0.012756473827102 w_k1_AXP_s001
                                 + 0.029796562819799 w_k1_AXP_s002
                                 + 0.0109388962271023 w_k1_AXP_s003
                                 + 0.0148954713601774 w_k1_AXP_s004
                                 + 0.0172502957716271 w_k1_AXP_s005
                                 + 0.00867963488801749 w_k1_AXP_s006
                                 + 0.0145043668059835 w_k1_AXP_s007
                                 + 0.0130008337870517 w_k1_AXP_s010
                                 + 0.0369867937027099 w_k1_AXP_s011
                                 + 0.0458851633284814 w_k1_AXP_s012
                                 + 0.0154538069947713 w_k1_AXP_s013
                                 + 0.0243759642738284 w_k1_AXP_s014
                                 + 0.0165400620095792 w_k1_AXP_s015
                                 + 0.0268036797653458 w_k1_AXP_s016
                                 + 0.0236041798121423 w_k1_AXP_s017
                                 + 0.0235864226763846 w_k1_AXP_s100
                                 + 0.011968574124883 w_k1_AXP_s101
                                 + 0.0351209214231733 w_k1_AXP_s102
                                 + 0.0229162459392959 w_k1_AXP_s103
                                 + 0.0124348036382603 w_k1_AXP_s104
                                 + 0.0305366126507107 w_k1_AXP_s105
                                 + 0.0395806639968584 w_k1_AXP_s106
                                 + 0.0175123270404478 w_k1_AXP_s107
                                 + 0.0422055672531547 w_k1_AXP_s110
                                 + 0.0753854231522872 w_k1_AXP_s111
                                 + 0.0809338306741109 w_k1_AXP_s112
                                 + 0.112519711566269 w_k1_AXP_s113
                                 + 0.0491579643807942 w_k1_AXP_s114
                                 + 0.0528948916348221 w_k1_AXP_s115
                                 + 0.0478529181691479 w_k1_AXP_s116
                                 + 0.0453940917338894 w_k1_AXP_s117
                                 + 0.0166315071513899 w_k1_DIS_s000
                                 + 0.0125519416798714 w_k1_DIS_s001
                                 + 0.0303765415666574 w_k1_DIS_s002
                                 + 0.0101346140929231 w_k1_DIS_s003
                                 + 0.0146499085356562 w_k1_DIS_s004
                                 + 0.0186108378762345 w_k1_DIS_s005
                                 + 0.0083743693919573 w_k1_DIS_s006
                                 + 0.0146391005791375 w_k1_DIS_s007
                                 + 0.0140006919840742 w_k1_DIS_s010
                                 + 0.0346393036165289 w_k1_DIS_s011
                                 + 0.0550126520529276 w_k1_DIS_s012
                                 + 0.0155387360493271 w_k1_DIS_s013
                                 + 0.0241074621262597 w_k1_DIS_s014
                                 + 0.0144518324378431 w_k1_DIS_s015
                                 + 0.0246500106427759 w_k1_DIS_s016
                                 + 0.0265133422364205 w_k1_DIS_s017
                                 + 0.0229449008980751 w_k1_DIS_s100
                                 + 0.0118585156145803 w_k1_DIS_s101
                                 + 0.0395842489760422 w_k1_DIS_s102
                                 + 0.0219962857252282 w_k1_DIS_s103
                                 + 0.0128281598135466 w_k1_DIS_s104
                                 + 0.0295112058847326 w_k1_DIS_s105
                                 + 0.0378895610726303 w_k1_DIS_s106
                                 + 0.0196385132047265 w_k1_DIS_s107
                                 + 0.041011480797794 w_k1_DIS_s110
                                 + 0.080578324218746 w_k1_DIS_s111
                                 + 0.0888009423161385 w_k1_DIS_s112
                                 + 0.116114108373058 w_k1_DIS_s113
                                 + 0.0538942403708685 w_k1_DIS_s114
                                 + 0.0542542225395985 w_k1_DIS_s115
                                 + 0.0420498336114303 w_k1_DIS_s116
                                 + 0.0482820105244016 w_k1_DIS_s117
                                 + 0.0159083878420212 w_k1_FDX_s000
                                 + 0.0116434954646925 w_k1_FDX_s001
                                 + 0.0304928923908187 w_k1_FDX_s002
                                 + 0.00997356254605963 w_k1_FDX_s003
                                 + 0.0138862839242086 w_k1_FDX_s004
                                 + 0.0184017111952468 w_k1_FDX_s005
                                 + 0.0078932510127809 w_k1_FDX_s006
                                 + 0.013486254767431 w_k1_FDX_s007
                                 + 0.0134399797414203 w_k1_FDX_s010
                                 + 0.0355118685837537 w_k1_FDX_s011
                                 + 0.0516060885216409 w_k1_FDX_s012
                                 + 0.0148770720476715 w_k1_FDX_s013
                                 + 0.0233741809223388 w_k1_FDX_s014
                                 + 0.0160511392067811 w_k1_FDX_s015
                                 + 0.0263293643119145 w_k1_FDX_s016
                                 + 0.0246597130830562 w_k1_FDX_s017
                                 + 0.0239993975028204 w_k1_FDX_s100
                                 + 0.0120837192924925 w_k1_FDX_s101
                                 + 0.0358513710298596 w_k1_FDX_s102
                                 + 0.0219531819985528 w_k1_FDX_s103
                                 + 0.0123248450697837 w_k1_FDX_s104
                                 + 0.0307461535032867 w_k1_FDX_s105
                                 + 0.0372570944340917 w_k1_FDX_s106
                                 + 0.0187743157207426 w_k1_FDX_s107
                                 + 0.0390133711881806 w_k1_FDX_s110
                                 + 0.0757611264610566 w_k1_FDX_s111
                                 + 0.0773212996473924 w_k1_FDX_s112
                                 + 0.110298559062351 w_k1_FDX_s113
                                 + 0.0524716576026208 w_k1_FDX_s114
                                 + 0.0515584593124555 w_k1_FDX_s115
                                 + 0.0429794842428893 w_k1_FDX_s116
                                 + 0.0465796180589525 w_k1_FDX_s117
                                 + 0.0153029659257813 w_k1_IBM_s000
                                 + 0.0125308376154952 w_k1_IBM_s001
                                 + 0.0313551782481451 w_k1_IBM_s002
                                 + 0.0101646605295372 w_k1_IBM_s003
                                 + 0.0140897499056455 w_k1_IBM_s004
                                 + 0.0185467429365202 w_k1_IBM_s005
                                 + 0.00785310300463832 w_k1_IBM_s006
                                 + 0.0144346069802131 w_k1_IBM_s007
                                 + 0.0135835620822931 w_k1_IBM_s010
                                 + 0.0350255445532342 w_k1_IBM_s011
                                 + 0.0524542258386618 w_k1_IBM_s012
                                 + 0.01506545956698 w_k1_IBM_s013
                                 + 0.0249082926340882 w_k1_IBM_s014
                                 + 0.0157897577291903 w_k1_IBM_s015
                                 + 0.0261877813593427 w_k1_IBM_s016
                                 + 0.0249738420065168 w_k1_IBM_s017
                                 + 0.0228548790227854 w_k1_IBM_s100
                                 + 0.0128537028448518 w_k1_IBM_s101
                                 + 0.03865231204919 w_k1_IBM_s102
                                 + 0.0211599452735277 w_k1_IBM_s103
                                 + 0.0123607996566965 w_k1_IBM_s104
                                 + 0.0322302386900542 w_k1_IBM_s105
                                 + 0.0357685529683384 w_k1_IBM_s106
                                 + 0.0185588979330146 w_k1_IBM_s107
                                 + 0.0387794456564601 w_k1_IBM_s110
                                 + 0.0724179385052943 w_k1_IBM_s111
                                 + 0.081893791431543 w_k1_IBM_s112
                                 + 0.11194628994736 w_k1_IBM_s113
                                 + 0.0520130364351741 w_k1_IBM_s114
                                 + 0.0523349087566947 w_k1_IBM_s115
                                 + 0.0435465134434644 w_k1_IBM_s116
                                 + 0.0464832498463095 w_k1_IBM_s117
                                 + 0.0158211911753551 w_k1_INTC_s000
                                 + 0.0111238155424883 w_k1_INTC_s001
                                 + 0.0290690038089729 w_k1_INTC_s002
                                 + 0.00982258719121113 w_k1_INTC_s003
                                 + 0.0128675621765566 w_k1_INTC_s004
                                 + 0.0178055614839607 w_k1_INTC_s005
                                 + 0.00706754139688506 w_k1_INTC_s006
                                 + 0.0144658765753657 w_k1_INTC_s007
                                 + 0.014676021760844 w_k1_INTC_s010
                                 + 0.0343280733129719 w_k1_INTC_s011
                                 + 0.0568324134729292 w_k1_INTC_s012
                                 + 0.0136249529215193 w_k1_INTC_s013
                                 + 0.0205692698229527 w_k1_INTC_s014
                                 + 0.0166330492559604 w_k1_INTC_s015
                                 + 0.022732893833157 w_k1_INTC_s016
                                 + 0.0245081456122282 w_k1_INTC_s017
                                 + 0.0242880514631386 w_k1_INTC_s100
                                 + 0.0117247108259906 w_k1_INTC_s101
                                 + 0.0414478695499542 w_k1_INTC_s102
                                 + 0.0258296473975249 w_k1_INTC_s103
                                 + 0.010438678817886 w_k1_INTC_s104
                                 + 0.02782633907149 w_k1_INTC_s105
                                 + 0.0350086425007037 w_k1_INTC_s106
                                 + 0.0175143483792255 w_k1_INTC_s107
                                 + 0.0388479370392039 w_k1_INTC_s110
                                 + 0.0690951321451455 w_k1_INTC_s111
                                 + 0.0797704491724186 w_k1_INTC_s112
                                 + 0.106502421804544 w_k1_INTC_s113
                                 + 0.0484474885934851 w_k1_INTC_s114
                                 + 0.0531151144325989 w_k1_INTC_s115
                                 + 0.0416570217689442 w_k1_INTC_s116
                                 + 0.0455612490330924 w_k1_INTC_s117
                                 + 0.0164494451706375 w_k1_KO_s000
                                 + 0.0119520127438448 w_k1_KO_s001
                                 + 0.027831153473184 w_k1_KO_s002
                                 + 0.00935684251320665 w_k1_KO_s003
                                 + 0.0133073507649965 w_k1_KO_s004
                                 + 0.0195589982261897 w_k1_KO_s005
                                 + 0.00769289364429695 w_k1_KO_s006
                                 + 0.0139624895269192 w_k1_KO_s007
                                 + 0.0139520800293897 w_k1_KO_s010
                                 + 0.0364361264853711 w_k1_KO_s011
                                 + 0.0505499027162327 w_k1_KO_s012
                                 + 0.0147308041064609 w_k1_KO_s013
                                 + 0.0231450720463887 w_k1_KO_s014
                                 + 0.0161625850216027 w_k1_KO_s015
                                 + 0.0247377209132948 w_k1_KO_s016
                                 + 0.0262186477809226 w_k1_KO_s017
                                 + 0.0209897726342809 w_k1_KO_s100
                                 + 0.0132378657576956 w_k1_KO_s101
                                 + 0.0379761166637835 w_k1_KO_s102
                                 + 0.0198863179787078 w_k1_KO_s103
                                 + 0.0112242180596261 w_k1_KO_s104
                                 + 0.029320624313009 w_k1_KO_s105
                                 + 0.042269830415482 w_k1_KO_s106
                                 + 0.0181053874112001 w_k1_KO_s107
                                 + 0.0447820330669691 w_k1_KO_s110
                                 + 0.0758335792492598 w_k1_KO_s111
                                 + 0.0752374933399182 w_k1_KO_s112
                                 + 0.107774843092863 w_k1_KO_s113
                                 + 0.0558830196982701 w_k1_KO_s114
                                 + 0.0496748854783562 w_k1_KO_s115
                                 + 0.0443125272865643 w_k1_KO_s116
                                 + 0.05082597466073 w_k1_KO_s117
                                 + 0.0176270595642437 w_k1_MAT_s000
                                 + 0.0132662507259045 w_k1_MAT_s001
                                 + 0.0363714850956566 w_k1_MAT_s002
                                 + 0.0113924111530704 w_k1_MAT_s003
                                 + 0.0138409472994277 w_k1_MAT_s004
                                 + 0.0257022894493145 w_k1_MAT_s005
                                 + 0.00916369158927665 w_k1_MAT_s006
                                 + 0.0126749668731794 w_k1_MAT_s007
                                 + 0.0153550311477012 w_k1_MAT_s010
                                 + 0.0329300087275038 w_k1_MAT_s011
                                 + 0.0480479973171598 w_k1_MAT_s012
                                 + 0.0195095761415983 w_k1_MAT_s013
                                 + 0.0313501905216362 w_k1_MAT_s014
                                 + 0.0173813049415277 w_k1_MAT_s015
                                 + 0.0267200070260706 w_k1_MAT_s016
                                 + 0.0326377476855248 w_k1_MAT_s017
                                 + 0.0239371992676307 w_k1_MAT_s100
                                 + 0.0128024925219041 w_k1_MAT_s101
                                 + 0.0434328005171781 w_k1_MAT_s102
                                 + 0.0255490872913364 w_k1_MAT_s103
                                 + 0.0112126071960754 w_k1_MAT_s104
                                 + 0.0362153818800005 w_k1_MAT_s105
                                 + 0.0349276841342218 w_k1_MAT_s106
                                 + 0.0188752037882742 w_k1_MAT_s107
                                 + 0.0448843605745319 w_k1_MAT_s110
                                 + 0.0752173372299442 w_k1_MAT_s111
                                 + 0.0806407131878209 w_k1_MAT_s112
                                 + 0.1035912490182 w_k1_MAT_s113
                                 + 0.0597616738011218 w_k1_MAT_s114
                                 + 0.0641793759882146 w_k1_MAT_s115
                                 + 0.0486548623246811 w_k1_MAT_s116
                                 + 0.0543863104879656 w_k1_MAT_s117
                                 + 0.0176047806707467 w_k1_MSFT_s000
                                 + 0.0111599353095516 w_k1_MSFT_s001
                                 + 0.0282697041181052 w_k1_MSFT_s002
                                 + 0.0100264421196081 w_k1_MSFT_s003
                                 + 0.0124841879730994 w_k1_MSFT_s004
                                 + 0.0171702136352634 w_k1_MSFT_s005
                                 + 0.00814692479177472 w_k1_MSFT_s006
                                 + 0.0143964812321472 w_k1_MSFT_s007
                                 + 0.0151730013761873 w_k1_MSFT_s010
                                 + 0.0335455240824716 w_k1_MSFT_s011
                                 + 0.0508551763015393 w_k1_MSFT_s012
                                 + 0.0152673638334604 w_k1_MSFT_s013
                                 + 0.0250898062254766 w_k1_MSFT_s014
                                 + 0.0151349082559339 w_k1_MSFT_s015
                                 + 0.0251864511149417 w_k1_MSFT_s016
                                 + 0.0235395483218921 w_k1_MSFT_s017
                                 + 0.0221947948545961 w_k1_MSFT_s100
                                 + 0.0135509376083687 w_k1_MSFT_s101
                                 + 0.036121205186652 w_k1_MSFT_s102
                                 + 0.021679152882517 w_k1_MSFT_s103
                                 + 0.0115459953238268 w_k1_MSFT_s104
                                 + 0.0305964142411004 w_k1_MSFT_s105
                                 + 0.0379529151771094 w_k1_MSFT_s106
                                 + 0.017979221497655 w_k1_MSFT_s107
                                 + 0.042738838821133 w_k1_MSFT_s110
                                 + 0.0807634607208598 w_k1_MSFT_s111
                                 + 0.0812797150041671 w_k1_MSFT_s112
                                 + 0.111336750152089 w_k1_MSFT_s113
                                 + 0.0535744486416917 w_k1_MSFT_s114
                                 + 0.0532622747521752 w_k1_MSFT_s115
                                 + 0.0392670172626579 w_k1_MSFT_s116
                                 + 0.0458767108206001 w_k1_MSFT_s117
                                 + 0.0162468054978621 w_k1_PG_s000
                                 + 0.0112862346213318 w_k1_PG_s001
                                 + 0.0300967062244971 w_k1_PG_s002
                                 + 0.00991735007226445 w_k1_PG_s003
                                 + 0.0141241767209063 w_k1_PG_s004
                                 + 0.0176085871852404 w_k1_PG_s005
                                 + 0.00831396214982018 w_k1_PG_s006
                                 + 0.014673680052759 w_k1_PG_s007
                                 + 0.014394665587681 w_k1_PG_s010
                                 + 0.0324451564888953 w_k1_PG_s011
                                 + 0.0522761956517074 w_k1_PG_s012
                                 + 0.0141921398274359 w_k1_PG_s013
                                 + 0.0248036800108081 w_k1_PG_s014
                                 + 0.0159510755227204 w_k1_PG_s015
                                 + 0.0252714988210227 w_k1_PG_s016
                                 + 0.0242951722524235 w_k1_PG_s017
                                 + 0.0213999053651634 w_k1_PG_s100
                                 + 0.0130647795598989 w_k1_PG_s101
                                 + 0.0365373758809838 w_k1_PG_s102
                                 + 0.020582430184361 w_k1_PG_s103
                                 + 0.0117077981835145 w_k1_PG_s104
                                 + 0.0323653650852323 w_k1_PG_s105
                                 + 0.0370906825158778 w_k1_PG_s106
                                 + 0.0180102535466997 w_k1_PG_s107
                                 + 0.0419182552297539 w_k1_PG_s110
                                 + 0.0772699723940718 w_k1_PG_s111
                                 + 0.0760812298501619 w_k1_PG_s112
                                 + 0.111915181511121 w_k1_PG_s113
                                 + 0.0507794139011166 w_k1_PG_s114
                                 + 0.0505274471239201 w_k1_PG_s115
                                 + 0.0446516466257318 w_k1_PG_s116
                                 + 0.0441572230396871 w_k1_PG_s117 >= 1.219
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
