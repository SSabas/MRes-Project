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
 w_k1_ADM_s0_return_constraint:  1.10657 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.99051 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.90782 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.12407 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.808246 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.07562 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.21006 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.800069 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.38827 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.06659 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.15514 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.969756 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.00276 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.31753 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.22565 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.04018 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.05992 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.877047 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.87511 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.33805 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.09602 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.41167 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.15401 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.13266 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.972437 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.894836 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.1688 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.17125 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.919603 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.792336 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.178 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.34128 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.16422 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.942188 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.31007 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.11407 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.938574 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.810216 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.20544 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.914498 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.902499 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.31747 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.844792 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.25243 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.43361 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.786344 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.4543 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.02833 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.998246 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.84956 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.06597 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.32732 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.42015 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.979883 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.896757 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.9533 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.674698 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.39706 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.07641 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.34186 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.25429 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.982468 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.19635 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.742843 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.03028 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.38794 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.979758 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.688207 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.31405 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.48189 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.40585 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.965167 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.28394 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.01955 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.796469 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.877273 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.30453 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.875716 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.827247 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.33593 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.803568 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.30461 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.38227 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.682584 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.48506 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.899207 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.2026 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.875667 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.11525 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.36288 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.53389 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.13397 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.942007 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.930006 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.663922 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.79955 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.822663 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.28439 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.05556 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.05727 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.30293 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.697406 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.08673 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.57924 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.876212 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.712 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.14111 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.50509 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.63637 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.0255 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.61819 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.02503 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.71687 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.833186 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.20248 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.91667 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.864328 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.20652 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.862126 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.18047 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.20801 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.769735 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.18375 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.08998 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.26196 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.912707 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.10438 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.27256 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.12366 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.15868 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.919157 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.865422 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.789141 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.53293 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.03203 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.26007 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.945305 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.06613 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.09944 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.857414 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.03524 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.35137 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.869274 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.733774 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.19093 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.15361 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.24544 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.887646 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.39117 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.10964 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.83039 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.830729 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.07674 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.0023 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.971557 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.10443 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.889746 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.16217 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.51315 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.818264 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 2.16808 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.685403 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.555847 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.844451 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.07961 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.1239 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 2.16342 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.793684 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.980254 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.50611 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.738036 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.13878 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.661386 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.18409 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.66838 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.697622 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.35748 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.532741 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.982363 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.992462 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.05235 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.839575 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.02762 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.94747 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.33553 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.4385 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.818814 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.70559 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.615887 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.06258 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.13987 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.922662 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.975884 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.14548 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.943469 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.15743 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.03987 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.796451 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.939825 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.98019 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.900862 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.966751 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.12905 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.975562 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.10303 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.14966 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.902866 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.889545 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.728398 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.51764 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.938711 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.07632 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.942934 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.1108 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.08528 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.761738 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.03594 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.27374 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.772505 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.742873 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.04968 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.2953 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.14898 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.999501 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.33643 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.0456 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.88329 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.842812 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.15782 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.96107 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.989036 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.13981 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.801774 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.21682 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.13629 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.720948 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.38879 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.732564 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.691012 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.824254 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.949413 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.09105 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.59327 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.01472 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.981511 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.18966 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.795633 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.35105 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.750327 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.15427 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.27122 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.921799 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.37962 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.672441 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.12454 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.35305 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.01613 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.861204 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.960797 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.60583 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.28739 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.24652 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.20978 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.808606 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.658237 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.958772 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.14633 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.0125 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.933324 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.11585 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.775747 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.07274 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.35707 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.848981 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 2.06865 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.562929 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.675225 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.677766 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.992549 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.54847 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 2.16615 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.873873 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.23863 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.65687 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.793435 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.16237 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.900409 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.3589 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.64446 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.872481 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.307 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.552311 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.11748 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.988745 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.25809 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.74904 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.29321 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 2.17268 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.40313 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.58886 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.915481 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.713841 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.678133 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.04798 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.18841 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.86862 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.885399 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.17963 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.923348 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.19652 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.09043 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.764428 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.25025 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.968981 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.04329 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.92669 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.999081 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.23032 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.12579 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.1276 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.962153 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.956705 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.830688 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.55929 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.85571 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.10621 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.894549 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.06149 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.17013 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.760468 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.10501 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.29848 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.900438 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.767008 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.0784 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.17009 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.32919 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.04228 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.42916 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.04348 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.802047 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.873947 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.15673 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.938783 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.891065 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.03876 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.929502 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.09976 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.12117 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.82215 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.18402 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.956248 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.960288 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.913686 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.06529 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.12747 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.27955 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.03073 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.999671 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.05354 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.876988 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.25809 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.902271 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.21208 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.06354 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.934844 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.06122 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.800091 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.06375 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.23828 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.964098 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.873857 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.10424 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.32233 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.28953 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.04745 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.20731 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.0855 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.87914 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.949846 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.349000417366344 z_k1_s0000
                                 - 0.212000426994241 z_k1_s0001
                                 - 0.331001900164312 z_k1_s0002
                                 - 0.293000285616883 z_k1_s0003
                                 - 0.191000314788063 z_k1_s0004
                                 - 0.425001134456011 z_k1_s0005
                                 - 0.270000366077204 z_k1_s0006
                                 - 0.441000095525061 z_k1_s0007
                                 - 0.613999645949007 z_k1_s0010
                                 - 0.325000534409843 z_k1_s0011
                                 - 0.2280004539024 z_k1_s0012
                                 - 0.31500028930153 z_k1_s0013
                                 - 0.457998884983581 z_k1_s0014
                                 - 0.919001942145355 z_k1_s0015
                                 - 0.516001477653426 z_k1_s0016
                                 - 1.12900123421179 z_k1_s0017
                                 - 0.202000430295374 z_k1_s0020
                                 - 0.166999923932051 z_k1_s0021
                                 - 0.0690000180156679 z_k1_s0022
                                 - 0.111000044151301 z_k1_s0023
                                 - 0.203000610905856 z_k1_s0024
                                 - 0.116000016803144 z_k1_s0025
                                 - 0.0580000084015718 z_k1_s0026
                                 - 0.236999773657988 z_k1_s0027
                                 - 0.341999390941762 z_k1_s0030
                                 - 0.109999538540037 z_k1_s0031
                                 - 0.295999510265262 z_k1_s0032
                                 - 0.165999532319667 z_k1_s0033
                                 - 0.180999382175414 z_k1_s0034
                                 - 0.299999129107916 z_k1_s0035
                                 - 0.43599896171609 z_k1_s0036
                                 - 0.30099903381858 z_k1_s0037
                                 - 0.0839999292732189 z_k1_s0040
                                 - 0.111000060639548 z_k1_s0041
                                 - 0.0579999868981603 z_k1_s0042
                                 - 0.121000074553017 z_k1_s0043
                                 - 0.0869999334472597 z_k1_s0044
                                 - 0.23499985796672 z_k1_s0045
                                 - 0.101000046726078 z_k1_s0046
                                 - 0.141000102379956 z_k1_s0047
                                 - 0.356000317070175 z_k1_s0050
                                 - 0.914999797232575 z_k1_s0051
                                 - 0.498998850021823 z_k1_s0052
                                 - 0.603999197966969 z_k1_s0053
                                 - 0.394998613219135 z_k1_s0054
                                 - 0.275000372141432 z_k1_s0055
                                 - 0.21800007542823 z_k1_s0056
                                 - 0.512000294873776 z_k1_s0057
                                 - 0.176999677048572 z_k1_s0060
                                 - 0.380998649749794 z_k1_s0061
                                 - 0.652998310682912 z_k1_s0062
                                 - 0.513997699589819 z_k1_s0063
                                 - 0.56500014826028 z_k1_s0064
                                 - 0.234999618388698 z_k1_s0065
                                 - 0.308999394278091 z_k1_s0066
                                 - 0.2579994192032 z_k1_s0067
                                 - 0.130999865381093 z_k1_s0070
                                 - 0.46800018945436 z_k1_s0071
                                 - 0.29599914682617 z_k1_s0072
                                 - 0.153999895541877 z_k1_s0073
                                 - 0.191999104725202 z_k1_s0074
                                 - 0.245000203430131 z_k1_s0075
                                 - 0.22200071536916 z_k1_s0076
                                 - 0.0989998941269342 z_k1_s0077
                                 - 0.106000029124416 z_k1_s0100
                                 - 0.0840000297872733 z_k1_s0101
                                 - 0.147000437252734 z_k1_s0102
                                 - 0.247000218785199 z_k1_s0103
                                 - 0.227999928494025 z_k1_s0104
                                 - 0.153000092344677 z_k1_s0105
                                 - 0.117000028792988 z_k1_s0106
                                 - 0.102999964578441 z_k1_s0107
                                 - 0.37299890202015 z_k1_s0110
                                 - 0.133999554591883 z_k1_s0111
                                 - 0.162999316347382 z_k1_s0112
                                 - 0.316999434905806 z_k1_s0113
                                 - 0.158999379553405 z_k1_s0114
                                 - 0.16999933773634 z_k1_s0115
                                 - 0.158999379553405 z_k1_s0116
                                 - 0.284999236556664 z_k1_s0117
                                 - 0.34400037353396 z_k1_s0120
                                 - 0.540001936677477 z_k1_s0121
                                 - 0.829999653239905 z_k1_s0122
                                 - 0.226000380434728 z_k1_s0123
                                 - 0.220000476216269 z_k1_s0124
                                 - 0.786001528639679 z_k1_s0125
                                 - 0.238000540772185 z_k1_s0126
                                 - 0.335000517206273 z_k1_s0127
                                 - 0.344999581236634 z_k1_s0130
                                 - 0.0939997035983664 z_k1_s0131
                                 - 0.184998785796592 z_k1_s0132
                                 - 0.196999115003768 z_k1_s0133
                                 - 0.200000095802808 z_k1_s0134
                                 - 0.360998223185046 z_k1_s0135
                                 - 0.209998573147631 z_k1_s0136
                                 - 0.204999334475219 z_k1_s0137
                                 - 0.504998664541182 z_k1_s0140
                                 - 0.351999226620476 z_k1_s0141
                                 - 0.256999484144399 z_k1_s0142
                                 - 0.345999872387068 z_k1_s0143
                                 - 0.53999913889441 z_k1_s0144
                                 - 0.162999528007432 z_k1_s0145
                                 - 0.61499742879953 z_k1_s0146
                                 - 0.403000099591965 z_k1_s0147
                                 - 0.0709999634490766 z_k1_s0150
                                 - 0.105000020227475 z_k1_s0151
                                 - 0.187000136481269 z_k1_s0152
                                 - 0.126000200072894 z_k1_s0153
                                 - 0.0679999377568739 z_k1_s0154
                                 - 0.0759999769694271 z_k1_s0155
                                 - 0.127999924201156 z_k1_s0156
                                 - 0.117999545560609 z_k1_s0157
                                 - 0.262000155107529 z_k1_s0160
                                 - 0.20700094956483 z_k1_s0161
                                 - 0.160000584761204 z_k1_s0162
                                 - 0.177999957255763 z_k1_s0163
                                 - 0.103000335004261 z_k1_s0164
                                 - 0.104000291159484 z_k1_s0165
                                 - 0.429001241458008 z_k1_s0166
                                 - 0.174001102838126 z_k1_s0167
                                 - 0.150999492419979 z_k1_s0170
                                 - 0.205999022465252 z_k1_s0171
                                 - 0.349998210829722 z_k1_s0172
                                 - 0.113999719320717 z_k1_s0173
                                 - 0.218999748967825 z_k1_s0174
                                 - 0.123999668201566 z_k1_s0175
                                 - 0.409998659312407 z_k1_s0176
                                 - 0.313999829734085 z_k1_s0177
                                 - 0.525999151609747 z_k1_s1000
                                 - 0.773998301798804 z_k1_s1001
                                 - 0.672998590649283 z_k1_s1002
                                 - 0.29899965088374 z_k1_s1003
                                 - 0.241999926341823 z_k1_s1004
                                 - 0.564001807168302 z_k1_s1005
                                 - 0.745000633605201 z_k1_s1006
                                 - 0.908997400345688 z_k1_s1007
                                 - 0.361999370339381 z_k1_s1010
                                 - 0.695997551126892 z_k1_s1011
                                 - 1.02299876056956 z_k1_s1012
                                 - 0.367999040444561 z_k1_s1013
                                 - 0.730997758819371 z_k1_s1014
                                 - 0.389999024794421 z_k1_s1015
                                 - 0.868000033784055 z_k1_s1016
                                 - 0.678998912244668 z_k1_s1017
                                 - 0.712001479805975 z_k1_s1020
                                 - 0.248000269680671 z_k1_s1021
                                 - 0.636999012970496 z_k1_s1022
                                 - 0.388000069139026 z_k1_s1023
                                 - 1.0509989099389 z_k1_s1024
                                 - 0.93799810956084 z_k1_s1025
                                 - 0.780998977133418 z_k1_s1026
                                 - 0.215999946244383 z_k1_s1027
                                 - 0.57799959154572 z_k1_s1030
                                 - 0.515998001818246 z_k1_s1031
                                 - 0.523999378391823 z_k1_s1032
                                 - 0.487999236289224 z_k1_s1033
                                 - 0.388998442008512 z_k1_s1034
                                 - 0.457999117870392 z_k1_s1035
                                 - 0.211999357331669 z_k1_s1036
                                 - 0.869995364174799 z_k1_s1037
                                 - 0.223999879240882 z_k1_s1040
                                 - 0.172999804705323 z_k1_s1041
                                 - 0.646998642525648 z_k1_s1042
                                 - 0.241999820459388 z_k1_s1043
                                 - 0.239999891279499 z_k1_s1044
                                 - 0.305000048673786 z_k1_s1045
                                 - 0.522998983175995 z_k1_s1046
                                 - 0.539000152413617 z_k1_s1047
                                 - 0.504999013719706 z_k1_s1050
                                 - 0.366998535560498 z_k1_s1051
                                 - 0.326998658319165 z_k1_s1052
                                 - 0.868998347454138 z_k1_s1053
                                 - 0.248999123117719 z_k1_s1054
                                 - 0.239999263447996 z_k1_s1055
                                 - 0.368998754841717 z_k1_s1056
                                 - 0.830996435302498 z_k1_s1057
                                 - 0.15599992229865 z_k1_s1060
                                 - 0.128999962823859 z_k1_s1061
                                 - 0.346999524805605 z_k1_s1062
                                 - 0.110999931174052 z_k1_s1063
                                 - 0.213999965614749 z_k1_s1064
                                 - 0.151999797932133 z_k1_s1065
                                 - 0.219999800164844 z_k1_s1066
                                 - 0.430998968505345 z_k1_s1067
                                 - 0.364000544041862 z_k1_s1070
                                 - 1.10200047816297 z_k1_s1071
                                 - 0.764001764627513 z_k1_s1072
                                 - 0.550001562180048 z_k1_s1073
                                 - 0.390000516544726 z_k1_s1074
                                 - 0.347000740674946 z_k1_s1075
                                 - 0.287000696937363 z_k1_s1076
                                 - 0.841002076232896 z_k1_s1077
                                 - 0.509996511475783 z_k1_s1100
                                 - 0.705996250878433 z_k1_s1101
                                 - 0.145999584004781 z_k1_s1102
                                 - 0.390999258258095 z_k1_s1103
                                 - 0.446997112951715 z_k1_s1104
                                 - 0.624996506492133 z_k1_s1105
                                 - 0.59499713805091 z_k1_s1106
                                 - 0.203999329387811 z_k1_s1107
                                 - 0.774001804895997 z_k1_s1110
                                 - 0.235000349956088 z_k1_s1111
                                 - 0.414000745362764 z_k1_s1112
                                 - 0.358001421613957 z_k1_s1113
                                 - 0.336998915455506 z_k1_s1114
                                 - 0.171000266728059 z_k1_s1115
                                 - 0.22600008691753 z_k1_s1116
                                 - 0.639001013320656 z_k1_s1117
                                 - 0.230999831482977 z_k1_s1120
                                 - 0.224999237496008 z_k1_s1121
                                 - 0.380001052483703 z_k1_s1122
                                 - 0.390001311128164 z_k1_s1123
                                 - 0.276000995383053 z_k1_s1124
                                 - 0.300999447992749 z_k1_s1125
                                 - 0.294000583342501 z_k1_s1126
                                 - 0.0970000954496721 z_k1_s1127
                                 - 0.512000539791283 z_k1_s1130
                                 - 0.386998912739731 z_k1_s1131
                                 - 0.243000052074249 z_k1_s1132
                                 - 0.210000060004916 z_k1_s1133
                                 - 0.867999680926629 z_k1_s1134
                                 - 0.287000041500027 z_k1_s1135
                                 - 0.23699999827979 z_k1_s1136
                                 - 0.29400020552696 z_k1_s1137
                                 - 0.163000065048985 z_k1_s1140
                                 - 0.262000767962661 z_k1_s1141
                                 - 0.344999504090944 z_k1_s1142
                                 - 0.243000178968793 z_k1_s1143
                                 - 0.483999656464076 z_k1_s1144
                                 - 0.324000238625057 z_k1_s1145
                                 - 0.102000135122747 z_k1_s1146
                                 - 0.102000135122747 z_k1_s1147
                                 - 0.446999425066212 z_k1_s1150
                                 - 0.81199938323361 z_k1_s1151
                                 - 0.339999526711501 z_k1_s1152
                                 - 0.474000975456649 z_k1_s1153
                                 - 0.744999807608989 z_k1_s1154
                                 - 0.674999702330512 z_k1_s1155
                                 - 0.846999435872849 z_k1_s1156
                                 - 0.25499953015883 z_k1_s1157
                                 - 0.548000134493996 z_k1_s1160
                                 - 0.662000701982885 z_k1_s1161
                                 - 1.13600235424148 z_k1_s1162
                                 - 1.0940028524671 z_k1_s1163
                                 - 1.12999737327037 z_k1_s1164
                                 - 0.425000583154055 z_k1_s1165
                                 - 0.762003085914737 z_k1_s1166
                                 - 1.31600325027657 z_k1_s1167
                                 - 0.870002056254669 z_k1_s1170
                                 - 0.705002810931744 z_k1_s1171
                                 - 0.623000449907698 z_k1_s1172
                                 - 0.459000375164627 z_k1_s1173
                                 - 0.865002883851813 z_k1_s1174
                                 - 0.644000957403995 z_k1_s1175
                                 - 1.43900480520435 z_k1_s1176
                                 - 1.03399881149415 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.19035 w_k1_ADM_s000
                                 + 1.01238 w_k1_AXP_s000
                                 + 1.08276 w_k1_DIS_s000
                                 + 1.10545 w_k1_FDX_s000
                                 + 0.897366 w_k1_IBM_s000
                                 + 1.15836 w_k1_INTC_s000
                                 + 1.08477 w_k1_KO_s000 + 1.10088 w_k1_MAT_s000
                                 + 1.01399 w_k1_MSFT_s000
                                 + 1.14244 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.946603 w_k1_ADM_s000
                                 + 0.692908 w_k1_AXP_s000
                                 + 0.835536 w_k1_DIS_s000
                                 + 0.949211 w_k1_FDX_s000
                                 + 0.66011 w_k1_IBM_s000
                                 + 0.903873 w_k1_INTC_s000
                                 + 0.826915 w_k1_KO_s000
                                 + 0.852546 w_k1_MAT_s000
                                 + 0.996562 w_k1_MSFT_s000
                                 + 0.997805 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.03219 w_k1_ADM_s000
                                 + 1.28673 w_k1_AXP_s000
                                 + 1.40659 w_k1_DIS_s000
                                 + 1.24473 w_k1_FDX_s000 + 1.1613 w_k1_IBM_s000
                                 + 1.1982 w_k1_INTC_s000 + 1.44953 w_k1_KO_s000
                                 + 2.03975 w_k1_MAT_s000
                                 + 1.28939 w_k1_MSFT_s000
                                 + 1.18355 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.981464 w_k1_ADM_s000
                                 + 0.828291 w_k1_AXP_s000
                                 + 0.910132 w_k1_DIS_s000
                                 + 0.950971 w_k1_FDX_s000
                                 + 1.06715 w_k1_IBM_s000
                                 + 0.886947 w_k1_INTC_s000
                                 + 1.14903 w_k1_KO_s000 + 1.03675 w_k1_MAT_s000
                                 + 0.993828 w_k1_MSFT_s000
                                 + 0.998152 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.50548 w_k1_ADM_s000
                                 + 1.33011 w_k1_AXP_s000
                                 + 2.01348 w_k1_DIS_s000
                                 + 1.52192 w_k1_FDX_s000
                                 + 1.11438 w_k1_IBM_s000
                                 + 1.37141 w_k1_INTC_s000
                                 + 1.60173 w_k1_KO_s000 + 1.54636 w_k1_MAT_s000
                                 + 1.81095 w_k1_MSFT_s000
                                 + 1.41589 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.01174 w_k1_ADM_s000
                                 + 1.0218 w_k1_AXP_s000 + 1.04604 w_k1_DIS_s000
                                 + 0.908458 w_k1_FDX_s000
                                 + 1.42019 w_k1_IBM_s000
                                 + 1.00765 w_k1_INTC_s000 + 1.1788 w_k1_KO_s000
                                 + 1.6357 w_k1_MAT_s000
                                 + 1.03576 w_k1_MSFT_s000
                                 + 1.12344 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.2307 w_k1_ADM_s000
                                 + 0.733888 w_k1_AXP_s000
                                 + 1.29471 w_k1_DIS_s000
                                 + 1.09281 w_k1_FDX_s000
                                 + 0.795089 w_k1_IBM_s000
                                 + 1.3803 w_k1_INTC_s000 + 1.07747 w_k1_KO_s000
                                 + 0.434162 w_k1_MAT_s000
                                 + 1.33386 w_k1_MSFT_s000
                                 + 1.03761 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.768105 w_k1_ADM_s000
                                 + 0.662228 w_k1_AXP_s000
                                 + 0.717989 w_k1_DIS_s000
                                 + 0.707059 w_k1_FDX_s000
                                 + 0.873339 w_k1_IBM_s000
                                 + 0.928809 w_k1_INTC_s000
                                 + 0.807328 w_k1_KO_s000
                                 + 0.904598 w_k1_MAT_s000
                                 + 0.837821 w_k1_MSFT_s000
                                 + 0.95184 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.07267 w_k1_ADM_s001
                                 + 1.13335 w_k1_AXP_s001
                                 + 1.03862 w_k1_DIS_s001
                                 + 1.15688 w_k1_FDX_s001
                                 + 0.941606 w_k1_IBM_s001
                                 + 0.810318 w_k1_INTC_s001
                                 + 0.902189 w_k1_KO_s001
                                 + 1.00902 w_k1_MAT_s001
                                 + 0.986542 w_k1_MSFT_s001
                                 + 0.983362 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.735683 w_k1_ADM_s001
                                 + 0.541402 w_k1_AXP_s001
                                 + 0.611879 w_k1_DIS_s001
                                 + 0.677106 w_k1_FDX_s001
                                 + 0.735526 w_k1_IBM_s001
                                 + 0.718981 w_k1_INTC_s001
                                 + 0.752444 w_k1_KO_s001
                                 + 0.481813 w_k1_MAT_s001
                                 + 0.706659 w_k1_MSFT_s001
                                 + 0.805598 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.03273 w_k1_ADM_s001
                                 + 0.912095 w_k1_AXP_s001
                                 + 0.774302 w_k1_DIS_s001
                                 + 1.15016 w_k1_FDX_s001
                                 + 0.431295 w_k1_IBM_s001
                                 + 0.997104 w_k1_INTC_s001
                                 + 0.614996 w_k1_KO_s001
                                 + 0.356128 w_k1_MAT_s001
                                 + 0.952629 w_k1_MSFT_s001
                                 + 0.902833 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.05823 w_k1_ADM_s001
                                 + 1.30681 w_k1_AXP_s001 + 1.0466 w_k1_DIS_s001
                                 + 0.897122 w_k1_FDX_s001
                                 + 1.91272 w_k1_IBM_s001
                                 + 1.08643 w_k1_INTC_s001
                                 + 1.32457 w_k1_KO_s001 + 2.02098 w_k1_MAT_s001
                                 + 1.04431 w_k1_MSFT_s001
                                 + 1.16857 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.56209 w_k1_ADM_s001
                                 + 1.28404 w_k1_AXP_s001
                                 + 1.28269 w_k1_DIS_s001
                                 + 1.32469 w_k1_FDX_s001
                                 + 0.909535 w_k1_IBM_s001
                                 + 1.15178 w_k1_INTC_s001
                                 + 1.15673 w_k1_KO_s001 + 0.65615 w_k1_MAT_s001
                                 + 1.30651 w_k1_MSFT_s001
                                 + 1.18205 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.18436 w_k1_ADM_s001
                                 + 1.1673 w_k1_AXP_s001 + 1.23471 w_k1_DIS_s001
                                 + 1.19994 w_k1_FDX_s001
                                 + 1.52806 w_k1_IBM_s001
                                 + 1.14329 w_k1_INTC_s001
                                 + 1.35783 w_k1_KO_s001 + 1.31185 w_k1_MAT_s001
                                 + 1.13939 w_k1_MSFT_s001
                                 + 1.15507 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.841184 w_k1_ADM_s001
                                 + 0.872761 w_k1_AXP_s001
                                 + 0.73781 w_k1_DIS_s001
                                 + 0.802816 w_k1_FDX_s001
                                 + 1.14132 w_k1_IBM_s001
                                 + 0.893415 w_k1_INTC_s001
                                 + 0.969265 w_k1_KO_s001
                                 + 1.01679 w_k1_MAT_s001
                                 + 0.857077 w_k1_MSFT_s001
                                 + 0.935885 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.14925 w_k1_ADM_s001
                                 + 0.811467 w_k1_AXP_s001
                                 + 0.767468 w_k1_DIS_s001
                                 + 0.909986 w_k1_FDX_s001
                                 + 0.640336 w_k1_IBM_s001
                                 + 0.910943 w_k1_INTC_s001
                                 + 0.766413 w_k1_KO_s001
                                 + 0.591136 w_k1_MAT_s001
                                 + 0.983888 w_k1_MSFT_s001
                                 + 0.866655 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.1834 w_k1_ADM_s002
                                 + 1.58548 w_k1_AXP_s002
                                 + 1.90779 w_k1_DIS_s002
                                 + 1.44814 w_k1_FDX_s002
                                 + 1.12818 w_k1_IBM_s002
                                 + 1.40763 w_k1_INTC_s002
                                 + 1.52677 w_k1_KO_s002 + 1.611 w_k1_MAT_s002
                                 + 1.44388 w_k1_MSFT_s002
                                 + 1.27947 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.639377 w_k1_ADM_s002
                                 + 0.690662 w_k1_AXP_s002
                                 + 0.60748 w_k1_DIS_s002
                                 + 0.71529 w_k1_FDX_s002
                                 + 0.788043 w_k1_IBM_s002
                                 + 1.03985 w_k1_INTC_s002
                                 + 0.843725 w_k1_KO_s002
                                 + 0.712663 w_k1_MAT_s002
                                 + 0.73201 w_k1_MSFT_s002
                                 + 0.944303 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.907537 w_k1_ADM_s002
                                 + 1.57288 w_k1_AXP_s002
                                 + 1.16878 w_k1_DIS_s002
                                 + 1.15888 w_k1_FDX_s002
                                 + 1.04313 w_k1_IBM_s002
                                 + 1.34601 w_k1_INTC_s002
                                 + 1.09793 w_k1_KO_s002 + 1.31631 w_k1_MAT_s002
                                 + 1.00075 w_k1_MSFT_s002
                                 + 1.11126 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.575655 w_k1_ADM_s002
                                 + 0.994138 w_k1_AXP_s002
                                 + 0.840316 w_k1_DIS_s002
                                 + 0.786172 w_k1_FDX_s002
                                 + 1.18319 w_k1_IBM_s002
                                 + 1.12278 w_k1_INTC_s002
                                 + 1.03602 w_k1_KO_s002 + 1.38161 w_k1_MAT_s002
                                 + 0.832179 w_k1_MSFT_s002
                                 + 0.977667 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.920313 w_k1_ADM_s002
                                 + 0.991376 w_k1_AXP_s002
                                 + 0.811511 w_k1_DIS_s002
                                 + 0.959876 w_k1_FDX_s002
                                 + 0.799447 w_k1_IBM_s002
                                 + 1.14817 w_k1_INTC_s002
                                 + 0.765986 w_k1_KO_s002
                                 + 1.03092 w_k1_MAT_s002
                                 + 0.808903 w_k1_MSFT_s002
                                 + 0.963265 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.755708 w_k1_ADM_s002
                                 + 0.832861 w_k1_AXP_s002 + 1.034 w_k1_DIS_s002
                                 + 0.974183 w_k1_FDX_s002
                                 + 1.33813 w_k1_IBM_s002
                                 + 1.47867 w_k1_INTC_s002
                                 + 1.36269 w_k1_KO_s002 + 1.48243 w_k1_MAT_s002
                                 + 1.08385 w_k1_MSFT_s002
                                 + 1.05008 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.86055 w_k1_ADM_s002
                                 + 1.48669 w_k1_AXP_s002
                                 + 1.51299 w_k1_DIS_s002
                                 + 1.14186 w_k1_FDX_s002
                                 + 1.96423 w_k1_IBM_s002
                                 + 1.81676 w_k1_INTC_s002
                                 + 1.65703 w_k1_KO_s002 + 2.01658 w_k1_MAT_s002
                                 + 1.32125 w_k1_MSFT_s002
                                 + 1.46091 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.03513 w_k1_ADM_s002
                                 + 0.955544 w_k1_AXP_s002
                                 + 1.26875 w_k1_DIS_s002
                                 + 1.02915 w_k1_FDX_s002
                                 + 0.817549 w_k1_IBM_s002
                                 + 1.26218 w_k1_INTC_s002
                                 + 1.02664 w_k1_KO_s002
                                 + 0.920498 w_k1_MAT_s002
                                 + 0.915652 w_k1_MSFT_s002
                                 + 1.10515 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.03167 w_k1_ADM_s003
                                 + 0.745144 w_k1_AXP_s003
                                 + 0.907476 w_k1_DIS_s003
                                 + 0.873618 w_k1_FDX_s003
                                 + 1.03946 w_k1_IBM_s003
                                 + 0.871838 w_k1_INTC_s003
                                 + 1.03673 w_k1_KO_s003 + 1.18946 w_k1_MAT_s003
                                 + 0.894209 w_k1_MSFT_s003
                                 + 0.925447 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.23603 w_k1_ADM_s003
                                 + 1.00453 w_k1_AXP_s003
                                 + 1.18609 w_k1_DIS_s003
                                 + 1.23799 w_k1_FDX_s003 + 0.8026 w_k1_IBM_s003
                                 + 1.11895 w_k1_INTC_s003
                                 + 0.90881 w_k1_KO_s003 + 1.14475 w_k1_MAT_s003
                                 + 1.13959 w_k1_MSFT_s003
                                 + 0.988649 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.928793 w_k1_ADM_s003
                                 + 0.71993 w_k1_AXP_s003
                                 + 0.676206 w_k1_DIS_s003
                                 + 0.854817 w_k1_FDX_s003
                                 + 0.685626 w_k1_IBM_s003
                                 + 0.85186 w_k1_INTC_s003
                                 + 0.82654 w_k1_KO_s003
                                 + 0.760864 w_k1_MAT_s003
                                 + 0.9476 w_k1_MSFT_s003
                                 + 0.901901 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.850636 w_k1_ADM_s003
                                 + 0.71696 w_k1_AXP_s003
                                 + 0.89128 w_k1_DIS_s003
                                 + 0.947004 w_k1_FDX_s003
                                 + 0.76553 w_k1_IBM_s003
                                 + 0.930063 w_k1_INTC_s003
                                 + 0.878096 w_k1_KO_s003
                                 + 1.00536 w_k1_MAT_s003
                                 + 1.04288 w_k1_MSFT_s003
                                 + 0.917541 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.888406 w_k1_ADM_s003
                                 + 1.1258 w_k1_AXP_s003 + 1.07123 w_k1_DIS_s003
                                 + 0.881999 w_k1_FDX_s003
                                 + 1.40143 w_k1_IBM_s003
                                 + 0.936424 w_k1_INTC_s003
                                 + 1.18216 w_k1_KO_s003 + 2.06735 w_k1_MAT_s003
                                 + 0.951513 w_k1_MSFT_s003
                                 + 1.09722 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.41977 w_k1_ADM_s003
                                 + 1.26372 w_k1_AXP_s003
                                 + 1.68717 w_k1_DIS_s003 + 1.3112 w_k1_FDX_s003
                                 + 1.20424 w_k1_IBM_s003
                                 + 1.2129 w_k1_INTC_s003 + 1.353 w_k1_KO_s003
                                 + 1.71898 w_k1_MAT_s003
                                 + 1.29903 w_k1_MSFT_s003
                                 + 1.32995 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.03892 w_k1_ADM_s003
                                 + 1.16394 w_k1_AXP_s003
                                 + 1.20265 w_k1_DIS_s003
                                 + 1.06792 w_k1_FDX_s003
                                 + 1.24019 w_k1_IBM_s003
                                 + 1.11254 w_k1_INTC_s003
                                 + 1.27288 w_k1_KO_s003 + 1.88148 w_k1_MAT_s003
                                 + 1.13028 w_k1_MSFT_s003
                                 + 1.04327 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.981156 w_k1_ADM_s003
                                 + 0.870703 w_k1_AXP_s003
                                 + 1.17631 w_k1_DIS_s003
                                 + 1.04274 w_k1_FDX_s003
                                 + 0.829324 w_k1_IBM_s003
                                 + 0.988892 w_k1_INTC_s003
                                 + 1.18872 w_k1_KO_s003 + 1.14934 w_k1_MAT_s003
                                 + 1.13117 w_k1_MSFT_s003
                                 + 1.03244 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.47075 w_k1_ADM_s004
                                 + 1.39237 w_k1_AXP_s004
                                 + 1.34835 w_k1_DIS_s004
                                 + 1.38982 w_k1_FDX_s004
                                 + 0.881927 w_k1_IBM_s004
                                 + 1.53826 w_k1_INTC_s004
                                 + 1.33761 w_k1_KO_s004
                                 + 0.796587 w_k1_MAT_s004
                                 + 1.46563 w_k1_MSFT_s004
                                 + 1.03123 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.53579 w_k1_ADM_s004
                                 + 1.46507 w_k1_AXP_s004
                                 + 1.48696 w_k1_DIS_s004
                                 + 1.37141 w_k1_FDX_s004
                                 + 1.55285 w_k1_IBM_s004
                                 + 1.59414 w_k1_INTC_s004
                                 + 1.75002 w_k1_KO_s004 + 1.18823 w_k1_MAT_s004
                                 + 1.37608 w_k1_MSFT_s004
                                 + 1.29268 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.08555 w_k1_ADM_s004
                                 + 1.12739 w_k1_AXP_s004
                                 + 0.723404 w_k1_DIS_s004
                                 + 1.08209 w_k1_FDX_s004
                                 + 0.675018 w_k1_IBM_s004
                                 + 1.21464 w_k1_INTC_s004
                                 + 0.824994 w_k1_KO_s004
                                 + 0.581626 w_k1_MAT_s004
                                 + 1.12264 w_k1_MSFT_s004
                                 + 0.999955 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.15093 w_k1_ADM_s004
                                 + 0.937702 w_k1_AXP_s004
                                 + 0.920424 w_k1_DIS_s004
                                 + 1.02852 w_k1_FDX_s004
                                 + 0.856992 w_k1_IBM_s004
                                 + 1.18058 w_k1_INTC_s004
                                 + 1.01284 w_k1_KO_s004
                                 + 0.586802 w_k1_MAT_s004
                                 + 1.10253 w_k1_MSFT_s004
                                 + 1.08182 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.802245 w_k1_ADM_s004
                                 + 0.596233 w_k1_AXP_s004
                                 + 0.659331 w_k1_DIS_s004
                                 + 0.841304 w_k1_FDX_s004
                                 + 0.70888 w_k1_IBM_s004
                                 + 0.805695 w_k1_INTC_s004
                                 + 0.820744 w_k1_KO_s004
                                 + 0.631257 w_k1_MAT_s004
                                 + 0.843581 w_k1_MSFT_s004
                                 + 0.873519 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.886921 w_k1_ADM_s004
                                 + 0.829831 w_k1_AXP_s004
                                 + 0.919192 w_k1_DIS_s004
                                 + 0.921116 w_k1_FDX_s004
                                 + 0.98588 w_k1_IBM_s004
                                 + 0.96672 w_k1_INTC_s004
                                 + 1.01245 w_k1_KO_s004
                                 + 0.768539 w_k1_MAT_s004
                                 + 1.03528 w_k1_MSFT_s004
                                 + 0.999645 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.893557 w_k1_ADM_s004
                                 + 0.657881 w_k1_AXP_s004
                                 + 0.607811 w_k1_DIS_s004
                                 + 0.723002 w_k1_FDX_s004
                                 + 0.652751 w_k1_IBM_s004
                                 + 0.904294 w_k1_INTC_s004
                                 + 0.822506 w_k1_KO_s004
                                 + 0.392532 w_k1_MAT_s004
                                 + 0.831388 w_k1_MSFT_s004
                                 + 0.782917 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.22467 w_k1_ADM_s004
                                 + 1.00494 w_k1_AXP_s004
                                 + 1.18509 w_k1_DIS_s004
                                 + 1.10201 w_k1_FDX_s004
                                 + 1.23119 w_k1_IBM_s004
                                 + 1.31525 w_k1_INTC_s004
                                 + 1.37918 w_k1_KO_s004
                                 + 0.753715 w_k1_MAT_s004
                                 + 1.23916 w_k1_MSFT_s004
                                 + 1.16653 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.902377 w_k1_ADM_s005
                                 + 0.843811 w_k1_AXP_s005
                                 + 0.908716 w_k1_DIS_s005
                                 + 0.9318 w_k1_FDX_s005
                                 + 0.917062 w_k1_IBM_s005
                                 + 0.766038 w_k1_INTC_s005
                                 + 0.989374 w_k1_KO_s005
                                 + 1.23675 w_k1_MAT_s005
                                 + 0.91605 w_k1_MSFT_s005
                                 + 0.888677 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.99984 w_k1_ADM_s005
                                 + 1.22931 w_k1_AXP_s005
                                 + 1.28692 w_k1_DIS_s005
                                 + 1.02494 w_k1_FDX_s005
                                 + 1.49575 w_k1_IBM_s005
                                 + 1.12908 w_k1_INTC_s005
                                 + 1.27604 w_k1_KO_s005 + 1.83005 w_k1_MAT_s005
                                 + 1.11106 w_k1_MSFT_s005
                                 + 1.06998 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.04522 w_k1_ADM_s005
                                 + 0.660055 w_k1_AXP_s005
                                 + 0.828085 w_k1_DIS_s005
                                 + 0.950533 w_k1_FDX_s005
                                 + 0.573186 w_k1_IBM_s005
                                 + 0.794284 w_k1_INTC_s005
                                 + 0.730587 w_k1_KO_s005
                                 + 0.755028 w_k1_MAT_s005
                                 + 0.974627 w_k1_MSFT_s005
                                 + 0.889586 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.0569 w_k1_ADM_s005
                                 + 0.882297 w_k1_AXP_s005
                                 + 1.2043 w_k1_DIS_s005 + 1.16226 w_k1_FDX_s005
                                 + 0.863294 w_k1_IBM_s005
                                 + 0.88244 w_k1_INTC_s005
                                 + 0.968525 w_k1_KO_s005
                                 + 0.869879 w_k1_MAT_s005
                                 + 1.01583 w_k1_MSFT_s005
                                 + 1.04195 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.775718 w_k1_ADM_s005
                                 + 0.640795 w_k1_AXP_s005
                                 + 0.693204 w_k1_DIS_s005
                                 + 0.836931 w_k1_FDX_s005
                                 + 0.81098 w_k1_IBM_s005
                                 + 0.798682 w_k1_INTC_s005
                                 + 0.735467 w_k1_KO_s005
                                 + 0.741318 w_k1_MAT_s005
                                 + 0.73613 w_k1_MSFT_s005
                                 + 0.810912 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.0174 w_k1_ADM_s005
                                 + 0.758661 w_k1_AXP_s005
                                 + 1.21218 w_k1_DIS_s005
                                 + 1.04597 w_k1_FDX_s005
                                 + 0.956362 w_k1_IBM_s005
                                 + 0.954116 w_k1_INTC_s005
                                 + 1.03262 w_k1_KO_s005
                                 + 0.739492 w_k1_MAT_s005
                                 + 1.1326 w_k1_MSFT_s005 + 1.05067 w_k1_PG_s005
                                 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.11532 w_k1_ADM_s005
                                 + 1.26839 w_k1_AXP_s005
                                 + 1.72389 w_k1_DIS_s005
                                 + 1.45167 w_k1_FDX_s005
                                 + 0.834083 w_k1_IBM_s005
                                 + 1.00705 w_k1_INTC_s005
                                 + 1.09474 w_k1_KO_s005 + 1.21607 w_k1_MAT_s005
                                 + 1.45595 w_k1_MSFT_s005
                                 + 1.12379 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.983813 w_k1_ADM_s005
                                 + 0.787201 w_k1_AXP_s005
                                 + 0.843529 w_k1_DIS_s005
                                 + 0.832836 w_k1_FDX_s005
                                 + 1.2007 w_k1_IBM_s005
                                 + 0.76707 w_k1_INTC_s005
                                 + 0.959725 w_k1_KO_s005
                                 + 1.22592 w_k1_MAT_s005
                                 + 0.784085 w_k1_MSFT_s005
                                 + 0.923587 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.849993 w_k1_ADM_s006
                                 + 0.628402 w_k1_AXP_s006
                                 + 0.647822 w_k1_DIS_s006
                                 + 0.756521 w_k1_FDX_s006
                                 + 0.765002 w_k1_IBM_s006
                                 + 0.735031 w_k1_INTC_s006
                                 + 0.779164 w_k1_KO_s006
                                 + 0.865626 w_k1_MAT_s006
                                 + 0.945035 w_k1_MSFT_s006
                                 + 0.815555 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.38393 w_k1_ADM_s006
                                 + 1.24848 w_k1_AXP_s006
                                 + 0.984025 w_k1_DIS_s006
                                 + 1.1076 w_k1_FDX_s006
                                 + 0.809003 w_k1_IBM_s006
                                 + 0.859036 w_k1_INTC_s006
                                 + 0.88918 w_k1_KO_s006
                                 + 0.876879 w_k1_MAT_s006
                                 + 0.98815 w_k1_MSFT_s006
                                 + 1.00704 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.08434 w_k1_ADM_s006
                                 + 0.757797 w_k1_AXP_s006
                                 + 1.01097 w_k1_DIS_s006
                                 + 0.954988 w_k1_FDX_s006
                                 + 0.727838 w_k1_IBM_s006
                                 + 0.835498 w_k1_INTC_s006
                                 + 0.878008 w_k1_KO_s006
                                 + 0.436482 w_k1_MAT_s006
                                 + 1.14331 w_k1_MSFT_s006
                                 + 0.886436 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.913574 w_k1_ADM_s006
                                 + 0.827889 w_k1_AXP_s006
                                 + 0.829409 w_k1_DIS_s006
                                 + 0.846143 w_k1_FDX_s006
                                 + 1.34747 w_k1_IBM_s006
                                 + 0.739452 w_k1_INTC_s006
                                 + 1.16338 w_k1_KO_s006 + 1.36606 w_k1_MAT_s006
                                 + 0.836726 w_k1_MSFT_s006
                                 + 0.902554 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.00512 w_k1_ADM_s006
                                 + 0.980539 w_k1_AXP_s006
                                 + 0.867681 w_k1_DIS_s006
                                 + 0.909601 w_k1_FDX_s006
                                 + 1.04403 w_k1_IBM_s006
                                 + 0.809448 w_k1_INTC_s006
                                 + 1.00898 w_k1_KO_s006 + 1.04836 w_k1_MAT_s006
                                 + 0.998346 w_k1_MSFT_s006
                                 + 0.979477 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.49073 w_k1_ADM_s006
                                 + 1.30557 w_k1_AXP_s006
                                 + 1.14897 w_k1_DIS_s006
                                 + 1.05936 w_k1_FDX_s006
                                 + 1.60394 w_k1_IBM_s006
                                 + 1.09081 w_k1_INTC_s006
                                 + 1.43379 w_k1_KO_s006 + 1.61853 w_k1_MAT_s006
                                 + 1.03313 w_k1_MSFT_s006 + 1.0991 w_k1_PG_s006
                                 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.49306 w_k1_ADM_s006
                                 + 1.2 w_k1_AXP_s006 + 1.74757 w_k1_DIS_s006
                                 + 1.28544 w_k1_FDX_s006
                                 + 1.05705 w_k1_IBM_s006
                                 + 1.10426 w_k1_INTC_s006
                                 + 1.24022 w_k1_KO_s006 + 1.05555 w_k1_MAT_s006
                                 + 1.42906 w_k1_MSFT_s006
                                 + 0.975786 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.17979 w_k1_ADM_s006
                                 + 1.27882 w_k1_AXP_s006
                                 + 1.32229 w_k1_DIS_s006
                                 + 1.29185 w_k1_FDX_s006
                                 + 0.945819 w_k1_IBM_s006
                                 + 1.23522 w_k1_INTC_s006
                                 + 1.22596 w_k1_KO_s006 + 1.24423 w_k1_MAT_s006
                                 + 1.31257 w_k1_MSFT_s006
                                 + 1.18468 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.96323 w_k1_ADM_s007
                                 + 1.3043 w_k1_AXP_s007
                                 + 0.733741 w_k1_DIS_s007
                                 + 0.947508 w_k1_FDX_s007
                                 + 0.982378 w_k1_IBM_s007
                                 + 0.822171 w_k1_INTC_s007
                                 + 0.875894 w_k1_KO_s007
                                 + 0.974557 w_k1_MAT_s007
                                 + 0.850616 w_k1_MSFT_s007
                                 + 0.994102 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.787539 w_k1_ADM_s007
                                 + 0.77578 w_k1_AXP_s007
                                 + 0.76322 w_k1_DIS_s007
                                 + 0.795469 w_k1_FDX_s007
                                 + 0.813364 w_k1_IBM_s007
                                 + 0.949457 w_k1_INTC_s007
                                 + 0.89844 w_k1_KO_s007
                                 + 0.591222 w_k1_MAT_s007
                                 + 0.734478 w_k1_MSFT_s007
                                 + 0.84018 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.898986 w_k1_ADM_s007
                                 + 0.937277 w_k1_AXP_s007
                                 + 1.03579 w_k1_DIS_s007
                                 + 0.96544 w_k1_FDX_s007
                                 + 1.00074 w_k1_IBM_s007
                                 + 1.1299 w_k1_INTC_s007
                                 + 0.922534 w_k1_KO_s007
                                 + 0.77724 w_k1_MAT_s007
                                 + 1.01526 w_k1_MSFT_s007
                                 + 0.992926 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.2026 w_k1_ADM_s007
                                 + 1.08127 w_k1_AXP_s007 + 1.5518 w_k1_DIS_s007
                                 + 1.21963 w_k1_FDX_s007
                                 + 1.37153 w_k1_IBM_s007
                                 + 1.27423 w_k1_INTC_s007 + 1.4647 w_k1_KO_s007
                                 + 0.708568 w_k1_MAT_s007
                                 + 1.21326 w_k1_MSFT_s007
                                 + 1.21424 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.04129 w_k1_ADM_s007
                                 + 1.27895 w_k1_AXP_s007
                                 + 1.46886 w_k1_DIS_s007
                                 + 1.30988 w_k1_FDX_s007
                                 + 0.88415 w_k1_IBM_s007
                                 + 1.42178 w_k1_INTC_s007
                                 + 1.11741 w_k1_KO_s007 + 1.01493 w_k1_MAT_s007
                                 + 1.23862 w_k1_MSFT_s007
                                 + 1.17849 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.843254 w_k1_ADM_s007
                                 + 1.03818 w_k1_AXP_s007
                                 + 0.937442 w_k1_DIS_s007
                                 + 0.848345 w_k1_FDX_s007
                                 + 1.37749 w_k1_IBM_s007
                                 + 1.06923 w_k1_INTC_s007
                                 + 1.07847 w_k1_KO_s007 + 1.29243 w_k1_MAT_s007
                                 + 0.828581 w_k1_MSFT_s007
                                 + 0.980443 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.26593 w_k1_ADM_s007
                                 + 1.01751 w_k1_AXP_s007
                                 + 1.05593 w_k1_DIS_s007
                                 + 1.07832 w_k1_FDX_s007
                                 + 0.670924 w_k1_IBM_s007
                                 + 1.26263 w_k1_INTC_s007 + 0.9012 w_k1_KO_s007
                                 + 0.455537 w_k1_MAT_s007
                                 + 1.13603 w_k1_MSFT_s007
                                 + 1.02441 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.09078 w_k1_ADM_s007
                                 + 1.94641 w_k1_AXP_s007 + 1.726 w_k1_DIS_s007
                                 + 1.45902 w_k1_FDX_s007
                                 + 1.26541 w_k1_IBM_s007
                                 + 1.50936 w_k1_INTC_s007
                                 + 1.35077 w_k1_KO_s007 + 1.45711 w_k1_MAT_s007
                                 + 1.21259 w_k1_MSFT_s007
                                 + 1.19197 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.09258 w_k1_ADM_s010
                                 + 1.29247 w_k1_AXP_s010
                                 + 1.27588 w_k1_DIS_s010
                                 + 1.47542 w_k1_FDX_s010
                                 + 0.857781 w_k1_IBM_s010
                                 + 1.39675 w_k1_INTC_s010
                                 + 1.04412 w_k1_KO_s010 + 1.21013 w_k1_MAT_s010
                                 + 1.34778 w_k1_MSFT_s010
                                 + 1.25675 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.939762 w_k1_ADM_s010
                                 + 0.749633 w_k1_AXP_s010
                                 + 1.05051 w_k1_DIS_s010
                                 + 0.839681 w_k1_FDX_s010
                                 + 1.17642 w_k1_IBM_s010
                                 + 1.04805 w_k1_INTC_s010
                                 + 1.17725 w_k1_KO_s010 + 1.03844 w_k1_MAT_s010
                                 + 0.947397 w_k1_MSFT_s010
                                 + 0.928484 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.13111 w_k1_ADM_s010
                                 + 1.23086 w_k1_AXP_s010
                                 + 1.06415 w_k1_DIS_s010
                                 + 0.992393 w_k1_FDX_s010
                                 + 1.43576 w_k1_IBM_s010
                                 + 0.932304 w_k1_INTC_s010
                                 + 1.37642 w_k1_KO_s010 + 1.82629 w_k1_MAT_s010
                                 + 0.953174 w_k1_MSFT_s010
                                 + 1.01009 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.960198 w_k1_ADM_s010
                                 + 0.820026 w_k1_AXP_s010
                                 + 1.15419 w_k1_DIS_s010
                                 + 1.10252 w_k1_FDX_s010 + 0.8424 w_k1_IBM_s010
                                 + 0.85214 w_k1_INTC_s010
                                 + 0.97504 w_k1_KO_s010 + 1.08723 w_k1_MAT_s010
                                 + 1.09056 w_k1_MSFT_s010
                                 + 0.935302 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.903552 w_k1_ADM_s010
                                 + 1.05405 w_k1_AXP_s010
                                 + 0.886488 w_k1_DIS_s010
                                 + 0.970059 w_k1_FDX_s010
                                 + 0.931176 w_k1_IBM_s010
                                 + 0.916855 w_k1_INTC_s010
                                 + 0.915587 w_k1_KO_s010
                                 + 1.17039 w_k1_MAT_s010
                                 + 0.850145 w_k1_MSFT_s010
                                 + 1.02456 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.879828 w_k1_ADM_s010
                                 + 0.67661 w_k1_AXP_s010
                                 + 0.653306 w_k1_DIS_s010
                                 + 0.811362 w_k1_FDX_s010
                                 + 0.676637 w_k1_IBM_s010
                                 + 0.966791 w_k1_INTC_s010
                                 + 0.683118 w_k1_KO_s010
                                 + 0.559272 w_k1_MAT_s010
                                 + 0.827139 w_k1_MSFT_s010
                                 + 0.855384 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.19976 w_k1_ADM_s010
                                 + 1.38222 w_k1_AXP_s010
                                 + 1.59732 w_k1_DIS_s010
                                 + 1.33899 w_k1_FDX_s010
                                 + 1.52575 w_k1_IBM_s010
                                 + 1.45322 w_k1_INTC_s010
                                 + 1.59701 w_k1_KO_s010 + 1.53722 w_k1_MAT_s010
                                 + 1.3787 w_k1_MSFT_s010 + 1.28853 w_k1_PG_s010
                                 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.898464 w_k1_ADM_s010
                                 + 0.975512 w_k1_AXP_s010
                                 + 0.558161 w_k1_DIS_s010
                                 + 0.759321 w_k1_FDX_s010
                                 + 0.922923 w_k1_IBM_s010
                                 + 0.76726 w_k1_INTC_s010
                                 + 0.768281 w_k1_KO_s010
                                 + 1.03682 w_k1_MAT_s010
                                 + 0.610092 w_k1_MSFT_s010
                                 + 0.770622 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.899497 w_k1_ADM_s011
                                 + 0.690136 w_k1_AXP_s011
                                 + 0.688643 w_k1_DIS_s011
                                 + 0.778503 w_k1_FDX_s011
                                 + 0.884106 w_k1_IBM_s011
                                 + 0.819231 w_k1_INTC_s011
                                 + 0.727797 w_k1_KO_s011
                                 + 0.613457 w_k1_MAT_s011
                                 + 0.821486 w_k1_MSFT_s011
                                 + 0.891087 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.10549 w_k1_ADM_s011
                                 + 1.30277 w_k1_AXP_s011
                                 + 0.930205 w_k1_DIS_s011
                                 + 1.03058 w_k1_FDX_s011
                                 + 1.07737 w_k1_IBM_s011
                                 + 0.968736 w_k1_INTC_s011
                                 + 1.05497 w_k1_KO_s011 + 1.7893 w_k1_MAT_s011
                                 + 0.970518 w_k1_MSFT_s011
                                 + 0.997221 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.992968 w_k1_ADM_s011
                                 + 0.929135 w_k1_AXP_s011
                                 + 0.860347 w_k1_DIS_s011
                                 + 0.861314 w_k1_FDX_s011
                                 + 1.26869 w_k1_IBM_s011
                                 + 0.753609 w_k1_INTC_s011
                                 + 1.02897 w_k1_KO_s011 + 1.50219 w_k1_MAT_s011
                                 + 0.871433 w_k1_MSFT_s011
                                 + 0.988056 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.42693 w_k1_ADM_s011
                                 + 0.971722 w_k1_AXP_s011
                                 + 1.18293 w_k1_DIS_s011 + 1.0203 w_k1_FDX_s011
                                 + 1.34608 w_k1_IBM_s011
                                 + 0.94289 w_k1_INTC_s011 + 1.1783 w_k1_KO_s011
                                 + 0.884898 w_k1_MAT_s011
                                 + 1.05006 w_k1_MSFT_s011
                                 + 1.06826 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.42589 w_k1_ADM_s011
                                 + 1.18718 w_k1_AXP_s011
                                 + 0.982561 w_k1_DIS_s011
                                 + 1.22671 w_k1_FDX_s011
                                 + 0.763168 w_k1_IBM_s011
                                 + 1.03133 w_k1_INTC_s011
                                 + 0.839327 w_k1_KO_s011
                                 + 0.700963 w_k1_MAT_s011
                                 + 0.995158 w_k1_MSFT_s011
                                 + 1.08378 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.1843 w_k1_ADM_s011
                                 + 1.19352 w_k1_AXP_s011
                                 + 1.25597 w_k1_DIS_s011
                                 + 1.11527 w_k1_FDX_s011
                                 + 0.962932 w_k1_IBM_s011
                                 + 1.07863 w_k1_INTC_s011
                                 + 1.18767 w_k1_KO_s011 + 1.46806 w_k1_MAT_s011
                                 + 1.09781 w_k1_MSFT_s011
                                 + 1.21505 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.58211 w_k1_ADM_s011
                                 + 1.40589 w_k1_AXP_s011
                                 + 1.72279 w_k1_DIS_s011
                                 + 1.43896 w_k1_FDX_s011 + 1.2007 w_k1_IBM_s011
                                 + 1.02164 w_k1_INTC_s011
                                 + 1.16145 w_k1_KO_s011 + 1.57806 w_k1_MAT_s011
                                 + 1.32404 w_k1_MSFT_s011 + 1.4476 w_k1_PG_s011
                                 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.02537 w_k1_ADM_s011
                                 + 0.826747 w_k1_AXP_s011
                                 + 0.907818 w_k1_DIS_s011
                                 + 0.922964 w_k1_FDX_s011
                                 + 0.690664 w_k1_IBM_s011
                                 + 0.966747 w_k1_INTC_s011
                                 + 0.806644 w_k1_KO_s011
                                 + 0.691702 w_k1_MAT_s011
                                 + 0.84798 w_k1_MSFT_s011
                                 + 0.874563 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.766178 w_k1_ADM_s012
                                 + 0.647273 w_k1_AXP_s012
                                 + 0.750267 w_k1_DIS_s012
                                 + 0.888695 w_k1_FDX_s012
                                 + 0.629891 w_k1_IBM_s012
                                 + 1.04583 w_k1_INTC_s012
                                 + 0.713287 w_k1_KO_s012
                                 + 0.427054 w_k1_MAT_s012
                                 + 0.857112 w_k1_MSFT_s012
                                 + 0.935268 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.949995 w_k1_ADM_s012
                                 + 0.819103 w_k1_AXP_s012
                                 + 0.760118 w_k1_DIS_s012
                                 + 0.89253 w_k1_FDX_s012
                                 + 0.848372 w_k1_IBM_s012
                                 + 0.9681 w_k1_INTC_s012 + 0.81401 w_k1_KO_s012
                                 + 0.422821 w_k1_MAT_s012
                                 + 0.841237 w_k1_MSFT_s012
                                 + 0.883405 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.969788 w_k1_ADM_s012
                                 + 0.847404 w_k1_AXP_s012
                                 + 1.09967 w_k1_DIS_s012
                                 + 1.07714 w_k1_FDX_s012
                                 + 0.971351 w_k1_IBM_s012
                                 + 1.00924 w_k1_INTC_s012 + 1.0423 w_k1_KO_s012
                                 + 0.734942 w_k1_MAT_s012
                                 + 0.985296 w_k1_MSFT_s012
                                 + 1.06956 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.03023 w_k1_ADM_s012
                                 + 0.887311 w_k1_AXP_s012
                                 + 0.956166 w_k1_DIS_s012
                                 + 1.08165 w_k1_FDX_s012
                                 + 0.689811 w_k1_IBM_s012
                                 + 1.03559 w_k1_INTC_s012
                                 + 0.899995 w_k1_KO_s012
                                 + 0.627372 w_k1_MAT_s012
                                 + 0.980532 w_k1_MSFT_s012
                                 + 1.02396 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.11729 w_k1_ADM_s012
                                 + 1.26516 w_k1_AXP_s012
                                 + 1.52132 w_k1_DIS_s012
                                 + 1.25655 w_k1_FDX_s012
                                 + 1.71979 w_k1_IBM_s012
                                 + 1.28611 w_k1_INTC_s012
                                 + 1.51349 w_k1_KO_s012 + 1.29336 w_k1_MAT_s012
                                 + 1.16988 w_k1_MSFT_s012
                                 + 1.20569 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.15605 w_k1_ADM_s012
                                 + 1.16731 w_k1_AXP_s012
                                 + 1.37079 w_k1_DIS_s012
                                 + 1.28129 w_k1_FDX_s012
                                 + 1.11337 w_k1_IBM_s012
                                 + 1.40456 w_k1_INTC_s012 + 1.2749 w_k1_KO_s012
                                 + 0.95661 w_k1_MAT_s012
                                 + 1.23797 w_k1_MSFT_s012
                                 + 1.05374 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.883214 w_k1_ADM_s012
                                 + 0.997517 w_k1_AXP_s012
                                 + 0.949299 w_k1_DIS_s012
                                 + 0.92263 w_k1_FDX_s012
                                 + 1.34369 w_k1_IBM_s012
                                 + 1.10353 w_k1_INTC_s012 + 1.1458 w_k1_KO_s012
                                 + 1.0981 w_k1_MAT_s012
                                 + 0.866148 w_k1_MSFT_s012
                                 + 1.07314 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.727357 w_k1_ADM_s012
                                 + 0.869799 w_k1_AXP_s012
                                 + 0.688554 w_k1_DIS_s012
                                 + 0.772139 w_k1_FDX_s012
                                 + 1.05894 w_k1_IBM_s012
                                 + 0.961724 w_k1_INTC_s012
                                 + 0.888212 w_k1_KO_s012
                                 + 1.02364 w_k1_MAT_s012
                                 + 0.816895 w_k1_MSFT_s012
                                 + 0.927478 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.23741 w_k1_ADM_s013
                                 + 1.23627 w_k1_AXP_s013
                                 + 1.24266 w_k1_DIS_s013
                                 + 1.26311 w_k1_FDX_s013
                                 + 0.947499 w_k1_IBM_s013
                                 + 0.986064 w_k1_INTC_s013
                                 + 0.961094 w_k1_KO_s013
                                 + 0.997406 w_k1_MAT_s013
                                 + 0.973304 w_k1_MSFT_s013
                                 + 0.97074 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.885009 w_k1_ADM_s013
                                 + 0.795457 w_k1_AXP_s013
                                 + 0.828186 w_k1_DIS_s013
                                 + 0.829716 w_k1_FDX_s013
                                 + 1.15336 w_k1_IBM_s013
                                 + 1.00129 w_k1_INTC_s013
                                 + 0.904969 w_k1_KO_s013
                                 + 0.761568 w_k1_MAT_s013
                                 + 0.737115 w_k1_MSFT_s013
                                 + 0.884321 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.06046 w_k1_ADM_s013
                                 + 0.84179 w_k1_AXP_s013
                                 + 0.92349 w_k1_DIS_s013
                                 + 0.871729 w_k1_FDX_s013
                                 + 1.04735 w_k1_IBM_s013
                                 + 0.925503 w_k1_INTC_s013
                                 + 0.919224 w_k1_KO_s013 + 0.8751 w_k1_MAT_s013
                                 + 0.835939 w_k1_MSFT_s013
                                 + 0.870422 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.28925 w_k1_ADM_s013
                                 + 0.745608 w_k1_AXP_s013
                                 + 1.28189 w_k1_DIS_s013
                                 + 1.08409 w_k1_FDX_s013
                                 + 0.889566 w_k1_IBM_s013
                                 + 1.13332 w_k1_INTC_s013 + 1.2026 w_k1_KO_s013
                                 + 0.763051 w_k1_MAT_s013
                                 + 1.09269 w_k1_MSFT_s013
                                 + 1.14406 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.11841 w_k1_ADM_s013
                                 + 1.19921 w_k1_AXP_s013
                                 + 1.02724 w_k1_DIS_s013
                                 + 0.915869 w_k1_FDX_s013
                                 + 1.36748 w_k1_IBM_s013
                                 + 0.951195 w_k1_INTC_s013
                                 + 1.05965 w_k1_KO_s013 + 1.05402 w_k1_MAT_s013
                                 + 0.852766 w_k1_MSFT_s013
                                 + 1.13069 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.03479 w_k1_ADM_s013
                                 + 1.00308 w_k1_AXP_s013
                                 + 0.904921 w_k1_DIS_s013
                                 + 0.96775 w_k1_FDX_s013
                                 + 0.80808 w_k1_IBM_s013
                                 + 1.12821 w_k1_INTC_s013
                                 + 0.811356 w_k1_KO_s013
                                 + 0.698135 w_k1_MAT_s013
                                 + 0.944231 w_k1_MSFT_s013
                                 + 0.918235 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.856001 w_k1_ADM_s013
                                 + 0.55241 w_k1_AXP_s013
                                 + 0.616063 w_k1_DIS_s013
                                 + 0.705786 w_k1_FDX_s013
                                 + 0.682312 w_k1_IBM_s013
                                 + 0.862552 w_k1_INTC_s013
                                 + 0.620743 w_k1_KO_s013
                                 + 0.554241 w_k1_MAT_s013
                                 + 0.677945 w_k1_MSFT_s013
                                 + 0.775927 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.87875 w_k1_ADM_s013
                                 + 1.32026 w_k1_AXP_s013
                                 + 1.67755 w_k1_DIS_s013
                                 + 1.42624 w_k1_FDX_s013
                                 + 1.43098 w_k1_IBM_s013
                                 + 1.22815 w_k1_INTC_s013
                                 + 1.61833 w_k1_KO_s013 + 1.31304 w_k1_MAT_s013
                                 + 1.22611 w_k1_MSFT_s013
                                 + 1.29666 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.827661 w_k1_ADM_s014
                                 + 0.779166 w_k1_AXP_s014
                                 + 0.762885 w_k1_DIS_s014
                                 + 0.888494 w_k1_FDX_s014
                                 + 0.577934 w_k1_IBM_s014
                                 + 0.934973 w_k1_INTC_s014
                                 + 0.71706 w_k1_KO_s014
                                 + 0.635531 w_k1_MAT_s014
                                 + 0.766813 w_k1_MSFT_s014
                                 + 0.87278 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.779267 w_k1_ADM_s014
                                 + 0.9078 w_k1_AXP_s014
                                 + 0.688131 w_k1_DIS_s014
                                 + 0.795977 w_k1_FDX_s014
                                 + 1.00472 w_k1_IBM_s014
                                 + 1.12856 w_k1_INTC_s014
                                 + 0.789945 w_k1_KO_s014
                                 + 0.856715 w_k1_MAT_s014
                                 + 0.74772 w_k1_MSFT_s014
                                 + 0.924674 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.682514 w_k1_ADM_s014
                                 + 0.664642 w_k1_AXP_s014
                                 + 0.629078 w_k1_DIS_s014
                                 + 0.756524 w_k1_FDX_s014
                                 + 1.27265 w_k1_IBM_s014
                                 + 0.936235 w_k1_INTC_s014
                                 + 0.912017 w_k1_KO_s014
                                 + 0.947995 w_k1_MAT_s014
                                 + 0.717807 w_k1_MSFT_s014
                                 + 0.793683 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.995165 w_k1_ADM_s014
                                 + 1.18651 w_k1_AXP_s014
                                 + 1.16506 w_k1_DIS_s014
                                 + 1.15154 w_k1_FDX_s014
                                 + 0.975406 w_k1_IBM_s014
                                 + 1.29046 w_k1_INTC_s014
                                 + 1.07081 w_k1_KO_s014 + 1.15976 w_k1_MAT_s014
                                 + 1.06789 w_k1_MSFT_s014
                                 + 1.04884 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.88877 w_k1_ADM_s014
                                 + 0.840953 w_k1_AXP_s014
                                 + 0.911163 w_k1_DIS_s014
                                 + 0.877647 w_k1_FDX_s014
                                 + 1.05828 w_k1_IBM_s014
                                 + 0.987548 w_k1_INTC_s014
                                 + 0.865799 w_k1_KO_s014
                                 + 0.778568 w_k1_MAT_s014
                                 + 0.857286 w_k1_MSFT_s014
                                 + 0.961949 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.06065 w_k1_ADM_s014
                                 + 1.50219 w_k1_AXP_s014
                                 + 0.933071 w_k1_DIS_s014
                                 + 0.934273 w_k1_FDX_s014
                                 + 1.71654 w_k1_IBM_s014
                                 + 0.969907 w_k1_INTC_s014
                                 + 1.01728 w_k1_KO_s014 + 1.46564 w_k1_MAT_s014
                                 + 0.828187 w_k1_MSFT_s014
                                 + 1.07776 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.26812 w_k1_ADM_s014
                                 + 1.37001 w_k1_AXP_s014
                                 + 1.38153 w_k1_DIS_s014
                                 + 1.24967 w_k1_FDX_s014
                                 + 1.41392 w_k1_IBM_s014
                                 + 1.37174 w_k1_INTC_s014
                                 + 1.44834 w_k1_KO_s014 + 1.31282 w_k1_MAT_s014
                                 + 1.1197 w_k1_MSFT_s014 + 1.20717 w_k1_PG_s014
                                 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.04667 w_k1_ADM_s014
                                 + 1.02748 w_k1_AXP_s014
                                 + 1.06559 w_k1_DIS_s014
                                 + 1.10086 w_k1_FDX_s014
                                 + 0.733507 w_k1_IBM_s014
                                 + 1.06188 w_k1_INTC_s014
                                 + 0.900579 w_k1_KO_s014
                                 + 0.993343 w_k1_MAT_s014
                                 + 1.03906 w_k1_MSFT_s014
                                 + 1.00641 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.861566 w_k1_ADM_s015
                                 + 0.574598 w_k1_AXP_s015
                                 + 0.634206 w_k1_DIS_s015
                                 + 0.700025 w_k1_FDX_s015
                                 + 0.846329 w_k1_IBM_s015
                                 + 0.659028 w_k1_INTC_s015
                                 + 0.778605 w_k1_KO_s015
                                 + 0.71841 w_k1_MAT_s015
                                 + 0.704928 w_k1_MSFT_s015
                                 + 0.905498 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.799362 w_k1_ADM_s015
                                 + 1.08783 w_k1_AXP_s015
                                 + 1.34654 w_k1_DIS_s015
                                 + 1.09511 w_k1_FDX_s015
                                 + 1.05074 w_k1_IBM_s015
                                 + 1.03549 w_k1_INTC_s015 + 1.0722 w_k1_KO_s015
                                 + 1.40403 w_k1_MAT_s015
                                 + 1.05811 w_k1_MSFT_s015
                                 + 1.12525 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.0488 w_k1_ADM_s015
                                 + 0.715701 w_k1_AXP_s015
                                 + 0.811631 w_k1_DIS_s015
                                 + 0.898245 w_k1_FDX_s015
                                 + 0.625732 w_k1_IBM_s015
                                 + 0.749903 w_k1_INTC_s015
                                 + 0.916314 w_k1_KO_s015
                                 + 0.632499 w_k1_MAT_s015
                                 + 0.852501 w_k1_MSFT_s015
                                 + 0.861506 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.27803 w_k1_ADM_s015
                                 + 1.06896 w_k1_AXP_s015
                                 + 1.05775 w_k1_DIS_s015
                                 + 1.11317 w_k1_FDX_s015
                                 + 0.957154 w_k1_IBM_s015
                                 + 1.11915 w_k1_INTC_s015
                                 + 1.02492 w_k1_KO_s015
                                 + 0.868532 w_k1_MAT_s015
                                 + 1.05538 w_k1_MSFT_s015
                                 + 1.15304 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.59193 w_k1_ADM_s015
                                 + 1.4689 w_k1_AXP_s015 + 2.00837 w_k1_DIS_s015
                                 + 1.40843 w_k1_FDX_s015
                                 + 1.49329 w_k1_IBM_s015
                                 + 1.2074 w_k1_INTC_s015 + 1.49452 w_k1_KO_s015
                                 + 1.57491 w_k1_MAT_s015
                                 + 1.41719 w_k1_MSFT_s015 + 1.3744 w_k1_PG_s015
                                 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.34224 w_k1_ADM_s015
                                 + 1.54977 w_k1_AXP_s015
                                 + 1.31097 w_k1_DIS_s015
                                 + 1.32852 w_k1_FDX_s015
                                 + 0.906481 w_k1_IBM_s015
                                 + 0.93391 w_k1_INTC_s015
                                 + 1.10212 w_k1_KO_s015 + 1.4252 w_k1_MAT_s015
                                 + 1.11047 w_k1_MSFT_s015
                                 + 1.13018 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.04992 w_k1_ADM_s015
                                 + 1.17349 w_k1_AXP_s015
                                 + 1.20271 w_k1_DIS_s015
                                 + 0.962282 w_k1_FDX_s015
                                 + 1.44539 w_k1_IBM_s015
                                 + 0.859088 w_k1_INTC_s015
                                 + 1.12621 w_k1_KO_s015 + 1.25084 w_k1_MAT_s015
                                 + 0.870839 w_k1_MSFT_s015
                                 + 1.00947 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.894333 w_k1_ADM_s015
                                 + 0.95259 w_k1_AXP_s015
                                 + 0.840927 w_k1_DIS_s015
                                 + 0.840238 w_k1_FDX_s015
                                 + 1.20004 w_k1_IBM_s015
                                 + 0.885942 w_k1_INTC_s015
                                 + 1.00348 w_k1_KO_s015 + 1.21605 w_k1_MAT_s015
                                 + 0.773048 w_k1_MSFT_s015
                                 + 0.861573 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.967356 w_k1_ADM_s016
                                 + 0.887597 w_k1_AXP_s016
                                 + 0.841614 w_k1_DIS_s016
                                 + 0.938046 w_k1_FDX_s016
                                 + 0.892834 w_k1_IBM_s016
                                 + 0.866577 w_k1_INTC_s016
                                 + 0.958609 w_k1_KO_s016
                                 + 0.78364 w_k1_MAT_s016
                                 + 0.968023 w_k1_MSFT_s016
                                 + 1.05704 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.727233 w_k1_ADM_s016
                                 + 0.665091 w_k1_AXP_s016
                                 + 0.658258 w_k1_DIS_s016
                                 + 0.760419 w_k1_FDX_s016
                                 + 0.683317 w_k1_IBM_s016
                                 + 0.863868 w_k1_INTC_s016
                                 + 0.83626 w_k1_KO_s016
                                 + 0.594659 w_k1_MAT_s016
                                 + 0.821554 w_k1_MSFT_s016
                                 + 0.898931 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.922423 w_k1_ADM_s016
                                 + 0.887329 w_k1_AXP_s016
                                 + 0.749494 w_k1_DIS_s016
                                 + 0.856246 w_k1_FDX_s016
                                 + 1.37815 w_k1_IBM_s016
                                 + 0.76853 w_k1_INTC_s016
                                 + 1.02755 w_k1_KO_s016 + 1.00935 w_k1_MAT_s016
                                 + 0.848346 w_k1_MSFT_s016
                                 + 0.973836 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.16846 w_k1_ADM_s016
                                 + 0.780636 w_k1_AXP_s016
                                 + 1.00677 w_k1_DIS_s016
                                 + 1.03087 w_k1_FDX_s016
                                 + 0.688695 w_k1_IBM_s016
                                 + 0.913403 w_k1_INTC_s016
                                 + 0.881595 w_k1_KO_s016
                                 + 0.547595 w_k1_MAT_s016
                                 + 1.06441 w_k1_MSFT_s016
                                 + 1.07793 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.979029 w_k1_ADM_s016
                                 + 0.901594 w_k1_AXP_s016
                                 + 0.689706 w_k1_DIS_s016
                                 + 0.884854 w_k1_FDX_s016
                                 + 1.08934 w_k1_IBM_s016
                                 + 0.784819 w_k1_INTC_s016
                                 + 0.786551 w_k1_KO_s016
                                 + 1.20547 w_k1_MAT_s016
                                 + 0.839582 w_k1_MSFT_s016
                                 + 1.00565 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.03152 w_k1_ADM_s016
                                 + 1.21122 w_k1_AXP_s016
                                 + 0.990054 w_k1_DIS_s016
                                 + 0.961194 w_k1_FDX_s016
                                 + 1.74462 w_k1_IBM_s016
                                 + 1.00753 w_k1_INTC_s016
                                 + 1.20462 w_k1_KO_s016 + 1.06571 w_k1_MAT_s016
                                 + 0.987438 w_k1_MSFT_s016
                                 + 1.12968 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.04847 w_k1_ADM_s016
                                 + 1.06764 w_k1_AXP_s016
                                 + 1.11883 w_k1_DIS_s016 + 1.1329 w_k1_FDX_s016
                                 + 1.06364 w_k1_IBM_s016
                                 + 1.34668 w_k1_INTC_s016
                                 + 1.10373 w_k1_KO_s016
                                 + 0.933193 w_k1_MAT_s016
                                 + 1.15468 w_k1_MSFT_s016 + 1.119 w_k1_PG_s016
                                 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.412 w_k1_ADM_s016
                                 + 1.55162 w_k1_AXP_s016
                                 + 1.46481 w_k1_DIS_s016
                                 + 1.24543 w_k1_FDX_s016
                                 + 1.56701 w_k1_IBM_s016
                                 + 1.27615 w_k1_INTC_s016
                                 + 1.50426 w_k1_KO_s016 + 1.84309 w_k1_MAT_s016
                                 + 1.29268 w_k1_MSFT_s016
                                 + 1.24596 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.919702 w_k1_ADM_s017
                                 + 1.53731 w_k1_AXP_s017
                                 + 1.30477 w_k1_DIS_s017
                                 + 1.26926 w_k1_FDX_s017
                                 + 1.14467 w_k1_IBM_s017
                                 + 1.3641 w_k1_INTC_s017 + 1.17959 w_k1_KO_s017
                                 + 1.14482 w_k1_MAT_s017
                                 + 1.09597 w_k1_MSFT_s017
                                 + 1.03272 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.894239 w_k1_ADM_s017
                                 + 0.964373 w_k1_AXP_s017
                                 + 1.16952 w_k1_DIS_s017
                                 + 1.00126 w_k1_FDX_s017
                                 + 1.00573 w_k1_IBM_s017
                                 + 1.25305 w_k1_INTC_s017
                                 + 1.24954 w_k1_KO_s017 + 1.31583 w_k1_MAT_s017
                                 + 1.13453 w_k1_MSFT_s017
                                 + 1.02797 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.869737 w_k1_ADM_s017
                                 + 0.952707 w_k1_AXP_s017
                                 + 0.874284 w_k1_DIS_s017
                                 + 0.990537 w_k1_FDX_s017
                                 + 0.715985 w_k1_IBM_s017
                                 + 1.20128 w_k1_INTC_s017
                                 + 0.749149 w_k1_KO_s017
                                 + 0.703979 w_k1_MAT_s017
                                 + 0.952528 w_k1_MSFT_s017
                                 + 0.90354 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.697863 w_k1_ADM_s017
                                 + 1.04585 w_k1_AXP_s017
                                 + 1.20971 w_k1_DIS_s017
                                 + 0.910686 w_k1_FDX_s017
                                 + 1.68243 w_k1_IBM_s017
                                 + 1.19242 w_k1_INTC_s017
                                 + 1.44285 w_k1_KO_s017 + 1.35535 w_k1_MAT_s017
                                 + 1.08846 w_k1_MSFT_s017
                                 + 1.05604 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.32194 w_k1_ADM_s017
                                 + 1.4383 w_k1_AXP_s017 + 1.66052 w_k1_DIS_s017
                                 + 1.30537 w_k1_FDX_s017
                                 + 1.38396 w_k1_IBM_s017
                                 + 1.20369 w_k1_INTC_s017
                                 + 1.30528 w_k1_KO_s017 + 1.57149 w_k1_MAT_s017
                                 + 1.19941 w_k1_MSFT_s017
                                 + 1.20564 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.29282 w_k1_ADM_s017
                                 + 0.988704 w_k1_AXP_s017
                                 + 1.26809 w_k1_DIS_s017 + 1.2246 w_k1_FDX_s017
                                 + 0.842181 w_k1_IBM_s017
                                 + 1.16644 w_k1_INTC_s017
                                 + 1.13427 w_k1_KO_s017
                                 + 0.729401 w_k1_MAT_s017
                                 + 1.16079 w_k1_MSFT_s017
                                 + 1.17028 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.80965 w_k1_ADM_s017
                                 + 0.679825 w_k1_AXP_s017
                                 + 0.720691 w_k1_DIS_s017
                                 + 0.81277 w_k1_FDX_s017 + 0.972 w_k1_IBM_s017
                                 + 0.843088 w_k1_INTC_s017
                                 + 0.925927 w_k1_KO_s017
                                 + 0.839531 w_k1_MAT_s017
                                 + 0.87642 w_k1_MSFT_s017
                                 + 0.83805 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.890034 w_k1_ADM_s017
                                 + 1.1208 w_k1_AXP_s017
                                 + 0.997742 w_k1_DIS_s017
                                 + 0.912461 w_k1_FDX_s017
                                 + 1.21759 w_k1_IBM_s017
                                 + 1.07847 w_k1_INTC_s017
                                 + 0.988043 w_k1_KO_s017
                                 + 1.41178 w_k1_MAT_s017
                                 + 0.915908 w_k1_MSFT_s017
                                 + 0.903002 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.980029 w_k1_ADM_s100
                                 + 1.05075 w_k1_AXP_s100
                                 + 0.923841 w_k1_DIS_s100
                                 + 0.960053 w_k1_FDX_s100
                                 + 1.02675 w_k1_IBM_s100
                                 + 0.853817 w_k1_INTC_s100
                                 + 0.929578 w_k1_KO_s100
                                 + 1.52498 w_k1_MAT_s100
                                 + 1.05092 w_k1_MSFT_s100 + 1.0213 w_k1_PG_s100
                                 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.874761 w_k1_ADM_s100
                                 + 0.696353 w_k1_AXP_s100
                                 + 0.83389 w_k1_DIS_s100
                                 + 0.835562 w_k1_FDX_s100
                                 + 1.13395 w_k1_IBM_s100
                                 + 0.899339 w_k1_INTC_s100
                                 + 1.07235 w_k1_KO_s100 + 1.13902 w_k1_MAT_s100
                                 + 1.04542 w_k1_MSFT_s100
                                 + 1.01761 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.756777 w_k1_ADM_s100
                                 + 0.561617 w_k1_AXP_s100
                                 + 0.545104 w_k1_DIS_s100
                                 + 0.708789 w_k1_FDX_s100
                                 + 0.710715 w_k1_IBM_s100
                                 + 0.705489 w_k1_INTC_s100
                                 + 0.68661 w_k1_KO_s100
                                 + 0.770288 w_k1_MAT_s100
                                 + 0.806298 w_k1_MSFT_s100
                                 + 0.765818 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.820727 w_k1_ADM_s100
                                 + 1.1293 w_k1_AXP_s100 + 1.44033 w_k1_DIS_s100
                                 + 1.03416 w_k1_FDX_s100
                                 + 1.97774 w_k1_IBM_s100
                                 + 1.43468 w_k1_INTC_s100
                                 + 1.71181 w_k1_KO_s100 + 1.85832 w_k1_MAT_s100
                                 + 1.4699 w_k1_MSFT_s100 + 1.22607 w_k1_PG_s100
                                 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.1236 w_k1_ADM_s100
                                 + 1.68405 w_k1_AXP_s100
                                 + 1.21061 w_k1_DIS_s100
                                 + 1.33098 w_k1_FDX_s100 + 1.4125 w_k1_IBM_s100
                                 + 1.30233 w_k1_INTC_s100
                                 + 1.32177 w_k1_KO_s100 + 1.89408 w_k1_MAT_s100
                                 + 1.22049 w_k1_MSFT_s100 + 1.1242 w_k1_PG_s100
                                 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.922485 w_k1_ADM_s100
                                 + 0.9594 w_k1_AXP_s100 + 1.07606 w_k1_DIS_s100
                                 + 1.05378 w_k1_FDX_s100
                                 + 0.918037 w_k1_IBM_s100
                                 + 1.1792 w_k1_INTC_s100 + 1.07589 w_k1_KO_s100
                                 + 0.87412 w_k1_MAT_s100
                                 + 1.08829 w_k1_MSFT_s100 + 1.1586 w_k1_PG_s100
                                 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.15576 w_k1_ADM_s100
                                 + 1.0959 w_k1_AXP_s100 + 1.53219 w_k1_DIS_s100
                                 + 1.26474 w_k1_FDX_s100 + 1.1692 w_k1_IBM_s100
                                 + 1.24823 w_k1_INTC_s100
                                 + 1.21816 w_k1_KO_s100 + 1.17009 w_k1_MAT_s100
                                 + 1.33679 w_k1_MSFT_s100 + 1.203 w_k1_PG_s100
                                 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.871504 w_k1_ADM_s100
                                 + 0.792291 w_k1_AXP_s100
                                 + 0.836236 w_k1_DIS_s100
                                 + 0.929639 w_k1_FDX_s100
                                 + 0.673254 w_k1_IBM_s100
                                 + 0.920968 w_k1_INTC_s100
                                 + 0.793262 w_k1_KO_s100
                                 + 0.65957 w_k1_MAT_s100
                                 + 0.922941 w_k1_MSFT_s100
                                 + 0.856174 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.1352 w_k1_ADM_s101
                                 + 1.62584 w_k1_AXP_s101
                                 + 1.21493 w_k1_DIS_s101
                                 + 1.23843 w_k1_FDX_s101
                                 + 1.37987 w_k1_IBM_s101
                                 + 1.27797 w_k1_INTC_s101 + 1.2431 w_k1_KO_s101
                                 + 1.27746 w_k1_MAT_s101
                                 + 1.24789 w_k1_MSFT_s101 + 1.1286 w_k1_PG_s101
                                 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.01699 w_k1_ADM_s101
                                 + 1.02709 w_k1_AXP_s101
                                 + 1.23746 w_k1_DIS_s101
                                 + 1.09373 w_k1_FDX_s101
                                 + 1.32687 w_k1_IBM_s101
                                 + 1.00884 w_k1_INTC_s101
                                 + 1.14194 w_k1_KO_s101
                                 + 0.922431 w_k1_MAT_s101
                                 + 1.09102 w_k1_MSFT_s101
                                 + 1.11766 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.787624 w_k1_ADM_s101
                                 + 0.749821 w_k1_AXP_s101
                                 + 0.719908 w_k1_DIS_s101
                                 + 0.845006 w_k1_FDX_s101
                                 + 0.755604 w_k1_IBM_s101
                                 + 0.824415 w_k1_INTC_s101
                                 + 0.887554 w_k1_KO_s101
                                 + 0.622203 w_k1_MAT_s101
                                 + 0.880575 w_k1_MSFT_s101
                                 + 0.903685 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.20379 w_k1_ADM_s101
                                 + 1.45341 w_k1_AXP_s101
                                 + 1.50697 w_k1_DIS_s101 + 1.4544 w_k1_FDX_s101
                                 + 1.12928 w_k1_IBM_s101
                                 + 1.1671 w_k1_INTC_s101 + 1.24296 w_k1_KO_s101
                                 + 1.09328 w_k1_MAT_s101
                                 + 1.39545 w_k1_MSFT_s101
                                 + 1.14234 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.921262 w_k1_ADM_s101
                                 + 1.16899 w_k1_AXP_s101 + 0.977 w_k1_DIS_s101
                                 + 0.992402 w_k1_FDX_s101
                                 + 1.12036 w_k1_IBM_s101
                                 + 0.869341 w_k1_INTC_s101
                                 + 1.14655 w_k1_KO_s101 + 1.0244 w_k1_MAT_s101
                                 + 1.00443 w_k1_MSFT_s101
                                 + 1.08466 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.18841 w_k1_ADM_s101
                                 + 1.07036 w_k1_AXP_s101 + 1.2951 w_k1_DIS_s101
                                 + 1.23836 w_k1_FDX_s101
                                 + 0.817526 w_k1_IBM_s101
                                 + 1.00251 w_k1_INTC_s101 + 1.0976 w_k1_KO_s101
                                 + 0.749813 w_k1_MAT_s101
                                 + 1.18059 w_k1_MSFT_s101
                                 + 1.15159 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.866508 w_k1_ADM_s101
                                 + 0.947617 w_k1_AXP_s101
                                 + 0.846877 w_k1_DIS_s101
                                 + 0.9776 w_k1_FDX_s101
                                 + 0.928729 w_k1_IBM_s101
                                 + 0.770351 w_k1_INTC_s101
                                 + 0.97853 w_k1_KO_s101
                                 + 0.858367 w_k1_MAT_s101
                                 + 0.951734 w_k1_MSFT_s101
                                 + 0.984092 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.796972 w_k1_ADM_s101
                                 + 0.755247 w_k1_AXP_s101
                                 + 0.827959 w_k1_DIS_s101
                                 + 0.805665 w_k1_FDX_s101
                                 + 1.24903 w_k1_IBM_s101
                                 + 0.857393 w_k1_INTC_s101
                                 + 1.11427 w_k1_KO_s101
                                 + 0.909173 w_k1_MAT_s101
                                 + 0.877393 w_k1_MSFT_s101
                                 + 0.991907 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.36558 w_k1_ADM_s102
                                 + 1.32969 w_k1_AXP_s102
                                 + 1.43825 w_k1_DIS_s102
                                 + 1.28221 w_k1_FDX_s102
                                 + 1.55017 w_k1_IBM_s102
                                 + 1.36602 w_k1_INTC_s102
                                 + 1.55541 w_k1_KO_s102 + 1.98744 w_k1_MAT_s102
                                 + 1.23237 w_k1_MSFT_s102
                                 + 1.34241 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.688181 w_k1_ADM_s102
                                 + 0.784154 w_k1_AXP_s102
                                 + 0.982447 w_k1_DIS_s102
                                 + 0.817163 w_k1_FDX_s102
                                 + 1.39601 w_k1_IBM_s102
                                 + 1.03199 w_k1_INTC_s102
                                 + 1.23142 w_k1_KO_s102 + 1.66618 w_k1_MAT_s102
                                 + 1.00583 w_k1_MSFT_s102
                                 + 1.00069 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.890619 w_k1_ADM_s102
                                 + 0.724905 w_k1_AXP_s102
                                 + 0.588449 w_k1_DIS_s102
                                 + 0.817155 w_k1_FDX_s102
                                 + 0.668098 w_k1_IBM_s102
                                 + 0.926155 w_k1_INTC_s102 + 0.647 w_k1_KO_s102
                                 + 0.675089 w_k1_MAT_s102
                                 + 0.853006 w_k1_MSFT_s102
                                 + 0.775087 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.27531 w_k1_ADM_s102
                                 + 0.900088 w_k1_AXP_s102
                                 + 1.25109 w_k1_DIS_s102
                                 + 1.08012 w_k1_FDX_s102
                                 + 1.32085 w_k1_IBM_s102
                                 + 1.06548 w_k1_INTC_s102
                                 + 1.07443 w_k1_KO_s102
                                 + 0.945454 w_k1_MAT_s102
                                 + 0.997461 w_k1_MSFT_s102
                                 + 1.13264 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.20217 w_k1_ADM_s102
                                 + 0.799568 w_k1_AXP_s102
                                 + 0.864068 w_k1_DIS_s102
                                 + 0.951726 w_k1_FDX_s102
                                 + 0.827241 w_k1_IBM_s102
                                 + 0.907183 w_k1_INTC_s102
                                 + 0.815021 w_k1_KO_s102
                                 + 0.886769 w_k1_MAT_s102
                                 + 0.901964 w_k1_MSFT_s102
                                 + 0.941733 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.1023 w_k1_ADM_s102
                                 + 1.10994 w_k1_AXP_s102
                                 + 1.20947 w_k1_DIS_s102
                                 + 1.13528 w_k1_FDX_s102
                                 + 0.91602 w_k1_IBM_s102
                                 + 1.13723 w_k1_INTC_s102
                                 + 1.01145 w_k1_KO_s102
                                 + 0.927551 w_k1_MAT_s102
                                 + 1.2377 w_k1_MSFT_s102 + 1.05913 w_k1_PG_s102
                                 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.09728 w_k1_ADM_s102
                                 + 0.753425 w_k1_AXP_s102
                                 + 0.781023 w_k1_DIS_s102
                                 + 0.863313 w_k1_FDX_s102
                                 + 1.04494 w_k1_IBM_s102
                                 + 0.962288 w_k1_INTC_s102
                                 + 0.83003 w_k1_KO_s102 + 1.01077 w_k1_MAT_s102
                                 + 0.904782 w_k1_MSFT_s102
                                 + 0.881306 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.19444 w_k1_ADM_s102
                                 + 1.17454 w_k1_AXP_s102 + 1.0332 w_k1_DIS_s102
                                 + 0.983 w_k1_FDX_s102 + 1.23741 w_k1_IBM_s102
                                 + 0.948078 w_k1_INTC_s102
                                 + 1.06024 w_k1_KO_s102 + 1.40774 w_k1_MAT_s102
                                 + 0.838117 w_k1_MSFT_s102
                                 + 1.07275 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.977036 w_k1_ADM_s103
                                 + 0.844147 w_k1_AXP_s103
                                 + 0.815493 w_k1_DIS_s103
                                 + 0.962466 w_k1_FDX_s103
                                 + 0.962162 w_k1_IBM_s103
                                 + 0.931102 w_k1_INTC_s103
                                 + 0.946732 w_k1_KO_s103
                                 + 1.20099 w_k1_MAT_s103
                                 + 1.04859 w_k1_MSFT_s103
                                 + 0.923806 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.00687 w_k1_ADM_s103
                                 + 0.825823 w_k1_AXP_s103
                                 + 1.20683 w_k1_DIS_s103
                                 + 1.05096 w_k1_FDX_s103
                                 + 0.908783 w_k1_IBM_s103
                                 + 1.08544 w_k1_INTC_s103
                                 + 1.12105 w_k1_KO_s103 + 1.20328 w_k1_MAT_s103
                                 + 1.24586 w_k1_MSFT_s103
                                 + 1.13348 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.05206 w_k1_ADM_s103
                                 + 1.11879 w_k1_AXP_s103
                                 + 1.50227 w_k1_DIS_s103
                                 + 1.20752 w_k1_FDX_s103 + 1.4306 w_k1_IBM_s103
                                 + 1.13939 w_k1_INTC_s103 + 1.5308 w_k1_KO_s103
                                 + 1.54326 w_k1_MAT_s103
                                 + 1.44919 w_k1_MSFT_s103
                                 + 1.21456 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.998171 w_k1_ADM_s103
                                 + 0.954931 w_k1_AXP_s103
                                 + 0.819923 w_k1_DIS_s103
                                 + 0.790193 w_k1_FDX_s103
                                 + 1.26669 w_k1_IBM_s103
                                 + 1.02803 w_k1_INTC_s103
                                 + 1.03555 w_k1_KO_s103 + 1.08271 w_k1_MAT_s103
                                 + 1.03752 w_k1_MSFT_s103
                                 + 0.942733 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.772135 w_k1_ADM_s103
                                 + 0.908681 w_k1_AXP_s103
                                 + 1.00984 w_k1_DIS_s103
                                 + 0.91109 w_k1_FDX_s103
                                 + 1.19301 w_k1_IBM_s103
                                 + 1.17007 w_k1_INTC_s103
                                 + 1.26688 w_k1_KO_s103 + 1.22265 w_k1_MAT_s103
                                 + 1.20819 w_k1_MSFT_s103
                                 + 1.08077 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.29113 w_k1_ADM_s103
                                 + 1.19945 w_k1_AXP_s103
                                 + 1.20991 w_k1_DIS_s103
                                 + 1.30123 w_k1_FDX_s103
                                 + 0.80643 w_k1_IBM_s103
                                 + 1.24154 w_k1_INTC_s103
                                 + 1.10244 w_k1_KO_s103
                                 + 0.790931 w_k1_MAT_s103
                                 + 1.26175 w_k1_MSFT_s103
                                 + 1.05453 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.02553 w_k1_ADM_s103
                                 + 1.63287 w_k1_AXP_s103
                                 + 1.35566 w_k1_DIS_s103
                                 + 1.37815 w_k1_FDX_s103
                                 + 1.43144 w_k1_IBM_s103
                                 + 1.34412 w_k1_INTC_s103
                                 + 1.27661 w_k1_KO_s103 + 2.20679 w_k1_MAT_s103
                                 + 1.34893 w_k1_MSFT_s103
                                 + 1.14918 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.833491 w_k1_ADM_s103
                                 + 0.64926 w_k1_AXP_s103
                                 + 0.687205 w_k1_DIS_s103
                                 + 0.810831 w_k1_FDX_s103
                                 + 0.679769 w_k1_IBM_s103
                                 + 0.856943 w_k1_INTC_s103
                                 + 0.822801 w_k1_KO_s103
                                 + 0.552617 w_k1_MAT_s103
                                 + 0.885435 w_k1_MSFT_s103
                                 + 0.850481 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.915027 w_k1_ADM_s104
                                 + 0.750368 w_k1_AXP_s104
                                 + 0.808851 w_k1_DIS_s104
                                 + 0.953891 w_k1_FDX_s104
                                 + 0.714007 w_k1_IBM_s104
                                 + 0.846093 w_k1_INTC_s104
                                 + 0.797332 w_k1_KO_s104
                                 + 0.59893 w_k1_MAT_s104
                                 + 0.868238 w_k1_MSFT_s104
                                 + 0.857725 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.741329 w_k1_ADM_s104
                                 + 0.640333 w_k1_AXP_s104
                                 + 0.599197 w_k1_DIS_s104
                                 + 0.733025 w_k1_FDX_s104
                                 + 0.664614 w_k1_IBM_s104
                                 + 0.583327 w_k1_INTC_s104
                                 + 0.626349 w_k1_KO_s104
                                 + 0.494368 w_k1_MAT_s104
                                 + 0.725861 w_k1_MSFT_s104
                                 + 0.717525 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.767614 w_k1_ADM_s104
                                 + 0.825412 w_k1_AXP_s104
                                 + 0.791213 w_k1_DIS_s104
                                 + 0.848717 w_k1_FDX_s104
                                 + 0.967481 w_k1_IBM_s104
                                 + 0.884045 w_k1_INTC_s104
                                 + 0.822393 w_k1_KO_s104 + 0.9302 w_k1_MAT_s104
                                 + 0.843267 w_k1_MSFT_s104
                                 + 0.942669 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.03729 w_k1_ADM_s104
                                 + 1.13535 w_k1_AXP_s104
                                 + 1.34269 w_k1_DIS_s104
                                 + 1.06125 w_k1_FDX_s104
                                 + 1.63005 w_k1_IBM_s104
                                 + 0.952682 w_k1_INTC_s104
                                 + 1.49579 w_k1_KO_s104 + 1.74442 w_k1_MAT_s104
                                 + 1.2402 w_k1_MSFT_s104 + 1.15897 w_k1_PG_s104
                                 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.08192 w_k1_ADM_s104
                                 + 1.09389 w_k1_AXP_s104
                                 + 0.90006 w_k1_DIS_s104
                                 + 1.19261 w_k1_FDX_s104
                                 + 0.809061 w_k1_IBM_s104
                                 + 1.07615 w_k1_INTC_s104
                                 + 0.936499 w_k1_KO_s104
                                 + 0.730064 w_k1_MAT_s104
                                 + 1.09378 w_k1_MSFT_s104
                                 + 1.06704 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.84461 w_k1_ADM_s104
                                 + 0.995292 w_k1_AXP_s104
                                 + 0.832106 w_k1_DIS_s104
                                 + 0.845564 w_k1_FDX_s104
                                 + 1.18708 w_k1_IBM_s104
                                 + 0.941629 w_k1_INTC_s104
                                 + 1.03103 w_k1_KO_s104 + 1.36144 w_k1_MAT_s104
                                 + 0.89535 w_k1_MSFT_s104
                                 + 0.963635 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.893585 w_k1_ADM_s104
                                 + 1.19815 w_k1_AXP_s104
                                 + 1.42738 w_k1_DIS_s104
                                 + 1.22655 w_k1_FDX_s104
                                 + 0.96587 w_k1_IBM_s104 + 1.119 w_k1_INTC_s104
                                 + 1.1126 w_k1_KO_s104 + 0.945839 w_k1_MAT_s104
                                 + 1.12471 w_k1_MSFT_s104
                                 + 1.07608 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.18135 w_k1_ADM_s104
                                 + 1.13724 w_k1_AXP_s104
                                 + 1.20995 w_k1_DIS_s104
                                 + 1.16486 w_k1_FDX_s104
                                 + 1.21356 w_k1_IBM_s104
                                 + 0.998215 w_k1_INTC_s104
                                 + 1.27355 w_k1_KO_s104 + 1.20663 w_k1_MAT_s104
                                 + 1.15672 w_k1_MSFT_s104
                                 + 1.09725 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.32823 w_k1_ADM_s105
                                 + 1.11393 w_k1_AXP_s105
                                 + 1.13015 w_k1_DIS_s105
                                 + 1.19648 w_k1_FDX_s105
                                 + 0.975717 w_k1_IBM_s105
                                 + 1.01139 w_k1_INTC_s105
                                 + 0.912495 w_k1_KO_s105
                                 + 1.18915 w_k1_MAT_s105
                                 + 1.09763 w_k1_MSFT_s105
                                 + 0.874717 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.750705 w_k1_ADM_s105
                                 + 0.631966 w_k1_AXP_s105
                                 + 0.690235 w_k1_DIS_s105
                                 + 0.764471 w_k1_FDX_s105
                                 + 0.771022 w_k1_IBM_s105
                                 + 0.795807 w_k1_INTC_s105
                                 + 0.755469 w_k1_KO_s105
                                 + 1.09293 w_k1_MAT_s105
                                 + 0.91757 w_k1_MSFT_s105
                                 + 0.753024 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.02253 w_k1_ADM_s105
                                 + 0.902927 w_k1_AXP_s105
                                 + 0.66125 w_k1_DIS_s105
                                 + 0.939079 w_k1_FDX_s105
                                 + 0.827581 w_k1_IBM_s105
                                 + 0.862874 w_k1_INTC_s105
                                 + 0.752356 w_k1_KO_s105
                                 + 1.05737 w_k1_MAT_s105
                                 + 0.883526 w_k1_MSFT_s105
                                 + 0.810957 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.01049 w_k1_ADM_s105
                                 + 0.800288 w_k1_AXP_s105
                                 + 0.932827 w_k1_DIS_s105
                                 + 0.959992 w_k1_FDX_s105
                                 + 0.834624 w_k1_IBM_s105
                                 + 0.774319 w_k1_INTC_s105
                                 + 0.932215 w_k1_KO_s105
                                 + 0.984681 w_k1_MAT_s105
                                 + 0.959655 w_k1_MSFT_s105
                                 + 0.908681 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.976116 w_k1_ADM_s105
                                 + 0.638847 w_k1_AXP_s105
                                 + 0.608661 w_k1_DIS_s105
                                 + 0.834173 w_k1_FDX_s105
                                 + 0.499953 w_k1_IBM_s105
                                 + 0.935497 w_k1_INTC_s105
                                 + 0.674501 w_k1_KO_s105
                                 + 0.433891 w_k1_MAT_s105
                                 + 0.838518 w_k1_MSFT_s105
                                 + 0.783487 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.0877 w_k1_ADM_s105
                                 + 0.742338 w_k1_AXP_s105
                                 + 0.896507 w_k1_DIS_s105
                                 + 0.779363 w_k1_FDX_s105
                                 + 1.26297 w_k1_IBM_s105
                                 + 0.907299 w_k1_INTC_s105
                                 + 1.20932 w_k1_KO_s105 + 1.41592 w_k1_MAT_s105
                                 + 0.937039 w_k1_MSFT_s105
                                 + 0.958299 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.29038 w_k1_ADM_s105
                                 + 1.17531 w_k1_AXP_s105
                                 + 1.48712 w_k1_DIS_s105
                                 + 1.42632 w_k1_FDX_s105
                                 + 1.00847 w_k1_IBM_s105
                                 + 1.17304 w_k1_INTC_s105
                                 + 1.00374 w_k1_KO_s105 + 1.10163 w_k1_MAT_s105
                                 + 1.33761 w_k1_MSFT_s105
                                 + 1.09716 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.026 w_k1_ADM_s105
                                 + 1.17167 w_k1_AXP_s105
                                 + 1.18966 w_k1_DIS_s105
                                 + 1.05986 w_k1_FDX_s105
                                 + 1.55812 w_k1_IBM_s105
                                 + 1.00861 w_k1_INTC_s105
                                 + 1.41574 w_k1_KO_s105 + 1.96629 w_k1_MAT_s105
                                 + 1.1399 w_k1_MSFT_s105 + 1.10628 w_k1_PG_s105
                                 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.8855 w_k1_ADM_s106
                                 + 1.19264 w_k1_AXP_s106
                                 + 0.979499 w_k1_DIS_s106
                                 + 1.08353 w_k1_FDX_s106
                                 + 0.855299 w_k1_IBM_s106
                                 + 1.17437 w_k1_INTC_s106
                                 + 0.938436 w_k1_KO_s106
                                 + 1.16761 w_k1_MAT_s106
                                 + 1.00171 w_k1_MSFT_s106
                                 + 1.01259 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.63483 w_k1_ADM_s106
                                 + 1.8143 w_k1_AXP_s106 + 1.46255 w_k1_DIS_s106
                                 + 1.37952 w_k1_FDX_s106
                                 + 1.28951 w_k1_IBM_s106
                                 + 1.3892 w_k1_INTC_s106 + 1.31556 w_k1_KO_s106
                                 + 1.48613 w_k1_MAT_s106
                                 + 1.25875 w_k1_MSFT_s106
                                 + 1.25994 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.02008 w_k1_ADM_s106
                                 + 1.24587 w_k1_AXP_s106 + 1.1655 w_k1_DIS_s106
                                 + 1.01764 w_k1_FDX_s106
                                 + 1.36704 w_k1_IBM_s106
                                 + 1.17682 w_k1_INTC_s106
                                 + 1.15501 w_k1_KO_s106 + 1.14867 w_k1_MAT_s106
                                 + 0.926465 w_k1_MSFT_s106
                                 + 0.991685 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.0295 w_k1_ADM_s106
                                 + 1.19779 w_k1_AXP_s106
                                 + 1.11581 w_k1_DIS_s106
                                 + 1.19252 w_k1_FDX_s106
                                 + 0.707346 w_k1_IBM_s106
                                 + 1.23403 w_k1_INTC_s106
                                 + 0.911975 w_k1_KO_s106
                                 + 0.828814 w_k1_MAT_s106
                                 + 1.13666 w_k1_MSFT_s106
                                 + 0.955779 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.937979 w_k1_ADM_s106
                                 + 0.723165 w_k1_AXP_s106
                                 + 0.696209 w_k1_DIS_s106
                                 + 0.836732 w_k1_FDX_s106
                                 + 0.657343 w_k1_IBM_s106
                                 + 1.0754 w_k1_INTC_s106
                                 + 0.793953 w_k1_KO_s106
                                 + 0.467692 w_k1_MAT_s106
                                 + 0.888159 w_k1_MSFT_s106
                                 + 0.825704 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.915784 w_k1_ADM_s106
                                 + 0.875228 w_k1_AXP_s106
                                 + 1.09309 w_k1_DIS_s106
                                 + 0.991144 w_k1_FDX_s106
                                 + 0.940925 w_k1_IBM_s106
                                 + 1.38287 w_k1_INTC_s106
                                 + 1.01649 w_k1_KO_s106
                                 + 0.607439 w_k1_MAT_s106
                                 + 1.05657 w_k1_MSFT_s106
                                 + 0.912105 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.968831 w_k1_ADM_s106
                                 + 1.27651 w_k1_AXP_s106
                                 + 1.30623 w_k1_DIS_s106
                                 + 1.22269 w_k1_FDX_s106 + 1.0036 w_k1_IBM_s106
                                 + 1.37126 w_k1_INTC_s106
                                 + 1.06581 w_k1_KO_s106 + 1.18516 w_k1_MAT_s106
                                 + 1.22622 w_k1_MSFT_s106
                                 + 1.05447 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.707047 w_k1_ADM_s106
                                 + 0.716414 w_k1_AXP_s106
                                 + 0.734083 w_k1_DIS_s106
                                 + 0.831037 w_k1_FDX_s106
                                 + 0.940761 w_k1_IBM_s106
                                 + 0.951087 w_k1_INTC_s106
                                 + 0.834282 w_k1_KO_s106
                                 + 1.02083 w_k1_MAT_s106
                                 + 0.843093 w_k1_MSFT_s106
                                 + 0.890951 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.08996 w_k1_ADM_s107
                                 + 1.18854 w_k1_AXP_s107
                                 + 1.24533 w_k1_DIS_s107
                                 + 1.32572 w_k1_FDX_s107
                                 + 0.81834 w_k1_IBM_s107
                                 + 1.15834 w_k1_INTC_s107
                                 + 1.01645 w_k1_KO_s107 + 1.09646 w_k1_MAT_s107
                                 + 1.31294 w_k1_MSFT_s107
                                 + 1.05299 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.11144 w_k1_ADM_s107
                                 + 1.04048 w_k1_AXP_s107
                                 + 1.17879 w_k1_DIS_s107
                                 + 1.14315 w_k1_FDX_s107
                                 + 1.02344 w_k1_IBM_s107
                                 + 1.14305 w_k1_INTC_s107
                                 + 1.06756 w_k1_KO_s107 + 1.05059 w_k1_MAT_s107
                                 + 1.16006 w_k1_MSFT_s107
                                 + 1.03602 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.909315 w_k1_ADM_s107
                                 + 0.844802 w_k1_AXP_s107
                                 + 0.843444 w_k1_DIS_s107
                                 + 1.02479 w_k1_FDX_s107
                                 + 0.762839 w_k1_IBM_s107
                                 + 1.06108 w_k1_INTC_s107
                                 + 0.795998 w_k1_KO_s107
                                 + 0.678833 w_k1_MAT_s107
                                 + 1.09931 w_k1_MSFT_s107
                                 + 0.842539 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.825957 w_k1_ADM_s107
                                 + 0.76381 w_k1_AXP_s107
                                 + 0.745509 w_k1_DIS_s107
                                 + 0.819863 w_k1_FDX_s107
                                 + 0.943405 w_k1_IBM_s107
                                 + 0.996918 w_k1_INTC_s107
                                 + 0.875409 w_k1_KO_s107
                                 + 1.05732 w_k1_MAT_s107
                                 + 0.847596 w_k1_MSFT_s107
                                 + 0.881726 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.612354 w_k1_ADM_s107
                                 + 0.636498 w_k1_AXP_s107
                                 + 0.605572 w_k1_DIS_s107
                                 + 0.739548 w_k1_FDX_s107
                                 + 0.700927 w_k1_IBM_s107
                                 + 0.84977 w_k1_INTC_s107 + 0.6674 w_k1_KO_s107
                                 + 0.641014 w_k1_MAT_s107
                                 + 0.800872 w_k1_MSFT_s107
                                 + 0.751545 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.32307 w_k1_ADM_s107
                                 + 1.40081 w_k1_AXP_s107
                                 + 1.35459 w_k1_DIS_s107
                                 + 1.14803 w_k1_FDX_s107
                                 + 1.71494 w_k1_IBM_s107
                                 + 1.01236 w_k1_INTC_s107
                                 + 1.47364 w_k1_KO_s107 + 1.82282 w_k1_MAT_s107
                                 + 1.22668 w_k1_MSFT_s107
                                 + 1.18366 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.786205 w_k1_ADM_s107
                                 + 0.777876 w_k1_AXP_s107
                                 + 0.750895 w_k1_DIS_s107
                                 + 0.697569 w_k1_FDX_s107
                                 + 1.24343 w_k1_IBM_s107
                                 + 0.85164 w_k1_INTC_s107
                                 + 0.922023 w_k1_KO_s107
                                 + 0.98341 w_k1_MAT_s107
                                 + 0.808717 w_k1_MSFT_s107
                                 + 0.802388 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.21291 w_k1_ADM_s107
                                 + 1.28148 w_k1_AXP_s107
                                 + 1.26371 w_k1_DIS_s107
                                 + 1.21097 w_k1_FDX_s107
                                 + 1.35335 w_k1_IBM_s107
                                 + 1.12101 w_k1_INTC_s107
                                 + 1.33293 w_k1_KO_s107 + 1.27862 w_k1_MAT_s107
                                 + 1.22124 w_k1_MSFT_s107
                                 + 1.03932 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.17222 w_k1_ADM_s110
                                 + 1.09399 w_k1_AXP_s110
                                 + 1.40772 w_k1_DIS_s110
                                 + 1.07136 w_k1_FDX_s110
                                 + 1.42627 w_k1_IBM_s110
                                 + 1.29384 w_k1_INTC_s110
                                 + 1.35702 w_k1_KO_s110 + 1.17839 w_k1_MAT_s110
                                 + 1.2967 w_k1_MSFT_s110 + 1.09656 w_k1_PG_s110
                                 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.926182 w_k1_ADM_s110
                                 + 1.1369 w_k1_AXP_s110 + 1.00355 w_k1_DIS_s110
                                 + 0.992485 w_k1_FDX_s110
                                 + 1.17041 w_k1_IBM_s110
                                 + 1.1363 w_k1_INTC_s110 + 1.1018 w_k1_KO_s110
                                 + 1.382 w_k1_MAT_s110 + 1.04124 w_k1_MSFT_s110
                                 + 1.13148 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.907895 w_k1_ADM_s110
                                 + 0.681158 w_k1_AXP_s110
                                 + 0.725202 w_k1_DIS_s110
                                 + 0.823236 w_k1_FDX_s110
                                 + 0.568467 w_k1_IBM_s110
                                 + 0.870353 w_k1_INTC_s110
                                 + 0.743678 w_k1_KO_s110
                                 + 0.38771 w_k1_MAT_s110
                                 + 0.809245 w_k1_MSFT_s110
                                 + 0.810406 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.717679 w_k1_ADM_s110
                                 + 0.668495 w_k1_AXP_s110
                                 + 0.661728 w_k1_DIS_s110
                                 + 0.73634 w_k1_FDX_s110
                                 + 0.811465 w_k1_IBM_s110
                                 + 0.754855 w_k1_INTC_s110
                                 + 0.771888 w_k1_KO_s110
                                 + 0.680966 w_k1_MAT_s110
                                 + 0.823787 w_k1_MSFT_s110
                                 + 0.833142 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.875276 w_k1_ADM_s110
                                 + 0.707058 w_k1_AXP_s110
                                 + 0.81015 w_k1_DIS_s110
                                 + 0.778819 w_k1_FDX_s110
                                 + 1.06452 w_k1_IBM_s110
                                 + 0.847615 w_k1_INTC_s110
                                 + 0.992051 w_k1_KO_s110
                                 + 0.926747 w_k1_MAT_s110
                                 + 1.06748 w_k1_MSFT_s110
                                 + 0.922156 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.87309 w_k1_ADM_s110
                                 + 0.867314 w_k1_AXP_s110
                                 + 1.02571 w_k1_DIS_s110
                                 + 0.965566 w_k1_FDX_s110
                                 + 0.838974 w_k1_IBM_s110
                                 + 0.986062 w_k1_INTC_s110
                                 + 1.00431 w_k1_KO_s110
                                 + 0.749326 w_k1_MAT_s110
                                 + 0.976767 w_k1_MSFT_s110
                                 + 0.954898 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.1635 w_k1_ADM_s110
                                 + 1.01465 w_k1_AXP_s110
                                 + 1.16461 w_k1_DIS_s110
                                 + 1.17562 w_k1_FDX_s110
                                 + 0.889239 w_k1_IBM_s110
                                 + 0.968827 w_k1_INTC_s110
                                 + 1.04446 w_k1_KO_s110 + 1.14216 w_k1_MAT_s110
                                 + 1.21637 w_k1_MSFT_s110
                                 + 1.03089 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.13223 w_k1_ADM_s110
                                 + 1.7034 w_k1_AXP_s110 + 1.98358 w_k1_DIS_s110
                                 + 1.49825 w_k1_FDX_s110
                                 + 1.17345 w_k1_IBM_s110
                                 + 1.30513 w_k1_INTC_s110
                                 + 1.41919 w_k1_KO_s110 + 1.67239 w_k1_MAT_s110
                                 + 1.6245 w_k1_MSFT_s110 + 1.23883 w_k1_PG_s110
                                 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.98354 w_k1_ADM_s111
                                 + 1.138 w_k1_AXP_s111 + 1.37946 w_k1_DIS_s111
                                 + 1.18779 w_k1_FDX_s111
                                 + 1.02765 w_k1_IBM_s111
                                 + 1.03239 w_k1_INTC_s111
                                 + 1.16576 w_k1_KO_s111 + 1.31816 w_k1_MAT_s111
                                 + 1.38653 w_k1_MSFT_s111
                                 + 1.17382 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.779968 w_k1_ADM_s111
                                 + 0.999199 w_k1_AXP_s111
                                 + 0.837428 w_k1_DIS_s111
                                 + 0.986383 w_k1_FDX_s111
                                 + 0.663561 w_k1_IBM_s111
                                 + 0.89921 w_k1_INTC_s111
                                 + 0.756882 w_k1_KO_s111
                                 + 0.760085 w_k1_MAT_s111
                                 + 1.04085 w_k1_MSFT_s111
                                 + 0.832847 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.762568 w_k1_ADM_s111
                                 + 0.850281 w_k1_AXP_s111
                                 + 0.70557 w_k1_DIS_s111
                                 + 0.925417 w_k1_FDX_s111
                                 + 0.902429 w_k1_IBM_s111
                                 + 1.03727 w_k1_INTC_s111
                                 + 0.844054 w_k1_KO_s111
                                 + 0.913162 w_k1_MAT_s111
                                 + 1.00171 w_k1_MSFT_s111
                                 + 0.898207 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.729498 w_k1_ADM_s111
                                 + 0.763433 w_k1_AXP_s111
                                 + 0.676309 w_k1_DIS_s111
                                 + 0.755636 w_k1_FDX_s111
                                 + 1.11977 w_k1_IBM_s111
                                 + 0.728337 w_k1_INTC_s111
                                 + 0.720656 w_k1_KO_s111
                                 + 0.954537 w_k1_MAT_s111
                                 + 0.707365 w_k1_MSFT_s111
                                 + 0.950487 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.96968 w_k1_ADM_s111
                                 + 1.07789 w_k1_AXP_s111
                                 + 1.28216 w_k1_DIS_s111 + 1.2905 w_k1_FDX_s111
                                 + 0.751248 w_k1_IBM_s111
                                 + 1.12748 w_k1_INTC_s111
                                 + 0.986723 w_k1_KO_s111
                                 + 0.892337 w_k1_MAT_s111
                                 + 1.31679 w_k1_MSFT_s111
                                 + 0.990785 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.718321 w_k1_ADM_s111
                                 + 0.595307 w_k1_AXP_s111
                                 + 0.549656 w_k1_DIS_s111
                                 + 0.722639 w_k1_FDX_s111
                                 + 0.586836 w_k1_IBM_s111
                                 + 0.687487 w_k1_INTC_s111
                                 + 0.64885 w_k1_KO_s111 + 0.66283 w_k1_MAT_s111
                                 + 0.871913 w_k1_MSFT_s111
                                 + 0.708922 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.614903 w_k1_ADM_s111
                                 + 0.649262 w_k1_AXP_s111
                                 + 0.545127 w_k1_DIS_s111
                                 + 0.710461 w_k1_FDX_s111
                                 + 0.788617 w_k1_IBM_s111
                                 + 0.70863 w_k1_INTC_s111
                                 + 0.61357 w_k1_KO_s111
                                 + 0.678912 w_k1_MAT_s111
                                 + 0.68719 w_k1_MSFT_s111
                                 + 0.744681 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.952412 w_k1_ADM_s111
                                 + 1.06598 w_k1_AXP_s111
                                 + 1.24029 w_k1_DIS_s111
                                 + 1.09863 w_k1_FDX_s111
                                 + 1.48935 w_k1_IBM_s111
                                 + 1.08026 w_k1_INTC_s111
                                 + 1.12144 w_k1_KO_s111 + 1.64194 w_k1_MAT_s111
                                 + 1.30316 w_k1_MSFT_s111
                                 + 1.17078 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.937145 w_k1_ADM_s112
                                 + 0.982854 w_k1_AXP_s112
                                 + 0.801932 w_k1_DIS_s112
                                 + 0.855977 w_k1_FDX_s112
                                 + 1.29343 w_k1_IBM_s112
                                 + 0.893986 w_k1_INTC_s112
                                 + 1.02601 w_k1_KO_s112 + 1.32514 w_k1_MAT_s112
                                 + 0.834387 w_k1_MSFT_s112
                                 + 0.980252 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.914342 w_k1_ADM_s112
                                 + 1.04856 w_k1_AXP_s112
                                 + 1.43336 w_k1_DIS_s112 + 1.1635 w_k1_FDX_s112
                                 + 1.58531 w_k1_IBM_s112
                                 + 1.33936 w_k1_INTC_s112
                                 + 1.40157 w_k1_KO_s112 + 1.82349 w_k1_MAT_s112
                                 + 1.29477 w_k1_MSFT_s112
                                 + 1.25264 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.1278 w_k1_ADM_s112
                                 + 1.04618 w_k1_AXP_s112
                                 + 1.05082 w_k1_DIS_s112
                                 + 1.04692 w_k1_FDX_s112
                                 + 1.05769 w_k1_IBM_s112
                                 + 1.03668 w_k1_INTC_s112
                                 + 1.10924 w_k1_KO_s112 + 1.14275 w_k1_MAT_s112
                                 + 0.97889 w_k1_MSFT_s112
                                 + 1.01497 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.837228 w_k1_ADM_s112
                                 + 0.626627 w_k1_AXP_s112
                                 + 0.638655 w_k1_DIS_s112
                                 + 0.756421 w_k1_FDX_s112
                                 + 0.890783 w_k1_IBM_s112
                                 + 0.839331 w_k1_INTC_s112
                                 + 0.749674 w_k1_KO_s112
                                 + 0.735167 w_k1_MAT_s112
                                 + 0.817021 w_k1_MSFT_s112
                                 + 0.823381 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.30117 w_k1_ADM_s112
                                 + 0.888636 w_k1_AXP_s112
                                 + 1.10298 w_k1_DIS_s112
                                 + 1.15581 w_k1_FDX_s112
                                 + 0.837179 w_k1_IBM_s112
                                 + 0.962371 w_k1_INTC_s112
                                 + 0.968672 w_k1_KO_s112
                                 + 0.842466 w_k1_MAT_s112
                                 + 0.886047 w_k1_MSFT_s112
                                 + 0.882295 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.13732 w_k1_ADM_s112
                                 + 1.12735 w_k1_AXP_s112
                                 + 1.65161 w_k1_DIS_s112
                                 + 1.39335 w_k1_FDX_s112
                                 + 1.06119 w_k1_IBM_s112
                                 + 1.43037 w_k1_INTC_s112
                                 + 1.46393 w_k1_KO_s112 + 1.38292 w_k1_MAT_s112
                                 + 1.37056 w_k1_MSFT_s112
                                 + 1.17576 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.916637 w_k1_ADM_s112
                                 + 0.752582 w_k1_AXP_s112
                                 + 0.77573 w_k1_DIS_s112
                                 + 0.891026 w_k1_FDX_s112
                                 + 0.695493 w_k1_IBM_s112
                                 + 0.957503 w_k1_INTC_s112
                                 + 0.778941 w_k1_KO_s112
                                 + 0.782648 w_k1_MAT_s112
                                 + 0.830338 w_k1_MSFT_s112
                                 + 0.8436 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.901269 w_k1_ADM_s112
                                 + 0.70715 w_k1_AXP_s112
                                 + 1.15941 w_k1_DIS_s112
                                 + 0.827653 w_k1_FDX_s112
                                 + 1.28459 w_k1_IBM_s112
                                 + 0.964265 w_k1_INTC_s112
                                 + 1.23579 w_k1_KO_s112 + 1.24843 w_k1_MAT_s112
                                 + 1.01373 w_k1_MSFT_s112 + 1.0367 w_k1_PG_s112
                                 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.06941 w_k1_ADM_s113
                                 + 1.07513 w_k1_AXP_s113
                                 + 0.84704 w_k1_DIS_s113
                                 + 0.956731 w_k1_FDX_s113
                                 + 0.81495 w_k1_IBM_s113
                                 + 0.917794 w_k1_INTC_s113
                                 + 0.885058 w_k1_KO_s113
                                 + 0.853261 w_k1_MAT_s113
                                 + 0.878989 w_k1_MSFT_s113
                                 + 0.98829 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.33688 w_k1_ADM_s113
                                 + 1.34398 w_k1_AXP_s113
                                 + 1.24816 w_k1_DIS_s113
                                 + 1.28774 w_k1_FDX_s113
                                 + 1.18293 w_k1_IBM_s113
                                 + 0.988874 w_k1_INTC_s113
                                 + 1.02927 w_k1_KO_s113 + 1.40734 w_k1_MAT_s113
                                 + 1.09051 w_k1_MSFT_s113
                                 + 1.06589 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.05322 w_k1_ADM_s113
                                 + 1.44221 w_k1_AXP_s113
                                 + 1.69381 w_k1_DIS_s113
                                 + 1.31852 w_k1_FDX_s113
                                 + 1.11578 w_k1_IBM_s113
                                 + 1.05655 w_k1_INTC_s113
                                 + 1.16554 w_k1_KO_s113 + 1.18417 w_k1_MAT_s113
                                 + 1.1522 w_k1_MSFT_s113 + 1.13878 w_k1_PG_s113
                                 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.685942 w_k1_ADM_s113
                                 + 0.738967 w_k1_AXP_s113
                                 + 0.513361 w_k1_DIS_s113
                                 + 0.68233 w_k1_FDX_s113
                                 + 1.10333 w_k1_IBM_s113
                                 + 0.732012 w_k1_INTC_s113
                                 + 0.717868 w_k1_KO_s113
                                 + 0.98969 w_k1_MAT_s113
                                 + 0.618592 w_k1_MSFT_s113
                                 + 0.828664 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.914977 w_k1_ADM_s113
                                 + 0.996873 w_k1_AXP_s113
                                 + 1.0388 w_k1_DIS_s113 + 1.02215 w_k1_FDX_s113
                                 + 0.953052 w_k1_IBM_s113
                                 + 0.889467 w_k1_INTC_s113
                                 + 0.811161 w_k1_KO_s113
                                 + 0.999659 w_k1_MAT_s113
                                 + 0.832927 w_k1_MSFT_s113
                                 + 1.04286 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.895102 w_k1_ADM_s113
                                 + 1.02971 w_k1_AXP_s113
                                 + 0.821542 w_k1_DIS_s113
                                 + 0.92031 w_k1_FDX_s113
                                 + 1.23427 w_k1_IBM_s113
                                 + 0.92112 w_k1_INTC_s113
                                 + 0.880876 w_k1_KO_s113
                                 + 1.28669 w_k1_MAT_s113
                                 + 0.886781 w_k1_MSFT_s113
                                 + 1.01566 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.907083 w_k1_ADM_s113
                                 + 1.31191 w_k1_AXP_s113
                                 + 1.02524 w_k1_DIS_s113
                                 + 0.951574 w_k1_FDX_s113
                                 + 1.60662 w_k1_IBM_s113
                                 + 0.993699 w_k1_INTC_s113
                                 + 0.996499 w_k1_KO_s113
                                 + 1.43194 w_k1_MAT_s113
                                 + 0.795249 w_k1_MSFT_s113
                                 + 0.990034 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.797949 w_k1_ADM_s113
                                 + 0.82582 w_k1_AXP_s113
                                 + 0.883625 w_k1_DIS_s113
                                 + 0.893234 w_k1_FDX_s113
                                 + 0.633314 w_k1_IBM_s113
                                 + 0.886504 w_k1_INTC_s113
                                 + 0.714596 w_k1_KO_s113
                                 + 0.61108 w_k1_MAT_s113
                                 + 0.948577 w_k1_MSFT_s113
                                 + 0.804931 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.976908 w_k1_ADM_s114
                                 + 0.758724 w_k1_AXP_s114
                                 + 0.652313 w_k1_DIS_s114
                                 + 0.85339 w_k1_FDX_s114
                                 + 0.627581 w_k1_IBM_s114
                                 + 0.881528 w_k1_INTC_s114
                                 + 0.668956 w_k1_KO_s114
                                 + 0.564847 w_k1_MAT_s114
                                 + 0.828863 w_k1_MSFT_s114
                                 + 0.868164 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.22373 w_k1_ADM_s114
                                 + 1.53823 w_k1_AXP_s114
                                 + 1.01065 w_k1_DIS_s114
                                 + 1.08203 w_k1_FDX_s114
                                 + 1.43796 w_k1_IBM_s114
                                 + 1.03847 w_k1_INTC_s114
                                 + 1.04363 w_k1_KO_s114 + 1.57491 w_k1_MAT_s114
                                 + 0.873098 w_k1_MSFT_s114
                                 + 1.12573 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.751726 w_k1_ADM_s114
                                 + 0.698287 w_k1_AXP_s114
                                 + 0.708101 w_k1_DIS_s114
                                 + 0.729748 w_k1_FDX_s114
                                 + 0.941078 w_k1_IBM_s114
                                 + 0.86854 w_k1_INTC_s114
                                 + 0.840444 w_k1_KO_s114
                                 + 1.05991 w_k1_MAT_s114
                                 + 0.826312 w_k1_MSFT_s114
                                 + 0.891187 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.11198 w_k1_ADM_s114
                                 + 0.791186 w_k1_AXP_s114
                                 + 1.27198 w_k1_DIS_s114 + 1.1393 w_k1_FDX_s114
                                 + 1.00006 w_k1_IBM_s114
                                 + 1.0722 w_k1_INTC_s114
                                 + 0.999542 w_k1_KO_s114
                                 + 0.863042 w_k1_MAT_s114
                                 + 1.18859 w_k1_MSFT_s114
                                 + 1.01078 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.12251 w_k1_ADM_s114
                                 + 0.886812 w_k1_AXP_s114
                                 + 0.908355 w_k1_DIS_s114
                                 + 0.93483 w_k1_FDX_s114
                                 + 1.07414 w_k1_IBM_s114
                                 + 0.876642 w_k1_INTC_s114
                                 + 0.952054 w_k1_KO_s114
                                 + 1.06982 w_k1_MAT_s114
                                 + 0.914113 w_k1_MSFT_s114
                                 + 1.04912 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.21735 w_k1_ADM_s114
                                 + 1.28139 w_k1_AXP_s114
                                 + 1.47066 w_k1_DIS_s114
                                 + 1.36728 w_k1_FDX_s114
                                 + 1.17435 w_k1_IBM_s114
                                 + 0.957303 w_k1_INTC_s114
                                 + 1.06282 w_k1_KO_s114 + 1.77941 w_k1_MAT_s114
                                 + 1.06785 w_k1_MSFT_s114
                                 + 1.11569 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.05953 w_k1_ADM_s114
                                 + 0.9347 w_k1_AXP_s114
                                 + 0.797571 w_k1_DIS_s114
                                 + 1.06293 w_k1_FDX_s114
                                 + 0.620017 w_k1_IBM_s114
                                 + 1.05968 w_k1_INTC_s114
                                 + 0.654252 w_k1_KO_s114
                                 + 0.583637 w_k1_MAT_s114
                                 + 1.01948 w_k1_MSFT_s114
                                 + 0.870821 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.986637 w_k1_ADM_s114
                                 + 0.877979 w_k1_AXP_s114
                                 + 1.07049 w_k1_DIS_s114
                                 + 0.934907 w_k1_FDX_s114
                                 + 1.5515 w_k1_IBM_s114
                                 + 0.960298 w_k1_INTC_s114
                                 + 1.08554 w_k1_KO_s114 + 1.20083 w_k1_MAT_s114
                                 + 0.960747 w_k1_MSFT_s114
                                 + 1.09824 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.09152 w_k1_ADM_s115
                                 + 1.12484 w_k1_AXP_s115
                                 + 1.40912 w_k1_DIS_s115
                                 + 1.33839 w_k1_FDX_s115
                                 + 0.885869 w_k1_IBM_s115
                                 + 1.22284 w_k1_INTC_s115
                                 + 1.11982 w_k1_KO_s115 + 1.25014 w_k1_MAT_s115
                                 + 1.39035 w_k1_MSFT_s115
                                 + 1.04289 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.850696 w_k1_ADM_s115
                                 + 0.801948 w_k1_AXP_s115
                                 + 0.912345 w_k1_DIS_s115
                                 + 0.913197 w_k1_FDX_s115
                                 + 0.918471 w_k1_IBM_s115
                                 + 0.694042 w_k1_INTC_s115 + 1.008 w_k1_KO_s115
                                 + 0.977719 w_k1_MAT_s115
                                 + 0.977088 w_k1_MSFT_s115
                                 + 0.909688 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.94076 w_k1_ADM_s115
                                 + 1.57226 w_k1_AXP_s115
                                 + 1.67127 w_k1_DIS_s115
                                 + 1.25929 w_k1_FDX_s115
                                 + 1.67971 w_k1_IBM_s115
                                 + 1.38965 w_k1_INTC_s115
                                 + 1.57439 w_k1_KO_s115 + 2.24521 w_k1_MAT_s115
                                 + 1.29727 w_k1_MSFT_s115
                                 + 1.10378 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.80071 w_k1_ADM_s115
                                 + 0.543822 w_k1_AXP_s115
                                 + 0.722187 w_k1_DIS_s115
                                 + 0.775245 w_k1_FDX_s115
                                 + 0.716561 w_k1_IBM_s115
                                 + 0.732499 w_k1_INTC_s115
                                 + 0.792177 w_k1_KO_s115
                                 + 0.591016 w_k1_MAT_s115
                                 + 0.864654 w_k1_MSFT_s115
                                 + 0.798287 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.14994 w_k1_ADM_s115
                                 + 1.24118 w_k1_AXP_s115
                                 + 0.905634 w_k1_DIS_s115
                                 + 1.10992 w_k1_FDX_s115
                                 + 0.876422 w_k1_IBM_s115
                                 + 0.944532 w_k1_INTC_s115
                                 + 0.917494 w_k1_KO_s115
                                 + 0.930675 w_k1_MAT_s115
                                 + 1.00317 w_k1_MSFT_s115
                                 + 0.873826 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.930426 w_k1_ADM_s115
                                 + 0.932473 w_k1_AXP_s115
                                 + 1.2937 w_k1_DIS_s115 + 1.08595 w_k1_FDX_s115
                                 + 1.0275 w_k1_IBM_s115
                                 + 1.22708 w_k1_INTC_s115
                                 + 1.24995 w_k1_KO_s115 + 1.47464 w_k1_MAT_s115
                                 + 1.29706 w_k1_MSFT_s115
                                 + 1.01522 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.09446 w_k1_ADM_s115
                                 + 1.09804 w_k1_AXP_s115
                                 + 0.900652 w_k1_DIS_s115
                                 + 0.911769 w_k1_FDX_s115
                                 + 1.34776 w_k1_IBM_s115
                                 + 0.957399 w_k1_INTC_s115
                                 + 1.09573 w_k1_KO_s115 + 1.31564 w_k1_MAT_s115
                                 + 0.891781 w_k1_MSFT_s115
                                 + 0.896203 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.757758 w_k1_ADM_s115
                                 + 0.975109 w_k1_AXP_s115
                                 + 0.553709 w_k1_DIS_s115
                                 + 0.752634 w_k1_FDX_s115
                                 + 0.925502 w_k1_IBM_s115
                                 + 0.76437 w_k1_INTC_s115
                                 + 0.816757 w_k1_KO_s115
                                 + 1.60731 w_k1_MAT_s115
                                 + 0.744613 w_k1_MSFT_s115
                                 + 0.767736 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.13596 w_k1_ADM_s116
                                 + 1.00241 w_k1_AXP_s116
                                 + 1.25081 w_k1_DIS_s116
                                 + 1.23338 w_k1_FDX_s116
                                 + 0.645689 w_k1_IBM_s116
                                 + 1.2334 w_k1_INTC_s116
                                 + 0.973227 w_k1_KO_s116
                                 + 0.733191 w_k1_MAT_s116
                                 + 1.27878 w_k1_MSFT_s116
                                 + 0.917746 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.06924 w_k1_ADM_s116
                                 + 1.41091 w_k1_AXP_s116
                                 + 1.67565 w_k1_DIS_s116
                                 + 1.47133 w_k1_FDX_s116
                                 + 1.23472 w_k1_IBM_s116
                                 + 1.1856 w_k1_INTC_s116 + 1.24743 w_k1_KO_s116
                                 + 1.36969 w_k1_MAT_s116
                                 + 1.35607 w_k1_MSFT_s116
                                 + 1.09731 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.820452 w_k1_ADM_s116
                                 + 1.27144 w_k1_AXP_s116
                                 + 0.981701 w_k1_DIS_s116
                                 + 1.12824 w_k1_FDX_s116
                                 + 1.04199 w_k1_IBM_s116
                                 + 1.09653 w_k1_INTC_s116
                                 + 1.04549 w_k1_KO_s116 + 1.27799 w_k1_MAT_s116
                                 + 1.20907 w_k1_MSFT_s116
                                 + 0.983121 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.821074 w_k1_ADM_s116
                                 + 0.931319 w_k1_AXP_s116
                                 + 1.10319 w_k1_DIS_s116
                                 + 0.944093 w_k1_FDX_s116
                                 + 1.50508 w_k1_IBM_s116
                                 + 0.988857 w_k1_INTC_s116
                                 + 1.19599 w_k1_KO_s116
                                 + 0.989565 w_k1_MAT_s116
                                 + 0.992163 w_k1_MSFT_s116
                                 + 0.973156 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.884784 w_k1_ADM_s116
                                 + 0.729475 w_k1_AXP_s116
                                 + 0.837033 w_k1_DIS_s116
                                 + 0.956925 w_k1_FDX_s116
                                 + 0.790584 w_k1_IBM_s116
                                 + 0.899348 w_k1_INTC_s116
                                 + 0.842327 w_k1_KO_s116
                                 + 0.724273 w_k1_MAT_s116
                                 + 1.03691 w_k1_MSFT_s116
                                 + 0.931628 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.990624 w_k1_ADM_s116
                                 + 1.01036 w_k1_AXP_s116
                                 + 1.41965 w_k1_DIS_s116
                                 + 1.18406 w_k1_FDX_s116
                                 + 0.949499 w_k1_IBM_s116
                                 + 1.3692 w_k1_INTC_s116 + 1.21583 w_k1_KO_s116
                                 + 0.720882 w_k1_MAT_s116
                                 + 1.49859 w_k1_MSFT_s116
                                 + 1.06691 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.962275 w_k1_ADM_s116
                                 + 0.857469 w_k1_AXP_s116
                                 + 0.986748 w_k1_DIS_s116
                                 + 0.966028 w_k1_FDX_s116
                                 + 1.09195 w_k1_IBM_s116
                                 + 0.799635 w_k1_INTC_s116
                                 + 0.991757 w_k1_KO_s116
                                 + 0.816735 w_k1_MAT_s116
                                 + 1.05809 w_k1_MSFT_s116
                                 + 1.01004 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.770509 w_k1_ADM_s116
                                 + 0.648881 w_k1_AXP_s116
                                 + 0.670971 w_k1_DIS_s116
                                 + 0.780745 w_k1_FDX_s116
                                 + 0.80164 w_k1_IBM_s116
                                 + 0.716486 w_k1_INTC_s116
                                 + 0.817728 w_k1_KO_s116
                                 + 0.648068 w_k1_MAT_s116
                                 + 0.88314 w_k1_MSFT_s116
                                 + 0.858617 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.942601 w_k1_ADM_s117
                                 + 1.13022 w_k1_AXP_s117
                                 + 1.02239 w_k1_DIS_s117
                                 + 1.07123 w_k1_FDX_s117
                                 + 1.02123 w_k1_IBM_s117
                                 + 1.14658 w_k1_INTC_s117
                                 + 0.969821 w_k1_KO_s117
                                 + 1.19612 w_k1_MAT_s117
                                 + 1.03857 w_k1_MSFT_s117
                                 + 0.933615 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.0961 w_k1_ADM_s117
                                 + 0.750944 w_k1_AXP_s117
                                 + 1.0041 w_k1_DIS_s117 + 1.01632 w_k1_FDX_s117
                                 + 0.659424 w_k1_IBM_s117
                                 + 1.08962 w_k1_INTC_s117
                                 + 0.764306 w_k1_KO_s117
                                 + 0.56868 w_k1_MAT_s117
                                 + 1.06235 w_k1_MSFT_s117
                                 + 0.90328 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.07065 w_k1_ADM_s117
                                 + 0.820014 w_k1_AXP_s117
                                 + 1.07247 w_k1_DIS_s117
                                 + 1.01815 w_k1_FDX_s117
                                 + 0.938982 w_k1_IBM_s117
                                 + 1.03639 w_k1_INTC_s117
                                 + 0.926392 w_k1_KO_s117
                                 + 1.00606 w_k1_MAT_s117
                                 + 1.04921 w_k1_MSFT_s117
                                 + 1.03122 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.0797 w_k1_ADM_s117
                                 + 1.09429 w_k1_AXP_s117
                                 + 0.734037 w_k1_DIS_s117
                                 + 0.908411 w_k1_FDX_s117
                                 + 1.04793 w_k1_IBM_s117
                                 + 0.957683 w_k1_INTC_s117
                                 + 0.876511 w_k1_KO_s117 + 1.5407 w_k1_MAT_s117
                                 + 0.90855 w_k1_MSFT_s117
                                 + 0.982494 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.02799 w_k1_ADM_s117
                                 + 1.04083 w_k1_AXP_s117
                                 + 0.996322 w_k1_DIS_s117
                                 + 0.988596 w_k1_FDX_s117
                                 + 1.42625 w_k1_IBM_s117
                                 + 1.19474 w_k1_INTC_s117
                                 + 1.05243 w_k1_KO_s117 + 1.68978 w_k1_MAT_s117
                                 + 0.971927 w_k1_MSFT_s117
                                 + 1.22131 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.48354 w_k1_ADM_s117
                                 + 1.43598 w_k1_AXP_s117
                                 + 1.83247 w_k1_DIS_s117
                                 + 1.38198 w_k1_FDX_s117 + 1.4056 w_k1_IBM_s117
                                 + 1.30079 w_k1_INTC_s117
                                 + 1.25169 w_k1_KO_s117 + 1.63768 w_k1_MAT_s117
                                 + 1.35764 w_k1_MSFT_s117 + 1.2477 w_k1_PG_s117
                                 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.940981 w_k1_ADM_s117
                                 + 0.644537 w_k1_AXP_s117
                                 + 0.761733 w_k1_DIS_s117
                                 + 0.774732 w_k1_FDX_s117
                                 + 0.900747 w_k1_IBM_s117
                                 + 0.987502 w_k1_INTC_s117
                                 + 0.874322 w_k1_KO_s117
                                 + 0.762007 w_k1_MAT_s117
                                 + 0.813809 w_k1_MSFT_s117
                                 + 0.81136 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.38535 w_k1_ADM_s117
                                 + 1.20049 w_k1_AXP_s117 + 1.3743 w_k1_DIS_s117
                                 + 1.33299 w_k1_FDX_s117
                                 + 0.910469 w_k1_IBM_s117
                                 + 1.36567 w_k1_INTC_s117
                                 + 1.13633 w_k1_KO_s117
                                 + 0.827849 w_k1_MAT_s117
                                 + 1.3609 w_k1_MSFT_s117 + 1.17369 w_k1_PG_s117
                                 >= 1
 k1_return_constraint:           0.026339006012455 w_k1_ADM_s000
                                 + 0.0500194350712214 w_k1_ADM_s001
                                 + 0.0104206474420069 w_k1_ADM_s002
                                 + 0.0223994644257247 w_k1_ADM_s003
                                 + 0.0103738964546097 w_k1_ADM_s004
                                 + 0.0372907552882604 w_k1_ADM_s005
                                 + 0.0353932477099531 w_k1_ADM_s006
                                 + 0.017415993843467 w_k1_ADM_s007
                                 + 0.011717335998295 w_k1_ADM_s010
                                 + 0.0206966568634707 w_k1_ADM_s011
                                 + 0.0342265251374566 w_k1_ADM_s012
                                 + 0.0212240238367911 w_k1_ADM_s013
                                 + 0.0306652092765454 w_k1_ADM_s014
                                 + 0.00952443877516637 w_k1_ADM_s015
                                 + 0.0166316257720596 w_k1_ADM_s016
                                 + 0.0176828919320938 w_k1_ADM_s017
                                 + 0.0439269601894917 w_k1_ADM_s100
                                 + 0.0479768746812641 w_k1_ADM_s101
                                 + 0.0561751605564028 w_k1_ADM_s102
                                 + 0.0395689550881516 w_k1_ADM_s103
                                 + 0.0270224356939117 w_k1_ADM_s104
                                 + 0.0399159273664748 w_k1_ADM_s105
                                 + 0.0167507825560131 w_k1_ADM_s106
                                 + 0.0471417520682145 w_k1_ADM_s107
                                 + 0.035250507550248 w_k1_ADM_s110
                                 + 0.0271859408507616 w_k1_ADM_s111
                                 + 0.0223566369368285 w_k1_ADM_s112
                                 + 0.0296555987436884 w_k1_ADM_s113
                                 + 0.0215583615889922 w_k1_ADM_s114
                                 + 0.0448304600393652 w_k1_ADM_s115
                                 + 0.0632870480646069 w_k1_ADM_s116
                                 + 0.0738655845507629 w_k1_ADM_s117
                                 + 0.0234732763678233 w_k1_AXP_s000
                                 + 0.0451876910563212 w_k1_AXP_s001
                                 + 0.0126503982175931 w_k1_AXP_s002
                                 + 0.0204989197986281 w_k1_AXP_s003
                                 + 0.00913457510582318 w_k1_AXP_s004
                                 + 0.0342879020408739 w_k1_AXP_s005
                                 + 0.0306881657020646 w_k1_AXP_s006
                                 + 0.0189637086879209 w_k1_AXP_s007
                                 + 0.0118949680993296 w_k1_AXP_s010
                                 + 0.0174229065919935 w_k1_AXP_s011
                                 + 0.0329349295701592 w_k1_AXP_s012
                                 + 0.0179250954428298 w_k1_AXP_s013
                                 + 0.032499649736952 w_k1_AXP_s014
                                 + 0.00903823378587046 w_k1_AXP_s015
                                 + 0.0159798554703983 w_k1_AXP_s016
                                 + 0.0195170993150357 w_k1_AXP_s017
                                 + 0.0429257736667258 w_k1_AXP_s100
                                 + 0.0521263652583267 w_k1_AXP_s101
                                 + 0.0467580107623958 w_k1_AXP_s102
                                 + 0.0378013361828266 w_k1_AXP_s103
                                 + 0.0289335611955509 w_k1_AXP_s104
                                 + 0.0352974345086409 w_k1_AXP_s105
                                 + 0.0186276468962166 w_k1_AXP_s106
                                 + 0.0468006052721046 w_k1_AXP_s107
                                 + 0.0353073546352754 w_k1_AXP_s110
                                 + 0.0303389422109944 w_k1_AXP_s111
                                 + 0.0197934847425504 w_k1_AXP_s112
                                 + 0.0329075233466228 w_k1_AXP_s113
                                 + 0.0198913738302234 w_k1_AXP_s114
                                 + 0.0467911338507795 w_k1_AXP_s115
                                 + 0.0670760065326727 w_k1_AXP_s116
                                 + 0.0651975048609708 w_k1_AXP_s117
                                 + 0.0278261947143342 w_k1_DIS_s000
                                 + 0.0431214867172165 w_k1_DIS_s001
                                 + 0.0133387261053219 w_k1_DIS_s002
                                 + 0.0236739812356929 w_k1_DIS_s003
                                 + 0.00933501243240537 w_k1_DIS_s004
                                 + 0.0405650423655814 w_k1_DIS_s005
                                 + 0.0321744802993465 w_k1_DIS_s006
                                 + 0.0191586138932636 w_k1_DIS_s007
                                 + 0.012114540144341 w_k1_DIS_s010
                                 + 0.0179912488854094 w_k1_DIS_s011
                                 + 0.0366611070019096 w_k1_DIS_s012
                                 + 0.0193533409842457 w_k1_DIS_s013
                                 + 0.0311544846074543 w_k1_DIS_s014
                                 + 0.00960845666941824 w_k1_DIS_s015
                                 + 0.0156475139258755 w_k1_DIS_s016
                                 + 0.0201151338499774 w_k1_DIS_s017
                                 + 0.047303676033532 w_k1_DIS_s100
                                 + 0.0510864660269867 w_k1_DIS_s101
                                 + 0.0500371209622153 w_k1_DIS_s102
                                 + 0.041136149458138 w_k1_DIS_s103
                                 + 0.0299017616185053 w_k1_DIS_s104
                                 + 0.0375495857629619 w_k1_DIS_s105
                                 + 0.0178865095883284 w_k1_DIS_s106
                                 + 0.0479125900834811 w_k1_DIS_s107
                                 + 0.0389183765865621 w_k1_DIS_s110
                                 + 0.0324054966044246 w_k1_DIS_s111
                                 + 0.0229822528661606 w_k1_DIS_s112
                                 + 0.0307635172564383 w_k1_DIS_s113
                                 + 0.0203118417236608 w_k1_DIS_s114
                                 + 0.0473323930004643 w_k1_DIS_s115
                                 + 0.0730093538555524 w_k1_DIS_s116
                                 + 0.0716154312722625 w_k1_DIS_s117
                                 + 0.0256133348477273 w_k1_FDX_s000
                                 + 0.0462629770666281 w_k1_FDX_s001
                                 + 0.0119720301172256 w_k1_FDX_s002
                                 + 0.0217765866298968 w_k1_FDX_s003
                                 + 0.00974245857152657 w_k1_FDX_s004
                                 + 0.038069651400457 w_k1_FDX_s005
                                 + 0.0310778981006225 w_k1_FDX_s006
                                 + 0.0181316649386651 w_k1_FDX_s007
                                 + 0.012253164263413 w_k1_FDX_s010
                                 + 0.0176878521464614 w_k1_FDX_s011
                                 + 0.0368794937879663 w_k1_FDX_s012
                                 + 0.018617212782344 w_k1_FDX_s013
                                 + 0.0316013575142007 w_k1_FDX_s014
                                 + 0.00891980268834741 w_k1_FDX_s015
                                 + 0.016174967434469 w_k1_FDX_s016
                                 + 0.019058932280906 w_k1_FDX_s017
                                 + 0.0464164053149971 w_k1_FDX_s100
                                 + 0.0521320749875301 w_k1_FDX_s101
                                 + 0.0500693362440439 w_k1_FDX_s102
                                 + 0.0406491243289622 w_k1_FDX_s103
                                 + 0.0295989315885474 w_k1_FDX_s104
                                 + 0.0382789506945136 w_k1_FDX_s105
                                 + 0.0178936081127362 w_k1_FDX_s106
                                 + 0.0488160438274976 w_k1_FDX_s107
                                 + 0.0361191881094153 w_k1_FDX_s110
                                 + 0.0322585253718163 w_k1_FDX_s111
                                 + 0.0223299991431795 w_k1_FDX_s112
                                 + 0.0309137943592942 w_k1_FDX_s113
                                 + 0.020504442075541 w_k1_FDX_s114
                                 + 0.0465949510973067 w_k1_FDX_s115
                                 + 0.0731256583149768 w_k1_FDX_s116
                                 + 0.0693803763796175 w_k1_FDX_s117
                                 + 0.0256643866727673 w_k1_IBM_s000
                                 + 0.0465075396716828 w_k1_IBM_s001
                                 + 0.0118800171386646 w_k1_IBM_s002
                                 + 0.0217907908919461 w_k1_IBM_s003
                                 + 0.00922175351179622 w_k1_IBM_s004
                                 + 0.0388245814016067 w_k1_IBM_s005
                                 + 0.0314895857740173 w_k1_IBM_s006
                                 + 0.0179824195742734 w_k1_IBM_s007
                                 + 0.0119828187325806 w_k1_IBM_s010
                                 + 0.0178043017792194 w_k1_IBM_s011
                                 + 0.0356493333916505 w_k1_IBM_s012
                                 + 0.018061505186945 w_k1_IBM_s013
                                 + 0.0332650336327971 w_k1_IBM_s014
                                 + 0.00905080761492828 w_k1_IBM_s015
                                 + 0.0174107012230895 w_k1_IBM_s016
                                 + 0.0201077280929479 w_k1_IBM_s017
                                 + 0.0483003906572091 w_k1_IBM_s100
                                 + 0.0540360806024073 w_k1_IBM_s101
                                 + 0.0520003383054233 w_k1_IBM_s102
                                 + 0.0412112021712313 w_k1_IBM_s103
                                 + 0.0301083833765218 w_k1_IBM_s104
                                 + 0.0386612088937622 w_k1_IBM_s105
                                 + 0.0176260151663149 w_k1_IBM_s106
                                 + 0.0489087491921334 w_k1_IBM_s107
                                 + 0.0372264681386652 w_k1_IBM_s110
                                 + 0.0320926809603373 w_k1_IBM_s111
                                 + 0.0228436604621553 w_k1_IBM_s112
                                 + 0.0312632918458685 w_k1_IBM_s113
                                 + 0.0216859666905853 w_k1_IBM_s114
                                 + 0.0477658298757064 w_k1_IBM_s115
                                 + 0.0718541365915095 w_k1_IBM_s116
                                 + 0.0679587567910269 w_k1_IBM_s117
                                 + 0.0272485028437196 w_k1_INTC_s000
                                 + 0.0436842902086647 w_k1_INTC_s001
                                 + 0.0148461350626825 w_k1_INTC_s002
                                 + 0.0214387514505942 w_k1_INTC_s003
                                 + 0.0109352133495526 w_k1_INTC_s004
                                 + 0.0342529517640987 w_k1_INTC_s005
                                 + 0.0275662710270046 w_k1_INTC_s006
                                 + 0.0204740436717001 w_k1_INTC_s007
                                 + 0.0118963615536857 w_k1_INTC_s010
                                 + 0.0164242530383731 w_k1_INTC_s011
                                 + 0.039260042632614 w_k1_INTC_s012
                                 + 0.0185921811696383 w_k1_INTC_s013
                                 + 0.035194470424996 w_k1_INTC_s014
                                 + 0.00804346737730906 w_k1_INTC_s015
                                 + 0.0167681468157827 w_k1_INTC_s016
                                 + 0.0211303720559593 w_k1_INTC_s017
                                 + 0.0479628015199207 w_k1_INTC_s100
                                 + 0.047149395614775 w_k1_INTC_s101
                                 + 0.0520840686927047 w_k1_INTC_s102
                                 + 0.0425124759878967 w_k1_INTC_s103
                                 + 0.0276171164794096 w_k1_INTC_s104
                                 + 0.0347954216617194 w_k1_INTC_s105
                                 + 0.0205966756847462 w_k1_INTC_s106
                                 + 0.0491014490631749 w_k1_INTC_s107
                                 + 0.037221580474203 w_k1_INTC_s110
                                 + 0.0304852029640906 w_k1_INTC_s111
                                 + 0.0230034144758426 w_k1_INTC_s112
                                 + 0.0279562602510953 w_k1_INTC_s113
                                 + 0.0191645857624562 w_k1_INTC_s114
                                 + 0.0446764123249084 w_k1_INTC_s115
                                 + 0.0693863835682786 w_k1_INTC_s116
                                 + 0.0755524524395334 w_k1_INTC_s117
                                 + 0.0282422610942147 w_k1_KO_s000
                                 + 0.0449900058152485 w_k1_KO_s001
                                 + 0.0129305455369591 w_k1_KO_s002
                                 + 0.0237759565092354 w_k1_KO_s003
                                 + 0.0106388277342542 w_k1_KO_s004
                                 + 0.0377385640993466 w_k1_KO_s005
                                 + 0.032545386894464 w_k1_KO_s006
                                 + 0.0184640523090461 w_k1_KO_s007
                                 + 0.0123198924002553 w_k1_KO_s010
                                 + 0.0170399317193989 w_k1_KO_s011
                                 + 0.0365874019048994 w_k1_KO_s012
                                 + 0.01790550198014 w_k1_KO_s013
                                 + 0.0313206807037532 w_k1_KO_s014
                                 + 0.00916306256555268 w_k1_KO_s015
                                 + 0.0168713508038303 w_k1_KO_s016
                                 + 0.0197858607283651 w_k1_KO_s017
                                 + 0.048481476875747 w_k1_KO_s100
                                 + 0.0548230419104655 w_k1_KO_s101
                                 + 0.0492445272396096 w_k1_KO_s102
                                 + 0.0441736057574147 w_k1_KO_s103
                                 + 0.0298858476361528 w_k1_KO_s104
                                 + 0.0379921755905079 w_k1_KO_s105
                                 + 0.0173658507530902 w_k1_KO_s106
                                 + 0.0479331883494103 w_k1_KO_s107
                                 + 0.0386242514273108 w_k1_KO_s110
                                 + 0.0298634901020188 w_k1_KO_s111
                                 + 0.0232312394321771 w_k1_KO_s112
                                 + 0.0268861571832351 w_k1_KO_s113
                                 + 0.0189792014272303 w_k1_KO_s114
                                 + 0.0489344287460409 w_k1_KO_s115
                                 + 0.0715563399219636 w_k1_KO_s116
                                 + 0.0651160762542398 w_k1_KO_s117
                                 + 0.0305055817322991 w_k1_MAT_s000
                                 + 0.0419210111779665 w_k1_MAT_s001
                                 + 0.0140498031400874 w_k1_MAT_s002
                                 + 0.0298097636019757 w_k1_MAT_s003
                                 + 0.00684989755177976 w_k1_MAT_s004
                                 + 0.0440589800349227 w_k1_MAT_s005
                                 + 0.0309433135935021 w_k1_MAT_s006
                                 + 0.0150043515856565 w_k1_MAT_s007
                                 + 0.0139157899408805 w_k1_MAT_s010
                                 + 0.0180301931211085 w_k1_MAT_s011
                                 + 0.0276772387169866 w_k1_MAT_s012
                                 + 0.0157629636858687 w_k1_MAT_s013
                                 + 0.0313443929188492 w_k1_MAT_s014
                                 + 0.0094515258895167 w_k1_MAT_s015
                                 + 0.0154341586831693 w_k1_MAT_s016
                                 + 0.0206693450425025 w_k1_MAT_s017
                                 + 0.0518041518844963 w_k1_MAT_s100
                                 + 0.0454693882743126 w_k1_MAT_s101
                                 + 0.0552065699172922 w_k1_MAT_s102
                                 + 0.0443855767424622 w_k1_MAT_s103
                                 + 0.0297917441525799 w_k1_MAT_s104
                                 + 0.0469140633444853 w_k1_MAT_s105
                                 + 0.0175757234249816 w_k1_MAT_s106
                                 + 0.0489709709180256 w_k1_MAT_s107
                                 + 0.0380285204281692 w_k1_MAT_s110
                                 + 0.0353535001459541 w_k1_MAT_s111
                                 + 0.0243732968180545 w_k1_MAT_s112
                                 + 0.032331086730523 w_k1_MAT_s113
                                 + 0.0235642316426571 w_k1_MAT_s114
                                 + 0.0560917547372898 w_k1_MAT_s115
                                 + 0.0644292713219813 w_k1_MAT_s116
                                 + 0.0724436424639075 w_k1_MAT_s117
                                 + 0.027988499550567 w_k1_MSFT_s000
                                 + 0.0458010429875025 w_k1_MSFT_s001
                                 + 0.0115890991860908 w_k1_MSFT_s002
                                 + 0.0227999315304234 w_k1_MSFT_s003
                                 + 0.0104975115479967 w_k1_MSFT_s004
                                 + 0.0376371673429859 w_k1_MSFT_s005
                                 + 0.033074812091353 w_k1_MSFT_s006
                                 + 0.0175558663066622 w_k1_MSFT_s007
                                 + 0.0117646438552792 w_k1_MSFT_s010
                                 + 0.0170842543424522 w_k1_MSFT_s011
                                 + 0.034987368101502 w_k1_MSFT_s012
                                 + 0.0168012293854693 w_k1_MSFT_s013
                                 + 0.0290968393151713 w_k1_MSFT_s014
                                 + 0.00836998493043327 w_k1_MSFT_s015
                                 + 0.0165834339071788 w_k1_MSFT_s016
                                 + 0.0191020425972993 w_k1_MSFT_s017
                                 + 0.0508808972104724 w_k1_MSFT_s100
                                 + 0.0524195310139769 w_k1_MSFT_s101
                                 + 0.0505386477642561 w_k1_MSFT_s102
                                 + 0.0461879054203741 w_k1_MSFT_s103
                                 + 0.0291306311734748 w_k1_MSFT_s104
                                 + 0.0388840751702522 w_k1_MSFT_s105
                                 + 0.0175010227216143 w_k1_MSFT_s106
                                 + 0.050595185334234 w_k1_MSFT_s107
                                 + 0.039794435722079 w_k1_MSFT_s110
                                 + 0.0356660291672566 w_k1_MSFT_s111
                                 + 0.0217422343113806 w_k1_MSFT_s112
                                 + 0.0272680044254768 w_k1_MSFT_s113
                                 + 0.019281593367271 w_k1_MSFT_s114
                                 + 0.048338890694315 w_k1_MSFT_s115
                                 + 0.0783451657833262 w_k1_MSFT_s116
                                 + 0.0701648602644558 w_k1_MSFT_s117
                                 + 0.0274227856387 w_k1_PG_s000
                                 + 0.0450381168508454 w_k1_PG_s001
                                 + 0.0126535583314513 w_k1_PG_s002
                                 + 0.022077349384516 w_k1_PG_s003
                                 + 0.00973476513897637 w_k1_PG_s004
                                 + 0.0369574700968075 w_k1_PG_s005
                                 + 0.0298964250781873 w_k1_PG_s006
                                 + 0.0181623212538819 w_k1_PG_s007
                                 + 0.0118531721408596 w_k1_PG_s010
                                 + 0.0182398724333065 w_k1_PG_s011
                                 + 0.0357752986587382 w_k1_PG_s012
                                 + 0.0179081393771427 w_k1_PG_s013
                                 + 0.0317622767436286 w_k1_PG_s014
                                 + 0.00899056682310647 w_k1_PG_s015
                                 + 0.0172863017703213 w_k1_PG_s016
                                 + 0.018406176976749 w_k1_PG_s017
                                 + 0.048068247955347 w_k1_PG_s100
                                 + 0.0530098844050277 w_k1_PG_s101
                                 + 0.0504039866975685 w_k1_PG_s102
                                 + 0.0410224617952722 w_k1_PG_s103
                                 + 0.0291084294550841 w_k1_PG_s104
                                 + 0.0352215810628895 w_k1_PG_s105
                                 + 0.0170202435602461 w_k1_PG_s106
                                 + 0.0446182545643778 w_k1_PG_s107
                                 + 0.0367724532808469 w_k1_PG_s110
                                 + 0.031879390984389 w_k1_PG_s111
                                 + 0.0216108672971911 w_k1_PG_s112
                                 + 0.0303723091872853 w_k1_PG_s113
                                 + 0.0205963374082828 w_k1_PG_s114
                                 + 0.0424963997402759 w_k1_PG_s115
                                 + 0.0681657905680194 w_k1_PG_s116
                                 + 0.0678357921780264 w_k1_PG_s117 >= 1.229
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
