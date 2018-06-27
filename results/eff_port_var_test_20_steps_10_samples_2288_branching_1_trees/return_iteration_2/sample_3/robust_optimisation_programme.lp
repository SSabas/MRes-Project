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
 w_k1_ADM_s0_return_constraint:  1.22227 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.96706 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.12996 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.901325 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.894804 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.13368 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.09191 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.765058 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.886213 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.937066 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.36696 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.05344 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.20658 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.781099 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.15414 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.40649 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.855481 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.93597 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.967093 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.87909 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.0819 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.723589 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.1828 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.15858 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.86341 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.024 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.09984 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.892946 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.775933 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.45544 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.1032 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.02403 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.10809 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.36326 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.06253 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.16485 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.826375 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.717813 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.32928 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.906589 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.21064 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.797756 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.945676 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.23826 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.20579 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.768784 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.753935 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.922604 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.70578 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.06533 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.37027 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.11215 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.37706 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.37376 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.10711 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.779225 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.955962 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.763718 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.14064 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.684702 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.05953 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.16449 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.614532 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.02274 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.42314 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.811501 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.573559 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.68571 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.05759 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.03178 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.08314 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.56922 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.867903 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.18541 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.772737 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.61023 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.40389 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.922267 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.20844 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.761111 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.871366 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.21138 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.13403 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.666657 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.712667 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.930206 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.99691 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.34777 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.25811 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.93865 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.36883 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.66463 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.968666 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.824119 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.09753 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.7325 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.2645 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.621776 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.02834 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.38771 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.529167 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.844315 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.6193 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.87969 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.587189 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.83646 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.10716 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.970139 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.04011 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.87843 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.912856 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.43445 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.848718 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.596914 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.28019 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.936388 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.09829 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.83194 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.862141 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.21066 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.06074 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.773134 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.765046 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.01335 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.42973 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.10302 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.33268 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.888631 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.32834 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.39488 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.93976 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.846192 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.13666 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.873703 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.1035 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.685461 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.0606 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.30993 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.800784 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.08683 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.19728 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.966881 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.699109 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.45773 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.934253 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.95926 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.11969 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.44151 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.998717 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.19564 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.86295 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.726857 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.1497 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.988729 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.20444 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.754145 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.07845 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.961999 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.47126 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.666478 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.11053 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.726158 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.55639 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.04948 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.873456 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.50015 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.01807 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.56797 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.34157 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.22139 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.842857 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.722695 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.38105 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.43512 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.20526 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.950569 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.446912 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.737505 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 2.09699 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.68647 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.784645 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.38256 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.55807 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.09179 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.799378 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.55935 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.623927 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.13624 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.855891 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.638807 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.16653 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.997876 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.999343 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.970436 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.856929 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.08899 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.1264 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.932201 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.856485 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.953618 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.57743 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.12974 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.40642 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.10456 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.05617 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.07078 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.854634 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.807692 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.912187 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.873682 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.93294 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.697049 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.13929 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.35947 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.749395 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.05342 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.06164 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.00704 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.769571 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.43217 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.924617 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.02282 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.05447 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.20911 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.00025 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.12807 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.854787 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.677818 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.21388 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.01113 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.08363 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.815259 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.935943 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.966243 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.38329 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.755322 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.870309 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.913274 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.65044 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.10678 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.13979 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.15237 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.05942 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.3452 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.1799 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.978866 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.858355 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.694784 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.07124 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.828242 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.10757 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.0378 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.563055 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.782055 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.53222 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.860523 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.670085 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.45503 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.26738 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.12241 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.03683 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.73818 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.874753 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.22008 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.935797 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.625681 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.25874 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.855319 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.35076 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.712506 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.30661 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.23686 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.45948 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.651806 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.990485 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.600589 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.15118 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.02366 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.906918 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.37175 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.965733 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.50953 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.51549 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.928099 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.879293 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.695273 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.20443 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.19587 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.09876 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.786438 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.345561 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.597979 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 2.24385 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.640106 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.575952 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.33197 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.37464 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.01221 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.869217 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.60134 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.513141 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.11694 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.847682 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.548663 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.19096 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.927318 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.07465 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.888194 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.932903 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.19341 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.14011 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.776811 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.870416 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.06746 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.39353 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.19857 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.38587 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.946659 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.22893 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.22005 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.93596 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.812409 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.03263 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.893574 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.03639 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.676908 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.03588 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.25822 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.747521 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.971858 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.28931 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.00209 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.76831 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.52299 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.963338 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.939363 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.10046 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.23949 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.03557 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.25824 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.891947 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.777592 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.16097 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.976351 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.12797 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.919962 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.929738 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.05367 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.02366 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.796637 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.887901 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.936972 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.14691 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.04892 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.09676 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.951108 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.10016 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.19828 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.02603 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.846547 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.937339 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.827489 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.01376 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.849305 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.05299 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.07194 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.784108 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.00918 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.16117 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.967341 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.837531 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.36883 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.982223 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.955832 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.02591 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.25293 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.945655 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.16648 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.86848 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.798426 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.0739999528892791 z_k1_s0000
                                 - 0.108000297081437 z_k1_s0001
                                 - 0.133999741934741 z_k1_s0002
                                 - 0.120999549508363 z_k1_s0003
                                 - 0.22000029183274 z_k1_s0004
                                 - 0.132999582517412 z_k1_s0005
                                 - 0.0879999287332056 z_k1_s0006
                                 - 0.0619999198802305 z_k1_s0007
                                 - 0.637999748899126 z_k1_s0010
                                 - 0.277000543429016 z_k1_s0011
                                 - 0.239000475495082 z_k1_s0012
                                 - 0.284001086433469 z_k1_s0013
                                 - 0.182000242693975 z_k1_s0014
                                 - 0.372000058762822 z_k1_s0015
                                 - 0.130000135952781 z_k1_s0016
                                 - 0.496001823686507 z_k1_s0017
                                 - 0.503000208202857 z_k1_s0020
                                 - 0.537999720575251 z_k1_s0021
                                 - 0.373998397404805 z_k1_s0022
                                 - 0.600998195246107 z_k1_s0023
                                 - 0.198999864143661 z_k1_s0024
                                 - 0.156999801697337 z_k1_s0025
                                 - 0.170999930446021 z_k1_s0026
                                 - 0.695000493671766 z_k1_s0027
                                 - 0.194999399173636 z_k1_s0030
                                 - 0.470999890618881 z_k1_s0031
                                 - 0.297999238130574 z_k1_s0032
                                 - 0.392999889510073 z_k1_s0033
                                 - 0.582997570678771 z_k1_s0034
                                 - 0.672997107651716 z_k1_s0035
                                 - 0.193999538451355 z_k1_s0036
                                 - 0.210999585123663 z_k1_s0037
                                 - 0.230000183981216 z_k1_s0040
                                 - 0.241999447284876 z_k1_s0041
                                 - 0.140999389814855 z_k1_s0042
                                 - 0.126999891694058 z_k1_s0043
                                 - 0.0939999061097807 z_k1_s0044
                                 - 0.129999975719868 z_k1_s0045
                                 - 0.268000086108598 z_k1_s0046
                                 - 0.108999923938988 z_k1_s0047
                                 - 0.106000109737446 z_k1_s0050
                                 - 0.0899999989279273 z_k1_s0051
                                 - 0.166000109022731 z_k1_s0052
                                 - 0.320000366188552 z_k1_s0053
                                 - 0.129000072463436 z_k1_s0054
                                 - 0.427000771914378 z_k1_s0055
                                 - 0.270999367771249 z_k1_s0056
                                 - 0.34099973693778 z_k1_s0057
                                 - 0.268000110681458 z_k1_s0060
                                 - 0.0789999356949136 z_k1_s0061
                                 - 0.315000555644223 z_k1_s0062
                                 - 0.1260000638577 z_k1_s0063
                                 - 0.130999959534599 z_k1_s0064
                                 - 0.239999744490881 z_k1_s0065
                                 - 0.25699923139235 z_k1_s0066
                                 - 0.168000613143565 z_k1_s0067
                                 - 0.181000254603817 z_k1_s0070
                                 - 0.1430000002738 z_k1_s0071
                                 - 0.166000037526217 z_k1_s0072
                                 - 0.259999328679857 z_k1_s0073
                                 - 0.131999988883511 z_k1_s0074
                                 - 0.162999698510794 z_k1_s0075
                                 - 0.17900052126004 z_k1_s0076
                                 - 0.254000128648526 z_k1_s0077
                                 - 0.162000995372909 z_k1_s0100
                                 - 0.14800064466109 z_k1_s0101
                                 - 0.137000560530794 z_k1_s0102
                                 - 0.127000252593654 z_k1_s0103
                                 - 0.0850001384602525 z_k1_s0104
                                 - 0.117000212657102 z_k1_s0105
                                 - 0.142999753690903 z_k1_s0106
                                 - 0.421000783526877 z_k1_s0107
                                 - 0.0650000588196174 z_k1_s0110
                                 - 0.118000393580517 z_k1_s0111
                                 - 0.0700000633442033 z_k1_s0112
                                 - 0.0500000452458595 z_k1_s0113
                                 - 0.0390000352917704 z_k1_s0114
                                 - 0.0670000606294517 z_k1_s0115
                                 - 0.178999946879961 z_k1_s0116
                                 - 0.128999901634101 z_k1_s0117
                                 - 0.110999817533107 z_k1_s0120
                                 - 0.073000079990381 z_k1_s0121
                                 - 0.0920003042619922 z_k1_s0122
                                 - 0.0939997956586321 z_k1_s0123
                                 - 0.102000131246847 z_k1_s0124
                                 - 0.0470000558294264 z_k1_s0125
                                 - 0.0980003584530156 z_k1_s0126
                                 - 0.172999850839981 z_k1_s0127
                                 - 0.0580000988241813 z_k1_s0130
                                 - 0.185000800250698 z_k1_s0131
                                 - 0.164000212275111 z_k1_s0132
                                 - 0.245001089034778 z_k1_s0133
                                 - 0.0500000963863898 z_k1_s0134
                                 - 0.196000438426792 z_k1_s0135
                                 - 0.102000274532421 z_k1_s0136
                                 - 0.0820002143378132 z_k1_s0137
                                 - 0.328000590902858 z_k1_s0140
                                 - 0.120000419713916 z_k1_s0141
                                 - 0.305001499620138 z_k1_s0142
                                 - 0.161000451001632 z_k1_s0143
                                 - 0.124000439381065 z_k1_s0144
                                 - 0.235001155445024 z_k1_s0145
                                 - 0.15600059671826 z_k1_s0146
                                 - 0.27400168780086 z_k1_s0147
                                 - 0.151000354802133 z_k1_s0150
                                 - 0.173000227414037 z_k1_s0151
                                 - 0.164000334563813 z_k1_s0152
                                 - 0.1680002196749 z_k1_s0153
                                 - 0.186000408976086 z_k1_s0154
                                 - 0.529001504921858 z_k1_s0155
                                 - 0.346000858428811 z_k1_s0156
                                 - 0.301001394177692 z_k1_s0157
                                 - 0.0660000340088525 z_k1_s0160
                                 - 0.144000202783075 z_k1_s0161
                                 - 0.0580000914987479 z_k1_s0162
                                 - 0.0890003217758397 z_k1_s0163
                                 - 0.175000344660081 z_k1_s0164
                                 - 0.127999787362355 z_k1_s0165
                                 - 0.106000542160142 z_k1_s0166
                                 - 0.11799996972483 z_k1_s0167
                                 - 0.106000165190395 z_k1_s0170
                                 - 0.0300000337614224 z_k1_s0171
                                 - 0.0370000186890698 z_k1_s0172
                                 - 0.0619999320734995 z_k1_s0173
                                 - 0.0320000360121839 z_k1_s0174
                                 - 0.0830001163566197 z_k1_s0175
                                 - 0.0270000074352623 z_k1_s0176
                                 - 0.0819999545811144 z_k1_s0177
                                 - 1.44699943133133 z_k1_s1000
                                 - 1.19400228521845 z_k1_s1001
                                 - 2.13700660157261 z_k1_s1002
                                 - 1.58300674933921 z_k1_s1003
                                 - 1.00900120629716 z_k1_s1004
                                 - 2.84300718980271 z_k1_s1005
                                 - 1.53400160070729 z_k1_s1006
                                 - 0.906001823026201 z_k1_s1007
                                 - 0.543000653768987 z_k1_s1010
                                 - 0.255000266385398 z_k1_s1011
                                 - 0.520000792303789 z_k1_s1012
                                 - 0.291000398383425 z_k1_s1013
                                 - 0.428001346442998 z_k1_s1014
                                 - 0.406001042910662 z_k1_s1015
                                 - 0.647001480830369 z_k1_s1016
                                 - 0.253000147618737 z_k1_s1017
                                 - 0.24799987471647 z_k1_s1020
                                 - 0.221000082963023 z_k1_s1021
                                 - 0.490999170497248 z_k1_s1022
                                 - 0.373999195399162 z_k1_s1023
                                 - 0.358999798591588 z_k1_s1024
                                 - 0.466999550605253 z_k1_s1025
                                 - 0.308999859233421 z_k1_s1026
                                 - 0.455998627946656 z_k1_s1027
                                 - 0.410002208396702 z_k1_s1030
                                 - 0.281001402766909 z_k1_s1031
                                 - 1.11200498836429 z_k1_s1032
                                 - 0.234001207057348 z_k1_s1033
                                 - 0.296001760543864 z_k1_s1034
                                 - 0.887003827732228 z_k1_s1035
                                 - 0.444000327503555 z_k1_s1036
                                 - 0.542004852111233 z_k1_s1037
                                 - 0.324000355259741 z_k1_s1040
                                 - 0.226000060025463 z_k1_s1041
                                 - 0.20100010329437 z_k1_s1042
                                 - 0.680001420767928 z_k1_s1043
                                 - 0.903002183258249 z_k1_s1044
                                 - 0.353001084372467 z_k1_s1045
                                 - 0.532999952515644 z_k1_s1046
                                 - 0.19800012215865 z_k1_s1047
                                 - 1.3959981169252 z_k1_s1050
                                 - 1.84699588566963 z_k1_s1051
                                 - 0.607999447740254 z_k1_s1052
                                 - 0.858001247655376 z_k1_s1053
                                 - 0.375999344208061 z_k1_s1054
                                 - 0.374999629075347 z_k1_s1055
                                 - 0.678999694145178 z_k1_s1056
                                 - 0.684999349740277 z_k1_s1057
                                 - 0.33100039524199 z_k1_s1060
                                 - 0.697003553564198 z_k1_s1061
                                 - 0.3380007290976 z_k1_s1062
                                 - 0.550002011607196 z_k1_s1063
                                 - 1.07499970572388 z_k1_s1064
                                 - 0.658999397343386 z_k1_s1065
                                 - 0.985001663592686 z_k1_s1066
                                 - 0.834002274722831 z_k1_s1067
                                 - 0.407000380219728 z_k1_s1070
                                 - 0.7119986794042 z_k1_s1071
                                 - 0.254000024366628 z_k1_s1072
                                 - 0.82199977881486 z_k1_s1073
                                 - 0.229999896240658 z_k1_s1074
                                 - 0.205000121292743 z_k1_s1075
                                 - 0.205000121292743 z_k1_s1076
                                 - 0.237999836515989 z_k1_s1077
                                 - 0.510003226029412 z_k1_s1100
                                 - 0.322000863555289 z_k1_s1101
                                 - 0.287000818592862 z_k1_s1102
                                 - 0.53899934271851 z_k1_s1103
                                 - 0.353000839107588 z_k1_s1104
                                 - 1.07900196457149 z_k1_s1105
                                 - 1.05300218825709 z_k1_s1106
                                 - 0.356000778689945 z_k1_s1107
                                 - 0.522999638414861 z_k1_s1110
                                 - 0.36399853882577 z_k1_s1111
                                 - 0.957996770424955 z_k1_s1112
                                 - 0.518999314760851 z_k1_s1113
                                 - 0.65899680070084 z_k1_s1114
                                 - 0.802999224278292 z_k1_s1115
                                 - 0.332999029596438 z_k1_s1116
                                 - 0.454998288881799 z_k1_s1117
                                 - 0.830000468810908 z_k1_s1120
                                 - 0.408001282292457 z_k1_s1121
                                 - 0.185999930827527 z_k1_s1122
                                 - 0.204000105546297 z_k1_s1123
                                 - 0.226999837831454 z_k1_s1124
                                 - 0.229000035777961 z_k1_s1125
                                 - 0.214000024078969 z_k1_s1126
                                 - 0.380000117841151 z_k1_s1127
                                 - 0.817998778996234 z_k1_s1130
                                 - 0.394999855708094 z_k1_s1131
                                 - 0.234999847726388 z_k1_s1132
                                 - 0.207999732546441 z_k1_s1133
                                 - 0.323998912561339 z_k1_s1134
                                 - 0.326999474158672 z_k1_s1135
                                 - 0.427999859283694 z_k1_s1136
                                 - 0.352001067138914 z_k1_s1137
                                 - 0.191999913917268 z_k1_s1140
                                 - 0.745999434474506 z_k1_s1141
                                 - 0.18999998277226 z_k1_s1142
                                 - 0.201999895842133 z_k1_s1143
                                 - 0.316999687978079 z_k1_s1144
                                 - 0.204999955659558 z_k1_s1145
                                 - 0.597999310546713 z_k1_s1146
                                 - 0.811999445658546 z_k1_s1147
                                 - 0.211999710397262 z_k1_s1150
                                 - 0.75900051865073 z_k1_s1151
                                 - 0.681996969083482 z_k1_s1152
                                 - 0.655996675799478 z_k1_s1153
                                 - 0.564998330300851 z_k1_s1154
                                 - 1.05399580331 z_k1_s1155
                                 - 0.876996487718126 z_k1_s1156
                                 - 0.556998240059619 z_k1_s1157
                                 - 0.725000021076424 z_k1_s1160
                                 - 0.212000456541753 z_k1_s1161
                                 - 0.412002329779882 z_k1_s1162
                                 - 0.556999148389617 z_k1_s1163
                                 - 0.363000548810237 z_k1_s1164
                                 - 0.724000537335149 z_k1_s1165
                                 - 0.43299945236053 z_k1_s1166
                                 - 0.555999664648342 z_k1_s1167
                                 - 1.63100203748804 z_k1_s1170
                                 - 0.680000519697389 z_k1_s1171
                                 - 0.380000257801456 z_k1_s1172
                                 - 0.653001893467854 z_k1_s1173
                                 - 0.403000381386893 z_k1_s1174
                                 - 0.532000471822126 z_k1_s1175
                                 - 0.80200170562925 z_k1_s1176
                                 - 0.464000419852387 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.28153 w_k1_ADM_s000
                                 + 1.33453 w_k1_AXP_s000
                                 + 1.56488 w_k1_DIS_s000
                                 + 1.23558 w_k1_FDX_s000 + 1.424 w_k1_IBM_s000
                                 + 1.29846 w_k1_INTC_s000
                                 + 1.18762 w_k1_KO_s000 + 1.06366 w_k1_MAT_s000
                                 + 1.14798 w_k1_MSFT_s000
                                 + 1.14568 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.980871 w_k1_ADM_s000
                                 + 0.936651 w_k1_AXP_s000
                                 + 0.694937 w_k1_DIS_s000
                                 + 0.841227 w_k1_FDX_s000
                                 + 0.841411 w_k1_IBM_s000
                                 + 0.940535 w_k1_INTC_s000
                                 + 0.77068 w_k1_KO_s000
                                 + 0.851623 w_k1_MAT_s000
                                 + 0.795081 w_k1_MSFT_s000
                                 + 1.02194 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.880121 w_k1_ADM_s000
                                 + 0.789736 w_k1_AXP_s000
                                 + 0.701192 w_k1_DIS_s000
                                 + 0.835587 w_k1_FDX_s000
                                 + 0.68633 w_k1_IBM_s000
                                 + 0.798275 w_k1_INTC_s000
                                 + 0.760421 w_k1_KO_s000
                                 + 0.844927 w_k1_MAT_s000
                                 + 0.827581 w_k1_MSFT_s000
                                 + 0.967079 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.16233 w_k1_ADM_s000
                                 + 1.43452 w_k1_AXP_s000
                                 + 1.52174 w_k1_DIS_s000
                                 + 1.28317 w_k1_FDX_s000
                                 + 1.15059 w_k1_IBM_s000
                                 + 1.34678 w_k1_INTC_s000
                                 + 1.27753 w_k1_KO_s000 + 1.29871 w_k1_MAT_s000
                                 + 1.38096 w_k1_MSFT_s000 + 1.1867 w_k1_PG_s000
                                 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.03624 w_k1_ADM_s000
                                 + 0.946758 w_k1_AXP_s000
                                 + 1.04454 w_k1_DIS_s000 + 1.0136 w_k1_FDX_s000
                                 + 1.12759 w_k1_IBM_s000
                                 + 0.848508 w_k1_INTC_s000
                                 + 1.01161 w_k1_KO_s000 + 1.14207 w_k1_MAT_s000
                                 + 0.888862 w_k1_MSFT_s000
                                 + 0.977754 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.904509 w_k1_ADM_s000
                                 + 1.34423 w_k1_AXP_s000
                                 + 1.08054 w_k1_DIS_s000
                                 + 1.15666 w_k1_FDX_s000
                                 + 0.978389 w_k1_IBM_s000
                                 + 1.24956 w_k1_INTC_s000
                                 + 0.951233 w_k1_KO_s000
                                 + 0.973433 w_k1_MAT_s000
                                 + 1.12543 w_k1_MSFT_s000
                                 + 1.13693 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.0188 w_k1_ADM_s000
                                 + 0.96217 w_k1_AXP_s000 + 1.044 w_k1_DIS_s000
                                 + 1.19892 w_k1_FDX_s000
                                 + 0.70421 w_k1_IBM_s000
                                 + 0.948196 w_k1_INTC_s000
                                 + 0.82067 w_k1_KO_s000
                                 + 0.718421 w_k1_MAT_s000
                                 + 0.927495 w_k1_MSFT_s000
                                 + 0.93517 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.17634 w_k1_ADM_s000
                                 + 1.52795 w_k1_AXP_s000
                                 + 1.42435 w_k1_DIS_s000
                                 + 1.05402 w_k1_FDX_s000
                                 + 1.91629 w_k1_IBM_s000
                                 + 1.06729 w_k1_INTC_s000 + 1.2832 w_k1_KO_s000
                                 + 1.49327 w_k1_MAT_s000
                                 + 0.946606 w_k1_MSFT_s000
                                 + 1.29033 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.893049 w_k1_ADM_s001
                                 + 0.689195 w_k1_AXP_s001
                                 + 0.769136 w_k1_DIS_s001
                                 + 0.852388 w_k1_FDX_s001
                                 + 0.879231 w_k1_IBM_s001
                                 + 0.938227 w_k1_INTC_s001
                                 + 0.878605 w_k1_KO_s001
                                 + 0.584939 w_k1_MAT_s001
                                 + 0.943945 w_k1_MSFT_s001
                                 + 0.978953 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.959547 w_k1_ADM_s001
                                 + 0.876938 w_k1_AXP_s001
                                 + 0.823197 w_k1_DIS_s001
                                 + 0.8893 w_k1_FDX_s001 + 1.24656 w_k1_IBM_s001
                                 + 0.985483 w_k1_INTC_s001
                                 + 1.02538 w_k1_KO_s001 + 0.93897 w_k1_MAT_s001
                                 + 0.92514 w_k1_MSFT_s001
                                 + 0.85567 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.27712 w_k1_ADM_s001
                                 + 1.18527 w_k1_AXP_s001
                                 + 1.64266 w_k1_DIS_s001
                                 + 1.39006 w_k1_FDX_s001
                                 + 1.00722 w_k1_IBM_s001
                                 + 1.24915 w_k1_INTC_s001
                                 + 1.22868 w_k1_KO_s001
                                 + 0.837927 w_k1_MAT_s001
                                 + 1.42332 w_k1_MSFT_s001
                                 + 1.07607 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.02525 w_k1_ADM_s001
                                 + 1.28506 w_k1_AXP_s001
                                 + 0.913653 w_k1_DIS_s001
                                 + 0.945906 w_k1_FDX_s001
                                 + 1.34271 w_k1_IBM_s001
                                 + 0.883534 w_k1_INTC_s001
                                 + 1.10514 w_k1_KO_s001 + 1.34405 w_k1_MAT_s001
                                 + 1.09244 w_k1_MSFT_s001
                                 + 0.964976 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.768786 w_k1_ADM_s001
                                 + 0.602638 w_k1_AXP_s001
                                 + 0.558334 w_k1_DIS_s001
                                 + 0.688391 w_k1_FDX_s001
                                 + 0.768662 w_k1_IBM_s001
                                 + 0.814384 w_k1_INTC_s001
                                 + 0.729837 w_k1_KO_s001
                                 + 0.555488 w_k1_MAT_s001
                                 + 0.783187 w_k1_MSFT_s001
                                 + 0.839885 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.936619 w_k1_ADM_s001
                                 + 0.835049 w_k1_AXP_s001
                                 + 1.02244 w_k1_DIS_s001
                                 + 1.09587 w_k1_FDX_s001
                                 + 0.734327 w_k1_IBM_s001
                                 + 1.03422 w_k1_INTC_s001
                                 + 0.935637 w_k1_KO_s001
                                 + 0.584094 w_k1_MAT_s001
                                 + 1.15742 w_k1_MSFT_s001
                                 + 0.984991 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.38724 w_k1_ADM_s001
                                 + 0.889601 w_k1_AXP_s001
                                 + 1.14826 w_k1_DIS_s001
                                 + 1.35358 w_k1_FDX_s001
                                 + 0.687101 w_k1_IBM_s001
                                 + 1.19829 w_k1_INTC_s001
                                 + 0.917244 w_k1_KO_s001
                                 + 0.54807 w_k1_MAT_s001
                                 + 1.29226 w_k1_MSFT_s001
                                 + 1.03996 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.2008 w_k1_ADM_s001
                                 + 1.09427 w_k1_AXP_s001
                                 + 1.33544 w_k1_DIS_s001
                                 + 1.18376 w_k1_FDX_s001
                                 + 1.31141 w_k1_IBM_s001
                                 + 1.05667 w_k1_INTC_s001
                                 + 1.19136 w_k1_KO_s001 + 1.08345 w_k1_MAT_s001
                                 + 1.18486 w_k1_MSFT_s001
                                 + 1.02608 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.802471 w_k1_ADM_s002
                                 + 0.828798 w_k1_AXP_s002
                                 + 0.961857 w_k1_DIS_s002
                                 + 0.970977 w_k1_FDX_s002
                                 + 0.645284 w_k1_IBM_s002
                                 + 0.949754 w_k1_INTC_s002
                                 + 0.880138 w_k1_KO_s002
                                 + 0.749969 w_k1_MAT_s002
                                 + 0.919996 w_k1_MSFT_s002
                                 + 0.896025 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.940637 w_k1_ADM_s002
                                 + 1.22317 w_k1_AXP_s002
                                 + 1.16588 w_k1_DIS_s002
                                 + 1.11969 w_k1_FDX_s002
                                 + 0.858382 w_k1_IBM_s002
                                 + 1.35375 w_k1_INTC_s002 + 1.0852 w_k1_KO_s002
                                 + 0.854037 w_k1_MAT_s002
                                 + 1.13872 w_k1_MSFT_s002
                                 + 1.12985 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.848168 w_k1_ADM_s002
                                 + 0.952425 w_k1_AXP_s002
                                 + 0.990584 w_k1_DIS_s002
                                 + 0.980764 w_k1_FDX_s002
                                 + 1.18376 w_k1_IBM_s002
                                 + 0.965079 w_k1_INTC_s002
                                 + 1.22178 w_k1_KO_s002 + 1.14046 w_k1_MAT_s002
                                 + 0.889772 w_k1_MSFT_s002
                                 + 0.979132 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.13386 w_k1_ADM_s002
                                 + 1.30228 w_k1_AXP_s002
                                 + 1.49547 w_k1_DIS_s002
                                 + 1.29612 w_k1_FDX_s002
                                 + 1.12554 w_k1_IBM_s002
                                 + 1.48656 w_k1_INTC_s002
                                 + 1.54449 w_k1_KO_s002 + 1.35722 w_k1_MAT_s002
                                 + 1.29811 w_k1_MSFT_s002
                                 + 1.23407 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.861077 w_k1_ADM_s002
                                 + 1.11279 w_k1_AXP_s002 + 1.1016 w_k1_DIS_s002
                                 + 0.924561 w_k1_FDX_s002
                                 + 1.48498 w_k1_IBM_s002
                                 + 1.06739 w_k1_INTC_s002
                                 + 1.32944 w_k1_KO_s002 + 1.31604 w_k1_MAT_s002
                                 + 1.00455 w_k1_MSFT_s002
                                 + 0.928988 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.06199 w_k1_ADM_s002
                                 + 1.35411 w_k1_AXP_s002
                                 + 1.43626 w_k1_DIS_s002 + 1.1692 w_k1_FDX_s002
                                 + 1.89137 w_k1_IBM_s002
                                 + 1.19013 w_k1_INTC_s002
                                 + 1.54572 w_k1_KO_s002 + 2.0183 w_k1_MAT_s002
                                 + 1.07119 w_k1_MSFT_s002
                                 + 1.04943 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.690593 w_k1_ADM_s002
                                 + 0.678957 w_k1_AXP_s002
                                 + 0.594185 w_k1_DIS_s002
                                 + 0.635407 w_k1_FDX_s002
                                 + 0.754105 w_k1_IBM_s002
                                 + 0.765335 w_k1_INTC_s002
                                 + 0.79991 w_k1_KO_s002
                                 + 0.648473 w_k1_MAT_s002
                                 + 0.718668 w_k1_MSFT_s002
                                 + 0.696482 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.769231 w_k1_ADM_s002
                                 + 0.810962 w_k1_AXP_s002
                                 + 0.849499 w_k1_DIS_s002
                                 + 0.83548 w_k1_FDX_s002
                                 + 0.966134 w_k1_IBM_s002
                                 + 1.24156 w_k1_INTC_s002
                                 + 1.04241 w_k1_KO_s002 + 0.7515 w_k1_MAT_s002
                                 + 0.889873 w_k1_MSFT_s002
                                 + 0.818555 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.2509 w_k1_ADM_s003
                                 + 0.938802 w_k1_AXP_s003
                                 + 1.46431 w_k1_DIS_s003
                                 + 1.06192 w_k1_FDX_s003
                                 + 1.39028 w_k1_IBM_s003
                                 + 1.17243 w_k1_INTC_s003
                                 + 1.39887 w_k1_KO_s003 + 1.1713 w_k1_MAT_s003
                                 + 1.19078 w_k1_MSFT_s003
                                 + 1.07215 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.01824 w_k1_ADM_s003
                                 + 1.2833 w_k1_AXP_s003 + 1.0236 w_k1_DIS_s003
                                 + 1.10655 w_k1_FDX_s003
                                 + 1.05782 w_k1_IBM_s003
                                 + 1.21307 w_k1_INTC_s003
                                 + 1.09847 w_k1_KO_s003 + 1.20174 w_k1_MAT_s003
                                 + 0.891442 w_k1_MSFT_s003
                                 + 1.1208 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.32543 w_k1_ADM_s003
                                 + 0.880895 w_k1_AXP_s003
                                 + 1.14456 w_k1_DIS_s003
                                 + 1.11318 w_k1_FDX_s003
                                 + 0.71903 w_k1_IBM_s003
                                 + 1.25373 w_k1_INTC_s003
                                 + 0.906745 w_k1_KO_s003
                                 + 0.504309 w_k1_MAT_s003
                                 + 1.16161 w_k1_MSFT_s003
                                 + 0.947842 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.1552 w_k1_ADM_s003
                                 + 1.0761 w_k1_AXP_s003 + 1.57655 w_k1_DIS_s003
                                 + 1.33162 w_k1_FDX_s003
                                 + 1.02892 w_k1_IBM_s003
                                 + 1.35629 w_k1_INTC_s003
                                 + 1.20894 w_k1_KO_s003 + 1.08946 w_k1_MAT_s003
                                 + 1.17432 w_k1_MSFT_s003
                                 + 1.13957 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.966759 w_k1_ADM_s003
                                 + 0.940576 w_k1_AXP_s003
                                 + 0.83146 w_k1_DIS_s003
                                 + 0.878085 w_k1_FDX_s003
                                 + 1.27045 w_k1_IBM_s003
                                 + 1.17997 w_k1_INTC_s003
                                 + 1.06607 w_k1_KO_s003 + 1.41969 w_k1_MAT_s003
                                 + 0.908377 w_k1_MSFT_s003
                                 + 0.893892 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.774827 w_k1_ADM_s003
                                 + 0.79697 w_k1_AXP_s003
                                 + 0.841999 w_k1_DIS_s003
                                 + 0.876972 w_k1_FDX_s003
                                 + 0.755875 w_k1_IBM_s003
                                 + 0.98676 w_k1_INTC_s003
                                 + 0.746731 w_k1_KO_s003
                                 + 0.699119 w_k1_MAT_s003
                                 + 0.925988 w_k1_MSFT_s003
                                 + 0.847725 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.53662 w_k1_ADM_s003
                                 + 1.44847 w_k1_AXP_s003
                                 + 1.53781 w_k1_DIS_s003
                                 + 1.49209 w_k1_FDX_s003
                                 + 0.859993 w_k1_IBM_s003
                                 + 1.46059 w_k1_INTC_s003
                                 + 1.00424 w_k1_KO_s003 + 0.85112 w_k1_MAT_s003
                                 + 1.11136 w_k1_MSFT_s003
                                 + 0.961381 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.15398 w_k1_ADM_s003
                                 + 0.713291 w_k1_AXP_s003
                                 + 0.825049 w_k1_DIS_s003
                                 + 0.892831 w_k1_FDX_s003
                                 + 0.894914 w_k1_IBM_s003
                                 + 0.914886 w_k1_INTC_s003
                                 + 0.920596 w_k1_KO_s003
                                 + 0.783096 w_k1_MAT_s003
                                 + 0.872616 w_k1_MSFT_s003
                                 + 0.991403 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.926822 w_k1_ADM_s004
                                 + 1.01931 w_k1_AXP_s004
                                 + 0.997939 w_k1_DIS_s004
                                 + 1.01237 w_k1_FDX_s004
                                 + 1.56565 w_k1_IBM_s004
                                 + 0.748301 w_k1_INTC_s004
                                 + 1.39341 w_k1_KO_s004 + 2.8561 w_k1_MAT_s004
                                 + 0.946921 w_k1_MSFT_s004
                                 + 1.3391 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.610186 w_k1_ADM_s004
                                 + 0.701162 w_k1_AXP_s004
                                 + 0.56491 w_k1_DIS_s004
                                 + 0.768491 w_k1_FDX_s004
                                 + 0.777379 w_k1_IBM_s004
                                 + 0.756752 w_k1_INTC_s004
                                 + 0.675474 w_k1_KO_s004
                                 + 1.06525 w_k1_MAT_s004
                                 + 0.75381 w_k1_MSFT_s004
                                 + 0.912598 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.01704 w_k1_ADM_s004
                                 + 1.26013 w_k1_AXP_s004
                                 + 1.61247 w_k1_DIS_s004 + 1.4547 w_k1_FDX_s004
                                 + 1.15309 w_k1_IBM_s004
                                 + 1.2546 w_k1_INTC_s004 + 1.28797 w_k1_KO_s004
                                 + 1.51263 w_k1_MAT_s004
                                 + 1.43164 w_k1_MSFT_s004
                                 + 1.21193 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.725917 w_k1_ADM_s004
                                 + 0.786708 w_k1_AXP_s004
                                 + 0.845353 w_k1_DIS_s004
                                 + 1.04665 w_k1_FDX_s004
                                 + 0.505395 w_k1_IBM_s004
                                 + 1.00772 w_k1_INTC_s004
                                 + 0.764983 w_k1_KO_s004
                                 + 0.624064 w_k1_MAT_s004
                                 + 1.02237 w_k1_MSFT_s004
                                 + 0.947001 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.10885 w_k1_ADM_s004
                                 + 1.20762 w_k1_AXP_s004
                                 + 0.910072 w_k1_DIS_s004
                                 + 1.0787 w_k1_FDX_s004
                                 + 0.943369 w_k1_IBM_s004
                                 + 0.853625 w_k1_INTC_s004
                                 + 1.02551 w_k1_KO_s004 + 1.89983 w_k1_MAT_s004
                                 + 0.886483 w_k1_MSFT_s004
                                 + 1.08632 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.04148 w_k1_ADM_s004
                                 + 0.886158 w_k1_AXP_s004
                                 + 1.20593 w_k1_DIS_s004
                                 + 1.27715 w_k1_FDX_s004
                                 + 0.690923 w_k1_IBM_s004
                                 + 1.1185 w_k1_INTC_s004 + 1.03008 w_k1_KO_s004
                                 + 0.788893 w_k1_MAT_s004
                                 + 1.26613 w_k1_MSFT_s004
                                 + 1.04385 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.694896 w_k1_ADM_s004
                                 + 0.679092 w_k1_AXP_s004
                                 + 0.905748 w_k1_DIS_s004
                                 + 0.820724 w_k1_FDX_s004
                                 + 1.14193 w_k1_IBM_s004
                                 + 0.957739 w_k1_INTC_s004
                                 + 0.995399 w_k1_KO_s004
                                 + 1.21745 w_k1_MAT_s004
                                 + 0.951951 w_k1_MSFT_s004
                                 + 0.967797 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.03349 w_k1_ADM_s004
                                 + 0.963327 w_k1_AXP_s004
                                 + 1.15274 w_k1_DIS_s004
                                 + 1.19381 w_k1_FDX_s004
                                 + 0.935053 w_k1_IBM_s004
                                 + 1.09824 w_k1_INTC_s004 + 1.0939 w_k1_KO_s004
                                 + 1.05467 w_k1_MAT_s004
                                 + 1.25036 w_k1_MSFT_s004
                                 + 1.29438 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.19288 w_k1_ADM_s005
                                 + 0.788831 w_k1_AXP_s005
                                 + 1.00007 w_k1_DIS_s005
                                 + 1.12277 w_k1_FDX_s005
                                 + 0.68579 w_k1_IBM_s005 + 1.127 w_k1_INTC_s005
                                 + 0.837121 w_k1_KO_s005
                                 + 0.388866 w_k1_MAT_s005
                                 + 1.00024 w_k1_MSFT_s005
                                 + 0.880633 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.76544 w_k1_ADM_s005
                                 + 0.729431 w_k1_AXP_s005
                                 + 1.02127 w_k1_DIS_s005
                                 + 0.848077 w_k1_FDX_s005
                                 + 1.59727 w_k1_IBM_s005
                                 + 1.0621 w_k1_INTC_s005 + 1.22121 w_k1_KO_s005
                                 + 1.15959 w_k1_MAT_s005
                                 + 1.02599 w_k1_MSFT_s005
                                 + 0.961609 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.829389 w_k1_ADM_s005
                                 + 0.617643 w_k1_AXP_s005
                                 + 0.544445 w_k1_DIS_s005
                                 + 0.729558 w_k1_FDX_s005
                                 + 0.811937 w_k1_IBM_s005
                                 + 0.857835 w_k1_INTC_s005
                                 + 0.784951 w_k1_KO_s005
                                 + 0.653892 w_k1_MAT_s005
                                 + 0.756218 w_k1_MSFT_s005
                                 + 0.87755 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.00935 w_k1_ADM_s005
                                 + 1.26925 w_k1_AXP_s005 + 1.3989 w_k1_DIS_s005
                                 + 1.13507 w_k1_FDX_s005
                                 + 1.43774 w_k1_IBM_s005
                                 + 1.19755 w_k1_INTC_s005
                                 + 1.55087 w_k1_KO_s005 + 1.66968 w_k1_MAT_s005
                                 + 1.174 w_k1_MSFT_s005 + 1.16212 w_k1_PG_s005
                                 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.765127 w_k1_ADM_s005
                                 + 0.687106 w_k1_AXP_s005
                                 + 0.916465 w_k1_DIS_s005
                                 + 0.948377 w_k1_FDX_s005
                                 + 0.882138 w_k1_IBM_s005
                                 + 1.09226 w_k1_INTC_s005 + 1.0717 w_k1_KO_s005
                                 + 0.80017 w_k1_MAT_s005
                                 + 1.02759 w_k1_MSFT_s005
                                 + 0.944378 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.04618 w_k1_ADM_s005
                                 + 0.996494 w_k1_AXP_s005
                                 + 0.986942 w_k1_DIS_s005
                                 + 0.935718 w_k1_FDX_s005
                                 + 1.14763 w_k1_IBM_s005
                                 + 1.05321 w_k1_INTC_s005
                                 + 1.12529 w_k1_KO_s005
                                 + 0.877572 w_k1_MAT_s005
                                 + 0.942268 w_k1_MSFT_s005
                                 + 0.939494 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.930431 w_k1_ADM_s005
                                 + 0.90622 w_k1_AXP_s005
                                 + 0.730253 w_k1_DIS_s005
                                 + 0.966428 w_k1_FDX_s005
                                 + 0.789596 w_k1_IBM_s005
                                 + 1.02927 w_k1_INTC_s005
                                 + 0.832704 w_k1_KO_s005
                                 + 0.946846 w_k1_MAT_s005
                                 + 0.895676 w_k1_MSFT_s005
                                 + 0.933659 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.17615 w_k1_ADM_s005
                                 + 1.1911 w_k1_AXP_s005 + 1.36589 w_k1_DIS_s005
                                 + 1.22983 w_k1_FDX_s005
                                 + 1.05459 w_k1_IBM_s005
                                 + 1.03513 w_k1_INTC_s005
                                 + 1.31019 w_k1_KO_s005 + 1.05446 w_k1_MAT_s005
                                 + 1.15776 w_k1_MSFT_s005
                                 + 1.21652 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.00694 w_k1_ADM_s006
                                 + 1.30722 w_k1_AXP_s006 + 1.1953 w_k1_DIS_s006
                                 + 1.15668 w_k1_FDX_s006
                                 + 0.977522 w_k1_IBM_s006
                                 + 1.17482 w_k1_INTC_s006
                                 + 1.01995 w_k1_KO_s006 + 1.21518 w_k1_MAT_s006
                                 + 1.03772 w_k1_MSFT_s006
                                 + 1.05643 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.13003 w_k1_ADM_s006
                                 + 0.84645 w_k1_AXP_s006
                                 + 1.00632 w_k1_DIS_s006
                                 + 1.09929 w_k1_FDX_s006
                                 + 0.616804 w_k1_IBM_s006
                                 + 0.819143 w_k1_INTC_s006
                                 + 0.688536 w_k1_KO_s006
                                 + 0.506852 w_k1_MAT_s006
                                 + 0.881673 w_k1_MSFT_s006
                                 + 0.97842 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.829492 w_k1_ADM_s006
                                 + 0.883048 w_k1_AXP_s006
                                 + 0.944951 w_k1_DIS_s006
                                 + 0.973368 w_k1_FDX_s006
                                 + 0.94988 w_k1_IBM_s006
                                 + 0.859074 w_k1_INTC_s006
                                 + 1.01045 w_k1_KO_s006 + 1.01906 w_k1_MAT_s006
                                 + 0.930554 w_k1_MSFT_s006
                                 + 0.933412 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.606132 w_k1_ADM_s006
                                 + 0.884102 w_k1_AXP_s006
                                 + 0.660061 w_k1_DIS_s006
                                 + 0.708523 w_k1_FDX_s006
                                 + 0.926678 w_k1_IBM_s006
                                 + 0.748008 w_k1_INTC_s006
                                 + 0.817619 w_k1_KO_s006
                                 + 1.39141 w_k1_MAT_s006
                                 + 0.692672 w_k1_MSFT_s006
                                 + 0.845934 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.07662 w_k1_ADM_s006
                                 + 1.51865 w_k1_AXP_s006
                                 + 1.20147 w_k1_DIS_s006
                                 + 1.09479 w_k1_FDX_s006
                                 + 1.58672 w_k1_IBM_s006
                                 + 1.17233 w_k1_INTC_s006
                                 + 1.20949 w_k1_KO_s006 + 1.48574 w_k1_MAT_s006
                                 + 0.964106 w_k1_MSFT_s006
                                 + 1.09026 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.874323 w_k1_ADM_s006
                                 + 0.761289 w_k1_AXP_s006
                                 + 0.8091 w_k1_DIS_s006
                                 + 0.797322 w_k1_FDX_s006
                                 + 0.87581 w_k1_IBM_s006
                                 + 0.920954 w_k1_INTC_s006
                                 + 0.849986 w_k1_KO_s006
                                 + 0.732346 w_k1_MAT_s006
                                 + 0.777729 w_k1_MSFT_s006
                                 + 0.787739 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.12111 w_k1_ADM_s006
                                 + 1.09693 w_k1_AXP_s006
                                 + 1.55041 w_k1_DIS_s006
                                 + 1.34196 w_k1_FDX_s006
                                 + 1.08055 w_k1_IBM_s006
                                 + 1.23478 w_k1_INTC_s006
                                 + 1.13988 w_k1_KO_s006
                                 + 0.967195 w_k1_MAT_s006
                                 + 1.18935 w_k1_MSFT_s006
                                 + 0.99868 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.818289 w_k1_ADM_s006
                                 + 0.713735 w_k1_AXP_s006
                                 + 0.764884 w_k1_DIS_s006
                                 + 0.938822 w_k1_FDX_s006
                                 + 0.64455 w_k1_IBM_s006
                                 + 0.84024 w_k1_INTC_s006
                                 + 0.732984 w_k1_KO_s006
                                 + 0.761598 w_k1_MAT_s006
                                 + 0.727281 w_k1_MSFT_s006
                                 + 0.772395 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.16893 w_k1_ADM_s007
                                 + 1.13969 w_k1_AXP_s007
                                 + 1.68325 w_k1_DIS_s007
                                 + 1.25416 w_k1_FDX_s007
                                 + 1.50908 w_k1_IBM_s007
                                 + 1.2153 w_k1_INTC_s007 + 1.58275 w_k1_KO_s007
                                 + 1.30297 w_k1_MAT_s007
                                 + 1.40262 w_k1_MSFT_s007
                                 + 1.51553 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.13021 w_k1_ADM_s007
                                 + 0.843957 w_k1_AXP_s007
                                 + 0.654734 w_k1_DIS_s007
                                 + 0.924183 w_k1_FDX_s007
                                 + 0.617068 w_k1_IBM_s007
                                 + 0.88242 w_k1_INTC_s007
                                 + 0.675859 w_k1_KO_s007
                                 + 0.697365 w_k1_MAT_s007
                                 + 0.923758 w_k1_MSFT_s007
                                 + 0.832706 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.882444 w_k1_ADM_s007
                                 + 0.709151 w_k1_AXP_s007
                                 + 0.685897 w_k1_DIS_s007
                                 + 0.831697 w_k1_FDX_s007
                                 + 0.945551 w_k1_IBM_s007
                                 + 0.779109 w_k1_INTC_s007
                                 + 0.872103 w_k1_KO_s007
                                 + 0.723949 w_k1_MAT_s007
                                 + 0.860608 w_k1_MSFT_s007
                                 + 0.937249 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.932899 w_k1_ADM_s007
                                 + 0.978806 w_k1_AXP_s007
                                 + 1.12089 w_k1_DIS_s007
                                 + 0.863915 w_k1_FDX_s007
                                 + 1.4245 w_k1_IBM_s007
                                 + 0.878957 w_k1_INTC_s007
                                 + 1.12241 w_k1_KO_s007 + 1.70109 w_k1_MAT_s007
                                 + 1.03621 w_k1_MSFT_s007
                                 + 1.12062 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.20203 w_k1_ADM_s007
                                 + 1.08429 w_k1_AXP_s007 + 1.1394 w_k1_DIS_s007
                                 + 1.30191 w_k1_FDX_s007
                                 + 0.638391 w_k1_IBM_s007
                                 + 1.18737 w_k1_INTC_s007
                                 + 1.01727 w_k1_KO_s007
                                 + 0.649669 w_k1_MAT_s007
                                 + 1.20031 w_k1_MSFT_s007
                                 + 1.09921 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.54111 w_k1_ADM_s007
                                 + 1.15677 w_k1_AXP_s007
                                 + 1.28505 w_k1_DIS_s007
                                 + 1.23824 w_k1_FDX_s007
                                 + 1.02393 w_k1_IBM_s007
                                 + 0.937531 w_k1_INTC_s007
                                 + 1.26205 w_k1_KO_s007 + 1.1508 w_k1_MAT_s007
                                 + 1.14837 w_k1_MSFT_s007
                                 + 1.24779 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.15466 w_k1_ADM_s007
                                 + 0.85386 w_k1_AXP_s007
                                 + 1.03022 w_k1_DIS_s007
                                 + 1.09448 w_k1_FDX_s007
                                 + 0.738371 w_k1_IBM_s007
                                 + 0.823805 w_k1_INTC_s007
                                 + 0.892742 w_k1_KO_s007
                                 + 0.926996 w_k1_MAT_s007
                                 + 1.02827 w_k1_MSFT_s007
                                 + 1.07824 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.36426 w_k1_ADM_s007
                                 + 0.850088 w_k1_AXP_s007
                                 + 1.02974 w_k1_DIS_s007 + 1.0407 w_k1_FDX_s007
                                 + 0.90313 w_k1_IBM_s007
                                 + 1.00704 w_k1_INTC_s007
                                 + 1.05791 w_k1_KO_s007
                                 + 0.775629 w_k1_MAT_s007
                                 + 1.0959 w_k1_MSFT_s007 + 1.03742 w_k1_PG_s007
                                 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.12526 w_k1_ADM_s010
                                 + 0.947209 w_k1_AXP_s010
                                 + 1.02978 w_k1_DIS_s010
                                 + 1.10454 w_k1_FDX_s010
                                 + 0.685136 w_k1_IBM_s010
                                 + 1.23669 w_k1_INTC_s010
                                 + 0.893296 w_k1_KO_s010
                                 + 0.67216 w_k1_MAT_s010
                                 + 1.13842 w_k1_MSFT_s010
                                 + 0.923935 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.15929 w_k1_ADM_s010
                                 + 0.968889 w_k1_AXP_s010
                                 + 1.28965 w_k1_DIS_s010
                                 + 1.22388 w_k1_FDX_s010
                                 + 0.759855 w_k1_IBM_s010
                                 + 0.890809 w_k1_INTC_s010
                                 + 0.93364 w_k1_KO_s010 + 1.1541 w_k1_MAT_s010
                                 + 1.1478 w_k1_MSFT_s010 + 1.0474 w_k1_PG_s010
                                 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.31843 w_k1_ADM_s010
                                 + 1.36879 w_k1_AXP_s010
                                 + 1.21442 w_k1_DIS_s010
                                 + 1.08283 w_k1_FDX_s010
                                 + 1.22726 w_k1_IBM_s010
                                 + 0.953475 w_k1_INTC_s010
                                 + 1.20221 w_k1_KO_s010 + 1.63092 w_k1_MAT_s010
                                 + 1.04325 w_k1_MSFT_s010
                                 + 1.13806 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.994138 w_k1_ADM_s010
                                 + 0.878192 w_k1_AXP_s010
                                 + 0.962046 w_k1_DIS_s010
                                 + 0.859223 w_k1_FDX_s010
                                 + 1.36676 w_k1_IBM_s010
                                 + 0.935721 w_k1_INTC_s010
                                 + 1.03888 w_k1_KO_s010 + 1.28126 w_k1_MAT_s010
                                 + 0.82443 w_k1_MSFT_s010
                                 + 0.976714 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.825589 w_k1_ADM_s010
                                 + 0.910405 w_k1_AXP_s010
                                 + 1.24613 w_k1_DIS_s010
                                 + 1.08092 w_k1_FDX_s010
                                 + 1.19193 w_k1_IBM_s010
                                 + 1.23937 w_k1_INTC_s010
                                 + 1.28428 w_k1_KO_s010 + 1.60806 w_k1_MAT_s010
                                 + 1.03596 w_k1_MSFT_s010
                                 + 1.00378 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.7333 w_k1_ADM_s010
                                 + 1.33975 w_k1_AXP_s010
                                 + 1.75031 w_k1_DIS_s010 + 1.4607 w_k1_FDX_s010
                                 + 1.4547 w_k1_IBM_s010
                                 + 1.24889 w_k1_INTC_s010
                                 + 1.63157 w_k1_KO_s010 + 1.26031 w_k1_MAT_s010
                                 + 1.33282 w_k1_MSFT_s010
                                 + 1.16236 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.19368 w_k1_ADM_s010
                                 + 1.20692 w_k1_AXP_s010
                                 + 1.02463 w_k1_DIS_s010
                                 + 1.12164 w_k1_FDX_s010
                                 + 0.863156 w_k1_IBM_s010
                                 + 1.03607 w_k1_INTC_s010
                                 + 1.03917 w_k1_KO_s010 + 1.01577 w_k1_MAT_s010
                                 + 1.0599 w_k1_MSFT_s010 + 1.01735 w_k1_PG_s010
                                 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.883312 w_k1_ADM_s010
                                 + 0.79654 w_k1_AXP_s010
                                 + 0.723057 w_k1_DIS_s010
                                 + 0.831023 w_k1_FDX_s010
                                 + 0.887538 w_k1_IBM_s010
                                 + 0.897258 w_k1_INTC_s010
                                 + 0.967873 w_k1_KO_s010
                                 + 0.730915 w_k1_MAT_s010
                                 + 0.787619 w_k1_MSFT_s010
                                 + 0.859493 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.25765 w_k1_ADM_s011
                                 + 1.39437 w_k1_AXP_s011
                                 + 1.89744 w_k1_DIS_s011
                                 + 1.44469 w_k1_FDX_s011
                                 + 0.972839 w_k1_IBM_s011
                                 + 1.63782 w_k1_INTC_s011
                                 + 1.58895 w_k1_KO_s011 + 1.09384 w_k1_MAT_s011
                                 + 1.8299 w_k1_MSFT_s011 + 1.31834 w_k1_PG_s011
                                 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.06708 w_k1_ADM_s011
                                 + 0.894217 w_k1_AXP_s011
                                 + 1.16513 w_k1_DIS_s011
                                 + 1.01925 w_k1_FDX_s011
                                 + 1.06049 w_k1_IBM_s011
                                 + 1.04456 w_k1_INTC_s011
                                 + 1.15898 w_k1_KO_s011
                                 + 0.882298 w_k1_MAT_s011
                                 + 1.16003 w_k1_MSFT_s011
                                 + 1.01674 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.957434 w_k1_ADM_s011
                                 + 1.36289 w_k1_AXP_s011
                                 + 1.26213 w_k1_DIS_s011
                                 + 1.27009 w_k1_FDX_s011
                                 + 0.734867 w_k1_IBM_s011
                                 + 1.16165 w_k1_INTC_s011
                                 + 1.01891 w_k1_KO_s011 + 1.236 w_k1_MAT_s011
                                 + 1.22347 w_k1_MSFT_s011
                                 + 1.09446 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.811509 w_k1_ADM_s011
                                 + 1.4292 w_k1_AXP_s011 + 1.03992 w_k1_DIS_s011
                                 + 1.07343 w_k1_FDX_s011
                                 + 1.22115 w_k1_IBM_s011
                                 + 1.16095 w_k1_INTC_s011
                                 + 1.22332 w_k1_KO_s011 + 1.8217 w_k1_MAT_s011
                                 + 1.02808 w_k1_MSFT_s011
                                 + 1.06088 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.812524 w_k1_ADM_s011
                                 + 1.23472 w_k1_AXP_s011
                                 + 1.23044 w_k1_DIS_s011
                                 + 0.99323 w_k1_FDX_s011
                                 + 1.74509 w_k1_IBM_s011 + 1.072 w_k1_INTC_s011
                                 + 1.17911 w_k1_KO_s011 + 1.9176 w_k1_MAT_s011
                                 + 0.99454 w_k1_MSFT_s011 + 1.0099 w_k1_PG_s011
                                 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.691027 w_k1_ADM_s011
                                 + 0.993394 w_k1_AXP_s011
                                 + 0.693244 w_k1_DIS_s011
                                 + 0.757409 w_k1_FDX_s011
                                 + 0.920664 w_k1_IBM_s011
                                 + 0.928575 w_k1_INTC_s011
                                 + 0.827848 w_k1_KO_s011
                                 + 1.03212 w_k1_MAT_s011
                                 + 0.82789 w_k1_MSFT_s011
                                 + 0.823465 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.837916 w_k1_ADM_s011
                                 + 0.921276 w_k1_AXP_s011
                                 + 0.953771 w_k1_DIS_s011
                                 + 0.912972 w_k1_FDX_s011
                                 + 0.727588 w_k1_IBM_s011
                                 + 1.01054 w_k1_INTC_s011
                                 + 1.02356 w_k1_KO_s011
                                 + 0.849401 w_k1_MAT_s011
                                 + 1.09777 w_k1_MSFT_s011
                                 + 1.00879 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.814451 w_k1_ADM_s011
                                 + 1.13801 w_k1_AXP_s011
                                 + 0.738538 w_k1_DIS_s011
                                 + 0.763855 w_k1_FDX_s011
                                 + 1.29777 w_k1_IBM_s011
                                 + 0.855847 w_k1_INTC_s011
                                 + 1.09453 w_k1_KO_s011 + 1.4558 w_k1_MAT_s011
                                 + 0.802262 w_k1_MSFT_s011
                                 + 0.893971 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.882699 w_k1_ADM_s012
                                 + 1.07837 w_k1_AXP_s012
                                 + 1.04072 w_k1_DIS_s012
                                 + 1.09255 w_k1_FDX_s012
                                 + 0.771764 w_k1_IBM_s012
                                 + 1.32849 w_k1_INTC_s012
                                 + 0.82984 w_k1_KO_s012
                                 + 0.912246 w_k1_MAT_s012
                                 + 1.09131 w_k1_MSFT_s012
                                 + 0.904855 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.783999 w_k1_ADM_s012
                                 + 1.03891 w_k1_AXP_s012
                                 + 1.29163 w_k1_DIS_s012
                                 + 1.13681 w_k1_FDX_s012
                                 + 1.66188 w_k1_IBM_s012
                                 + 1.34552 w_k1_INTC_s012
                                 + 1.18919 w_k1_KO_s012 + 1.39699 w_k1_MAT_s012
                                 + 1.15545 w_k1_MSFT_s012 + 1.0973 w_k1_PG_s012
                                 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.10525 w_k1_ADM_s012
                                 + 0.945315 w_k1_AXP_s012
                                 + 0.898068 w_k1_DIS_s012
                                 + 1.05545 w_k1_FDX_s012
                                 + 0.642903 w_k1_IBM_s012
                                 + 1.18416 w_k1_INTC_s012
                                 + 0.722722 w_k1_KO_s012
                                 + 0.636606 w_k1_MAT_s012
                                 + 0.857016 w_k1_MSFT_s012
                                 + 0.886807 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.26634 w_k1_ADM_s012
                                 + 1.57644 w_k1_AXP_s012 + 1.3735 w_k1_DIS_s012
                                 + 1.43118 w_k1_FDX_s012
                                 + 1.02523 w_k1_IBM_s012
                                 + 1.43818 w_k1_INTC_s012
                                 + 0.953789 w_k1_KO_s012
                                 + 0.93787 w_k1_MAT_s012
                                 + 1.24683 w_k1_MSFT_s012
                                 + 0.980088 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.795212 w_k1_ADM_s012
                                 + 0.756283 w_k1_AXP_s012
                                 + 0.71075 w_k1_DIS_s012
                                 + 0.822542 w_k1_FDX_s012
                                 + 0.707725 w_k1_IBM_s012
                                 + 0.81521 w_k1_INTC_s012
                                 + 0.633587 w_k1_KO_s012
                                 + 0.83964 w_k1_MAT_s012
                                 + 0.933972 w_k1_MSFT_s012
                                 + 0.831566 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.28982 w_k1_ADM_s012
                                 + 0.977227 w_k1_AXP_s012
                                 + 0.820423 w_k1_DIS_s012
                                 + 0.908047 w_k1_FDX_s012
                                 + 1.36455 w_k1_IBM_s012
                                 + 1.10506 w_k1_INTC_s012 + 1.1713 w_k1_KO_s012
                                 + 1.26788 w_k1_MAT_s012
                                 + 0.989585 w_k1_MSFT_s012
                                 + 0.957813 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.26544 w_k1_ADM_s012
                                 + 1.51013 w_k1_AXP_s012
                                 + 1.23865 w_k1_DIS_s012
                                 + 1.20137 w_k1_FDX_s012
                                 + 1.22542 w_k1_IBM_s012
                                 + 1.28464 w_k1_INTC_s012 + 1.1662 w_k1_KO_s012
                                 + 1.84902 w_k1_MAT_s012
                                 + 1.34703 w_k1_MSFT_s012
                                 + 1.10782 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.777266 w_k1_ADM_s012
                                 + 0.956217 w_k1_AXP_s012
                                 + 0.926437 w_k1_DIS_s012
                                 + 0.895177 w_k1_FDX_s012
                                 + 1.10557 w_k1_IBM_s012
                                 + 1.04483 w_k1_INTC_s012
                                 + 0.855029 w_k1_KO_s012
                                 + 1.13038 w_k1_MAT_s012
                                 + 0.995307 w_k1_MSFT_s012
                                 + 0.846925 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.523976 w_k1_ADM_s013
                                 + 0.660711 w_k1_AXP_s013
                                 + 0.515074 w_k1_DIS_s013
                                 + 0.694246 w_k1_FDX_s013
                                 + 0.594507 w_k1_IBM_s013
                                 + 0.750007 w_k1_INTC_s013
                                 + 0.63821 w_k1_KO_s013 + 0.70132 w_k1_MAT_s013
                                 + 0.694083 w_k1_MSFT_s013
                                 + 0.807452 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.781156 w_k1_ADM_s013
                                 + 0.916559 w_k1_AXP_s013
                                 + 0.886503 w_k1_DIS_s013
                                 + 0.840088 w_k1_FDX_s013
                                 + 0.984724 w_k1_IBM_s013
                                 + 0.941866 w_k1_INTC_s013
                                 + 0.993368 w_k1_KO_s013
                                 + 1.10091 w_k1_MAT_s013
                                 + 0.843389 w_k1_MSFT_s013
                                 + 0.921737 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.863725 w_k1_ADM_s013
                                 + 1.06137 w_k1_AXP_s013
                                 + 0.794381 w_k1_DIS_s013
                                 + 0.842298 w_k1_FDX_s013
                                 + 1.35574 w_k1_IBM_s013
                                 + 0.99448 w_k1_INTC_s013
                                 + 0.988847 w_k1_KO_s013
                                 + 1.10509 w_k1_MAT_s013
                                 + 0.967105 w_k1_MSFT_s013
                                 + 1.06632 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.85875 w_k1_ADM_s013
                                 + 0.748914 w_k1_AXP_s013
                                 + 0.830546 w_k1_DIS_s013
                                 + 0.946005 w_k1_FDX_s013
                                 + 0.734566 w_k1_IBM_s013
                                 + 0.949299 w_k1_INTC_s013
                                 + 0.9443 w_k1_KO_s013 + 0.800444 w_k1_MAT_s013
                                 + 1.03296 w_k1_MSFT_s013
                                 + 0.883349 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.34908 w_k1_ADM_s013
                                 + 1.65584 w_k1_AXP_s013
                                 + 1.82251 w_k1_DIS_s013
                                 + 1.40907 w_k1_FDX_s013 + 1.5053 w_k1_IBM_s013
                                 + 1.28127 w_k1_INTC_s013
                                 + 1.57094 w_k1_KO_s013 + 2.01734 w_k1_MAT_s013
                                 + 1.42766 w_k1_MSFT_s013
                                 + 1.45258 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.98545 w_k1_ADM_s013
                                 + 1.32768 w_k1_AXP_s013
                                 + 1.26275 w_k1_DIS_s013
                                 + 1.09738 w_k1_FDX_s013
                                 + 1.21097 w_k1_IBM_s013
                                 + 1.03792 w_k1_INTC_s013
                                 + 1.11568 w_k1_KO_s013 + 1.41238 w_k1_MAT_s013
                                 + 1.10338 w_k1_MSFT_s013
                                 + 1.20083 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.16037 w_k1_ADM_s013
                                 + 1.13897 w_k1_AXP_s013
                                 + 1.27132 w_k1_DIS_s013
                                 + 1.29146 w_k1_FDX_s013
                                 + 0.73513 w_k1_IBM_s013
                                 + 1.21697 w_k1_INTC_s013
                                 + 1.01827 w_k1_KO_s013
                                 + 0.812978 w_k1_MAT_s013
                                 + 1.30469 w_k1_MSFT_s013 + 1.0994 w_k1_PG_s013
                                 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.950159 w_k1_ADM_s013
                                 + 1.33571 w_k1_AXP_s013
                                 + 0.815491 w_k1_DIS_s013
                                 + 0.998746 w_k1_FDX_s013
                                 + 0.831939 w_k1_IBM_s013
                                 + 0.966312 w_k1_INTC_s013
                                 + 0.837862 w_k1_KO_s013
                                 + 0.99461 w_k1_MAT_s013
                                 + 0.907812 w_k1_MSFT_s013
                                 + 1.07847 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.24673 w_k1_ADM_s014
                                 + 1.35669 w_k1_AXP_s014 + 1.2533 w_k1_DIS_s014
                                 + 1.08119 w_k1_FDX_s014
                                 + 1.21742 w_k1_IBM_s014
                                 + 1.07909 w_k1_INTC_s014
                                 + 1.15896 w_k1_KO_s014
                                 + 0.834945 w_k1_MAT_s014
                                 + 1.08658 w_k1_MSFT_s014
                                 + 1.11944 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.761114 w_k1_ADM_s014
                                 + 0.693799 w_k1_AXP_s014
                                 + 0.681566 w_k1_DIS_s014
                                 + 0.663243 w_k1_FDX_s014
                                 + 1.17098 w_k1_IBM_s014
                                 + 0.751174 w_k1_INTC_s014
                                 + 0.897295 w_k1_KO_s014
                                 + 0.773462 w_k1_MAT_s014
                                 + 0.716411 w_k1_MSFT_s014
                                 + 0.898125 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.893038 w_k1_ADM_s014
                                 + 0.857563 w_k1_AXP_s014
                                 + 0.868105 w_k1_DIS_s014
                                 + 0.881466 w_k1_FDX_s014
                                 + 0.914354 w_k1_IBM_s014
                                 + 1.02604 w_k1_INTC_s014
                                 + 1.05234 w_k1_KO_s014
                                 + 0.694198 w_k1_MAT_s014
                                 + 1.02677 w_k1_MSFT_s014
                                 + 1.06023 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.898772 w_k1_ADM_s014
                                 + 1.03003 w_k1_AXP_s014
                                 + 0.986915 w_k1_DIS_s014
                                 + 0.834686 w_k1_FDX_s014
                                 + 1.59371 w_k1_IBM_s014
                                 + 1.19743 w_k1_INTC_s014
                                 + 1.32617 w_k1_KO_s014 + 1.08808 w_k1_MAT_s014
                                 + 0.936921 w_k1_MSFT_s014
                                 + 1.0234 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.58388 w_k1_ADM_s014
                                 + 1.61047 w_k1_AXP_s014
                                 + 1.75105 w_k1_DIS_s014
                                 + 1.34004 w_k1_FDX_s014
                                 + 1.23084 w_k1_IBM_s014
                                 + 1.55399 w_k1_INTC_s014
                                 + 1.56293 w_k1_KO_s014
                                 + 0.849761 w_k1_MAT_s014
                                 + 1.4955 w_k1_MSFT_s014 + 1.33382 w_k1_PG_s014
                                 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.12365 w_k1_ADM_s014
                                 + 1.14233 w_k1_AXP_s014
                                 + 1.06753 w_k1_DIS_s014
                                 + 1.16832 w_k1_FDX_s014
                                 + 0.884505 w_k1_IBM_s014
                                 + 1.30731 w_k1_INTC_s014
                                 + 1.08826 w_k1_KO_s014
                                 + 0.771248 w_k1_MAT_s014
                                 + 1.19484 w_k1_MSFT_s014
                                 + 1.06274 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.99063 w_k1_ADM_s014
                                 + 1.20261 w_k1_AXP_s014
                                 + 0.952406 w_k1_DIS_s014
                                 + 0.894867 w_k1_FDX_s014
                                 + 1.20698 w_k1_IBM_s014
                                 + 1.13437 w_k1_INTC_s014
                                 + 1.28772 w_k1_KO_s014 + 1.40033 w_k1_MAT_s014
                                 + 0.97992 w_k1_MSFT_s014
                                 + 1.07605 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.953147 w_k1_ADM_s014
                                 + 0.859146 w_k1_AXP_s014
                                 + 0.647946 w_k1_DIS_s014
                                 + 0.817558 w_k1_FDX_s014
                                 + 0.719214 w_k1_IBM_s014
                                 + 0.904924 w_k1_INTC_s014
                                 + 0.836129 w_k1_KO_s014
                                 + 0.687821 w_k1_MAT_s014
                                 + 0.879482 w_k1_MSFT_s014
                                 + 0.845267 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.23156 w_k1_ADM_s015
                                 + 1.12525 w_k1_AXP_s015
                                 + 0.960086 w_k1_DIS_s015
                                 + 1.17554 w_k1_FDX_s015
                                 + 0.563079 w_k1_IBM_s015
                                 + 0.892426 w_k1_INTC_s015
                                 + 0.81218 w_k1_KO_s015
                                 + 0.424696 w_k1_MAT_s015
                                 + 1.03866 w_k1_MSFT_s015
                                 + 0.803291 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.00832 w_k1_ADM_s015
                                 + 0.767214 w_k1_AXP_s015
                                 + 0.72203 w_k1_DIS_s015
                                 + 0.790366 w_k1_FDX_s015
                                 + 0.855239 w_k1_IBM_s015
                                 + 0.752439 w_k1_INTC_s015
                                 + 0.851659 w_k1_KO_s015
                                 + 0.767054 w_k1_MAT_s015
                                 + 0.839242 w_k1_MSFT_s015
                                 + 0.852197 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.19065 w_k1_ADM_s015
                                 + 1.78497 w_k1_AXP_s015
                                 + 1.21264 w_k1_DIS_s015
                                 + 1.40184 w_k1_FDX_s015
                                 + 1.02526 w_k1_IBM_s015
                                 + 1.05825 w_k1_INTC_s015
                                 + 1.19471 w_k1_KO_s015 + 1.62239 w_k1_MAT_s015
                                 + 0.948209 w_k1_MSFT_s015
                                 + 1.10169 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.00365 w_k1_ADM_s015
                                 + 0.893398 w_k1_AXP_s015
                                 + 0.862153 w_k1_DIS_s015
                                 + 0.993448 w_k1_FDX_s015
                                 + 0.849002 w_k1_IBM_s015
                                 + 0.783913 w_k1_INTC_s015
                                 + 0.893117 w_k1_KO_s015
                                 + 0.942253 w_k1_MAT_s015
                                 + 0.913313 w_k1_MSFT_s015
                                 + 0.951647 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.669907 w_k1_ADM_s015
                                 + 0.641773 w_k1_AXP_s015
                                 + 0.604845 w_k1_DIS_s015
                                 + 0.708994 w_k1_FDX_s015
                                 + 0.59748 w_k1_IBM_s015
                                 + 0.661732 w_k1_INTC_s015
                                 + 0.745224 w_k1_KO_s015
                                 + 0.610396 w_k1_MAT_s015
                                 + 0.743864 w_k1_MSFT_s015
                                 + 0.849745 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.875114 w_k1_ADM_s015
                                 + 0.885072 w_k1_AXP_s015
                                 + 0.848298 w_k1_DIS_s015
                                 + 0.807654 w_k1_FDX_s015
                                 + 1.14518 w_k1_IBM_s015
                                 + 0.948412 w_k1_INTC_s015
                                 + 1.16774 w_k1_KO_s015 + 1.05383 w_k1_MAT_s015
                                 + 0.832704 w_k1_MSFT_s015
                                 + 0.941254 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.06729 w_k1_ADM_s015
                                 + 1.23401 w_k1_AXP_s015
                                 + 1.50847 w_k1_DIS_s015
                                 + 1.26735 w_k1_FDX_s015
                                 + 1.22385 w_k1_IBM_s015
                                 + 1.00742 w_k1_INTC_s015
                                 + 1.38232 w_k1_KO_s015 + 1.62884 w_k1_MAT_s015
                                 + 1.15873 w_k1_MSFT_s015
                                 + 1.24677 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.863166 w_k1_ADM_s015
                                 + 0.793987 w_k1_AXP_s015
                                 + 1.13299 w_k1_DIS_s015
                                 + 1.02345 w_k1_FDX_s015
                                 + 0.920704 w_k1_IBM_s015
                                 + 0.875405 w_k1_INTC_s015
                                 + 0.989328 w_k1_KO_s015
                                 + 0.911822 w_k1_MAT_s015
                                 + 1.08125 w_k1_MSFT_s015
                                 + 0.971337 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.93954 w_k1_ADM_s016
                                 + 0.911004 w_k1_AXP_s016
                                 + 1.07141 w_k1_DIS_s016 + 1.1318 w_k1_FDX_s016
                                 + 0.676427 w_k1_IBM_s016
                                 + 1.0081 w_k1_INTC_s016 + 1.1681 w_k1_KO_s016
                                 + 1.02041 w_k1_MAT_s016
                                 + 1.22418 w_k1_MSFT_s016
                                 + 1.18341 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.09098 w_k1_ADM_s016
                                 + 0.999119 w_k1_AXP_s016
                                 + 0.884227 w_k1_DIS_s016
                                 + 0.898585 w_k1_FDX_s016
                                 + 0.797189 w_k1_IBM_s016
                                 + 0.903975 w_k1_INTC_s016
                                 + 0.935884 w_k1_KO_s016
                                 + 0.89654 w_k1_MAT_s016
                                 + 0.96689 w_k1_MSFT_s016
                                 + 0.880344 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.822542 w_k1_ADM_s016
                                 + 1.36471 w_k1_AXP_s016
                                 + 1.54975 w_k1_DIS_s016
                                 + 1.30847 w_k1_FDX_s016
                                 + 1.03813 w_k1_IBM_s016
                                 + 1.33859 w_k1_INTC_s016
                                 + 1.62208 w_k1_KO_s016 + 1.78997 w_k1_MAT_s016
                                 + 1.39883 w_k1_MSFT_s016
                                 + 1.34081 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.937299 w_k1_ADM_s016
                                 + 1.37017 w_k1_AXP_s016
                                 + 1.23526 w_k1_DIS_s016
                                 + 1.12484 w_k1_FDX_s016
                                 + 1.54977 w_k1_IBM_s016
                                 + 1.07326 w_k1_INTC_s016
                                 + 1.41729 w_k1_KO_s016 + 2.36338 w_k1_MAT_s016
                                 + 1.22168 w_k1_MSFT_s016
                                 + 1.24895 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.960144 w_k1_ADM_s016
                                 + 1.24645 w_k1_AXP_s016
                                 + 1.01203 w_k1_DIS_s016
                                 + 1.06675 w_k1_FDX_s016
                                 + 0.857397 w_k1_IBM_s016
                                 + 1.05452 w_k1_INTC_s016
                                 + 1.02753 w_k1_KO_s016 + 1.16681 w_k1_MAT_s016
                                 + 0.989689 w_k1_MSFT_s016
                                 + 1.01032 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.825046 w_k1_ADM_s016
                                 + 1.01782 w_k1_AXP_s016
                                 + 0.784884 w_k1_DIS_s016
                                 + 0.809181 w_k1_FDX_s016
                                 + 1.18434 w_k1_IBM_s016
                                 + 0.966654 w_k1_INTC_s016
                                 + 1.12993 w_k1_KO_s016 + 1.48672 w_k1_MAT_s016
                                 + 0.879799 w_k1_MSFT_s016
                                 + 0.965492 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.800674 w_k1_ADM_s016
                                 + 0.712332 w_k1_AXP_s016
                                 + 0.670993 w_k1_DIS_s016
                                 + 0.746286 w_k1_FDX_s016
                                 + 0.799444 w_k1_IBM_s016
                                 + 0.697165 w_k1_INTC_s016
                                 + 0.832982 w_k1_KO_s016
                                 + 0.640474 w_k1_MAT_s016
                                 + 0.672647 w_k1_MSFT_s016
                                 + 0.88213 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.834182 w_k1_ADM_s016
                                 + 1.09725 w_k1_AXP_s016
                                 + 1.00769 w_k1_DIS_s016
                                 + 1.00255 w_k1_FDX_s016
                                 + 1.16411 w_k1_IBM_s016
                                 + 1.1952 w_k1_INTC_s016 + 1.24164 w_k1_KO_s016
                                 + 1.11858 w_k1_MAT_s016
                                 + 1.06377 w_k1_MSFT_s016
                                 + 1.02804 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.986521 w_k1_ADM_s017
                                 + 1.54629 w_k1_AXP_s017
                                 + 1.25346 w_k1_DIS_s017
                                 + 1.12382 w_k1_FDX_s017
                                 + 1.45536 w_k1_IBM_s017
                                 + 1.11474 w_k1_INTC_s017
                                 + 1.44475 w_k1_KO_s017 + 1.77199 w_k1_MAT_s017
                                 + 1.22653 w_k1_MSFT_s017
                                 + 1.12163 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.804046 w_k1_ADM_s017
                                 + 0.821183 w_k1_AXP_s017
                                 + 0.479394 w_k1_DIS_s017
                                 + 0.643138 w_k1_FDX_s017
                                 + 0.738209 w_k1_IBM_s017
                                 + 0.809134 w_k1_INTC_s017
                                 + 0.725644 w_k1_KO_s017
                                 + 0.617539 w_k1_MAT_s017
                                 + 0.720771 w_k1_MSFT_s017
                                 + 0.763856 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.92199 w_k1_ADM_s017
                                 + 0.634057 w_k1_AXP_s017
                                 + 0.626573 w_k1_DIS_s017
                                 + 0.830122 w_k1_FDX_s017
                                 + 0.602818 w_k1_IBM_s017
                                 + 0.72856 w_k1_INTC_s017
                                 + 0.806298 w_k1_KO_s017
                                 + 0.607959 w_k1_MAT_s017
                                 + 0.883448 w_k1_MSFT_s017
                                 + 0.795495 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.960029 w_k1_ADM_s017
                                 + 0.87449 w_k1_AXP_s017
                                 + 0.906904 w_k1_DIS_s017
                                 + 0.832674 w_k1_FDX_s017
                                 + 0.94391 w_k1_IBM_s017
                                 + 0.917713 w_k1_INTC_s017
                                 + 1.04519 w_k1_KO_s017
                                 + 0.864341 w_k1_MAT_s017
                                 + 0.917592 w_k1_MSFT_s017
                                 + 0.882323 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.831856 w_k1_ADM_s017
                                 + 1.14191 w_k1_AXP_s017
                                 + 0.917044 w_k1_DIS_s017
                                 + 1.06104 w_k1_FDX_s017
                                 + 0.583213 w_k1_IBM_s017
                                 + 0.975245 w_k1_INTC_s017
                                 + 0.809832 w_k1_KO_s017
                                 + 0.683103 w_k1_MAT_s017
                                 + 1.01398 w_k1_MSFT_s017
                                 + 0.924943 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.06233 w_k1_ADM_s017
                                 + 1.30422 w_k1_AXP_s017
                                 + 1.02748 w_k1_DIS_s017
                                 + 1.08058 w_k1_FDX_s017
                                 + 0.896929 w_k1_IBM_s017
                                 + 1.0307 w_k1_INTC_s017 + 1.09919 w_k1_KO_s017
                                 + 1.06983 w_k1_MAT_s017
                                 + 1.17058 w_k1_MSFT_s017
                                 + 1.02917 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.22813 w_k1_ADM_s017
                                 + 1.40271 w_k1_AXP_s017
                                 + 1.19101 w_k1_DIS_s017
                                 + 1.20568 w_k1_FDX_s017
                                 + 0.672567 w_k1_IBM_s017
                                 + 1.24274 w_k1_INTC_s017
                                 + 1.04765 w_k1_KO_s017 + 1.06065 w_k1_MAT_s017
                                 + 1.19339 w_k1_MSFT_s017
                                 + 0.970468 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.25706 w_k1_ADM_s017
                                 + 1.43451 w_k1_AXP_s017
                                 + 1.31705 w_k1_DIS_s017 + 1.2533 w_k1_FDX_s017
                                 + 1.01005 w_k1_IBM_s017
                                 + 1.45312 w_k1_INTC_s017
                                 + 1.27343 w_k1_KO_s017 + 1.24896 w_k1_MAT_s017
                                 + 1.34292 w_k1_MSFT_s017 + 1.0973 w_k1_PG_s017
                                 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.778891 w_k1_ADM_s100
                                 + 0.708761 w_k1_AXP_s100
                                 + 0.766187 w_k1_DIS_s100
                                 + 0.784024 w_k1_FDX_s100
                                 + 0.684748 w_k1_IBM_s100
                                 + 0.740063 w_k1_INTC_s100
                                 + 0.711744 w_k1_KO_s100
                                 + 0.674128 w_k1_MAT_s100
                                 + 0.764726 w_k1_MSFT_s100
                                 + 0.855401 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.755951 w_k1_ADM_s100
                                 + 0.92918 w_k1_AXP_s100
                                 + 0.710116 w_k1_DIS_s100
                                 + 0.806406 w_k1_FDX_s100
                                 + 0.899463 w_k1_IBM_s100
                                 + 0.889082 w_k1_INTC_s100
                                 + 0.841183 w_k1_KO_s100
                                 + 0.888615 w_k1_MAT_s100
                                 + 0.828917 w_k1_MSFT_s100
                                 + 0.856418 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.942812 w_k1_ADM_s100
                                 + 0.959888 w_k1_AXP_s100
                                 + 1.1071 w_k1_DIS_s100 + 1.02329 w_k1_FDX_s100
                                 + 0.9592 w_k1_IBM_s100
                                 + 1.01746 w_k1_INTC_s100
                                 + 0.942056 w_k1_KO_s100
                                 + 0.859462 w_k1_MAT_s100
                                 + 1.07373 w_k1_MSFT_s100
                                 + 1.08077 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.824286 w_k1_ADM_s100
                                 + 1.00903 w_k1_AXP_s100
                                 + 0.919762 w_k1_DIS_s100
                                 + 1.03416 w_k1_FDX_s100
                                 + 0.720592 w_k1_IBM_s100
                                 + 0.875753 w_k1_INTC_s100
                                 + 0.821598 w_k1_KO_s100
                                 + 0.803793 w_k1_MAT_s100
                                 + 1.03731 w_k1_MSFT_s100
                                 + 1.03276 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.09938 w_k1_ADM_s100
                                 + 1.47047 w_k1_AXP_s100 + 1.3234 w_k1_DIS_s100
                                 + 1.2977 w_k1_FDX_s100
                                 + 0.944344 w_k1_IBM_s100
                                 + 0.953865 w_k1_INTC_s100
                                 + 0.970901 w_k1_KO_s100
                                 + 1.02205 w_k1_MAT_s100
                                 + 1.13264 w_k1_MSFT_s100 + 1.0949 w_k1_PG_s100
                                 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.917485 w_k1_ADM_s100
                                 + 1.11383 w_k1_AXP_s100
                                 + 1.02169 w_k1_DIS_s100
                                 + 0.948049 w_k1_FDX_s100
                                 + 1.19743 w_k1_IBM_s100
                                 + 0.809992 w_k1_INTC_s100
                                 + 0.990258 w_k1_KO_s100
                                 + 1.23523 w_k1_MAT_s100
                                 + 0.879596 w_k1_MSFT_s100
                                 + 1.10982 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.29123 w_k1_ADM_s100
                                 + 1.64636 w_k1_AXP_s100
                                 + 1.87438 w_k1_DIS_s100 + 1.3266 w_k1_FDX_s100
                                 + 1.32804 w_k1_IBM_s100
                                 + 1.16318 w_k1_INTC_s100 + 1.4188 w_k1_KO_s100
                                 + 1.56588 w_k1_MAT_s100
                                 + 1.46183 w_k1_MSFT_s100
                                 + 1.25803 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.822202 w_k1_ADM_s100
                                 + 1.04136 w_k1_AXP_s100
                                 + 0.97392 w_k1_DIS_s100
                                 + 0.923992 w_k1_FDX_s100
                                 + 1.61062 w_k1_IBM_s100
                                 + 0.909969 w_k1_INTC_s100
                                 + 1.23403 w_k1_KO_s100 + 1.41961 w_k1_MAT_s100
                                 + 0.971106 w_k1_MSFT_s100
                                 + 1.10003 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.2877 w_k1_ADM_s101
                                 + 1.13115 w_k1_AXP_s101
                                 + 1.01537 w_k1_DIS_s101
                                 + 1.07905 w_k1_FDX_s101
                                 + 1.13485 w_k1_IBM_s101
                                 + 0.971049 w_k1_INTC_s101
                                 + 1.17314 w_k1_KO_s101 + 1.00862 w_k1_MAT_s101
                                 + 0.985093 w_k1_MSFT_s101
                                 + 1.02297 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.31125 w_k1_ADM_s101
                                 + 1.35199 w_k1_AXP_s101 + 1.4442 w_k1_DIS_s101
                                 + 1.27638 w_k1_FDX_s101
                                 + 0.972101 w_k1_IBM_s101
                                 + 1.04142 w_k1_INTC_s101
                                 + 1.16029 w_k1_KO_s101 + 1.09968 w_k1_MAT_s101
                                 + 1.14359 w_k1_MSFT_s101
                                 + 1.14219 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.15662 w_k1_ADM_s101
                                 + 1.05136 w_k1_AXP_s101
                                 + 0.957105 w_k1_DIS_s101
                                 + 1.0355 w_k1_FDX_s101
                                 + 0.661135 w_k1_IBM_s101
                                 + 1.05483 w_k1_INTC_s101
                                 + 0.863556 w_k1_KO_s101
                                 + 0.77232 w_k1_MAT_s101
                                 + 1.0898 w_k1_MSFT_s101 + 1.02724 w_k1_PG_s101
                                 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.15172 w_k1_ADM_s101
                                 + 1.37189 w_k1_AXP_s101
                                 + 1.49547 w_k1_DIS_s101
                                 + 1.22165 w_k1_FDX_s101 + 1.5834 w_k1_IBM_s101
                                 + 1.1746 w_k1_INTC_s101 + 1.54797 w_k1_KO_s101
                                 + 1.49918 w_k1_MAT_s101
                                 + 1.16183 w_k1_MSFT_s101
                                 + 1.21697 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.942315 w_k1_ADM_s101
                                 + 0.956078 w_k1_AXP_s101
                                 + 1.05754 w_k1_DIS_s101
                                 + 0.836891 w_k1_FDX_s101
                                 + 1.29468 w_k1_IBM_s101
                                 + 0.893549 w_k1_INTC_s101
                                 + 1.16745 w_k1_KO_s101 + 1.40641 w_k1_MAT_s101
                                 + 0.953828 w_k1_MSFT_s101
                                 + 0.968461 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.971702 w_k1_ADM_s101
                                 + 0.852332 w_k1_AXP_s101
                                 + 1.08408 w_k1_DIS_s101
                                 + 0.934596 w_k1_FDX_s101
                                 + 0.905161 w_k1_IBM_s101
                                 + 0.901015 w_k1_INTC_s101
                                 + 1.23018 w_k1_KO_s101
                                 + 0.943886 w_k1_MAT_s101
                                 + 1.01582 w_k1_MSFT_s101
                                 + 1.10563 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.870854 w_k1_ADM_s101
                                 + 0.700449 w_k1_AXP_s101
                                 + 0.672208 w_k1_DIS_s101
                                 + 0.746006 w_k1_FDX_s101
                                 + 0.827144 w_k1_IBM_s101
                                 + 0.711315 w_k1_INTC_s101
                                 + 0.831629 w_k1_KO_s101
                                 + 0.657252 w_k1_MAT_s101
                                 + 0.793856 w_k1_MSFT_s101
                                 + 0.97883 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.00541 w_k1_ADM_s101
                                 + 0.800253 w_k1_AXP_s101
                                 + 0.76672 w_k1_DIS_s101
                                 + 0.918232 w_k1_FDX_s101
                                 + 0.601492 w_k1_IBM_s101
                                 + 0.763661 w_k1_INTC_s101
                                 + 0.755202 w_k1_KO_s101
                                 + 0.702339 w_k1_MAT_s101
                                 + 0.868706 w_k1_MSFT_s101
                                 + 0.943454 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.10979 w_k1_ADM_s102
                                 + 1.37553 w_k1_AXP_s102
                                 + 1.15008 w_k1_DIS_s102
                                 + 1.07058 w_k1_FDX_s102
                                 + 1.37022 w_k1_IBM_s102
                                 + 1.26142 w_k1_INTC_s102
                                 + 1.26096 w_k1_KO_s102 + 2.02435 w_k1_MAT_s102
                                 + 1.1148 w_k1_MSFT_s102 + 1.20106 w_k1_PG_s102
                                 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.00089 w_k1_ADM_s102
                                 + 0.89726 w_k1_AXP_s102 + 1.5966 w_k1_DIS_s102
                                 + 1.12266 w_k1_FDX_s102
                                 + 1.19178 w_k1_IBM_s102
                                 + 1.14976 w_k1_INTC_s102
                                 + 1.20819 w_k1_KO_s102 + 1.33435 w_k1_MAT_s102
                                 + 1.28858 w_k1_MSFT_s102
                                 + 1.18744 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.78857 w_k1_ADM_s102
                                 + 0.695325 w_k1_AXP_s102
                                 + 0.812845 w_k1_DIS_s102
                                 + 0.894892 w_k1_FDX_s102
                                 + 0.8028 w_k1_IBM_s102
                                 + 0.95686 w_k1_INTC_s102
                                 + 0.806824 w_k1_KO_s102
                                 + 0.834421 w_k1_MAT_s102
                                 + 0.871345 w_k1_MSFT_s102
                                 + 0.905164 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.13842 w_k1_ADM_s102
                                 + 1.47226 w_k1_AXP_s102
                                 + 1.56968 w_k1_DIS_s102
                                 + 1.33862 w_k1_FDX_s102 + 1.347 w_k1_IBM_s102
                                 + 1.17125 w_k1_INTC_s102
                                 + 1.27762 w_k1_KO_s102 + 1.99352 w_k1_MAT_s102
                                 + 1.19665 w_k1_MSFT_s102 + 1.0521 w_k1_PG_s102
                                 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.933456 w_k1_ADM_s102
                                 + 1.13424 w_k1_AXP_s102
                                 + 0.904254 w_k1_DIS_s102
                                 + 1.12195 w_k1_FDX_s102
                                 + 0.750471 w_k1_IBM_s102
                                 + 1.22234 w_k1_INTC_s102
                                 + 0.858922 w_k1_KO_s102
                                 + 0.818892 w_k1_MAT_s102
                                 + 1.09591 w_k1_MSFT_s102
                                 + 0.953417 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.843567 w_k1_ADM_s102
                                 + 0.798501 w_k1_AXP_s102
                                 + 0.602775 w_k1_DIS_s102
                                 + 0.726381 w_k1_FDX_s102
                                 + 0.892329 w_k1_IBM_s102
                                 + 0.890577 w_k1_INTC_s102
                                 + 0.702383 w_k1_KO_s102
                                 + 0.780624 w_k1_MAT_s102
                                 + 0.829704 w_k1_MSFT_s102
                                 + 0.831779 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.80023 w_k1_ADM_s102
                                 + 0.792313 w_k1_AXP_s102
                                 + 1.06544 w_k1_DIS_s102
                                 + 0.956358 w_k1_FDX_s102
                                 + 0.960234 w_k1_IBM_s102
                                 + 1.18771 w_k1_INTC_s102 + 1.2213 w_k1_KO_s102
                                 + 1.15386 w_k1_MAT_s102
                                 + 1.14159 w_k1_MSFT_s102
                                 + 1.02431 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.937027 w_k1_ADM_s102
                                 + 1.17649 w_k1_AXP_s102
                                 + 0.908333 w_k1_DIS_s102
                                 + 0.947187 w_k1_FDX_s102
                                 + 1.22655 w_k1_IBM_s102
                                 + 1.08944 w_k1_INTC_s102
                                 + 1.07903 w_k1_KO_s102 + 1.07519 w_k1_MAT_s102
                                 + 0.988486 w_k1_MSFT_s102
                                 + 0.927473 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.1391 w_k1_ADM_s103
                                 + 1.52404 w_k1_AXP_s103 + 1.7011 w_k1_DIS_s103
                                 + 1.35259 w_k1_FDX_s103 + 1.3518 w_k1_IBM_s103
                                 + 1.27875 w_k1_INTC_s103
                                 + 1.46603 w_k1_KO_s103 + 1.54107 w_k1_MAT_s103
                                 + 1.22397 w_k1_MSFT_s103
                                 + 1.14866 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.980057 w_k1_ADM_s103
                                 + 0.734858 w_k1_AXP_s103
                                 + 0.759205 w_k1_DIS_s103
                                 + 0.867093 w_k1_FDX_s103
                                 + 0.553778 w_k1_IBM_s103
                                 + 0.757258 w_k1_INTC_s103
                                 + 0.757669 w_k1_KO_s103
                                 + 0.89617 w_k1_MAT_s103
                                 + 0.887762 w_k1_MSFT_s103
                                 + 0.903916 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.02293 w_k1_ADM_s103
                                 + 0.798816 w_k1_AXP_s103
                                 + 0.770055 w_k1_DIS_s103
                                 + 0.781455 w_k1_FDX_s103
                                 + 0.905887 w_k1_IBM_s103
                                 + 0.878394 w_k1_INTC_s103
                                 + 0.955403 w_k1_KO_s103
                                 + 0.767418 w_k1_MAT_s103
                                 + 0.860892 w_k1_MSFT_s103
                                 + 0.810211 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.16608 w_k1_ADM_s103
                                 + 1.03711 w_k1_AXP_s103
                                 + 0.880466 w_k1_DIS_s103
                                 + 1.09282 w_k1_FDX_s103
                                 + 0.618152 w_k1_IBM_s103
                                 + 1.1539 w_k1_INTC_s103 + 0.82366 w_k1_KO_s103
                                 + 1.0341 w_k1_MAT_s103
                                 + 1.04554 w_k1_MSFT_s103
                                 + 0.871282 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.36267 w_k1_ADM_s103
                                 + 1.34209 w_k1_AXP_s103
                                 + 1.17556 w_k1_DIS_s103
                                 + 1.10768 w_k1_FDX_s103
                                 + 0.997473 w_k1_IBM_s103
                                 + 1.08735 w_k1_INTC_s103
                                 + 1.02383 w_k1_KO_s103 + 1.24781 w_k1_MAT_s103
                                 + 1.04306 w_k1_MSFT_s103
                                 + 0.976882 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.969767 w_k1_ADM_s103
                                 + 0.856672 w_k1_AXP_s103
                                 + 0.951516 w_k1_DIS_s103
                                 + 0.916368 w_k1_FDX_s103
                                 + 1.14879 w_k1_IBM_s103
                                 + 1.00434 w_k1_INTC_s103
                                 + 1.01841 w_k1_KO_s103 + 1.29018 w_k1_MAT_s103
                                 + 1.06977 w_k1_MSFT_s103
                                 + 0.98091 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.18345 w_k1_ADM_s103
                                 + 1.20965 w_k1_AXP_s103 + 1.0601 w_k1_DIS_s103
                                 + 0.985067 w_k1_FDX_s103
                                 + 1.60485 w_k1_IBM_s103
                                 + 0.935265 w_k1_INTC_s103
                                 + 1.16353 w_k1_KO_s103 + 1.88556 w_k1_MAT_s103
                                 + 1.0103 w_k1_MSFT_s103 + 1.12542 w_k1_PG_s103
                                 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.05412 w_k1_ADM_s103
                                 + 1.04648 w_k1_AXP_s103
                                 + 1.31067 w_k1_DIS_s103
                                 + 1.21392 w_k1_FDX_s103
                                 + 0.790419 w_k1_IBM_s103
                                 + 1.05214 w_k1_INTC_s103
                                 + 1.03804 w_k1_KO_s103 + 1.07079 w_k1_MAT_s103
                                 + 1.21779 w_k1_MSFT_s103
                                 + 1.03591 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.05858 w_k1_ADM_s104
                                 + 0.7264 w_k1_AXP_s104
                                 + 0.743036 w_k1_DIS_s104
                                 + 0.861659 w_k1_FDX_s104
                                 + 0.625435 w_k1_IBM_s104
                                 + 1.08132 w_k1_INTC_s104
                                 + 0.783781 w_k1_KO_s104
                                 + 0.493013 w_k1_MAT_s104
                                 + 0.901768 w_k1_MSFT_s104
                                 + 0.883662 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.08253 w_k1_ADM_s104
                                 + 1.3374 w_k1_AXP_s104 + 1.11938 w_k1_DIS_s104
                                 + 1.22703 w_k1_FDX_s104
                                 + 0.730871 w_k1_IBM_s104
                                 + 1.3316 w_k1_INTC_s104 + 1.02754 w_k1_KO_s104
                                 + 0.646069 w_k1_MAT_s104
                                 + 1.08302 w_k1_MSFT_s104
                                 + 1.04011 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.915492 w_k1_ADM_s104
                                 + 0.947748 w_k1_AXP_s104
                                 + 0.872978 w_k1_DIS_s104
                                 + 0.890864 w_k1_FDX_s104
                                 + 1.39796 w_k1_IBM_s104
                                 + 1.27738 w_k1_INTC_s104
                                 + 1.15537 w_k1_KO_s104 + 1.09828 w_k1_MAT_s104
                                 + 0.809993 w_k1_MSFT_s104
                                 + 1.05752 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.999946 w_k1_ADM_s104
                                 + 1.03589 w_k1_AXP_s104
                                 + 0.868231 w_k1_DIS_s104
                                 + 0.907448 w_k1_FDX_s104
                                 + 1.12167 w_k1_IBM_s104
                                 + 1.05477 w_k1_INTC_s104
                                 + 1.06372 w_k1_KO_s104 + 1.08328 w_k1_MAT_s104
                                 + 0.770347 w_k1_MSFT_s104
                                 + 1.00162 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.895264 w_k1_ADM_s104
                                 + 0.652326 w_k1_AXP_s104
                                 + 0.600446 w_k1_DIS_s104
                                 + 0.778634 w_k1_FDX_s104
                                 + 0.804537 w_k1_IBM_s104
                                 + 0.946195 w_k1_INTC_s104
                                 + 0.696638 w_k1_KO_s104
                                 + 0.804835 w_k1_MAT_s104
                                 + 0.696341 w_k1_MSFT_s104
                                 + 0.825175 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.40319 w_k1_ADM_s104
                                 + 1.45754 w_k1_AXP_s104
                                 + 1.74262 w_k1_DIS_s104
                                 + 1.28746 w_k1_FDX_s104
                                 + 1.74005 w_k1_IBM_s104
                                 + 1.6373 w_k1_INTC_s104 + 1.7719 w_k1_KO_s104
                                 + 1.57244 w_k1_MAT_s104
                                 + 1.35305 w_k1_MSFT_s104 + 1.6657 w_k1_PG_s104
                                 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.25634 w_k1_ADM_s104
                                 + 1.00927 w_k1_AXP_s104
                                 + 1.41317 w_k1_DIS_s104
                                 + 1.14444 w_k1_FDX_s104
                                 + 1.00854 w_k1_IBM_s104
                                 + 1.19442 w_k1_INTC_s104 + 1.1516 w_k1_KO_s104
                                 + 0.809096 w_k1_MAT_s104
                                 + 1.1069 w_k1_MSFT_s104 + 1.21741 w_k1_PG_s104
                                 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.42682 w_k1_ADM_s104
                                 + 1.85747 w_k1_AXP_s104
                                 + 1.42528 w_k1_DIS_s104 + 1.5187 w_k1_FDX_s104
                                 + 1.01769 w_k1_IBM_s104
                                 + 1.32884 w_k1_INTC_s104
                                 + 1.24318 w_k1_KO_s104 + 1.41778 w_k1_MAT_s104
                                 + 1.10421 w_k1_MSFT_s104
                                 + 1.31557 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.853803 w_k1_ADM_s105
                                 + 0.831721 w_k1_AXP_s105
                                 + 0.719902 w_k1_DIS_s105
                                 + 0.7785 w_k1_FDX_s105
                                 + 0.949171 w_k1_IBM_s105
                                 + 0.871596 w_k1_INTC_s105
                                 + 0.858901 w_k1_KO_s105
                                 + 1.02388 w_k1_MAT_s105
                                 + 0.790567 w_k1_MSFT_s105
                                 + 0.862917 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.27991 w_k1_ADM_s105
                                 + 1.05193 w_k1_AXP_s105
                                 + 1.15877 w_k1_DIS_s105
                                 + 1.03516 w_k1_FDX_s105
                                 + 1.15511 w_k1_IBM_s105
                                 + 1.13621 w_k1_INTC_s105
                                 + 1.03689 w_k1_KO_s105 + 1.18469 w_k1_MAT_s105
                                 + 1.06091 w_k1_MSFT_s105
                                 + 0.99358 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.0536 w_k1_ADM_s105
                                 + 0.953002 w_k1_AXP_s105
                                 + 1.16731 w_k1_DIS_s105
                                 + 1.16644 w_k1_FDX_s105
                                 + 0.777324 w_k1_IBM_s105
                                 + 0.971265 w_k1_INTC_s105
                                 + 0.848594 w_k1_KO_s105
                                 + 0.971126 w_k1_MAT_s105
                                 + 1.10661 w_k1_MSFT_s105
                                 + 1.02775 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.966326 w_k1_ADM_s105
                                 + 0.843977 w_k1_AXP_s105
                                 + 0.871777 w_k1_DIS_s105
                                 + 0.971305 w_k1_FDX_s105
                                 + 0.682981 w_k1_IBM_s105
                                 + 0.789266 w_k1_INTC_s105
                                 + 0.712644 w_k1_KO_s105
                                 + 0.650386 w_k1_MAT_s105
                                 + 0.806789 w_k1_MSFT_s105
                                 + 0.831924 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.11115 w_k1_ADM_s105
                                 + 1.4884 w_k1_AXP_s105 + 1.30368 w_k1_DIS_s105
                                 + 1.09751 w_k1_FDX_s105
                                 + 1.85347 w_k1_IBM_s105
                                 + 1.04397 w_k1_INTC_s105
                                 + 1.48974 w_k1_KO_s105 + 2.17866 w_k1_MAT_s105
                                 + 1.01326 w_k1_MSFT_s105
                                 + 1.06895 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.2136 w_k1_ADM_s105
                                 + 1.34377 w_k1_AXP_s105
                                 + 1.54428 w_k1_DIS_s105
                                 + 1.21187 w_k1_FDX_s105
                                 + 1.49048 w_k1_IBM_s105
                                 + 1.16421 w_k1_INTC_s105
                                 + 1.34413 w_k1_KO_s105 + 1.40219 w_k1_MAT_s105
                                 + 1.131 w_k1_MSFT_s105 + 1.13714 w_k1_PG_s105
                                 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.27768 w_k1_ADM_s105
                                 + 1.37495 w_k1_AXP_s105
                                 + 1.45095 w_k1_DIS_s105
                                 + 1.29565 w_k1_FDX_s105
                                 + 1.14093 w_k1_IBM_s105
                                 + 1.0815 w_k1_INTC_s105 + 1.23888 w_k1_KO_s105
                                 + 1.33221 w_k1_MAT_s105
                                 + 1.23076 w_k1_MSFT_s105
                                 + 1.11423 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.96401 w_k1_ADM_s105
                                 + 0.9225 w_k1_AXP_s105
                                 + 0.861666 w_k1_DIS_s105
                                 + 0.762515 w_k1_FDX_s105
                                 + 1.40165 w_k1_IBM_s105
                                 + 0.827805 w_k1_INTC_s105
                                 + 1.03698 w_k1_KO_s105 + 1.27714 w_k1_MAT_s105
                                 + 0.866334 w_k1_MSFT_s105
                                 + 0.891694 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.14232 w_k1_ADM_s106
                                 + 1.51758 w_k1_AXP_s106
                                 + 1.58157 w_k1_DIS_s106
                                 + 1.24884 w_k1_FDX_s106
                                 + 1.78987 w_k1_IBM_s106
                                 + 1.42281 w_k1_INTC_s106 + 1.7317 w_k1_KO_s106
                                 + 1.48377 w_k1_MAT_s106
                                 + 1.28335 w_k1_MSFT_s106
                                 + 1.15627 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.06672 w_k1_ADM_s106
                                 + 1.07769 w_k1_AXP_s106
                                 + 0.79845 w_k1_DIS_s106
                                 + 0.997371 w_k1_FDX_s106
                                 + 0.774036 w_k1_IBM_s106
                                 + 1.11487 w_k1_INTC_s106
                                 + 0.927608 w_k1_KO_s106
                                 + 0.696262 w_k1_MAT_s106
                                 + 0.995441 w_k1_MSFT_s106
                                 + 0.850306 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.911416 w_k1_ADM_s106
                                 + 0.972539 w_k1_AXP_s106
                                 + 1.01648 w_k1_DIS_s106
                                 + 0.969887 w_k1_FDX_s106
                                 + 0.68662 w_k1_IBM_s106
                                 + 1.02546 w_k1_INTC_s106
                                 + 1.00287 w_k1_KO_s106
                                 + 0.674782 w_k1_MAT_s106
                                 + 1.10389 w_k1_MSFT_s106
                                 + 0.935875 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.89186 w_k1_ADM_s106
                                 + 1.42448 w_k1_AXP_s106
                                 + 0.894065 w_k1_DIS_s106
                                 + 0.834445 w_k1_FDX_s106
                                 + 1.51646 w_k1_IBM_s106
                                 + 0.955266 w_k1_INTC_s106
                                 + 1.26872 w_k1_KO_s106 + 2.00699 w_k1_MAT_s106
                                 + 0.811872 w_k1_MSFT_s106
                                 + 0.943863 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.11634 w_k1_ADM_s106
                                 + 1.14422 w_k1_AXP_s106
                                 + 1.14851 w_k1_DIS_s106
                                 + 1.05814 w_k1_FDX_s106
                                 + 0.948478 w_k1_IBM_s106
                                 + 1.0785 w_k1_INTC_s106 + 1.14103 w_k1_KO_s106
                                 + 0.995252 w_k1_MAT_s106
                                 + 1.06642 w_k1_MSFT_s106
                                 + 0.969673 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.04843 w_k1_ADM_s106
                                 + 1.48652 w_k1_AXP_s106
                                 + 1.16921 w_k1_DIS_s106
                                 + 1.15669 w_k1_FDX_s106
                                 + 1.02648 w_k1_IBM_s106
                                 + 1.09221 w_k1_INTC_s106
                                 + 1.15217 w_k1_KO_s106 + 1.55909 w_k1_MAT_s106
                                 + 0.972966 w_k1_MSFT_s106
                                 + 1.0318 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.728295 w_k1_ADM_s106
                                 + 0.815683 w_k1_AXP_s106
                                 + 0.651048 w_k1_DIS_s106
                                 + 0.771592 w_k1_FDX_s106
                                 + 0.73841 w_k1_IBM_s106
                                 + 0.863763 w_k1_INTC_s106
                                 + 0.786682 w_k1_KO_s106
                                 + 0.667732 w_k1_MAT_s106
                                 + 0.718449 w_k1_MSFT_s106
                                 + 0.808743 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.20435 w_k1_ADM_s106
                                 + 1.4924 w_k1_AXP_s106 + 1.24267 w_k1_DIS_s106
                                 + 1.20153 w_k1_FDX_s106
                                 + 1.28599 w_k1_IBM_s106
                                 + 1.17955 w_k1_INTC_s106
                                 + 1.49341 w_k1_KO_s106 + 1.22451 w_k1_MAT_s106
                                 + 1.09665 w_k1_MSFT_s106
                                 + 1.11997 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.851543 w_k1_ADM_s107
                                 + 0.781877 w_k1_AXP_s107
                                 + 0.781091 w_k1_DIS_s107
                                 + 0.823813 w_k1_FDX_s107
                                 + 0.970822 w_k1_IBM_s107
                                 + 0.789008 w_k1_INTC_s107
                                 + 0.855246 w_k1_KO_s107 + 0.8727 w_k1_MAT_s107
                                 + 0.703579 w_k1_MSFT_s107
                                 + 0.820831 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.89848 w_k1_ADM_s107
                                 + 1.15853 w_k1_AXP_s107
                                 + 1.13055 w_k1_DIS_s107
                                 + 1.06233 w_k1_FDX_s107
                                 + 1.37392 w_k1_IBM_s107
                                 + 0.998006 w_k1_INTC_s107
                                 + 1.06146 w_k1_KO_s107 + 1.46147 w_k1_MAT_s107
                                 + 0.901685 w_k1_MSFT_s107
                                 + 1.04564 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.727239 w_k1_ADM_s107
                                 + 0.754083 w_k1_AXP_s107
                                 + 1.08718 w_k1_DIS_s107
                                 + 0.960567 w_k1_FDX_s107
                                 + 1.05339 w_k1_IBM_s107
                                 + 0.987764 w_k1_INTC_s107
                                 + 1.14235 w_k1_KO_s107 + 1.11569 w_k1_MAT_s107
                                 + 0.883103 w_k1_MSFT_s107
                                 + 0.981716 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.850579 w_k1_ADM_s107
                                 + 0.985852 w_k1_AXP_s107
                                 + 0.952488 w_k1_DIS_s107
                                 + 1.09596 w_k1_FDX_s107
                                 + 0.798149 w_k1_IBM_s107
                                 + 1.03396 w_k1_INTC_s107
                                 + 0.850042 w_k1_KO_s107
                                 + 0.785486 w_k1_MAT_s107
                                 + 0.973211 w_k1_MSFT_s107
                                 + 0.956947 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.06472 w_k1_ADM_s107
                                 + 1.17075 w_k1_AXP_s107
                                 + 1.52201 w_k1_DIS_s107
                                 + 1.26396 w_k1_FDX_s107
                                 + 1.30095 w_k1_IBM_s107
                                 + 1.11867 w_k1_INTC_s107
                                 + 1.35881 w_k1_KO_s107 + 1.83056 w_k1_MAT_s107
                                 + 1.24654 w_k1_MSFT_s107 + 1.2179 w_k1_PG_s107
                                 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.793183 w_k1_ADM_s107
                                 + 0.70822 w_k1_AXP_s107
                                 + 0.934229 w_k1_DIS_s107
                                 + 1.0125 w_k1_FDX_s107 + 0.62071 w_k1_IBM_s107
                                 + 0.890568 w_k1_INTC_s107
                                 + 0.83154 w_k1_KO_s107
                                 + 0.531289 w_k1_MAT_s107
                                 + 0.861257 w_k1_MSFT_s107
                                 + 0.866778 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.915562 w_k1_ADM_s107
                                 + 0.74596 w_k1_AXP_s107
                                 + 0.595894 w_k1_DIS_s107
                                 + 0.785066 w_k1_FDX_s107
                                 + 0.657225 w_k1_IBM_s107
                                 + 0.747646 w_k1_INTC_s107
                                 + 0.68911 w_k1_KO_s107
                                 + 0.640868 w_k1_MAT_s107
                                 + 0.637126 w_k1_MSFT_s107
                                 + 0.794574 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.32386 w_k1_ADM_s107
                                 + 1.4076 w_k1_AXP_s107 + 1.65252 w_k1_DIS_s107
                                 + 1.48502 w_k1_FDX_s107
                                 + 1.31801 w_k1_IBM_s107
                                 + 1.20565 w_k1_INTC_s107
                                 + 1.22621 w_k1_KO_s107 + 1.32057 w_k1_MAT_s107
                                 + 1.24204 w_k1_MSFT_s107
                                 + 1.12291 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.737455 w_k1_ADM_s110
                                 + 0.62596 w_k1_AXP_s110
                                 + 0.624879 w_k1_DIS_s110
                                 + 0.736331 w_k1_FDX_s110
                                 + 0.673116 w_k1_IBM_s110
                                 + 0.799089 w_k1_INTC_s110
                                 + 0.754149 w_k1_KO_s110
                                 + 0.660582 w_k1_MAT_s110
                                 + 0.732093 w_k1_MSFT_s110
                                 + 0.952971 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.600631 w_k1_ADM_s110
                                 + 0.773473 w_k1_AXP_s110
                                 + 0.669069 w_k1_DIS_s110
                                 + 0.795251 w_k1_FDX_s110
                                 + 1.04016 w_k1_IBM_s110
                                 + 1.00704 w_k1_INTC_s110
                                 + 0.868909 w_k1_KO_s110
                                 + 1.01428 w_k1_MAT_s110
                                 + 0.840088 w_k1_MSFT_s110
                                 + 0.984571 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.0399 w_k1_ADM_s110
                                 + 1.24515 w_k1_AXP_s110
                                 + 1.43546 w_k1_DIS_s110
                                 + 1.21658 w_k1_FDX_s110
                                 + 1.47681 w_k1_IBM_s110
                                 + 1.07749 w_k1_INTC_s110
                                 + 1.28016 w_k1_KO_s110 + 1.27518 w_k1_MAT_s110
                                 + 1.09779 w_k1_MSFT_s110
                                 + 1.26268 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.820865 w_k1_ADM_s110
                                 + 0.799692 w_k1_AXP_s110
                                 + 0.751597 w_k1_DIS_s110
                                 + 0.822568 w_k1_FDX_s110
                                 + 0.903784 w_k1_IBM_s110
                                 + 1.02907 w_k1_INTC_s110
                                 + 0.878543 w_k1_KO_s110
                                 + 0.839678 w_k1_MAT_s110
                                 + 0.973606 w_k1_MSFT_s110
                                 + 0.910953 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.783782 w_k1_ADM_s110
                                 + 0.929042 w_k1_AXP_s110
                                 + 0.91477 w_k1_DIS_s110
                                 + 0.982523 w_k1_FDX_s110
                                 + 1.48236 w_k1_IBM_s110
                                 + 1.1101 w_k1_INTC_s110 + 1.12021 w_k1_KO_s110
                                 + 1.14849 w_k1_MAT_s110
                                 + 1.06163 w_k1_MSFT_s110
                                 + 1.11106 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.976799 w_k1_ADM_s110
                                 + 0.952298 w_k1_AXP_s110
                                 + 1.13882 w_k1_DIS_s110
                                 + 1.05806 w_k1_FDX_s110
                                 + 0.788437 w_k1_IBM_s110
                                 + 1.17909 w_k1_INTC_s110
                                 + 0.974206 w_k1_KO_s110
                                 + 0.540013 w_k1_MAT_s110
                                 + 1.07666 w_k1_MSFT_s110
                                 + 1.09889 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.758822 w_k1_ADM_s110
                                 + 1.21948 w_k1_AXP_s110
                                 + 1.19383 w_k1_DIS_s110
                                 + 1.10555 w_k1_FDX_s110
                                 + 1.08403 w_k1_IBM_s110
                                 + 1.35986 w_k1_INTC_s110
                                 + 1.25217 w_k1_KO_s110 + 1.29331 w_k1_MAT_s110
                                 + 1.32711 w_k1_MSFT_s110
                                 + 1.28302 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.909459 w_k1_ADM_s110
                                 + 1.7128 w_k1_AXP_s110 + 1.64897 w_k1_DIS_s110
                                 + 1.35594 w_k1_FDX_s110
                                 + 1.68573 w_k1_IBM_s110
                                 + 1.19417 w_k1_INTC_s110
                                 + 1.49267 w_k1_KO_s110 + 2.07178 w_k1_MAT_s110
                                 + 1.32279 w_k1_MSFT_s110 + 1.5643 w_k1_PG_s110
                                 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.04145 w_k1_ADM_s111
                                 + 1.39023 w_k1_AXP_s111
                                 + 1.24052 w_k1_DIS_s111
                                 + 1.22937 w_k1_FDX_s111
                                 + 1.34132 w_k1_IBM_s111
                                 + 1.00963 w_k1_INTC_s111
                                 + 1.07532 w_k1_KO_s111 + 1.54652 w_k1_MAT_s111
                                 + 1.17972 w_k1_MSFT_s111 + 1.1678 w_k1_PG_s111
                                 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.659486 w_k1_ADM_s111
                                 + 0.728843 w_k1_AXP_s111
                                 + 0.652588 w_k1_DIS_s111
                                 + 0.722286 w_k1_FDX_s111
                                 + 0.996814 w_k1_IBM_s111
                                 + 0.76209 w_k1_INTC_s111
                                 + 0.653976 w_k1_KO_s111
                                 + 0.962893 w_k1_MAT_s111
                                 + 0.68667 w_k1_MSFT_s111
                                 + 0.90282 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.879887 w_k1_ADM_s111
                                 + 0.892207 w_k1_AXP_s111
                                 + 0.792049 w_k1_DIS_s111
                                 + 0.886385 w_k1_FDX_s111
                                 + 0.77837 w_k1_IBM_s111
                                 + 0.973925 w_k1_INTC_s111
                                 + 0.714554 w_k1_KO_s111
                                 + 0.653778 w_k1_MAT_s111
                                 + 0.901271 w_k1_MSFT_s111
                                 + 0.893322 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.857037 w_k1_ADM_s111
                                 + 0.998845 w_k1_AXP_s111
                                 + 1.00119 w_k1_DIS_s111
                                 + 0.982719 w_k1_FDX_s111
                                 + 1.41665 w_k1_IBM_s111
                                 + 0.994003 w_k1_INTC_s111
                                 + 0.984365 w_k1_KO_s111
                                 + 1.13292 w_k1_MAT_s111
                                 + 0.990889 w_k1_MSFT_s111
                                 + 1.10385 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.01304 w_k1_ADM_s111
                                 + 1.09675 w_k1_AXP_s111
                                 + 1.35502 w_k1_DIS_s111
                                 + 1.19979 w_k1_FDX_s111
                                 + 0.931616 w_k1_IBM_s111
                                 + 1.14444 w_k1_INTC_s111
                                 + 1.07081 w_k1_KO_s111
                                 + 0.877758 w_k1_MAT_s111
                                 + 1.14365 w_k1_MSFT_s111 + 1.1315 w_k1_PG_s111
                                 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.05068 w_k1_ADM_s111
                                 + 0.934227 w_k1_AXP_s111
                                 + 1.0676 w_k1_DIS_s111 + 1.00259 w_k1_FDX_s111
                                 + 0.974037 w_k1_IBM_s111
                                 + 1.04998 w_k1_INTC_s111 + 1.0813 w_k1_KO_s111
                                 + 0.966016 w_k1_MAT_s111
                                 + 1.06639 w_k1_MSFT_s111
                                 + 1.11783 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.0861 w_k1_ADM_s111
                                 + 1.41347 w_k1_AXP_s111 + 1.8923 w_k1_DIS_s111
                                 + 1.3903 w_k1_FDX_s111 + 1.2777 w_k1_IBM_s111
                                 + 1.27959 w_k1_INTC_s111
                                 + 1.34512 w_k1_KO_s111 + 1.59435 w_k1_MAT_s111
                                 + 1.42515 w_k1_MSFT_s111
                                 + 1.28122 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.793888 w_k1_ADM_s111
                                 + 0.892631 w_k1_AXP_s111
                                 + 0.789612 w_k1_DIS_s111
                                 + 0.875163 w_k1_FDX_s111
                                 + 1.02181 w_k1_IBM_s111
                                 + 1.00562 w_k1_INTC_s111
                                 + 0.876366 w_k1_KO_s111
                                 + 0.846059 w_k1_MAT_s111
                                 + 0.97111 w_k1_MSFT_s111
                                 + 0.949777 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.04997 w_k1_ADM_s112
                                 + 0.899792 w_k1_AXP_s112
                                 + 0.931911 w_k1_DIS_s112
                                 + 0.957594 w_k1_FDX_s112
                                 + 0.992636 w_k1_IBM_s112
                                 + 0.924846 w_k1_INTC_s112
                                 + 0.907172 w_k1_KO_s112
                                 + 0.942711 w_k1_MAT_s112
                                 + 0.901306 w_k1_MSFT_s112
                                 + 0.930666 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.784062 w_k1_ADM_s112
                                 + 0.680949 w_k1_AXP_s112
                                 + 0.688648 w_k1_DIS_s112
                                 + 0.742074 w_k1_FDX_s112
                                 + 0.900692 w_k1_IBM_s112
                                 + 0.796866 w_k1_INTC_s112
                                 + 0.850907 w_k1_KO_s112
                                 + 0.793596 w_k1_MAT_s112
                                 + 0.763821 w_k1_MSFT_s112
                                 + 0.987703 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.756752 w_k1_ADM_s112
                                 + 0.860069 w_k1_AXP_s112
                                 + 0.608647 w_k1_DIS_s112
                                 + 0.830221 w_k1_FDX_s112
                                 + 0.773832 w_k1_IBM_s112
                                 + 0.819272 w_k1_INTC_s112
                                 + 0.695452 w_k1_KO_s112
                                 + 0.756404 w_k1_MAT_s112
                                 + 0.717038 w_k1_MSFT_s112
                                 + 0.898101 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.188 w_k1_ADM_s112
                                 + 1.80894 w_k1_AXP_s112
                                 + 1.68159 w_k1_DIS_s112
                                 + 1.39117 w_k1_FDX_s112
                                 + 1.28103 w_k1_IBM_s112
                                 + 1.31161 w_k1_INTC_s112
                                 + 1.24174 w_k1_KO_s112 + 1.38006 w_k1_MAT_s112
                                 + 1.28678 w_k1_MSFT_s112
                                 + 1.06257 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.990425 w_k1_ADM_s112
                                 + 1.12021 w_k1_AXP_s112
                                 + 1.32136 w_k1_DIS_s112
                                 + 1.11557 w_k1_FDX_s112 + 1.5313 w_k1_IBM_s112
                                 + 1.20321 w_k1_INTC_s112
                                 + 1.20086 w_k1_KO_s112 + 1.13641 w_k1_MAT_s112
                                 + 1.11251 w_k1_MSFT_s112 + 1.1659 w_k1_PG_s112
                                 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.19789 w_k1_ADM_s112
                                 + 1.43158 w_k1_AXP_s112
                                 + 1.10559 w_k1_DIS_s112
                                 + 1.08144 w_k1_FDX_s112
                                 + 1.32765 w_k1_IBM_s112
                                 + 1.14374 w_k1_INTC_s112
                                 + 1.14654 w_k1_KO_s112 + 1.49179 w_k1_MAT_s112
                                 + 1.06427 w_k1_MSFT_s112
                                 + 1.24631 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.914417 w_k1_ADM_s112
                                 + 0.976624 w_k1_AXP_s112
                                 + 0.67131 w_k1_DIS_s112
                                 + 0.988939 w_k1_FDX_s112
                                 + 0.707153 w_k1_IBM_s112
                                 + 0.922014 w_k1_INTC_s112
                                 + 0.606144 w_k1_KO_s112
                                 + 0.497579 w_k1_MAT_s112
                                 + 0.741669 w_k1_MSFT_s112
                                 + 0.855199 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.15752 w_k1_ADM_s112
                                 + 1.04456 w_k1_AXP_s112
                                 + 1.17757 w_k1_DIS_s112
                                 + 1.17473 w_k1_FDX_s112
                                 + 0.986246 w_k1_IBM_s112
                                 + 0.922541 w_k1_INTC_s112
                                 + 0.921922 w_k1_KO_s112
                                 + 0.908441 w_k1_MAT_s112
                                 + 1.13979 w_k1_MSFT_s112
                                 + 1.09083 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.813475 w_k1_ADM_s113
                                 + 0.895677 w_k1_AXP_s113
                                 + 0.786779 w_k1_DIS_s113
                                 + 0.952294 w_k1_FDX_s113
                                 + 0.683871 w_k1_IBM_s113
                                 + 1.14578 w_k1_INTC_s113
                                 + 0.695642 w_k1_KO_s113
                                 + 0.548576 w_k1_MAT_s113
                                 + 1.006 w_k1_MSFT_s113 + 0.919367 w_k1_PG_s113
                                 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.16128 w_k1_ADM_s113
                                 + 1.66105 w_k1_AXP_s113 + 1.6612 w_k1_DIS_s113
                                 + 1.45345 w_k1_FDX_s113
                                 + 1.04894 w_k1_IBM_s113
                                 + 1.37117 w_k1_INTC_s113
                                 + 1.19096 w_k1_KO_s113
                                 + 0.963462 w_k1_MAT_s113
                                 + 1.41951 w_k1_MSFT_s113
                                 + 1.16195 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.36123 w_k1_ADM_s113
                                 + 1.26813 w_k1_AXP_s113
                                 + 1.39252 w_k1_DIS_s113
                                 + 0.972109 w_k1_FDX_s113
                                 + 1.88776 w_k1_IBM_s113
                                 + 1.09169 w_k1_INTC_s113
                                 + 1.55754 w_k1_KO_s113 + 1.51944 w_k1_MAT_s113
                                 + 1.08997 w_k1_MSFT_s113
                                 + 1.28793 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.743497 w_k1_ADM_s113
                                 + 0.882934 w_k1_AXP_s113
                                 + 0.796501 w_k1_DIS_s113
                                 + 0.800504 w_k1_FDX_s113
                                 + 0.917854 w_k1_IBM_s113
                                 + 1.05162 w_k1_INTC_s113
                                 + 0.843943 w_k1_KO_s113
                                 + 1.10941 w_k1_MAT_s113
                                 + 0.893333 w_k1_MSFT_s113
                                 + 0.919695 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.03862 w_k1_ADM_s113
                                 + 1.09708 w_k1_AXP_s113 + 1.2103 w_k1_DIS_s113
                                 + 1.25331 w_k1_FDX_s113
                                 + 0.800378 w_k1_IBM_s113
                                 + 1.12848 w_k1_INTC_s113
                                 + 0.764647 w_k1_KO_s113
                                 + 0.825545 w_k1_MAT_s113
                                 + 1.27363 w_k1_MSFT_s113
                                 + 1.01315 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.01345 w_k1_ADM_s113
                                 + 1.04541 w_k1_AXP_s113
                                 + 0.921286 w_k1_DIS_s113
                                 + 0.872288 w_k1_FDX_s113
                                 + 1.32298 w_k1_IBM_s113
                                 + 1.09542 w_k1_INTC_s113
                                 + 1.00111 w_k1_KO_s113 + 1.03093 w_k1_MAT_s113
                                 + 0.927977 w_k1_MSFT_s113
                                 + 1.05124 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.01287 w_k1_ADM_s113
                                 + 1.20776 w_k1_AXP_s113
                                 + 1.19561 w_k1_DIS_s113 + 1.092 w_k1_FDX_s113
                                 + 1.3386 w_k1_IBM_s113
                                 + 1.14397 w_k1_INTC_s113
                                 + 1.11866 w_k1_KO_s113 + 1.03293 w_k1_MAT_s113
                                 + 1.20405 w_k1_MSFT_s113 + 1.1279 w_k1_PG_s113
                                 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.805288 w_k1_ADM_s113
                                 + 0.847964 w_k1_AXP_s113
                                 + 1.1663 w_k1_DIS_s113 + 1.00006 w_k1_FDX_s113
                                 + 0.767192 w_k1_IBM_s113
                                 + 1.28844 w_k1_INTC_s113
                                 + 0.901472 w_k1_KO_s113
                                 + 0.689736 w_k1_MAT_s113
                                 + 1.17283 w_k1_MSFT_s113
                                 + 1.03314 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.973637 w_k1_ADM_s114
                                 + 1.13843 w_k1_AXP_s114
                                 + 0.87652 w_k1_DIS_s114
                                 + 0.94243 w_k1_FDX_s114
                                 + 1.01421 w_k1_IBM_s114
                                 + 0.982673 w_k1_INTC_s114
                                 + 0.963076 w_k1_KO_s114
                                 + 1.50677 w_k1_MAT_s114
                                 + 0.888203 w_k1_MSFT_s114
                                 + 0.973869 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.947169 w_k1_ADM_s114
                                 + 0.87481 w_k1_AXP_s114
                                 + 0.929875 w_k1_DIS_s114
                                 + 1.07505 w_k1_FDX_s114
                                 + 0.751738 w_k1_IBM_s114
                                 + 0.982559 w_k1_INTC_s114
                                 + 0.866907 w_k1_KO_s114
                                 + 0.889604 w_k1_MAT_s114
                                 + 1.00156 w_k1_MSFT_s114
                                 + 0.952567 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.761329 w_k1_ADM_s114
                                 + 0.673279 w_k1_AXP_s114
                                 + 0.625608 w_k1_DIS_s114
                                 + 0.792532 w_k1_FDX_s114
                                 + 0.703675 w_k1_IBM_s114
                                 + 0.956955 w_k1_INTC_s114
                                 + 0.756125 w_k1_KO_s114
                                 + 0.54002 w_k1_MAT_s114
                                 + 0.832792 w_k1_MSFT_s114
                                 + 0.776447 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.703498 w_k1_ADM_s114
                                 + 0.680877 w_k1_AXP_s114
                                 + 0.798918 w_k1_DIS_s114
                                 + 0.825479 w_k1_FDX_s114
                                 + 0.865342 w_k1_IBM_s114
                                 + 0.85917 w_k1_INTC_s114
                                 + 0.830418 w_k1_KO_s114
                                 + 1.01328 w_k1_MAT_s114
                                 + 0.872377 w_k1_MSFT_s114
                                 + 0.873351 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.19585 w_k1_ADM_s114
                                 + 1.35122 w_k1_AXP_s114
                                 + 1.67262 w_k1_DIS_s114
                                 + 1.33616 w_k1_FDX_s114
                                 + 1.22098 w_k1_IBM_s114
                                 + 1.16363 w_k1_INTC_s114
                                 + 1.31681 w_k1_KO_s114 + 1.83907 w_k1_MAT_s114
                                 + 1.18302 w_k1_MSFT_s114
                                 + 1.18447 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.612533 w_k1_ADM_s114
                                 + 0.809138 w_k1_AXP_s114
                                 + 0.615746 w_k1_DIS_s114
                                 + 0.77313 w_k1_FDX_s114
                                 + 0.749468 w_k1_IBM_s114
                                 + 0.672459 w_k1_INTC_s114
                                 + 0.634301 w_k1_KO_s114
                                 + 1.02744 w_k1_MAT_s114
                                 + 0.617565 w_k1_MSFT_s114
                                 + 0.85727 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.748113 w_k1_ADM_s114
                                 + 0.919407 w_k1_AXP_s114
                                 + 0.909265 w_k1_DIS_s114
                                 + 0.924234 w_k1_FDX_s114
                                 + 1.30611 w_k1_IBM_s114
                                 + 0.787129 w_k1_INTC_s114
                                 + 1.0763 w_k1_KO_s114 + 1.64718 w_k1_MAT_s114
                                 + 0.980262 w_k1_MSFT_s114
                                 + 0.98699 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.07044 w_k1_ADM_s114
                                 + 1.17676 w_k1_AXP_s114 + 1.3141 w_k1_DIS_s114
                                 + 1.11641 w_k1_FDX_s114
                                 + 1.26658 w_k1_IBM_s114
                                 + 1.1506 w_k1_INTC_s114 + 1.09084 w_k1_KO_s114
                                 + 1.13695 w_k1_MAT_s114
                                 + 1.04838 w_k1_MSFT_s114
                                 + 1.04026 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.966639 w_k1_ADM_s115
                                 + 0.845663 w_k1_AXP_s115
                                 + 0.919242 w_k1_DIS_s115
                                 + 1.12213 w_k1_FDX_s115
                                 + 0.747488 w_k1_IBM_s115
                                 + 1.21178 w_k1_INTC_s115
                                 + 0.787267 w_k1_KO_s115
                                 + 0.608903 w_k1_MAT_s115
                                 + 1.05527 w_k1_MSFT_s115 + 1.0229 w_k1_PG_s115
                                 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.948369 w_k1_ADM_s115
                                 + 0.911966 w_k1_AXP_s115
                                 + 0.865535 w_k1_DIS_s115
                                 + 0.882248 w_k1_FDX_s115
                                 + 1.06897 w_k1_IBM_s115
                                 + 0.909731 w_k1_INTC_s115
                                 + 1.00202 w_k1_KO_s115
                                 + 0.973607 w_k1_MAT_s115
                                 + 0.766813 w_k1_MSFT_s115
                                 + 0.995999 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.28539 w_k1_ADM_s115
                                 + 1.67828 w_k1_AXP_s115
                                 + 1.42502 w_k1_DIS_s115
                                 + 1.35466 w_k1_FDX_s115
                                 + 1.25262 w_k1_IBM_s115
                                 + 1.21384 w_k1_INTC_s115
                                 + 1.24617 w_k1_KO_s115 + 1.37614 w_k1_MAT_s115
                                 + 1.13336 w_k1_MSFT_s115
                                 + 1.14319 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.970236 w_k1_ADM_s115
                                 + 1.20387 w_k1_AXP_s115
                                 + 1.03753 w_k1_DIS_s115
                                 + 1.13827 w_k1_FDX_s115
                                 + 0.69057 w_k1_IBM_s115
                                 + 1.19041 w_k1_INTC_s115
                                 + 0.877184 w_k1_KO_s115
                                 + 0.625024 w_k1_MAT_s115
                                 + 0.918437 w_k1_MSFT_s115
                                 + 0.956605 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.746887 w_k1_ADM_s115
                                 + 0.782273 w_k1_AXP_s115
                                 + 0.636861 w_k1_DIS_s115
                                 + 0.746513 w_k1_FDX_s115
                                 + 0.730513 w_k1_IBM_s115
                                 + 0.787694 w_k1_INTC_s115
                                 + 0.861641 w_k1_KO_s115
                                 + 0.904779 w_k1_MAT_s115
                                 + 0.713054 w_k1_MSFT_s115
                                 + 0.778274 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.18865 w_k1_ADM_s115
                                 + 1.03414 w_k1_AXP_s115
                                 + 1.21463 w_k1_DIS_s115
                                 + 1.09294 w_k1_FDX_s115
                                 + 1.25008 w_k1_IBM_s115
                                 + 1.03601 w_k1_INTC_s115
                                 + 1.22374 w_k1_KO_s115 + 1.14103 w_k1_MAT_s115
                                 + 0.968818 w_k1_MSFT_s115
                                 + 1.09663 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.14346 w_k1_ADM_s115
                                 + 0.890193 w_k1_AXP_s115
                                 + 0.91835 w_k1_DIS_s115
                                 + 0.941288 w_k1_FDX_s115
                                 + 0.779141 w_k1_IBM_s115
                                 + 1.04914 w_k1_INTC_s115
                                 + 0.94394 w_k1_KO_s115
                                 + 0.684063 w_k1_MAT_s115
                                 + 0.949491 w_k1_MSFT_s115
                                 + 0.910952 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.06607 w_k1_ADM_s115
                                 + 1.02238 w_k1_AXP_s115
                                 + 0.910974 w_k1_DIS_s115
                                 + 0.845176 w_k1_FDX_s115
                                 + 1.58524 w_k1_IBM_s115
                                 + 1.00492 w_k1_INTC_s115
                                 + 1.20115 w_k1_KO_s115 + 1.1982 w_k1_MAT_s115
                                 + 0.877195 w_k1_MSFT_s115
                                 + 0.986058 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.208 w_k1_ADM_s116
                                 + 1.0168 w_k1_AXP_s116
                                 + 0.880996 w_k1_DIS_s116
                                 + 0.924571 w_k1_FDX_s116
                                 + 1.03334 w_k1_IBM_s116
                                 + 0.975456 w_k1_INTC_s116
                                 + 1.01127 w_k1_KO_s116
                                 + 0.921433 w_k1_MAT_s116
                                 + 0.992744 w_k1_MSFT_s116
                                 + 1.07995 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.981411 w_k1_ADM_s116
                                 + 0.770745 w_k1_AXP_s116
                                 + 0.926611 w_k1_DIS_s116
                                 + 1.01166 w_k1_FDX_s116
                                 + 0.74383 w_k1_IBM_s116
                                 + 1.05026 w_k1_INTC_s116
                                 + 0.911189 w_k1_KO_s116
                                 + 0.640108 w_k1_MAT_s116
                                 + 1.01962 w_k1_MSFT_s116
                                 + 1.00976 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.15451 w_k1_ADM_s116
                                 + 1.26645 w_k1_AXP_s116 + 1.4656 w_k1_DIS_s116
                                 + 1.22307 w_k1_FDX_s116
                                 + 1.47255 w_k1_IBM_s116
                                 + 1.18748 w_k1_INTC_s116
                                 + 1.29165 w_k1_KO_s116 + 1.32994 w_k1_MAT_s116
                                 + 1.20155 w_k1_MSFT_s116 + 1.2421 w_k1_PG_s116
                                 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.07878 w_k1_ADM_s116
                                 + 0.97802 w_k1_AXP_s116
                                 + 1.05337 w_k1_DIS_s116
                                 + 1.05715 w_k1_FDX_s116
                                 + 0.856372 w_k1_IBM_s116
                                 + 1.03384 w_k1_INTC_s116
                                 + 0.877636 w_k1_KO_s116
                                 + 0.735264 w_k1_MAT_s116
                                 + 1.0182 w_k1_MSFT_s116 + 1.04386 w_k1_PG_s116
                                 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.866141 w_k1_ADM_s116
                                 + 1.02259 w_k1_AXP_s116
                                 + 0.990873 w_k1_DIS_s116
                                 + 0.951977 w_k1_FDX_s116
                                 + 1.45761 w_k1_IBM_s116
                                 + 0.940377 w_k1_INTC_s116
                                 + 1.22936 w_k1_KO_s116 + 1.3211 w_k1_MAT_s116
                                 + 0.934044 w_k1_MSFT_s116
                                 + 1.09413 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.33177 w_k1_ADM_s116
                                 + 1.20749 w_k1_AXP_s116
                                 + 1.40777 w_k1_DIS_s116
                                 + 1.18166 w_k1_FDX_s116
                                 + 0.987732 w_k1_IBM_s116
                                 + 1.18598 w_k1_INTC_s116
                                 + 1.15858 w_k1_KO_s116
                                 + 0.807697 w_k1_MAT_s116
                                 + 1.13586 w_k1_MSFT_s116
                                 + 1.23005 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.959937 w_k1_ADM_s116
                                 + 1.00181 w_k1_AXP_s116
                                 + 0.673161 w_k1_DIS_s116
                                 + 0.81795 w_k1_FDX_s116
                                 + 1.12786 w_k1_IBM_s116
                                 + 0.839528 w_k1_INTC_s116
                                 + 0.915688 w_k1_KO_s116
                                 + 1.09155 w_k1_MAT_s116
                                 + 0.793705 w_k1_MSFT_s116
                                 + 1.06274 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.820185 w_k1_ADM_s116
                                 + 0.788054 w_k1_AXP_s116
                                 + 0.657565 w_k1_DIS_s116
                                 + 0.784317 w_k1_FDX_s116
                                 + 0.745264 w_k1_IBM_s116
                                 + 0.933743 w_k1_INTC_s116
                                 + 0.727207 w_k1_KO_s116
                                 + 0.672763 w_k1_MAT_s116
                                 + 0.862393 w_k1_MSFT_s116
                                 + 0.808612 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.2973 w_k1_ADM_s117
                                 + 1.2269 w_k1_AXP_s117 + 1.22125 w_k1_DIS_s117
                                 + 1.04167 w_k1_FDX_s117
                                 + 1.34745 w_k1_IBM_s117
                                 + 1.25363 w_k1_INTC_s117
                                 + 1.54213 w_k1_KO_s117 + 1.29371 w_k1_MAT_s117
                                 + 1.12402 w_k1_MSFT_s117
                                 + 1.03157 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.32445 w_k1_ADM_s117
                                 + 1.2474 w_k1_AXP_s117 + 1.15519 w_k1_DIS_s117
                                 + 1.28677 w_k1_FDX_s117
                                 + 0.76214 w_k1_IBM_s117
                                 + 1.39766 w_k1_INTC_s117
                                 + 1.10015 w_k1_KO_s117
                                 + 0.724925 w_k1_MAT_s117
                                 + 1.21269 w_k1_MSFT_s117
                                 + 0.975809 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.729195 w_k1_ADM_s117
                                 + 0.69752 w_k1_AXP_s117
                                 + 0.508085 w_k1_DIS_s117
                                 + 0.683067 w_k1_FDX_s117
                                 + 0.851338 w_k1_IBM_s117
                                 + 0.877634 w_k1_INTC_s117
                                 + 0.745918 w_k1_KO_s117
                                 + 1.10118 w_k1_MAT_s117
                                 + 0.593094 w_k1_MSFT_s117
                                 + 0.797226 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.966413 w_k1_ADM_s117
                                 + 0.831142 w_k1_AXP_s117
                                 + 0.790787 w_k1_DIS_s117
                                 + 0.928274 w_k1_FDX_s117
                                 + 0.662904 w_k1_IBM_s117
                                 + 0.945677 w_k1_INTC_s117
                                 + 0.844939 w_k1_KO_s117
                                 + 0.73475 w_k1_MAT_s117
                                 + 0.913781 w_k1_MSFT_s117
                                 + 0.810754 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.197 w_k1_ADM_s117
                                 + 1.66196 w_k1_AXP_s117
                                 + 1.65004 w_k1_DIS_s117 + 1.473 w_k1_FDX_s117
                                 + 1.14485 w_k1_IBM_s117
                                 + 1.76248 w_k1_INTC_s117 + 1.51 w_k1_KO_s117
                                 + 1.27781 w_k1_MAT_s117
                                 + 1.41769 w_k1_MSFT_s117
                                 + 1.29167 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.17196 w_k1_ADM_s117
                                 + 1.05635 w_k1_AXP_s117
                                 + 0.848557 w_k1_DIS_s117
                                 + 1.03337 w_k1_FDX_s117
                                 + 0.695002 w_k1_IBM_s117
                                 + 1.08255 w_k1_INTC_s117
                                 + 0.966941 w_k1_KO_s117
                                 + 0.655197 w_k1_MAT_s117
                                 + 0.806556 w_k1_MSFT_s117
                                 + 0.976278 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.904586 w_k1_ADM_s117
                                 + 0.903674 w_k1_AXP_s117
                                 + 0.778114 w_k1_DIS_s117
                                 + 0.812494 w_k1_FDX_s117
                                 + 1.09342 w_k1_IBM_s117
                                 + 0.984402 w_k1_INTC_s117
                                 + 1.18848 w_k1_KO_s117 + 1.25888 w_k1_MAT_s117
                                 + 0.852165 w_k1_MSFT_s117
                                 + 0.842007 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.895361 w_k1_ADM_s117
                                 + 0.77099 w_k1_AXP_s117
                                 + 0.975993 w_k1_DIS_s117
                                 + 0.893175 w_k1_FDX_s117
                                 + 0.929528 w_k1_IBM_s117
                                 + 1.11092 w_k1_INTC_s117
                                 + 1.02237 w_k1_KO_s117
                                 + 0.933988 w_k1_MAT_s117
                                 + 0.923944 w_k1_MSFT_s117
                                 + 0.862487 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.00970204647213468 w_k1_ADM_s000
                                 + 0.0269624641943049 w_k1_ADM_s001
                                 + 0.0289916051476721 w_k1_ADM_s002
                                 + 0.0319915362998594 w_k1_ADM_s003
                                 + 0.0113493399488577 w_k1_ADM_s004
                                 + 0.0185464020347692 w_k1_ADM_s005
                                 + 0.014732673754631 w_k1_ADM_s006
                                 + 0.0172531087816216 w_k1_ADM_s007
                                 + 0.0147629297153031 w_k1_ADM_s010
                                 + 0.0064829744526937 w_k1_ADM_s011
                                 + 0.00776144271211706 w_k1_ADM_s012
                                 + 0.0098382497575289 w_k1_ADM_s013
                                 + 0.0179350642480388 w_k1_ADM_s014
                                 + 0.0194092302964475 w_k1_ADM_s015
                                 + 0.00807175009372691 w_k1_ADM_s016
                                 + 0.00473359493732539 w_k1_ADM_s017
                                 + 0.117926654205021 w_k1_ADM_s100
                                 + 0.0358582215655296 w_k1_ADM_s101
                                 + 0.0271298984350664 w_k1_ADM_s102
                                 + 0.0451312839998942 w_k1_ADM_s103
                                 + 0.0370750346112162 w_k1_ADM_s104
                                 + 0.0742637478955463 w_k1_ADM_s105
                                 + 0.0553298240452725 w_k1_ADM_s106
                                 + 0.0278044625718324 w_k1_ADM_s107
                                 + 0.0376385905879804 w_k1_ADM_s110
                                 + 0.0430663370542199 w_k1_ADM_s111
                                 + 0.0270916800234076 w_k1_ADM_s112
                                 + 0.0298354282952773 w_k1_ADM_s113
                                 + 0.0300150583767799 w_k1_ADM_s114
                                 + 0.0570927496291655 w_k1_ADM_s115
                                 + 0.0431068659077343 w_k1_ADM_s116
                                 + 0.0617149156269048 w_k1_ADM_s117
                                 + 0.0104578717146086 w_k1_AXP_s000
                                 + 0.0240958377509395 w_k1_AXP_s001
                                 + 0.033275846885492 w_k1_AXP_s002
                                 + 0.0298913185000606 w_k1_AXP_s003
                                 + 0.0119742766517759 w_k1_AXP_s004
                                 + 0.018238444966306 w_k1_AXP_s005
                                 + 0.0159023245791539 w_k1_AXP_s006
                                 + 0.0139962130255857 w_k1_AXP_s007
                                 + 0.0133796941330327 w_k1_AXP_s010
                                 + 0.00789437655884058 w_k1_AXP_s011
                                 + 0.00867183060326592 w_k1_AXP_s012
                                 + 0.0113415688333531 w_k1_AXP_s013
                                 + 0.0184680777762929 w_k1_AXP_s014
                                 + 0.0199900225686253 w_k1_AXP_s015
                                 + 0.00958491477496781 w_k1_AXP_s016
                                 + 0.00566515462758427 w_k1_AXP_s017
                                 + 0.139029337361939 w_k1_AXP_s100
                                 + 0.0331580758878508 w_k1_AXP_s101
                                 + 0.0299284782672666 w_k1_AXP_s102
                                 + 0.0422374284287723 w_k1_AXP_s103
                                 + 0.0344179560768891 w_k1_AXP_s104
                                 + 0.0703660254558957 w_k1_AXP_s105
                                 + 0.0662342515020491 w_k1_AXP_s106
                                 + 0.03047391317439 w_k1_AXP_s107
                                 + 0.0460605038564982 w_k1_AXP_s110
                                 + 0.0471528688288473 w_k1_AXP_s111
                                 + 0.0274170241460633 w_k1_AXP_s112
                                 + 0.0338314418566923 w_k1_AXP_s113
                                 + 0.0323618910105288 w_k1_AXP_s114
                                 + 0.0568790172339152 w_k1_AXP_s115
                                 + 0.0408448005209429 w_k1_AXP_s116
                                 + 0.0597134111292079 w_k1_AXP_s117
                                 + 0.0102263583544903 w_k1_DIS_s000
                                 + 0.0266442885203717 w_k1_DIS_s001
                                 + 0.035170275885974 w_k1_DIS_s002
                                 + 0.0325213852073805 w_k1_DIS_s003
                                 + 0.0131165775449286 w_k1_DIS_s004
                                 + 0.0193926354951222 w_k1_DIS_s005
                                 + 0.0165919903976892 w_k1_DIS_s006
                                 + 0.0160941277539645 w_k1_DIS_s007
                                 + 0.0140788798804304 w_k1_DIS_s010
                                 + 0.00761595618323959 w_k1_DIS_s011
                                 + 0.00814258198904031 w_k1_DIS_s012
                                 + 0.0106281208954491 w_k1_DIS_s013
                                 + 0.0171065438834161 w_k1_DIS_s014
                                 + 0.0203781576043199 w_k1_DIS_s015
                                 + 0.00865469738208834 w_k1_DIS_s016
                                 + 0.00481441721967212 w_k1_DIS_s017
                                 + 0.13776081170429 w_k1_DIS_s100
                                 + 0.0337416168174377 w_k1_DIS_s101
                                 + 0.0297377577588172 w_k1_DIS_s102
                                 + 0.0434616794525566 w_k1_DIS_s103
                                 + 0.034523668124752 w_k1_DIS_s104
                                 + 0.0724765735598591 w_k1_DIS_s105
                                 + 0.0559816087926424 w_k1_DIS_s106
                                 + 0.0323898014531081 w_k1_DIS_s107
                                 + 0.047470633870297 w_k1_DIS_s110
                                 + 0.049043718871764 w_k1_DIS_s111
                                 + 0.0265497442152554 w_k1_DIS_s112
                                 + 0.0340832768235612 w_k1_DIS_s113
                                 + 0.0340946108613494 w_k1_DIS_s114
                                 + 0.0545713706706511 w_k1_DIS_s115
                                 + 0.0406171882414993 w_k1_DIS_s116
                                 + 0.056801582685649 w_k1_DIS_s117
                                 + 0.00997198685035764 w_k1_FDX_s000
                                 + 0.0268708698216155 w_k1_FDX_s001
                                 + 0.0329342973214066 w_k1_FDX_s002
                                 + 0.0316328477896083 w_k1_FDX_s003
                                 + 0.0137436240208597 w_k1_FDX_s004
                                 + 0.0188283628898021 w_k1_FDX_s005
                                 + 0.0163009949359184 w_k1_FDX_s006
                                 + 0.0155577577841841 w_k1_FDX_s007
                                 + 0.0139057760156225 w_k1_FDX_s010
                                 + 0.00708196378493903 w_k1_FDX_s011
                                 + 0.00835070250025571 w_k1_FDX_s012
                                 + 0.0106475985247319 w_k1_FDX_s013
                                 + 0.016417878281358 w_k1_FDX_s014
                                 + 0.0201672958628361 w_k1_FDX_s015
                                 + 0.00867761966236703 w_k1_FDX_s016
                                 + 0.00479725024295162 w_k1_FDX_s017
                                 + 0.127980285828409 w_k1_FDX_s100
                                 + 0.0325798079635686 w_k1_FDX_s101
                                 + 0.0292307557602677 w_k1_FDX_s102
                                 + 0.0415894004808166 w_k1_FDX_s103
                                 + 0.0342088859028468 w_k1_FDX_s104
                                 + 0.0681047864634783 w_k1_FDX_s105
                                 + 0.0555716296482406 w_k1_FDX_s106
                                 + 0.0324917911442416 w_k1_FDX_s107
                                 + 0.0455946828855515 w_k1_FDX_s110
                                 + 0.047219585511706 w_k1_FDX_s111
                                 + 0.0269470500391633 w_k1_FDX_s112
                                 + 0.0325874478235809 w_k1_FDX_s113
                                 + 0.0334153042399056 w_k1_FDX_s114
                                 + 0.0544809386119691 w_k1_FDX_s115
                                 + 0.0396886825499411 w_k1_FDX_s116
                                 + 0.0564912851503873 w_k1_FDX_s117
                                 + 0.00986413199320627 w_k1_IBM_s000
                                 + 0.0268115544504922 w_k1_IBM_s001
                                 + 0.0329843015678044 w_k1_IBM_s002
                                 + 0.0299300838769316 w_k1_IBM_s003
                                 + 0.0136144978790548 w_k1_IBM_s004
                                 + 0.0198873676171474 w_k1_IBM_s005
                                 + 0.0153071744219877 w_k1_IBM_s006
                                 + 0.0150144694984302 w_k1_IBM_s007
                                 + 0.0133376563016061 w_k1_IBM_s010
                                 + 0.0072826465251262 w_k1_IBM_s011
                                 + 0.00810178731644454 w_k1_IBM_s012
                                 + 0.0107478574468065 w_k1_IBM_s013
                                 + 0.0182113978872475 w_k1_IBM_s014
                                 + 0.0196127698176883 w_k1_IBM_s015
                                 + 0.00881327727185377 w_k1_IBM_s016
                                 + 0.0045133274497585 w_k1_IBM_s017
                                 + 0.131088960522737 w_k1_IBM_s100
                                 + 0.0327763311136965 w_k1_IBM_s101
                                 + 0.0304330238526552 w_k1_IBM_s102
                                 + 0.0431705230732186 w_k1_IBM_s103
                                 + 0.0349133807090103 w_k1_IBM_s104
                                 + 0.0750778887023407 w_k1_IBM_s105
                                 + 0.0569400456710815 w_k1_IBM_s106
                                 + 0.0317187489540653 w_k1_IBM_s107
                                 + 0.0470481496119103 w_k1_IBM_s110
                                 + 0.0483174262050907 w_k1_IBM_s111
                                 + 0.0277437589440904 w_k1_IBM_s112
                                 + 0.0314318187099608 w_k1_IBM_s113
                                 + 0.0341422791048822 w_k1_IBM_s114
                                 + 0.0557370820134183 w_k1_IBM_s115
                                 + 0.0413751735189888 w_k1_IBM_s116
                                 + 0.0561167798394081 w_k1_IBM_s117
                                 + 0.00970068496390886 w_k1_INTC_s000
                                 + 0.0263387607318148 w_k1_INTC_s001
                                 + 0.0385341914163771 w_k1_INTC_s002
                                 + 0.035350120095736 w_k1_INTC_s003
                                 + 0.0126214895235666 w_k1_INTC_s004
                                 + 0.0196320194368724 w_k1_INTC_s005
                                 + 0.0157752423083675 w_k1_INTC_s006
                                 + 0.0141681593395308 w_k1_INTC_s007
                                 + 0.0135901982306801 w_k1_INTC_s010
                                 + 0.00764393417832073 w_k1_INTC_s011
                                 + 0.00931556944963256 w_k1_INTC_s012
                                 + 0.0108428612753458 w_k1_INTC_s013
                                 + 0.0187464176517809 w_k1_INTC_s014
                                 + 0.0180703977919499 w_k1_INTC_s015
                                 + 0.00893072347117713 w_k1_INTC_s016
                                 + 0.00495757407031409 w_k1_INTC_s017
                                 + 0.115670958966214 w_k1_INTC_s100
                                 + 0.0308484553180625 w_k1_INTC_s101
                                 + 0.0319329732187583 w_k1_INTC_s102
                                 + 0.0418210727689137 w_k1_INTC_s103
                                 + 0.0395740946645304 w_k1_INTC_s104
                                 + 0.0671353577578609 w_k1_INTC_s105
                                 + 0.0583373176289151 w_k1_INTC_s106
                                 + 0.0301258503693228 w_k1_INTC_s107
                                 + 0.051168819466677 w_k1_INTC_s110
                                 + 0.0473531268354129 w_k1_INTC_s111
                                 + 0.0259561975986997 w_k1_INTC_s112
                                 + 0.0362112156645382 w_k1_INTC_s113
                                 + 0.0318874884736203 w_k1_INTC_s114
                                 + 0.055729502671891 w_k1_INTC_s115
                                 + 0.0407763722965113 w_k1_INTC_s116
                                 + 0.0653726928088578 w_k1_INTC_s117
                                 + 0.0092843955105176 w_k1_KO_s000
                                 + 0.0264314275897189 w_k1_KO_s001
                                 + 0.0378022249041063 w_k1_KO_s002
                                 + 0.0304861215890145 w_k1_KO_s003
                                 + 0.0137901488647764 w_k1_KO_s004
                                 + 0.0211641051520833 w_k1_KO_s005
                                 + 0.0152758297945742 w_k1_KO_s006
                                 + 0.015882249250359 w_k1_KO_s007
                                 + 0.0143566998951841 w_k1_KO_s010
                                 + 0.00798394405274461 w_k1_KO_s011
                                 + 0.00716954698744791 w_k1_KO_s012
                                 + 0.0108410536799204 w_k1_KO_s013
                                 + 0.019018267974268 w_k1_KO_s014
                                 + 0.0214837456080312 w_k1_KO_s015
                                 + 0.00991341109876842 w_k1_KO_s016
                                 + 0.0051940314424409 w_k1_KO_s017
                                 + 0.124374684525424 w_k1_KO_s100
                                 + 0.0356065568378731 w_k1_KO_s101
                                 + 0.0295948987697444 w_k1_KO_s102
                                 + 0.0435475139395605 w_k1_KO_s103
                                 + 0.0355623081828932 w_k1_KO_s104
                                 + 0.0685726914825278 w_k1_KO_s105
                                 + 0.0626278546578303 w_k1_KO_s106
                                 + 0.0300883220222919 w_k1_KO_s107
                                 + 0.0480188362900866 w_k1_KO_s110
                                 + 0.0441647557487172 w_k1_KO_s111
                                 + 0.024979914728735 w_k1_KO_s112
                                 + 0.0295223758453524 w_k1_KO_s113
                                 + 0.0321987889656171 w_k1_KO_s114
                                 + 0.0562626107805657 w_k1_KO_s115
                                 + 0.0403323871706018 w_k1_KO_s116
                                 + 0.066490022834492 w_k1_KO_s117
                                 + 0.00977575184849377 w_k1_MAT_s000
                                 + 0.0214228705055731 w_k1_MAT_s001
                                 + 0.0329086940322664 w_k1_MAT_s002
                                 + 0.0300139426651247 w_k1_MAT_s003
                                 + 0.0192960497119854 w_k1_MAT_s004
                                 + 0.0188253845151845 w_k1_MAT_s005
                                 + 0.0160894390032507 w_k1_MAT_s006
                                 + 0.0153429790293684 w_k1_MAT_s007
                                 + 0.0140296793903667 w_k1_MAT_s010
                                 + 0.0083659569724345 w_k1_MAT_s011
                                 + 0.0087196107050485 w_k1_MAT_s012
                                 + 0.0116386701264653 w_k1_MAT_s013
                                 + 0.0144712235877422 w_k1_MAT_s014
                                 + 0.0213025226995846 w_k1_MAT_s015
                                 + 0.0110498403415917 w_k1_MAT_s016
                                 + 0.00524148557456924 w_k1_MAT_s017
                                 + 0.133768060136541 w_k1_MAT_s100
                                 + 0.0325406796350049 w_k1_MAT_s101
                                 + 0.034575657645586 w_k1_MAT_s102
                                 + 0.0491033161119768 w_k1_MAT_s103
                                 + 0.0325694388251187 w_k1_MAT_s104
                                 + 0.0789033389210506 w_k1_MAT_s105
                                 + 0.0608464712369752 w_k1_MAT_s106
                                 + 0.0330042492859874 w_k1_MAT_s107
                                 + 0.0456956815586212 w_k1_MAT_s110
                                 + 0.0464364517466078 w_k1_MAT_s111
                                 + 0.0257973668902942 w_k1_MAT_s112
                                 + 0.0270659795393948 w_k1_MAT_s113
                                 + 0.039620550810555 w_k1_MAT_s114
                                 + 0.0519774804831089 w_k1_MAT_s115
                                 + 0.0367225157461468 w_k1_MAT_s116
                                 + 0.0580775260956766 w_k1_MAT_s117
                                 + 0.00934351238050959 w_k1_MSFT_s000
                                 + 0.0283771648795511 w_k1_MSFT_s001
                                 + 0.0329776057407048 w_k1_MSFT_s002
                                 + 0.0301223076530162 w_k1_MSFT_s003
                                 + 0.0137125313617292 w_k1_MSFT_s004
                                 + 0.0187200918199204 w_k1_MSFT_s005
                                 + 0.0146896122650407 w_k1_MSFT_s006
                                 + 0.0160629131344964 w_k1_MSFT_s007
                                 + 0.0132907958804453 w_k1_MSFT_s010
                                 + 0.00787122894280107 w_k1_MSFT_s011
                                 + 0.00847503853491187 w_k1_MSFT_s012
                                 + 0.01103131890853 w_k1_MSFT_s013
                                 + 0.0176645869086171 w_k1_MSFT_s014
                                 + 0.0191621047276144 w_k1_MSFT_s015
                                 + 0.00892526234304666 w_k1_MSFT_s016
                                 + 0.00513160361726841 w_k1_MSFT_s017
                                 + 0.127987299535801 w_k1_MSFT_s100
                                 + 0.0328538415545332 w_k1_MSFT_s101
                                 + 0.0302102463008639 w_k1_MSFT_s102
                                 + 0.0431952589167949 w_k1_MSFT_s103
                                 + 0.0313861839135344 w_k1_MSFT_s104
                                 + 0.0666240425204794 w_k1_MSFT_s105
                                 + 0.0534812937820338 w_k1_MSFT_s106
                                 + 0.0284212094436076 w_k1_MSFT_s107
                                 + 0.0488855701602432 w_k1_MSFT_s110
                                 + 0.0477102441861302 w_k1_MSFT_s111
                                 + 0.0254369160003564 w_k1_MSFT_s112
                                 + 0.0346984219412756 w_k1_MSFT_s113
                                 + 0.0319124636940467 w_k1_MSFT_s114
                                 + 0.0492646887180534 w_k1_MSFT_s115
                                 + 0.0398266304274123 w_k1_MSFT_s116
                                 + 0.0559255130716157 w_k1_MSFT_s117
                                 + 0.00996941222122316 w_k1_PG_s000
                                 + 0.0255623677357776 w_k1_PG_s001
                                 + 0.0320405067784277 w_k1_PG_s002
                                 + 0.0295461879068631 w_k1_PG_s003
                                 + 0.0145826332643917 w_k1_PG_s004
                                 + 0.0188828771717073 w_k1_PG_s005
                                 + 0.0147933551127785 w_k1_PG_s006
                                 + 0.0164532752805576 w_k1_PG_s007
                                 + 0.0131329787952838 w_k1_PG_s010
                                 + 0.00725778083085599 w_k1_PG_s011
                                 + 0.00739178074605144 w_k1_PG_s012
                                 + 0.0111721877123027 w_k1_PG_s013
                                 + 0.0177770038066714 w_k1_PG_s014
                                 + 0.0198901691596826 w_k1_PG_s015
                                 + 0.0090900293537832 w_k1_PG_s016
                                 + 0.00457146599442816 w_k1_PG_s017
                                 + 0.133912393641746 w_k1_PG_s100
                                 + 0.0347042450755915 w_k1_PG_s101
                                 + 0.0286836138612964 w_k1_PG_s102
                                 + 0.0405017932542941 w_k1_PG_s103
                                 + 0.0365752760237772 w_k1_PG_s104
                                 + 0.0657415515808851 w_k1_PG_s105
                                 + 0.0526386960751681 w_k1_PG_s106
                                 + 0.0300248557005061 w_k1_PG_s107
                                 + 0.0514226872524141 w_k1_PG_s110
                                 + 0.0487014455249873 w_k1_PG_s111
                                 + 0.0270684070351204 w_k1_PG_s112
                                 + 0.0322339268296206 w_k1_PG_s113
                                 + 0.0320766575038789 w_k1_PG_s114
                                 + 0.053237012798576 w_k1_PG_s115
                                 + 0.0428769106886776 w_k1_PG_s116
                                 + 0.0529382262112415 w_k1_PG_s117 >= 1.05
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
