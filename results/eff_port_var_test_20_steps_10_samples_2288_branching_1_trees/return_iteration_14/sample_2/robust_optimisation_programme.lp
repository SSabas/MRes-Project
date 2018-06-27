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
 w_k1_ADM_s0_return_constraint:  1.15235 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.943692 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.790216 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.02191 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.00816 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.824769 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.33486 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.48295 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.06244 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.18952 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.35646 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.917764 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.23161 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.884464 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.918587 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.897209 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.694083 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.23 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.25827 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.32136 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.05234 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.950422 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.19401 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.885066 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.01286 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.51791 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.3444 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.02806 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.948589 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.19315 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.25172 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.860232 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.05665 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.57905 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.738628 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.44789 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.10282 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.40008 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.26916 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.896496 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.733928 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.08168 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.1231 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.843619 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.99406 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.48669 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.1473 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.36719 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.43471 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.835801 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.05429 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.83995 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.901195 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.78514 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.62844 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.42685 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.11304 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.31267 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.12069 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.01721 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.07616 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.750821 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.16183 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.72023 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.63967 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.05449 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.737462 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.18142 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.17631 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.805448 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.892397 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.39231 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.548132 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.59016 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.891787 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.50961 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.34266 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.858299 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.760382 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.15525 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.21576 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.814358 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.01105 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.76375 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.10251 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.43552 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.21199 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.773021 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.826831 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.867472 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.885944 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.72014 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.588723 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.52943 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.22757 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.34877 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.993607 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.805499 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.08062 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.726024 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.03935 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.65458 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.65376 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.917762 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.636434 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.31481 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.21777 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.617486 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.972669 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.45185 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.580015 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.76909 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.855985 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.13628 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.20376 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.919046 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.855458 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.06644 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.07904 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.933666 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.07946 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.47633 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.21343 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.22686 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.12649 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.76923 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.989691 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.894419 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.930064 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.845598 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.638152 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.23763 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.09176 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.38004 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.83075 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.845146 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.12531 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.751074 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.983946 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.28481 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.50612 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.839425 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.784709 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.17683 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.11707 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.766633 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.06002 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.42406 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.598324 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.51584 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.915486 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.15923 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.22045 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.937865 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.646121 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.11406 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.18969 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.713227 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.858202 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.25209 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.67403 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.07507 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.61189 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.31406 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.06881 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.853168 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.92197 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.664078 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.970042 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.55521 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.14682 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.855407 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.79864 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.14221 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.788201 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.01744 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.25686 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.89699 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.1204 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.67466 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.644622 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.11661 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.30738 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.993508 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.723213 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.834757 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.12029 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.35323 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.835832 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.17616 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.12522 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.960817 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.906891 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.07298 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.02171 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.916077 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.01382 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.24733 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.16528 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.30156 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.17853 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.795115 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.05103 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.968956 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.929306 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.807334 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.770932 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.28016 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.10278 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.997783 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.986053 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.906949 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.10578 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.850907 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.09375 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.21766 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.3295 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.03033 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.761378 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.11168 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.15591 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.865494 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.02013 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.46893 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.716985 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.33379 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.759965 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.0786 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.19431 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.942124 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.735703 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.06779 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.04865 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.759367 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.914592 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.50513 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.902984 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.19639 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.41239 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.91003 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.07774 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.827668 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.04373 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.821684 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.841536 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.39861 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.33815 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.14107 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.37387 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.09753 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.953106 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.01617 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.20227 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.64484 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.41973 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.27764 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.7141 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.23587 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.285 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.87147 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.908988 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.12787 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.845173 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.53098 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.841122 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.10516 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.24647 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.870066 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.742109 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.08062 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.2323 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.70314 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.613033 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.16019 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.606093 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.01145 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.22668 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.995704 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.05993 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.671768 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.972548 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.788011 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.970282 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.50344 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.15871 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.19194 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 2.25122 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.38843 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.82336 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.05283 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.51901 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 2.05305 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.54099 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.73948 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.669442 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.20037 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.36857 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.05749 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.853585 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.03079 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.01282 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.58222 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.899224 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.34123 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.19818 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.924398 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.838281 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.01211 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.07135 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.924149 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.15102 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.54952 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.13749 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.22723 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.23517 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.734278 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.0499 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.976709 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.940218 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.86066 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.71922 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.18708 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.16159 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.31194 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.942385 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.917751 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.06804 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.807135 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.06821 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.26692 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.31982 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.900146 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.812645 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.21293 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.12711 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.821519 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.05529 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.35682 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.651071 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.25976 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.907548 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.06589 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.14926 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.945251 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.866666 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.04555 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.10686 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.865191 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.934105 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.28764 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.96719 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.998374 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.20828 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.905469 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.97922 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.814541 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.970685 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.879552 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.830559 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.24065 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.11019 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.16964 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.1196 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.936196 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.980756 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.930829 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.05919 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.2734 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.19199 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.00543 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.819629 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.03082 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.08625 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.878613 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.956581 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.19579 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.784861 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.28873 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.01289 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.14363 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.0720001580629199 z_k1_s0000
                                 - 0.13200028978202 z_k1_s0001
                                 - 0.0280000221355264 z_k1_s0002
                                 - 0.0540000595471096 z_k1_s0003
                                 - 0.12700003297176 z_k1_s0004
                                 - 0.0649997985285562 z_k1_s0005
                                 - 0.079999978958532 z_k1_s0006
                                 - 0.0320000505834932 z_k1_s0007
                                 - 0.0530000797414204 z_k1_s0010
                                 - 0.1080000342804 z_k1_s0011
                                 - 0.150000473834655 z_k1_s0012
                                 - 0.033000071363565 z_k1_s0013
                                 - 0.0390000793569315 z_k1_s0014
                                 - 0.0600000799336649 z_k1_s0015
                                 - 0.0579998763355146 z_k1_s0016
                                 - 0.0470000717480539 z_k1_s0017
                                 - 0.0730003465295597 z_k1_s0020
                                 - 0.0470002322437343 z_k1_s0021
                                 - 0.124000294254235 z_k1_s0022
                                 - 0.173000643589722 z_k1_s0023
                                 - 0.0940007608889345 z_k1_s0024
                                 - 0.0890004681595521 z_k1_s0025
                                 - 0.0410001773699413 z_k1_s0026
                                 - 0.100000667561994 z_k1_s0027
                                 - 0.0429998725667506 z_k1_s0030
                                 - 0.0320000282358146 z_k1_s0031
                                 - 0.0389999095373044 z_k1_s0032
                                 - 0.04500003408161 z_k1_s0033
                                 - 0.0389999095373044 z_k1_s0034
                                 - 0.0519998793830725 z_k1_s0035
                                 - 0.0630000477142539 z_k1_s0036
                                 - 0.0470001955964694 z_k1_s0037
                                 - 0.0740000497310369 z_k1_s0040
                                 - 0.0170000114246977 z_k1_s0041
                                 - 0.0170000114246977 z_k1_s0042
                                 - 0.0490000329300109 z_k1_s0043
                                 - 0.0620000416665444 z_k1_s0044
                                 - 0.0550000369622572 z_k1_s0045
                                 - 0.0540000362902161 z_k1_s0046
                                 - 0.0230000505570026 z_k1_s0047
                                 - 0.12599988837923 z_k1_s0050
                                 - 0.0969998858697556 z_k1_s0051
                                 - 0.0839999255861536 z_k1_s0052
                                 - 0.247999930701814 z_k1_s0053
                                 - 0.114999851123952 z_k1_s0054
                                 - 0.285999389438958 z_k1_s0055
                                 - 0.278999593039898 z_k1_s0056
                                 - 0.0809999752437399 z_k1_s0057
                                 - 0.167000947493641 z_k1_s0060
                                 - 0.176000845268394 z_k1_s0061
                                 - 0.0700001394835663 z_k1_s0062
                                 - 0.0990002164253737 z_k1_s0063
                                 - 0.11600030776723 z_k1_s0064
                                 - 0.23000023191691 z_k1_s0065
                                 - 0.290000769418074 z_k1_s0066
                                 - 0.0710002093920597 z_k1_s0067
                                 - 0.0930003075422025 z_k1_s0070
                                 - 0.140000783483763 z_k1_s0071
                                 - 0.104000267272791 z_k1_s0072
                                 - 0.180999855786553 z_k1_s0073
                                 - 0.231000968747322 z_k1_s0074
                                 - 0.191001115181547 z_k1_s0075
                                 - 0.158000976789072 z_k1_s0076
                                 - 0.198000830354848 z_k1_s0077
                                 - 0.406998438602097 z_k1_s0100
                                 - 0.699000159239022 z_k1_s0101
                                 - 0.23500002840369 z_k1_s0102
                                 - 0.231999905717747 z_k1_s0103
                                 - 0.302999934686511 z_k1_s0104
                                 - 0.310999942449186 z_k1_s0105
                                 - 0.390999700676101 z_k1_s0106
                                 - 0.616000917125842 z_k1_s0107
                                 - 0.71400034334822 z_k1_s0110
                                 - 0.838997856141182 z_k1_s0111
                                 - 1.4229976132276 z_k1_s0112
                                 - 0.529999426393126 z_k1_s0113
                                 - 0.513999790083586 z_k1_s0114
                                 - 0.483999480454097 z_k1_s0115
                                 - 0.614998415011725 z_k1_s0116
                                 - 0.546999677396436 z_k1_s0117
                                 - 0.149999821492046 z_k1_s0120
                                 - 0.654998828135421 z_k1_s0121
                                 - 0.422999977699385 z_k1_s0122
                                 - 0.248999816272754 z_k1_s0123
                                 - 0.299001121569575 z_k1_s0124
                                 - 0.376999411458064 z_k1_s0125
                                 - 0.200999904103287 z_k1_s0126
                                 - 0.204999875459084 z_k1_s0127
                                 - 0.44300059295985 z_k1_s0130
                                 - 0.503001253778396 z_k1_s0131
                                 - 0.18399995114334 z_k1_s0132
                                 - 0.337998759777329 z_k1_s0133
                                 - 0.281000162249907 z_k1_s0134
                                 - 0.460000554608416 z_k1_s0135
                                 - 0.327999100807615 z_k1_s0136
                                 - 0.169999887185688 z_k1_s0137
                                 - 0.748002176629299 z_k1_s0140
                                 - 0.301000254353983 z_k1_s0141
                                 - 0.336000302864929 z_k1_s0142
                                 - 0.504999885742881 z_k1_s0143
                                 - 0.878998938555605 z_k1_s0144
                                 - 0.506998748633857 z_k1_s0145
                                 - 0.33800038705706 z_k1_s0146
                                 - 0.456998679332505 z_k1_s0147
                                 - 0.412999676982821 z_k1_s0150
                                 - 0.99400075227651 z_k1_s0151
                                 - 0.3849997857975 z_k1_s0152
                                 - 0.682997968384467 z_k1_s0153
                                 - 0.507999930475553 z_k1_s0154
                                 - 0.389999583223545 z_k1_s0155
                                 - 0.724001435275392 z_k1_s0156
                                 - 1.03099925322924 z_k1_s0157
                                 - 0.422000348211649 z_k1_s0160
                                 - 0.0989999114189308 z_k1_s0161
                                 - 0.467999619453109 z_k1_s0162
                                 - 0.151999921296714 z_k1_s0163
                                 - 0.227000331689607 z_k1_s0164
                                 - 0.198999898343066 z_k1_s0165
                                 - 0.28400043348831 z_k1_s0166
                                 - 0.249999967310337 z_k1_s0167
                                 - 0.382000147575225 z_k1_s0170
                                 - 0.23100030344981 z_k1_s0171
                                 - 0.296999091293403 z_k1_s0172
                                 - 0.338999559738571 z_k1_s0173
                                 - 0.460000368116648 z_k1_s0174
                                 - 0.195999988845456 z_k1_s0175
                                 - 0.203999807478034 z_k1_s0176
                                 - 0.163999805115699 z_k1_s0177
                                 - 0.858001167036926 z_k1_s1000
                                 - 1.47600330101391 z_k1_s1001
                                 - 0.669001394981341 z_k1_s1002
                                 - 1.2670053879396 z_k1_s1003
                                 - 0.71300086202841 z_k1_s1004
                                 - 1.35400485958354 z_k1_s1005
                                 - 1.89399912716253 z_k1_s1006
                                 - 0.661001249372383 z_k1_s1007
                                 - 2.30200202134233 z_k1_s1010
                                 - 1.5250006531307 z_k1_s1011
                                 - 1.61299522312058 z_k1_s1012
                                 - 0.474000597812474 z_k1_s1013
                                 - 0.511999802126458 z_k1_s1014
                                 - 1.91300149717977 z_k1_s1015
                                 - 1.38200095689619 z_k1_s1016
                                 - 1.63899826607266 z_k1_s1017
                                 - 0.403001404597247 z_k1_s1020
                                 - 0.57800155827601 z_k1_s1021
                                 - 0.388998846383271 z_k1_s1022
                                 - 0.349999953757572 z_k1_s1023
                                 - 0.60399822696014 z_k1_s1024
                                 - 0.454001126092242 z_k1_s1025
                                 - 0.589999204745713 z_k1_s1026
                                 - 0.16799993537164 z_k1_s1027
                                 - 0.347999875490812 z_k1_s1030
                                 - 0.550001158966125 z_k1_s1031
                                 - 0.222000189009807 z_k1_s1032
                                 - 0.195000003442306 z_k1_s1033
                                 - 0.417000664252378 z_k1_s1034
                                 - 0.189000119471839 z_k1_s1035
                                 - 0.156000049933872 z_k1_s1036
                                 - 0.281999736414877 z_k1_s1037
                                 - 0.70400148497112 z_k1_s1040
                                 - 0.643000453519159 z_k1_s1041
                                 - 1.18799855995162 z_k1_s1042
                                 - 0.445000693527222 z_k1_s1043
                                 - 0.428998076982658 z_k1_s1044
                                 - 0.254000078998713 z_k1_s1045
                                 - 0.290000073624516 z_k1_s1046
                                 - 0.25599977471684 z_k1_s1047
                                 - 0.54800040323029 z_k1_s1050
                                 - 0.384999968087103 z_k1_s1051
                                 - 0.18099993796045 z_k1_s1052
                                 - 0.252000004438821 z_k1_s1053
                                 - 0.519000941124808 z_k1_s1054
                                 - 0.349999464443029 z_k1_s1055
                                 - 0.378998926548511 z_k1_s1056
                                 - 0.172000047251875 z_k1_s1057
                                 - 0.871000728649464 z_k1_s1060
                                 - 0.635000193463781 z_k1_s1061
                                 - 1.61900358135255 z_k1_s1062
                                 - 1.89699997652183 z_k1_s1063
                                 - 1.19400306812449 z_k1_s1064
                                 - 0.946001184084146 z_k1_s1065
                                 - 1.08799883278168 z_k1_s1066
                                 - 0.611000154056776 z_k1_s1067
                                 - 1.17500055759822 z_k1_s1070
                                 - 0.273000061727061 z_k1_s1071
                                 - 0.551999564318336 z_k1_s1072
                                 - 0.331999876559608 z_k1_s1073
                                 - 0.447000241192691 z_k1_s1074
                                 - 0.340999948656119 z_k1_s1075
                                 - 0.705000334558932 z_k1_s1076
                                 - 0.728999919616167 z_k1_s1077
                                 - 0.495999073839891 z_k1_s1100
                                 - 0.304998795316953 z_k1_s1101
                                 - 0.156000009991539 z_k1_s1102
                                 - 0.238999874120389 z_k1_s1103
                                 - 0.275999974785035 z_k1_s1104
                                 - 0.302999799583595 z_k1_s1105
                                 - 0.526000737838042 z_k1_s1106
                                 - 0.486999411040334 z_k1_s1107
                                 - 0.187000353674204 z_k1_s1110
                                 - 0.413999885500027 z_k1_s1111
                                 - 0.374002100150197 z_k1_s1112
                                 - 0.511001566032419 z_k1_s1113
                                 - 0.513999571881309 z_k1_s1114
                                 - 0.397000739676238 z_k1_s1115
                                 - 0.628000398237491 z_k1_s1116
                                 - 0.456999158703219 z_k1_s1117
                                 - 0.444998670288132 z_k1_s1120
                                 - 0.192999524952889 z_k1_s1121
                                 - 0.233999613289205 z_k1_s1122
                                 - 0.754999174733712 z_k1_s1123
                                 - 0.32599949275414 z_k1_s1124
                                 - 0.249999606896126 z_k1_s1125
                                 - 0.86199978123519 z_k1_s1126
                                 - 0.285999676286563 z_k1_s1127
                                 - 0.121999999601819 z_k1_s1130
                                 - 0.0849999774291277 z_k1_s1131
                                 - 0.217999607914577 z_k1_s1132
                                 - 0.221000455055927 z_k1_s1133
                                 - 0.174999553559817 z_k1_s1134
                                 - 0.0940001012522598 z_k1_s1135
                                 - 0.265000791791138 z_k1_s1136
                                 - 0.330999785893379 z_k1_s1137
                                 - 0.17399983797704 z_k1_s1140
                                 - 0.135000081137469 z_k1_s1141
                                 - 0.112000006347349 z_k1_s1142
                                 - 0.299000274771186 z_k1_s1143
                                 - 0.119000120087869 z_k1_s1144
                                 - 0.360000867366932 z_k1_s1145
                                 - 0.120000056622186 z_k1_s1146
                                 - 0.0760000591107305 z_k1_s1147
                                 - 0.394999985061072 z_k1_s1150
                                 - 0.133000011972468 z_k1_s1151
                                 - 0.165000083023008 z_k1_s1152
                                 - 0.133000011972468 z_k1_s1153
                                 - 0.0830000478685451 z_k1_s1154
                                 - 0.389000916176893 z_k1_s1155
                                 - 0.11300009349092 z_k1_s1156
                                 - 0.268000385173206 z_k1_s1157
                                 - 0.434000441788566 z_k1_s1160
                                 - 0.38700103247829 z_k1_s1161
                                 - 0.325000609465114 z_k1_s1162
                                 - 0.482000374553031 z_k1_s1163
                                 - 0.743000233897638 z_k1_s1164
                                 - 0.359000411881393 z_k1_s1165
                                 - 0.286000349181029 z_k1_s1166
                                 - 0.582999258347677 z_k1_s1167
                                 - 0.286000188369232 z_k1_s1170
                                 - 0.058000023339221 z_k1_s1171
                                 - 0.234999963702071 z_k1_s1172
                                 - 0.153000674001482 z_k1_s1173
                                 - 0.21000003215927 z_k1_s1174
                                 - 0.249999315428005 z_k1_s1175
                                 - 0.131999911785873 z_k1_s1176
                                 - 0.194000318851776 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.15864 w_k1_ADM_s000
                                 + 1.14803 w_k1_AXP_s000
                                 + 0.953512 w_k1_DIS_s000
                                 + 1.08587 w_k1_FDX_s000
                                 + 0.869284 w_k1_IBM_s000
                                 + 1.09487 w_k1_INTC_s000
                                 + 1.16178 w_k1_KO_s000 + 1.39387 w_k1_MAT_s000
                                 + 0.998902 w_k1_MSFT_s000
                                 + 1.12713 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.05266 w_k1_ADM_s000
                                 + 1.04994 w_k1_AXP_s000
                                 + 1.33731 w_k1_DIS_s000 + 1.2393 w_k1_FDX_s000
                                 + 1.07772 w_k1_IBM_s000
                                 + 1.19262 w_k1_INTC_s000
                                 + 1.38896 w_k1_KO_s000 + 1.40791 w_k1_MAT_s000
                                 + 1.21437 w_k1_MSFT_s000
                                 + 1.23085 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.992573 w_k1_ADM_s000
                                 + 1.46005 w_k1_AXP_s000
                                 + 1.63592 w_k1_DIS_s000
                                 + 1.28792 w_k1_FDX_s000
                                 + 1.80204 w_k1_IBM_s000
                                 + 1.38163 w_k1_INTC_s000
                                 + 1.83426 w_k1_KO_s000 + 1.89841 w_k1_MAT_s000
                                 + 1.3914 w_k1_MSFT_s000 + 1.47251 w_k1_PG_s000
                                 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.926841 w_k1_ADM_s000
                                 + 0.694238 w_k1_AXP_s000
                                 + 0.497412 w_k1_DIS_s000
                                 + 0.716764 w_k1_FDX_s000
                                 + 0.753055 w_k1_IBM_s000
                                 + 0.819282 w_k1_INTC_s000
                                 + 0.636623 w_k1_KO_s000
                                 + 0.618454 w_k1_MAT_s000
                                 + 0.647393 w_k1_MSFT_s000
                                 + 0.804698 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.89299 w_k1_ADM_s000
                                 + 1.03744 w_k1_AXP_s000
                                 + 1.08479 w_k1_DIS_s000
                                 + 1.01441 w_k1_FDX_s000
                                 + 1.21066 w_k1_IBM_s000
                                 + 0.972363 w_k1_INTC_s000
                                 + 1.3357 w_k1_KO_s000 + 1.53943 w_k1_MAT_s000
                                 + 1.05054 w_k1_MSFT_s000
                                 + 1.16686 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.755219 w_k1_ADM_s000
                                 + 0.689231 w_k1_AXP_s000
                                 + 0.729486 w_k1_DIS_s000
                                 + 0.836101 w_k1_FDX_s000
                                 + 0.882764 w_k1_IBM_s000
                                 + 0.839119 w_k1_INTC_s000
                                 + 0.85337 w_k1_KO_s000 + 1.00338 w_k1_MAT_s000
                                 + 0.838247 w_k1_MSFT_s000 + 0.901 w_k1_PG_s000
                                 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.814322 w_k1_ADM_s000
                                 + 0.763998 w_k1_AXP_s000
                                 + 0.877892 w_k1_DIS_s000
                                 + 0.92732 w_k1_FDX_s000
                                 + 0.83342 w_k1_IBM_s000
                                 + 0.920282 w_k1_INTC_s000
                                 + 0.976737 w_k1_KO_s000 + 1.1113 w_k1_MAT_s000
                                 + 1.00862 w_k1_MSFT_s000
                                 + 0.981659 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.755778 w_k1_ADM_s000
                                 + 0.720179 w_k1_AXP_s000
                                 + 0.727255 w_k1_DIS_s000
                                 + 0.940601 w_k1_FDX_s000
                                 + 0.764359 w_k1_IBM_s000
                                 + 0.853904 w_k1_INTC_s000
                                 + 0.854308 w_k1_KO_s000
                                 + 1.16522 w_k1_MAT_s000
                                 + 0.762737 w_k1_MSFT_s000
                                 + 0.946197 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.951637 w_k1_ADM_s001
                                 + 0.76112 w_k1_AXP_s001
                                 + 0.842853 w_k1_DIS_s001
                                 + 0.815063 w_k1_FDX_s001
                                 + 0.889626 w_k1_IBM_s001
                                 + 1.17659 w_k1_INTC_s001
                                 + 0.903988 w_k1_KO_s001
                                 + 0.763661 w_k1_MAT_s001
                                 + 0.897083 w_k1_MSFT_s001
                                 + 0.862832 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.980783 w_k1_ADM_s001
                                 + 1.09464 w_k1_AXP_s001
                                 + 0.882559 w_k1_DIS_s001
                                 + 1.02895 w_k1_FDX_s001
                                 + 1.18254 w_k1_IBM_s001
                                 + 0.906407 w_k1_INTC_s001
                                 + 1.00429 w_k1_KO_s001 + 1.44968 w_k1_MAT_s001
                                 + 0.845865 w_k1_MSFT_s001
                                 + 1.07481 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.18649 w_k1_ADM_s001
                                 + 1.39486 w_k1_AXP_s001
                                 + 1.29665 w_k1_DIS_s001
                                 + 1.13729 w_k1_FDX_s001
                                 + 1.01798 w_k1_IBM_s001
                                 + 1.36409 w_k1_INTC_s001
                                 + 1.14325 w_k1_KO_s001 + 1.14741 w_k1_MAT_s001
                                 + 1.0107 w_k1_MSFT_s001 + 1.05296 w_k1_PG_s001
                                 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.96557 w_k1_ADM_s001
                                 + 0.646944 w_k1_AXP_s001
                                 + 0.533341 w_k1_DIS_s001
                                 + 0.678657 w_k1_FDX_s001
                                 + 0.78982 w_k1_IBM_s001
                                 + 0.733236 w_k1_INTC_s001
                                 + 0.743283 w_k1_KO_s001
                                 + 0.561276 w_k1_MAT_s001
                                 + 0.688567 w_k1_MSFT_s001
                                 + 0.880654 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.871141 w_k1_ADM_s001
                                 + 0.869641 w_k1_AXP_s001
                                 + 0.54374 w_k1_DIS_s001
                                 + 0.784445 w_k1_FDX_s001
                                 + 1.06192 w_k1_IBM_s001
                                 + 0.965712 w_k1_INTC_s001
                                 + 0.708333 w_k1_KO_s001
                                 + 0.961327 w_k1_MAT_s001
                                 + 0.615539 w_k1_MSFT_s001
                                 + 0.828008 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.917601 w_k1_ADM_s001
                                 + 0.959556 w_k1_AXP_s001
                                 + 0.867752 w_k1_DIS_s001
                                 + 0.853002 w_k1_FDX_s001
                                 + 0.827541 w_k1_IBM_s001
                                 + 1.02404 w_k1_INTC_s001
                                 + 0.772751 w_k1_KO_s001 + 1.0593 w_k1_MAT_s001
                                 + 0.843021 w_k1_MSFT_s001
                                 + 0.893476 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.09568 w_k1_ADM_s001
                                 + 0.937577 w_k1_AXP_s001
                                 + 0.724711 w_k1_DIS_s001
                                 + 0.98308 w_k1_FDX_s001
                                 + 0.560702 w_k1_IBM_s001
                                 + 1.02184 w_k1_INTC_s001
                                 + 0.654157 w_k1_KO_s001
                                 + 0.538684 w_k1_MAT_s001
                                 + 0.798552 w_k1_MSFT_s001
                                 + 0.876005 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.17943 w_k1_ADM_s001
                                 + 1.48389 w_k1_AXP_s001
                                 + 1.50745 w_k1_DIS_s001
                                 + 1.53765 w_k1_FDX_s001
                                 + 0.940357 w_k1_IBM_s001
                                 + 1.31035 w_k1_INTC_s001
                                 + 1.00786 w_k1_KO_s001 + 1.17949 w_k1_MAT_s001
                                 + 1.24072 w_k1_MSFT_s001
                                 + 1.08293 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.0597 w_k1_ADM_s002
                                 + 1.20354 w_k1_AXP_s002
                                 + 0.916373 w_k1_DIS_s002
                                 + 0.903975 w_k1_FDX_s002
                                 + 1.34215 w_k1_IBM_s002
                                 + 1.0743 w_k1_INTC_s002 + 1.07441 w_k1_KO_s002
                                 + 1.1696 w_k1_MAT_s002
                                 + 0.973511 w_k1_MSFT_s002
                                 + 1.0287 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.12028 w_k1_ADM_s002
                                 + 1.07124 w_k1_AXP_s002
                                 + 1.04261 w_k1_DIS_s002
                                 + 1.19053 w_k1_FDX_s002
                                 + 0.848551 w_k1_IBM_s002
                                 + 0.979795 w_k1_INTC_s002
                                 + 1.06472 w_k1_KO_s002 + 1.62395 w_k1_MAT_s002
                                 + 0.951776 w_k1_MSFT_s002
                                 + 1.08936 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.27026 w_k1_ADM_s002
                                 + 1.03483 w_k1_AXP_s002
                                 + 1.05003 w_k1_DIS_s002
                                 + 1.05892 w_k1_FDX_s002
                                 + 1.16841 w_k1_IBM_s002
                                 + 0.99525 w_k1_INTC_s002
                                 + 0.996055 w_k1_KO_s002
                                 + 0.854522 w_k1_MAT_s002
                                 + 0.977845 w_k1_MSFT_s002
                                 + 0.954197 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.938694 w_k1_ADM_s002
                                 + 0.625372 w_k1_AXP_s002
                                 + 0.632308 w_k1_DIS_s002
                                 + 0.763542 w_k1_FDX_s002
                                 + 0.639718 w_k1_IBM_s002
                                 + 0.708639 w_k1_INTC_s002
                                 + 0.79127 w_k1_KO_s002
                                 + 0.756438 w_k1_MAT_s002
                                 + 0.877757 w_k1_MSFT_s002
                                 + 0.92682 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.64053 w_k1_ADM_s002
                                 + 1.46126 w_k1_AXP_s002
                                 + 1.29664 w_k1_DIS_s002
                                 + 1.39678 w_k1_FDX_s002
                                 + 1.13326 w_k1_IBM_s002
                                 + 1.36285 w_k1_INTC_s002
                                 + 1.36219 w_k1_KO_s002 + 1.02602 w_k1_MAT_s002
                                 + 1.28133 w_k1_MSFT_s002
                                 + 1.12559 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.06764 w_k1_ADM_s002
                                 + 0.779636 w_k1_AXP_s002
                                 + 0.8411 w_k1_DIS_s002
                                 + 0.885527 w_k1_FDX_s002
                                 + 0.925477 w_k1_IBM_s002
                                 + 0.932871 w_k1_INTC_s002
                                 + 0.950482 w_k1_KO_s002
                                 + 0.951251 w_k1_MAT_s002
                                 + 0.983004 w_k1_MSFT_s002
                                 + 0.962799 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.989873 w_k1_ADM_s002
                                 + 0.837178 w_k1_AXP_s002
                                 + 1.05136 w_k1_DIS_s002
                                 + 1.02531 w_k1_FDX_s002
                                 + 0.775701 w_k1_IBM_s002
                                 + 0.97734 w_k1_INTC_s002
                                 + 0.902176 w_k1_KO_s002
                                 + 0.559272 w_k1_MAT_s002
                                 + 1.08946 w_k1_MSFT_s002
                                 + 0.94445 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.27166 w_k1_ADM_s002
                                 + 1.18488 w_k1_AXP_s002
                                 + 1.46455 w_k1_DIS_s002
                                 + 1.21226 w_k1_FDX_s002
                                 + 1.21355 w_k1_IBM_s002
                                 + 1.17347 w_k1_INTC_s002
                                 + 1.31484 w_k1_KO_s002 + 1.63901 w_k1_MAT_s002
                                 + 1.32886 w_k1_MSFT_s002
                                 + 1.16927 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.922421 w_k1_ADM_s003
                                 + 0.998241 w_k1_AXP_s003
                                 + 0.940738 w_k1_DIS_s003
                                 + 0.95574 w_k1_FDX_s003
                                 + 0.825406 w_k1_IBM_s003
                                 + 0.764682 w_k1_INTC_s003
                                 + 1.01664 w_k1_KO_s003
                                 + 0.977545 w_k1_MAT_s003
                                 + 1.03647 w_k1_MSFT_s003
                                 + 0.99973 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.12475 w_k1_ADM_s003
                                 + 0.699606 w_k1_AXP_s003
                                 + 0.945332 w_k1_DIS_s003
                                 + 1.13791 w_k1_FDX_s003
                                 + 0.79228 w_k1_IBM_s003
                                 + 1.03006 w_k1_INTC_s003
                                 + 0.890151 w_k1_KO_s003
                                 + 0.554463 w_k1_MAT_s003
                                 + 1.14485 w_k1_MSFT_s003
                                 + 0.980062 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.784936 w_k1_ADM_s003
                                 + 0.664592 w_k1_AXP_s003
                                 + 0.665045 w_k1_DIS_s003
                                 + 0.770248 w_k1_FDX_s003
                                 + 0.834824 w_k1_IBM_s003
                                 + 0.6192 w_k1_INTC_s003
                                 + 0.745156 w_k1_KO_s003
                                 + 0.646332 w_k1_MAT_s003
                                 + 0.726158 w_k1_MSFT_s003
                                 + 0.867229 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.931086 w_k1_ADM_s003
                                 + 1.22287 w_k1_AXP_s003
                                 + 1.11163 w_k1_DIS_s003
                                 + 0.962644 w_k1_FDX_s003
                                 + 1.61277 w_k1_IBM_s003
                                 + 0.92372 w_k1_INTC_s003
                                 + 1.22117 w_k1_KO_s003 + 1.32088 w_k1_MAT_s003
                                 + 0.881179 w_k1_MSFT_s003
                                 + 1.22247 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.853109 w_k1_ADM_s003
                                 + 0.74168 w_k1_AXP_s003
                                 + 0.738134 w_k1_DIS_s003
                                 + 0.840765 w_k1_FDX_s003
                                 + 0.986302 w_k1_IBM_s003
                                 + 0.758788 w_k1_INTC_s003
                                 + 0.909487 w_k1_KO_s003
                                 + 1.07248 w_k1_MAT_s003
                                 + 0.804291 w_k1_MSFT_s003
                                 + 1.01174 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.22676 w_k1_ADM_s003
                                 + 1.13139 w_k1_AXP_s003
                                 + 0.880385 w_k1_DIS_s003
                                 + 1.10518 w_k1_FDX_s003
                                 + 0.960819 w_k1_IBM_s003
                                 + 0.784613 w_k1_INTC_s003
                                 + 0.969235 w_k1_KO_s003
                                 + 1.53174 w_k1_MAT_s003
                                 + 0.97006 w_k1_MSFT_s003
                                 + 1.16401 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.901679 w_k1_ADM_s003
                                 + 0.918744 w_k1_AXP_s003
                                 + 1.13354 w_k1_DIS_s003
                                 + 1.05957 w_k1_FDX_s003
                                 + 1.20034 w_k1_IBM_s003
                                 + 1.19981 w_k1_INTC_s003
                                 + 1.20215 w_k1_KO_s003 + 1.05483 w_k1_MAT_s003
                                 + 1.1364 w_k1_MSFT_s003 + 1.14708 w_k1_PG_s003
                                 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.25062 w_k1_ADM_s003
                                 + 1.34295 w_k1_AXP_s003
                                 + 1.56265 w_k1_DIS_s003
                                 + 1.26812 w_k1_FDX_s003
                                 + 1.22263 w_k1_IBM_s003
                                 + 1.15619 w_k1_INTC_s003
                                 + 1.21854 w_k1_KO_s003 + 1.4704 w_k1_MAT_s003
                                 + 1.24947 w_k1_MSFT_s003
                                 + 1.34966 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.966142 w_k1_ADM_s004
                                 + 0.984827 w_k1_AXP_s004
                                 + 1.0833 w_k1_DIS_s004
                                 + 0.977764 w_k1_FDX_s004
                                 + 0.89917 w_k1_IBM_s004
                                 + 0.968765 w_k1_INTC_s004
                                 + 0.95689 w_k1_KO_s004 + 1.42732 w_k1_MAT_s004
                                 + 0.869142 w_k1_MSFT_s004
                                 + 0.965137 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.823105 w_k1_ADM_s004
                                 + 0.645025 w_k1_AXP_s004
                                 + 0.695373 w_k1_DIS_s004
                                 + 0.780967 w_k1_FDX_s004
                                 + 0.666602 w_k1_IBM_s004
                                 + 0.710036 w_k1_INTC_s004
                                 + 0.745583 w_k1_KO_s004
                                 + 0.704216 w_k1_MAT_s004
                                 + 0.734901 w_k1_MSFT_s004
                                 + 0.879876 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.0414 w_k1_ADM_s004
                                 + 1.10498 w_k1_AXP_s004
                                 + 1.79006 w_k1_DIS_s004
                                 + 1.46109 w_k1_FDX_s004
                                 + 1.31196 w_k1_IBM_s004
                                 + 1.24308 w_k1_INTC_s004
                                 + 1.46389 w_k1_KO_s004 + 1.85884 w_k1_MAT_s004
                                 + 1.42869 w_k1_MSFT_s004
                                 + 1.34206 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.773689 w_k1_ADM_s004
                                 + 0.834324 w_k1_AXP_s004
                                 + 0.814275 w_k1_DIS_s004
                                 + 0.847524 w_k1_FDX_s004
                                 + 0.97876 w_k1_IBM_s004
                                 + 0.741313 w_k1_INTC_s004
                                 + 0.793194 w_k1_KO_s004
                                 + 0.995715 w_k1_MAT_s004
                                 + 0.733355 w_k1_MSFT_s004
                                 + 0.843348 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.01771 w_k1_ADM_s004
                                 + 1.40774 w_k1_AXP_s004
                                 + 1.39048 w_k1_DIS_s004
                                 + 1.13612 w_k1_FDX_s004
                                 + 1.17633 w_k1_IBM_s004
                                 + 0.914562 w_k1_INTC_s004
                                 + 1.10815 w_k1_KO_s004 + 1.46719 w_k1_MAT_s004
                                 + 0.951467 w_k1_MSFT_s004
                                 + 1.01008 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.922639 w_k1_ADM_s004
                                 + 1.09872 w_k1_AXP_s004
                                 + 1.00883 w_k1_DIS_s004
                                 + 0.939292 w_k1_FDX_s004
                                 + 1.38712 w_k1_IBM_s004
                                 + 1.25879 w_k1_INTC_s004
                                 + 1.35197 w_k1_KO_s004 + 1.32894 w_k1_MAT_s004
                                 + 0.971495 w_k1_MSFT_s004
                                 + 0.964246 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.04957 w_k1_ADM_s004
                                 + 1.1928 w_k1_AXP_s004 + 1.05867 w_k1_DIS_s004
                                 + 1.17102 w_k1_FDX_s004
                                 + 0.833025 w_k1_IBM_s004
                                 + 1.02326 w_k1_INTC_s004
                                 + 1.03945 w_k1_KO_s004 + 1.11262 w_k1_MAT_s004
                                 + 1.0085 w_k1_MSFT_s004
                                 + 0.959073 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.667546 w_k1_ADM_s004
                                 + 0.864854 w_k1_AXP_s004
                                 + 0.650249 w_k1_DIS_s004
                                 + 0.896789 w_k1_FDX_s004
                                 + 0.629021 w_k1_IBM_s004
                                 + 0.896249 w_k1_INTC_s004
                                 + 0.737211 w_k1_KO_s004
                                 + 0.932109 w_k1_MAT_s004
                                 + 0.882645 w_k1_MSFT_s004
                                 + 0.888234 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.13049 w_k1_ADM_s005
                                 + 1.41906 w_k1_AXP_s005
                                 + 1.24681 w_k1_DIS_s005
                                 + 1.18465 w_k1_FDX_s005
                                 + 1.27557 w_k1_IBM_s005
                                 + 1.23092 w_k1_INTC_s005
                                 + 1.37167 w_k1_KO_s005 + 1.73199 w_k1_MAT_s005
                                 + 1.26121 w_k1_MSFT_s005
                                 + 1.03656 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.03563 w_k1_ADM_s005
                                 + 1.02984 w_k1_AXP_s005
                                 + 0.914588 w_k1_DIS_s005
                                 + 0.959083 w_k1_FDX_s005
                                 + 1.17574 w_k1_IBM_s005
                                 + 1.0833 w_k1_INTC_s005 + 1.21811 w_k1_KO_s005
                                 + 1.1961 w_k1_MAT_s005
                                 + 1.11103 w_k1_MSFT_s005
                                 + 0.91253 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.05411 w_k1_ADM_s005
                                 + 1.40151 w_k1_AXP_s005
                                 + 1.13017 w_k1_DIS_s005
                                 + 0.893148 w_k1_FDX_s005
                                 + 1.65686 w_k1_IBM_s005 + 1.129 w_k1_INTC_s005
                                 + 1.40005 w_k1_KO_s005 + 1.61429 w_k1_MAT_s005
                                 + 0.961602 w_k1_MSFT_s005
                                 + 0.976617 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.16473 w_k1_ADM_s005
                                 + 1.15419 w_k1_AXP_s005
                                 + 1.18824 w_k1_DIS_s005
                                 + 1.23202 w_k1_FDX_s005
                                 + 0.92075 w_k1_IBM_s005
                                 + 1.27462 w_k1_INTC_s005
                                 + 1.24441 w_k1_KO_s005 + 0.81908 w_k1_MAT_s005
                                 + 1.28854 w_k1_MSFT_s005
                                 + 0.988538 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.45967 w_k1_ADM_s005
                                 + 1.59477 w_k1_AXP_s005
                                 + 1.69101 w_k1_DIS_s005
                                 + 1.48209 w_k1_FDX_s005
                                 + 1.57998 w_k1_IBM_s005
                                 + 1.28357 w_k1_INTC_s005
                                 + 1.62384 w_k1_KO_s005 + 1.83929 w_k1_MAT_s005
                                 + 1.5663 w_k1_MSFT_s005 + 1.1809 w_k1_PG_s005
                                 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.701536 w_k1_ADM_s005
                                 + 0.864936 w_k1_AXP_s005
                                 + 0.784506 w_k1_DIS_s005
                                 + 0.862602 w_k1_FDX_s005
                                 + 0.932933 w_k1_IBM_s005
                                 + 0.981221 w_k1_INTC_s005
                                 + 1.02021 w_k1_KO_s005
                                 + 0.968123 w_k1_MAT_s005
                                 + 0.9585 w_k1_MSFT_s005
                                 + 0.944325 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.897491 w_k1_ADM_s005
                                 + 0.632031 w_k1_AXP_s005
                                 + 0.648656 w_k1_DIS_s005
                                 + 0.795511 w_k1_FDX_s005
                                 + 0.631163 w_k1_IBM_s005
                                 + 0.962868 w_k1_INTC_s005
                                 + 0.773613 w_k1_KO_s005
                                 + 0.723355 w_k1_MAT_s005
                                 + 0.982705 w_k1_MSFT_s005
                                 + 0.793076 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.19339 w_k1_ADM_s005
                                 + 1.11733 w_k1_AXP_s005
                                 + 0.866026 w_k1_DIS_s005
                                 + 0.976373 w_k1_FDX_s005
                                 + 1.16116 w_k1_IBM_s005
                                 + 0.916632 w_k1_INTC_s005
                                 + 1.11695 w_k1_KO_s005 + 1.36727 w_k1_MAT_s005
                                 + 1.01963 w_k1_MSFT_s005
                                 + 0.945518 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.896375 w_k1_ADM_s006
                                 + 0.919589 w_k1_AXP_s006
                                 + 0.841588 w_k1_DIS_s006
                                 + 1.03333 w_k1_FDX_s006
                                 + 0.739855 w_k1_IBM_s006
                                 + 1.00715 w_k1_INTC_s006
                                 + 0.70795 w_k1_KO_s006
                                 + 0.628651 w_k1_MAT_s006
                                 + 0.978917 w_k1_MSFT_s006
                                 + 0.908689 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.11201 w_k1_ADM_s006
                                 + 1.44509 w_k1_AXP_s006
                                 + 1.66607 w_k1_DIS_s006
                                 + 1.38133 w_k1_FDX_s006
                                 + 1.42231 w_k1_IBM_s006
                                 + 1.26783 w_k1_INTC_s006
                                 + 1.17744 w_k1_KO_s006 + 1.49201 w_k1_MAT_s006
                                 + 1.14867 w_k1_MSFT_s006
                                 + 1.24193 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.04643 w_k1_ADM_s006
                                 + 1.56607 w_k1_AXP_s006
                                 + 1.14744 w_k1_DIS_s006
                                 + 1.08372 w_k1_FDX_s006
                                 + 1.27647 w_k1_IBM_s006
                                 + 0.919605 w_k1_INTC_s006
                                 + 0.85792 w_k1_KO_s006
                                 + 0.933564 w_k1_MAT_s006
                                 + 0.78553 w_k1_MSFT_s006
                                 + 0.97435 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.773829 w_k1_ADM_s006
                                 + 1.26368 w_k1_AXP_s006
                                 + 1.31778 w_k1_DIS_s006
                                 + 1.01767 w_k1_FDX_s006
                                 + 1.22748 w_k1_IBM_s006
                                 + 0.872513 w_k1_INTC_s006
                                 + 1.05144 w_k1_KO_s006 + 1.60069 w_k1_MAT_s006
                                 + 0.954231 w_k1_MSFT_s006
                                 + 1.02009 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.956284 w_k1_ADM_s006
                                 + 1.33824 w_k1_AXP_s006 + 1.184 w_k1_DIS_s006
                                 + 1.27753 w_k1_FDX_s006
                                 + 0.837301 w_k1_IBM_s006
                                 + 0.979963 w_k1_INTC_s006
                                 + 0.861382 w_k1_KO_s006
                                 + 0.783388 w_k1_MAT_s006
                                 + 0.960936 w_k1_MSFT_s006
                                 + 1.03728 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.69714 w_k1_ADM_s006
                                 + 0.826094 w_k1_AXP_s006
                                 + 0.620717 w_k1_DIS_s006
                                 + 0.791531 w_k1_FDX_s006
                                 + 0.766503 w_k1_IBM_s006
                                 + 0.72165 w_k1_INTC_s006
                                 + 0.774961 w_k1_KO_s006
                                 + 0.947606 w_k1_MAT_s006
                                 + 0.755446 w_k1_MSFT_s006
                                 + 0.909178 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.945626 w_k1_ADM_s006
                                 + 0.982215 w_k1_AXP_s006
                                 + 0.928726 w_k1_DIS_s006
                                 + 0.907721 w_k1_FDX_s006
                                 + 1.08366 w_k1_IBM_s006
                                 + 0.815129 w_k1_INTC_s006
                                 + 0.913933 w_k1_KO_s006
                                 + 1.08586 w_k1_MAT_s006
                                 + 0.810088 w_k1_MSFT_s006
                                 + 0.956423 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.958159 w_k1_ADM_s006
                                 + 0.995422 w_k1_AXP_s006
                                 + 1.36998 w_k1_DIS_s006
                                 + 1.02329 w_k1_FDX_s006
                                 + 1.57824 w_k1_IBM_s006
                                 + 0.948224 w_k1_INTC_s006
                                 + 1.11256 w_k1_KO_s006 + 1.09268 w_k1_MAT_s006
                                 + 0.822733 w_k1_MSFT_s006
                                 + 1.01923 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.06975 w_k1_ADM_s007
                                 + 1.0112 w_k1_AXP_s007 + 1.31322 w_k1_DIS_s007
                                 + 1.18124 w_k1_FDX_s007
                                 + 0.838122 w_k1_IBM_s007
                                 + 1.02489 w_k1_INTC_s007
                                 + 1.28698 w_k1_KO_s007 + 1.09122 w_k1_MAT_s007
                                 + 1.1556 w_k1_MSFT_s007 + 1.10294 w_k1_PG_s007
                                 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.48466 w_k1_ADM_s007
                                 + 1.25374 w_k1_AXP_s007 + 1.2631 w_k1_DIS_s007
                                 + 1.13641 w_k1_FDX_s007
                                 + 1.43967 w_k1_IBM_s007
                                 + 1.01557 w_k1_INTC_s007
                                 + 1.49662 w_k1_KO_s007 + 1.47858 w_k1_MAT_s007
                                 + 0.974579 w_k1_MSFT_s007
                                 + 1.10132 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.08203 w_k1_ADM_s007
                                 + 0.751678 w_k1_AXP_s007
                                 + 0.668794 w_k1_DIS_s007
                                 + 0.847092 w_k1_FDX_s007
                                 + 0.794166 w_k1_IBM_s007
                                 + 0.704382 w_k1_INTC_s007
                                 + 0.881279 w_k1_KO_s007
                                 + 0.943414 w_k1_MAT_s007
                                 + 0.822619 w_k1_MSFT_s007
                                 + 1.00896 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.741731 w_k1_ADM_s007
                                 + 0.590311 w_k1_AXP_s007
                                 + 0.487444 w_k1_DIS_s007
                                 + 0.671892 w_k1_FDX_s007
                                 + 0.713334 w_k1_IBM_s007
                                 + 0.870366 w_k1_INTC_s007
                                 + 0.728871 w_k1_KO_s007
                                 + 0.554586 w_k1_MAT_s007
                                 + 0.760019 w_k1_MSFT_s007
                                 + 0.869715 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.39636 w_k1_ADM_s007
                                 + 1.38025 w_k1_AXP_s007 + 1.8365 w_k1_DIS_s007
                                 + 1.27641 w_k1_FDX_s007
                                 + 1.46184 w_k1_IBM_s007
                                 + 1.35153 w_k1_INTC_s007
                                 + 1.69589 w_k1_KO_s007 + 1.30963 w_k1_MAT_s007
                                 + 1.38276 w_k1_MSFT_s007
                                 + 1.39997 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.11212 w_k1_ADM_s007
                                 + 0.873744 w_k1_AXP_s007
                                 + 0.834977 w_k1_DIS_s007
                                 + 0.957047 w_k1_FDX_s007
                                 + 0.673867 w_k1_IBM_s007
                                 + 0.706406 w_k1_INTC_s007
                                 + 0.783742 w_k1_KO_s007
                                 + 0.673858 w_k1_MAT_s007
                                 + 0.820832 w_k1_MSFT_s007
                                 + 1.00512 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.912324 w_k1_ADM_s007
                                 + 0.96837 w_k1_AXP_s007 + 0.8423 w_k1_DIS_s007
                                 + 0.867838 w_k1_FDX_s007
                                 + 1.09674 w_k1_IBM_s007
                                 + 1.04571 w_k1_INTC_s007 + 1.1083 w_k1_KO_s007
                                 + 1.33218 w_k1_MAT_s007
                                 + 0.950876 w_k1_MSFT_s007
                                 + 0.993502 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.14667 w_k1_ADM_s007
                                 + 0.888018 w_k1_AXP_s007
                                 + 1.02488 w_k1_DIS_s007
                                 + 1.00749 w_k1_FDX_s007
                                 + 1.10052 w_k1_IBM_s007
                                 + 0.91039 w_k1_INTC_s007
                                 + 1.20081 w_k1_KO_s007 + 0.90537 w_k1_MAT_s007
                                 + 0.924591 w_k1_MSFT_s007
                                 + 1.2366 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.897463 w_k1_ADM_s010
                                 + 0.702501 w_k1_AXP_s010
                                 + 0.769252 w_k1_DIS_s010
                                 + 0.789426 w_k1_FDX_s010
                                 + 0.715914 w_k1_IBM_s010
                                 + 0.842774 w_k1_INTC_s010
                                 + 0.747338 w_k1_KO_s010
                                 + 0.495507 w_k1_MAT_s010
                                 + 0.897114 w_k1_MSFT_s010
                                 + 0.754456 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.978177 w_k1_ADM_s010
                                 + 1.01519 w_k1_AXP_s010
                                 + 1.09287 w_k1_DIS_s010
                                 + 0.98776 w_k1_FDX_s010
                                 + 1.01173 w_k1_IBM_s010
                                 + 1.12993 w_k1_INTC_s010
                                 + 0.98016 w_k1_KO_s010
                                 + 0.677543 w_k1_MAT_s010
                                 + 1.00514 w_k1_MSFT_s010
                                 + 0.980102 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.25129 w_k1_ADM_s010
                                 + 1.33051 w_k1_AXP_s010
                                 + 1.65549 w_k1_DIS_s010
                                 + 1.25641 w_k1_FDX_s010
                                 + 0.885264 w_k1_IBM_s010
                                 + 1.17025 w_k1_INTC_s010 + 1.0405 w_k1_KO_s010
                                 + 1.04971 w_k1_MAT_s010
                                 + 1.21187 w_k1_MSFT_s010
                                 + 1.17545 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.22651 w_k1_ADM_s010
                                 + 0.775198 w_k1_AXP_s010
                                 + 0.907545 w_k1_DIS_s010
                                 + 1.02095 w_k1_FDX_s010
                                 + 0.672041 w_k1_IBM_s010
                                 + 0.983933 w_k1_INTC_s010
                                 + 0.756451 w_k1_KO_s010
                                 + 0.666868 w_k1_MAT_s010
                                 + 1.00744 w_k1_MSFT_s010
                                 + 0.920551 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.709087 w_k1_ADM_s010
                                 + 0.886636 w_k1_AXP_s010
                                 + 0.667382 w_k1_DIS_s010
                                 + 0.7462 w_k1_FDX_s010
                                 + 0.951459 w_k1_IBM_s010
                                 + 0.952496 w_k1_INTC_s010
                                 + 0.826872 w_k1_KO_s010
                                 + 0.896379 w_k1_MAT_s010
                                 + 0.830253 w_k1_MSFT_s010
                                 + 0.844551 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.03621 w_k1_ADM_s010
                                 + 0.899895 w_k1_AXP_s010
                                 + 0.784575 w_k1_DIS_s010
                                 + 0.848207 w_k1_FDX_s010
                                 + 1.06428 w_k1_IBM_s010
                                 + 0.80531 w_k1_INTC_s010
                                 + 0.886103 w_k1_KO_s010
                                 + 0.977499 w_k1_MAT_s010
                                 + 0.800965 w_k1_MSFT_s010
                                 + 0.856146 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.48899 w_k1_ADM_s010
                                 + 1.16501 w_k1_AXP_s010 + 1.1338 w_k1_DIS_s010
                                 + 1.21033 w_k1_FDX_s010
                                 + 0.838912 w_k1_IBM_s010
                                 + 1.28906 w_k1_INTC_s010
                                 + 0.841412 w_k1_KO_s010
                                 + 0.600301 w_k1_MAT_s010
                                 + 1.16545 w_k1_MSFT_s010
                                 + 0.964643 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.05986 w_k1_ADM_s010
                                 + 1.10382 w_k1_AXP_s010
                                 + 1.23984 w_k1_DIS_s010
                                 + 0.981613 w_k1_FDX_s010
                                 + 1.44278 w_k1_IBM_s010
                                 + 1.21357 w_k1_INTC_s010
                                 + 1.22984 w_k1_KO_s010 + 1.17738 w_k1_MAT_s010
                                 + 1.01096 w_k1_MSFT_s010
                                 + 1.01847 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.87734 w_k1_ADM_s011
                                 + 0.950921 w_k1_AXP_s011
                                 + 0.813312 w_k1_DIS_s011
                                 + 0.939909 w_k1_FDX_s011
                                 + 1.31769 w_k1_IBM_s011
                                 + 0.977359 w_k1_INTC_s011
                                 + 0.991165 w_k1_KO_s011
                                 + 1.16084 w_k1_MAT_s011
                                 + 0.993877 w_k1_MSFT_s011
                                 + 0.907522 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.710644 w_k1_ADM_s011
                                 + 0.740233 w_k1_AXP_s011
                                 + 0.746516 w_k1_DIS_s011
                                 + 0.76957 w_k1_FDX_s011
                                 + 0.956355 w_k1_IBM_s011
                                 + 0.903478 w_k1_INTC_s011
                                 + 0.910112 w_k1_KO_s011
                                 + 0.829034 w_k1_MAT_s011
                                 + 0.856566 w_k1_MSFT_s011
                                 + 0.848257 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.13797 w_k1_ADM_s011
                                 + 1.35184 w_k1_AXP_s011 + 1.2257 w_k1_DIS_s011
                                 + 1.20667 w_k1_FDX_s011
                                 + 1.32917 w_k1_IBM_s011
                                 + 1.31089 w_k1_INTC_s011
                                 + 1.33724 w_k1_KO_s011 + 1.34499 w_k1_MAT_s011
                                 + 1.21977 w_k1_MSFT_s011 + 1.1757 w_k1_PG_s011
                                 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.907675 w_k1_ADM_s011
                                 + 0.843049 w_k1_AXP_s011
                                 + 0.741641 w_k1_DIS_s011
                                 + 0.993037 w_k1_FDX_s011
                                 + 0.683255 w_k1_IBM_s011
                                 + 1.08423 w_k1_INTC_s011
                                 + 0.663574 w_k1_KO_s011
                                 + 0.408874 w_k1_MAT_s011
                                 + 0.878888 w_k1_MSFT_s011
                                 + 0.848265 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.02104 w_k1_ADM_s011
                                 + 1.06747 w_k1_AXP_s011
                                 + 0.773347 w_k1_DIS_s011
                                 + 0.902683 w_k1_FDX_s011
                                 + 0.914803 w_k1_IBM_s011
                                 + 1.01535 w_k1_INTC_s011
                                 + 0.890458 w_k1_KO_s011
                                 + 0.785004 w_k1_MAT_s011
                                 + 0.932253 w_k1_MSFT_s011
                                 + 0.935325 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.78622 w_k1_ADM_s011
                                 + 0.603412 w_k1_AXP_s011
                                 + 0.532242 w_k1_DIS_s011
                                 + 0.719825 w_k1_FDX_s011
                                 + 0.692328 w_k1_IBM_s011
                                 + 0.878291 w_k1_INTC_s011
                                 + 0.68651 w_k1_KO_s011
                                 + 0.653875 w_k1_MAT_s011
                                 + 0.818028 w_k1_MSFT_s011
                                 + 0.733682 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.37649 w_k1_ADM_s011
                                 + 1.5802 w_k1_AXP_s011 + 1.20984 w_k1_DIS_s011
                                 + 1.29009 w_k1_FDX_s011
                                 + 0.970426 w_k1_IBM_s011
                                 + 1.47968 w_k1_INTC_s011
                                 + 0.978783 w_k1_KO_s011
                                 + 0.986145 w_k1_MAT_s011
                                 + 1.2682 w_k1_MSFT_s011 + 1.13257 w_k1_PG_s011
                                 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.0175 w_k1_ADM_s011
                                 + 0.945604 w_k1_AXP_s011
                                 + 1.19571 w_k1_DIS_s011
                                 + 1.22092 w_k1_FDX_s011
                                 + 0.718092 w_k1_IBM_s011
                                 + 1.12297 w_k1_INTC_s011 + 1.0073 w_k1_KO_s011
                                 + 0.735015 w_k1_MAT_s011
                                 + 1.23777 w_k1_MSFT_s011
                                 + 0.957652 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.13369 w_k1_ADM_s012
                                 + 1.38127 w_k1_AXP_s012
                                 + 1.70714 w_k1_DIS_s012 + 1.5015 w_k1_FDX_s012
                                 + 1.17857 w_k1_IBM_s012
                                 + 1.25125 w_k1_INTC_s012
                                 + 1.24311 w_k1_KO_s012
                                 + 0.933022 w_k1_MAT_s012
                                 + 1.4302 w_k1_MSFT_s012 + 1.14451 w_k1_PG_s012
                                 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.999338 w_k1_ADM_s012
                                 + 0.841329 w_k1_AXP_s012
                                 + 0.713222 w_k1_DIS_s012
                                 + 0.845444 w_k1_FDX_s012
                                 + 0.803229 w_k1_IBM_s012
                                 + 0.823756 w_k1_INTC_s012
                                 + 0.838034 w_k1_KO_s012
                                 + 0.72272 w_k1_MAT_s012
                                 + 0.960541 w_k1_MSFT_s012
                                 + 0.887773 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.14356 w_k1_ADM_s012
                                 + 1.21808 w_k1_AXP_s012
                                 + 1.19302 w_k1_DIS_s012
                                 + 1.12917 w_k1_FDX_s012
                                 + 1.57477 w_k1_IBM_s012
                                 + 1.08838 w_k1_INTC_s012
                                 + 1.36163 w_k1_KO_s012 + 1.24714 w_k1_MAT_s012
                                 + 1.03358 w_k1_MSFT_s012
                                 + 1.13923 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.950534 w_k1_ADM_s012
                                 + 0.746083 w_k1_AXP_s012
                                 + 0.637596 w_k1_DIS_s012
                                 + 0.767102 w_k1_FDX_s012
                                 + 1.27336 w_k1_IBM_s012
                                 + 0.883032 w_k1_INTC_s012
                                 + 1.04106 w_k1_KO_s012
                                 + 0.967812 w_k1_MAT_s012
                                 + 0.854348 w_k1_MSFT_s012
                                 + 0.896797 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.2483 w_k1_ADM_s012
                                 + 0.74995 w_k1_AXP_s012
                                 + 1.06975 w_k1_DIS_s012
                                 + 0.996311 w_k1_FDX_s012
                                 + 1.00279 w_k1_IBM_s012
                                 + 0.937612 w_k1_INTC_s012
                                 + 1.0722 w_k1_KO_s012 + 0.696299 w_k1_MAT_s012
                                 + 1.07504 w_k1_MSFT_s012
                                 + 0.979152 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.30316 w_k1_ADM_s012
                                 + 1.16425 w_k1_AXP_s012
                                 + 1.06982 w_k1_DIS_s012
                                 + 1.07341 w_k1_FDX_s012
                                 + 0.974996 w_k1_IBM_s012
                                 + 1.0267 w_k1_INTC_s012 + 1.01924 w_k1_KO_s012
                                 + 0.945373 w_k1_MAT_s012
                                 + 1.06881 w_k1_MSFT_s012
                                 + 0.953165 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.874601 w_k1_ADM_s012
                                 + 0.767821 w_k1_AXP_s012
                                 + 0.620431 w_k1_DIS_s012
                                 + 0.845736 w_k1_FDX_s012
                                 + 0.565772 w_k1_IBM_s012
                                 + 0.950268 w_k1_INTC_s012
                                 + 0.655557 w_k1_KO_s012
                                 + 0.389415 w_k1_MAT_s012
                                 + 0.913147 w_k1_MSFT_s012
                                 + 0.704825 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.19116 w_k1_ADM_s012
                                 + 1.46031 w_k1_AXP_s012
                                 + 0.976461 w_k1_DIS_s012
                                 + 1.07148 w_k1_FDX_s012
                                 + 1.16879 w_k1_IBM_s012
                                 + 1.02011 w_k1_INTC_s012
                                 + 1.05164 w_k1_KO_s012 + 1.29658 w_k1_MAT_s012
                                 + 0.99796 w_k1_MSFT_s012
                                 + 0.983734 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.20655 w_k1_ADM_s013
                                 + 1.15736 w_k1_AXP_s013 + 1.0406 w_k1_DIS_s013
                                 + 1.05482 w_k1_FDX_s013
                                 + 1.08997 w_k1_IBM_s013
                                 + 0.973043 w_k1_INTC_s013
                                 + 1.2447 w_k1_KO_s013 + 1.66248 w_k1_MAT_s013
                                 + 1.14789 w_k1_MSFT_s013
                                 + 1.20639 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.887485 w_k1_ADM_s013
                                 + 0.856464 w_k1_AXP_s013
                                 + 0.701521 w_k1_DIS_s013
                                 + 0.829467 w_k1_FDX_s013
                                 + 0.948502 w_k1_IBM_s013
                                 + 0.760945 w_k1_INTC_s013
                                 + 0.862607 w_k1_KO_s013
                                 + 0.975962 w_k1_MAT_s013
                                 + 0.954199 w_k1_MSFT_s013
                                 + 0.903355 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.39059 w_k1_ADM_s013
                                 + 1.50208 w_k1_AXP_s013 + 1.1243 w_k1_DIS_s013
                                 + 1.19897 w_k1_FDX_s013
                                 + 0.906727 w_k1_IBM_s013
                                 + 0.862594 w_k1_INTC_s013
                                 + 1.01867 w_k1_KO_s013 + 1.33604 w_k1_MAT_s013
                                 + 1.08421 w_k1_MSFT_s013
                                 + 0.957481 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.993247 w_k1_ADM_s013
                                 + 0.627431 w_k1_AXP_s013
                                 + 0.805139 w_k1_DIS_s013
                                 + 0.807158 w_k1_FDX_s013
                                 + 0.943943 w_k1_IBM_s013
                                 + 0.872772 w_k1_INTC_s013
                                 + 1.00839 w_k1_KO_s013
                                 + 0.904491 w_k1_MAT_s013
                                 + 1.0493 w_k1_MSFT_s013
                                 + 0.868874 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.28305 w_k1_ADM_s013
                                 + 1.1393 w_k1_AXP_s013 + 1.72304 w_k1_DIS_s013
                                 + 1.45795 w_k1_FDX_s013 + 1.0937 w_k1_IBM_s013
                                 + 1.09978 w_k1_INTC_s013
                                 + 1.41722 w_k1_KO_s013 + 1.40931 w_k1_MAT_s013
                                 + 1.44214 w_k1_MSFT_s013
                                 + 1.13236 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.41077 w_k1_ADM_s013
                                 + 1.03713 w_k1_AXP_s013
                                 + 1.39274 w_k1_DIS_s013
                                 + 1.06548 w_k1_FDX_s013
                                 + 1.45666 w_k1_IBM_s013
                                 + 0.962055 w_k1_INTC_s013
                                 + 1.40414 w_k1_KO_s013 + 1.25292 w_k1_MAT_s013
                                 + 1.14079 w_k1_MSFT_s013
                                 + 1.10912 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.806077 w_k1_ADM_s013
                                 + 0.635142 w_k1_AXP_s013
                                 + 0.574043 w_k1_DIS_s013
                                 + 0.703795 w_k1_FDX_s013
                                 + 0.71328 w_k1_IBM_s013
                                 + 0.933716 w_k1_INTC_s013
                                 + 0.79391 w_k1_KO_s013
                                 + 0.767269 w_k1_MAT_s013
                                 + 0.811666 w_k1_MSFT_s013
                                 + 0.706353 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.989464 w_k1_ADM_s013
                                 + 0.821345 w_k1_AXP_s013
                                 + 0.798849 w_k1_DIS_s013
                                 + 1.02286 w_k1_FDX_s013
                                 + 0.721857 w_k1_IBM_s013
                                 + 1.02926 w_k1_INTC_s013
                                 + 0.903791 w_k1_KO_s013
                                 + 0.813976 w_k1_MAT_s013
                                 + 1.16529 w_k1_MSFT_s013 + 0.9759 w_k1_PG_s013
                                 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.727006 w_k1_ADM_s014
                                 + 0.743374 w_k1_AXP_s014
                                 + 0.613491 w_k1_DIS_s014
                                 + 0.781984 w_k1_FDX_s014
                                 + 0.786505 w_k1_IBM_s014
                                 + 0.828284 w_k1_INTC_s014
                                 + 0.733144 w_k1_KO_s014
                                 + 0.906431 w_k1_MAT_s014
                                 + 0.747892 w_k1_MSFT_s014
                                 + 0.825411 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.800838 w_k1_ADM_s014
                                 + 0.648025 w_k1_AXP_s014
                                 + 0.898182 w_k1_DIS_s014
                                 + 0.978875 w_k1_FDX_s014
                                 + 0.651654 w_k1_IBM_s014
                                 + 0.907001 w_k1_INTC_s014
                                 + 0.709574 w_k1_KO_s014
                                 + 0.458811 w_k1_MAT_s014
                                 + 1.06331 w_k1_MSFT_s014
                                 + 0.813618 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.3433 w_k1_ADM_s014
                                 + 1.30821 w_k1_AXP_s014
                                 + 1.64206 w_k1_DIS_s014
                                 + 1.32218 w_k1_FDX_s014
                                 + 1.01778 w_k1_IBM_s014
                                 + 1.14704 w_k1_INTC_s014
                                 + 1.11515 w_k1_KO_s014
                                 + 0.994723 w_k1_MAT_s014
                                 + 1.14883 w_k1_MSFT_s014
                                 + 1.14297 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.992442 w_k1_ADM_s014
                                 + 1.10626 w_k1_AXP_s014
                                 + 0.913934 w_k1_DIS_s014
                                 + 1.10431 w_k1_FDX_s014
                                 + 0.901638 w_k1_IBM_s014
                                 + 0.847776 w_k1_INTC_s014
                                 + 0.83411 w_k1_KO_s014 + 1.30295 w_k1_MAT_s014
                                 + 0.844688 w_k1_MSFT_s014
                                 + 0.988156 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.762035 w_k1_ADM_s014
                                 + 1.04027 w_k1_AXP_s014
                                 + 0.983246 w_k1_DIS_s014
                                 + 0.907129 w_k1_FDX_s014
                                 + 1.09426 w_k1_IBM_s014
                                 + 1.00803 w_k1_INTC_s014
                                 + 0.888913 w_k1_KO_s014
                                 + 1.19948 w_k1_MAT_s014
                                 + 0.944282 w_k1_MSFT_s014
                                 + 0.865039 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.994667 w_k1_ADM_s014
                                 + 1.56389 w_k1_AXP_s014
                                 + 1.54933 w_k1_DIS_s014 + 1.2527 w_k1_FDX_s014
                                 + 1.40879 w_k1_IBM_s014
                                 + 1.34903 w_k1_INTC_s014
                                 + 1.48395 w_k1_KO_s014 + 2.2731 w_k1_MAT_s014
                                 + 1.26828 w_k1_MSFT_s014
                                 + 1.17799 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.09102 w_k1_ADM_s014
                                 + 1.18653 w_k1_AXP_s014
                                 + 0.99656 w_k1_DIS_s014
                                 + 1.13524 w_k1_FDX_s014
                                 + 0.76687 w_k1_IBM_s014
                                 + 0.942333 w_k1_INTC_s014
                                 + 0.86921 w_k1_KO_s014
                                 + 0.997694 w_k1_MAT_s014
                                 + 1.02153 w_k1_MSFT_s014
                                 + 0.974328 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.01999 w_k1_ADM_s014
                                 + 0.986695 w_k1_AXP_s014
                                 + 1.21107 w_k1_DIS_s014
                                 + 1.10879 w_k1_FDX_s014
                                 + 1.05461 w_k1_IBM_s014
                                 + 1.04199 w_k1_INTC_s014
                                 + 1.15143 w_k1_KO_s014 + 1.22342 w_k1_MAT_s014
                                 + 1.0344 w_k1_MSFT_s014 + 1.0017 w_k1_PG_s014
                                 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.96954 w_k1_ADM_s015
                                 + 0.912243 w_k1_AXP_s015
                                 + 0.953969 w_k1_DIS_s015
                                 + 0.884904 w_k1_FDX_s015
                                 + 1.2386 w_k1_IBM_s015
                                 + 1.17327 w_k1_INTC_s015
                                 + 1.07032 w_k1_KO_s015
                                 + 0.789249 w_k1_MAT_s015
                                 + 0.983318 w_k1_MSFT_s015
                                 + 0.922989 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.09684 w_k1_ADM_s015
                                 + 1.36436 w_k1_AXP_s015
                                 + 1.28825 w_k1_DIS_s015
                                 + 1.14355 w_k1_FDX_s015
                                 + 1.29458 w_k1_IBM_s015
                                 + 1.33175 w_k1_INTC_s015
                                 + 1.19529 w_k1_KO_s015 + 1.16403 w_k1_MAT_s015
                                 + 1.14539 w_k1_MSFT_s015
                                 + 1.14693 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.72538 w_k1_ADM_s015
                                 + 0.610526 w_k1_AXP_s015
                                 + 0.831267 w_k1_DIS_s015
                                 + 0.731788 w_k1_FDX_s015
                                 + 0.77115 w_k1_IBM_s015
                                 + 0.938849 w_k1_INTC_s015
                                 + 0.749564 w_k1_KO_s015
                                 + 0.328365 w_k1_MAT_s015
                                 + 0.7873 w_k1_MSFT_s015
                                 + 0.658229 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.25626 w_k1_ADM_s015
                                 + 1.29496 w_k1_AXP_s015
                                 + 0.997856 w_k1_DIS_s015
                                 + 1.0957 w_k1_FDX_s015 + 1.02084 w_k1_IBM_s015
                                 + 1.29012 w_k1_INTC_s015 + 1.1102 w_k1_KO_s015
                                 + 0.985993 w_k1_MAT_s015
                                 + 0.983595 w_k1_MSFT_s015
                                 + 1.06456 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.98369 w_k1_ADM_s015
                                 + 0.770346 w_k1_AXP_s015
                                 + 0.85753 w_k1_DIS_s015
                                 + 0.97942 w_k1_FDX_s015
                                 + 0.775073 w_k1_IBM_s015
                                 + 1.22201 w_k1_INTC_s015
                                 + 0.916378 w_k1_KO_s015
                                 + 0.585159 w_k1_MAT_s015
                                 + 1.04803 w_k1_MSFT_s015
                                 + 0.921231 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.39061 w_k1_ADM_s015
                                 + 1.67516 w_k1_AXP_s015
                                 + 2.02459 w_k1_DIS_s015
                                 + 1.48856 w_k1_FDX_s015 + 1.41 w_k1_IBM_s015
                                 + 2.04937 w_k1_INTC_s015
                                 + 1.59282 w_k1_KO_s015
                                 + 0.884047 w_k1_MAT_s015
                                 + 1.55904 w_k1_MSFT_s015
                                 + 1.32021 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.895308 w_k1_ADM_s015
                                 + 0.967052 w_k1_AXP_s015
                                 + 1.01639 w_k1_DIS_s015
                                 + 0.978221 w_k1_FDX_s015
                                 + 0.833429 w_k1_IBM_s015
                                 + 1.39025 w_k1_INTC_s015
                                 + 1.01036 w_k1_KO_s015
                                 + 0.685085 w_k1_MAT_s015
                                 + 1.15383 w_k1_MSFT_s015
                                 + 1.03409 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.801994 w_k1_ADM_s015
                                 + 0.758343 w_k1_AXP_s015
                                 + 0.593397 w_k1_DIS_s015
                                 + 0.767771 w_k1_FDX_s015
                                 + 0.704129 w_k1_IBM_s015
                                 + 0.922365 w_k1_INTC_s015
                                 + 0.800428 w_k1_KO_s015
                                 + 0.665197 w_k1_MAT_s015
                                 + 0.756718 w_k1_MSFT_s015
                                 + 0.90737 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.766224 w_k1_ADM_s016
                                 + 0.833339 w_k1_AXP_s016
                                 + 0.75094 w_k1_DIS_s016
                                 + 0.956228 w_k1_FDX_s016
                                 + 0.788106 w_k1_IBM_s016
                                 + 1.00509 w_k1_INTC_s016
                                 + 0.86252 w_k1_KO_s016
                                 + 0.695726 w_k1_MAT_s016
                                 + 1.00166 w_k1_MSFT_s016
                                 + 0.830087 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.865046 w_k1_ADM_s016
                                 + 1.03572 w_k1_AXP_s016
                                 + 0.806928 w_k1_DIS_s016
                                 + 1.07155 w_k1_FDX_s016
                                 + 0.603496 w_k1_IBM_s016
                                 + 1.04686 w_k1_INTC_s016
                                 + 0.762212 w_k1_KO_s016
                                 + 0.572693 w_k1_MAT_s016
                                 + 1.00996 w_k1_MSFT_s016
                                 + 0.948115 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.924139 w_k1_ADM_s016
                                 + 0.916226 w_k1_AXP_s016
                                 + 1.10022 w_k1_DIS_s016
                                 + 1.11129 w_k1_FDX_s016 + 1.0455 w_k1_IBM_s016
                                 + 1.10155 w_k1_INTC_s016
                                 + 0.985544 w_k1_KO_s016
                                 + 0.87402 w_k1_MAT_s016
                                 + 1.16108 w_k1_MSFT_s016
                                 + 1.05691 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.803546 w_k1_ADM_s016
                                 + 0.601645 w_k1_AXP_s016
                                 + 0.572588 w_k1_DIS_s016
                                 + 0.779458 w_k1_FDX_s016
                                 + 0.629837 w_k1_IBM_s016
                                 + 0.622601 w_k1_INTC_s016
                                 + 0.548203 w_k1_KO_s016
                                 + 0.46369 w_k1_MAT_s016
                                 + 0.674508 w_k1_MSFT_s016
                                 + 0.675228 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.84824 w_k1_ADM_s016
                                 + 0.559278 w_k1_AXP_s016
                                 + 0.66473 w_k1_DIS_s016
                                 + 0.773378 w_k1_FDX_s016
                                 + 0.874087 w_k1_IBM_s016
                                 + 0.665151 w_k1_INTC_s016
                                 + 0.796267 w_k1_KO_s016
                                 + 0.491959 w_k1_MAT_s016
                                 + 0.774612 w_k1_MSFT_s016
                                 + 0.82171 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.644106 w_k1_ADM_s016
                                 + 0.912229 w_k1_AXP_s016
                                 + 0.716209 w_k1_DIS_s016
                                 + 0.761562 w_k1_FDX_s016
                                 + 1.16865 w_k1_IBM_s016
                                 + 0.949477 w_k1_INTC_s016
                                 + 0.980399 w_k1_KO_s016
                                 + 0.925609 w_k1_MAT_s016
                                 + 0.792041 w_k1_MSFT_s016
                                 + 0.93986 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.20788 w_k1_ADM_s016
                                 + 1.48393 w_k1_AXP_s016
                                 + 1.69653 w_k1_DIS_s016
                                 + 1.39527 w_k1_FDX_s016
                                 + 1.71043 w_k1_IBM_s016
                                 + 1.57474 w_k1_INTC_s016
                                 + 1.41942 w_k1_KO_s016 + 1.28287 w_k1_MAT_s016
                                 + 1.66281 w_k1_MSFT_s016
                                 + 1.16373 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.01562 w_k1_ADM_s016
                                 + 1.45708 w_k1_AXP_s016
                                 + 1.43025 w_k1_DIS_s016 + 1.4099 w_k1_FDX_s016
                                 + 1.03522 w_k1_IBM_s016
                                 + 1.06303 w_k1_INTC_s016
                                 + 0.937593 w_k1_KO_s016
                                 + 1.07155 w_k1_MAT_s016
                                 + 1.23287 w_k1_MSFT_s016
                                 + 1.13144 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.09485 w_k1_ADM_s017
                                 + 1.2011 w_k1_AXP_s017
                                 + 0.921055 w_k1_DIS_s017
                                 + 1.1258 w_k1_FDX_s017
                                 + 0.861252 w_k1_IBM_s017
                                 + 1.02862 w_k1_INTC_s017
                                 + 0.751066 w_k1_KO_s017
                                 + 0.891803 w_k1_MAT_s017
                                 + 0.974941 w_k1_MSFT_s017
                                 + 1.07895 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.779264 w_k1_ADM_s017
                                 + 0.638619 w_k1_AXP_s017
                                 + 0.69501 w_k1_DIS_s017
                                 + 0.740379 w_k1_FDX_s017
                                 + 1.21137 w_k1_IBM_s017
                                 + 0.73648 w_k1_INTC_s017
                                 + 0.943131 w_k1_KO_s017
                                 + 1.10894 w_k1_MAT_s017
                                 + 0.786221 w_k1_MSFT_s017
                                 + 0.890548 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.927999 w_k1_ADM_s017
                                 + 1.16969 w_k1_AXP_s017
                                 + 0.940086 w_k1_DIS_s017
                                 + 0.964991 w_k1_FDX_s017
                                 + 1.23831 w_k1_IBM_s017
                                 + 0.851799 w_k1_INTC_s017
                                 + 0.858325 w_k1_KO_s017
                                 + 1.18959 w_k1_MAT_s017
                                 + 0.873307 w_k1_MSFT_s017
                                 + 1.04578 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.03575 w_k1_ADM_s017
                                 + 0.764835 w_k1_AXP_s017
                                 + 1.05071 w_k1_DIS_s017
                                 + 0.960259 w_k1_FDX_s017
                                 + 1.04119 w_k1_IBM_s017
                                 + 0.933981 w_k1_INTC_s017
                                 + 0.928378 w_k1_KO_s017
                                 + 0.834608 w_k1_MAT_s017
                                 + 0.913406 w_k1_MSFT_s017
                                 + 1.0107 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.79505 w_k1_ADM_s017
                                 + 0.838399 w_k1_AXP_s017
                                 + 0.730037 w_k1_DIS_s017
                                 + 0.867966 w_k1_FDX_s017
                                 + 0.752837 w_k1_IBM_s017
                                 + 0.776372 w_k1_INTC_s017
                                 + 0.749235 w_k1_KO_s017
                                 + 0.856686 w_k1_MAT_s017
                                 + 0.836579 w_k1_MSFT_s017
                                 + 0.909617 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.42227 w_k1_ADM_s017
                                 + 1.44321 w_k1_AXP_s017
                                 + 1.43488 w_k1_DIS_s017
                                 + 1.25331 w_k1_FDX_s017
                                 + 1.48699 w_k1_IBM_s017
                                 + 1.23727 w_k1_INTC_s017
                                 + 1.43956 w_k1_KO_s017 + 1.30992 w_k1_MAT_s017
                                 + 1.12832 w_k1_MSFT_s017
                                 + 1.31965 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.07008 w_k1_ADM_s017
                                 + 1.45605 w_k1_AXP_s017
                                 + 1.45103 w_k1_DIS_s017
                                 + 1.50082 w_k1_FDX_s017
                                 + 1.02918 w_k1_IBM_s017
                                 + 1.21032 w_k1_INTC_s017
                                 + 1.08543 w_k1_KO_s017 + 1.14648 w_k1_MAT_s017
                                 + 1.23237 w_k1_MSFT_s017
                                 + 1.13288 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.00187 w_k1_ADM_s017
                                 + 1.01735 w_k1_AXP_s017
                                 + 1.38631 w_k1_DIS_s017
                                 + 1.18882 w_k1_FDX_s017
                                 + 1.04006 w_k1_IBM_s017
                                 + 0.927466 w_k1_INTC_s017
                                 + 1.08361 w_k1_KO_s017 + 1.18035 w_k1_MAT_s017
                                 + 1.17158 w_k1_MSFT_s017
                                 + 1.13141 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.799398 w_k1_ADM_s100
                                 + 0.662356 w_k1_AXP_s100
                                 + 0.613505 w_k1_DIS_s100
                                 + 0.807976 w_k1_FDX_s100
                                 + 0.656194 w_k1_IBM_s100
                                 + 0.886872 w_k1_INTC_s100
                                 + 0.665763 w_k1_KO_s100
                                 + 0.678003 w_k1_MAT_s100
                                 + 0.806842 w_k1_MSFT_s100
                                 + 0.811803 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.978704 w_k1_ADM_s100
                                 + 1.34911 w_k1_AXP_s100
                                 + 0.962908 w_k1_DIS_s100
                                 + 0.969617 w_k1_FDX_s100
                                 + 1.38986 w_k1_IBM_s100
                                 + 0.929646 w_k1_INTC_s100
                                 + 1.2237 w_k1_KO_s100 + 1.6788 w_k1_MAT_s100
                                 + 0.965522 w_k1_MSFT_s100
                                 + 1.13809 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.32502 w_k1_ADM_s100
                                 + 1.71076 w_k1_AXP_s100
                                 + 1.73007 w_k1_DIS_s100
                                 + 1.42895 w_k1_FDX_s100
                                 + 1.22869 w_k1_IBM_s100
                                 + 1.48553 w_k1_INTC_s100
                                 + 1.58685 w_k1_KO_s100 + 1.64237 w_k1_MAT_s100
                                 + 1.46547 w_k1_MSFT_s100
                                 + 1.25212 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.04448 w_k1_ADM_s100
                                 + 1.05206 w_k1_AXP_s100
                                 + 1.36551 w_k1_DIS_s100
                                 + 1.13807 w_k1_FDX_s100
                                 + 1.19262 w_k1_IBM_s100
                                 + 1.08443 w_k1_INTC_s100
                                 + 1.44231 w_k1_KO_s100 + 1.05767 w_k1_MAT_s100
                                 + 1.2125 w_k1_MSFT_s100 + 1.10938 w_k1_PG_s100
                                 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.02887 w_k1_ADM_s100
                                 + 1.13899 w_k1_AXP_s100
                                 + 0.92725 w_k1_DIS_s100
                                 + 1.13137 w_k1_FDX_s100
                                 + 0.808499 w_k1_IBM_s100
                                 + 0.965807 w_k1_INTC_s100
                                 + 0.826319 w_k1_KO_s100
                                 + 0.762878 w_k1_MAT_s100
                                 + 0.878178 w_k1_MSFT_s100
                                 + 0.945356 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.789144 w_k1_ADM_s100
                                 + 0.771233 w_k1_AXP_s100
                                 + 0.674275 w_k1_DIS_s100
                                 + 0.774232 w_k1_FDX_s100
                                 + 1.01854 w_k1_IBM_s100
                                 + 0.936709 w_k1_INTC_s100
                                 + 0.929519 w_k1_KO_s100
                                 + 1.06703 w_k1_MAT_s100
                                 + 0.870905 w_k1_MSFT_s100
                                 + 0.855987 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.873476 w_k1_ADM_s100
                                 + 0.963516 w_k1_AXP_s100
                                 + 0.964343 w_k1_DIS_s100
                                 + 0.970546 w_k1_FDX_s100
                                 + 0.888068 w_k1_IBM_s100
                                 + 0.947489 w_k1_INTC_s100
                                 + 1.0228 w_k1_KO_s100 + 0.888059 w_k1_MAT_s100
                                 + 0.902828 w_k1_MSFT_s100
                                 + 1.03392 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.15159 w_k1_ADM_s100
                                 + 1.06002 w_k1_AXP_s100
                                 + 1.24507 w_k1_DIS_s100 + 1.2988 w_k1_FDX_s100
                                 + 0.787044 w_k1_IBM_s100
                                 + 1.3856 w_k1_INTC_s100 + 1.09219 w_k1_KO_s100
                                 + 0.774685 w_k1_MAT_s100
                                 + 1.2563 w_k1_MSFT_s100 + 1.05189 w_k1_PG_s100
                                 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.937149 w_k1_ADM_s101
                                 + 0.895519 w_k1_AXP_s101
                                 + 0.743035 w_k1_DIS_s101
                                 + 0.866507 w_k1_FDX_s101
                                 + 0.814404 w_k1_IBM_s101
                                 + 0.775896 w_k1_INTC_s101
                                 + 0.739148 w_k1_KO_s101
                                 + 0.645235 w_k1_MAT_s101
                                 + 0.882689 w_k1_MSFT_s101
                                 + 0.811853 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.926632 w_k1_ADM_s101
                                 + 0.808256 w_k1_AXP_s101
                                 + 0.896533 w_k1_DIS_s101
                                 + 0.947353 w_k1_FDX_s101
                                 + 0.6918 w_k1_IBM_s101
                                 + 0.938278 w_k1_INTC_s101
                                 + 0.801131 w_k1_KO_s101
                                 + 0.474648 w_k1_MAT_s101
                                 + 0.98993 w_k1_MSFT_s101
                                 + 0.821073 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.1132 w_k1_ADM_s101
                                 + 1.31619 w_k1_AXP_s101
                                 + 1.35314 w_k1_DIS_s101
                                 + 1.26718 w_k1_FDX_s101
                                 + 1.16616 w_k1_IBM_s101
                                 + 1.28366 w_k1_INTC_s101
                                 + 1.22471 w_k1_KO_s101
                                 + 0.850478 w_k1_MAT_s101
                                 + 1.26897 w_k1_MSFT_s101
                                 + 1.05947 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.1965 w_k1_ADM_s101
                                 + 1.6603 w_k1_AXP_s101 + 1.59789 w_k1_DIS_s101
                                 + 1.31848 w_k1_FDX_s101
                                 + 1.68153 w_k1_IBM_s101
                                 + 1.18199 w_k1_INTC_s101
                                 + 1.20174 w_k1_KO_s101 + 2.03665 w_k1_MAT_s101
                                 + 1.16364 w_k1_MSFT_s101
                                 + 1.22087 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.13736 w_k1_ADM_s101
                                 + 1.50037 w_k1_AXP_s101 + 1.536 w_k1_DIS_s101
                                 + 1.48807 w_k1_FDX_s101
                                 + 0.839279 w_k1_IBM_s101
                                 + 1.29556 w_k1_INTC_s101
                                 + 0.993747 w_k1_KO_s101
                                 + 0.925247 w_k1_MAT_s101
                                 + 1.24719 w_k1_MSFT_s101
                                 + 0.931057 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.999487 w_k1_ADM_s101
                                 + 1.21978 w_k1_AXP_s101
                                 + 0.934342 w_k1_DIS_s101
                                 + 1.04182 w_k1_FDX_s101
                                 + 1.37041 w_k1_IBM_s101
                                 + 1.04942 w_k1_INTC_s101 + 1.0032 w_k1_KO_s101
                                 + 1.42162 w_k1_MAT_s101
                                 + 0.981201 w_k1_MSFT_s101
                                 + 0.987977 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.902938 w_k1_ADM_s101
                                 + 1.3018 w_k1_AXP_s101 + 1.15746 w_k1_DIS_s101
                                 + 1.16295 w_k1_FDX_s101
                                 + 0.955008 w_k1_IBM_s101
                                 + 1.1445 w_k1_INTC_s101 + 1.02784 w_k1_KO_s101
                                 + 1.08701 w_k1_MAT_s101
                                 + 1.22708 w_k1_MSFT_s101
                                 + 0.980198 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.782551 w_k1_ADM_s101
                                 + 0.835381 w_k1_AXP_s101
                                 + 0.725828 w_k1_DIS_s101
                                 + 0.772131 w_k1_FDX_s101
                                 + 1.03262 w_k1_IBM_s101
                                 + 0.818129 w_k1_INTC_s101
                                 + 0.774862 w_k1_KO_s101
                                 + 0.710078 w_k1_MAT_s101
                                 + 0.777984 w_k1_MSFT_s101
                                 + 0.736048 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.13003 w_k1_ADM_s102
                                 + 0.681505 w_k1_AXP_s102
                                 + 0.747797 w_k1_DIS_s102
                                 + 0.822303 w_k1_FDX_s102
                                 + 0.727181 w_k1_IBM_s102
                                 + 0.820286 w_k1_INTC_s102
                                 + 0.78756 w_k1_KO_s102
                                 + 0.690774 w_k1_MAT_s102
                                 + 0.836331 w_k1_MSFT_s102
                                 + 0.87082 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.986318 w_k1_ADM_s102
                                 + 1.01395 w_k1_AXP_s102
                                 + 1.16586 w_k1_DIS_s102
                                 + 1.04952 w_k1_FDX_s102
                                 + 1.25436 w_k1_IBM_s102
                                 + 1.15022 w_k1_INTC_s102
                                 + 1.24668 w_k1_KO_s102 + 0.99836 w_k1_MAT_s102
                                 + 1.01587 w_k1_MSFT_s102
                                 + 1.05688 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.23484 w_k1_ADM_s102
                                 + 1.28957 w_k1_AXP_s102
                                 + 1.44152 w_k1_DIS_s102
                                 + 1.32687 w_k1_FDX_s102
                                 + 1.02755 w_k1_IBM_s102
                                 + 1.21618 w_k1_INTC_s102
                                 + 1.06439 w_k1_KO_s102 + 0.9453 w_k1_MAT_s102
                                 + 1.10452 w_k1_MSFT_s102
                                 + 1.04882 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.95631 w_k1_ADM_s102
                                 + 0.79486 w_k1_AXP_s102
                                 + 0.654977 w_k1_DIS_s102
                                 + 0.751065 w_k1_FDX_s102
                                 + 0.850078 w_k1_IBM_s102
                                 + 0.593103 w_k1_INTC_s102
                                 + 0.719426 w_k1_KO_s102
                                 + 0.791458 w_k1_MAT_s102
                                 + 0.741431 w_k1_MSFT_s102
                                 + 0.754976 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.00446 w_k1_ADM_s102
                                 + 1.07414 w_k1_AXP_s102
                                 + 0.985094 w_k1_DIS_s102
                                 + 1.0386 w_k1_FDX_s102
                                 + 0.853851 w_k1_IBM_s102
                                 + 1.01995 w_k1_INTC_s102
                                 + 1.00811 w_k1_KO_s102
                                 + 0.769775 w_k1_MAT_s102
                                 + 0.948929 w_k1_MSFT_s102
                                 + 0.971007 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.35325 w_k1_ADM_s102
                                 + 1.33303 w_k1_AXP_s102
                                 + 1.51917 w_k1_DIS_s102
                                 + 1.22475 w_k1_FDX_s102
                                 + 1.53996 w_k1_IBM_s102
                                 + 1.18427 w_k1_INTC_s102
                                 + 1.50083 w_k1_KO_s102 + 1.65595 w_k1_MAT_s102
                                 + 1.24196 w_k1_MSFT_s102
                                 + 1.10439 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.863864 w_k1_ADM_s102
                                 + 0.737535 w_k1_AXP_s102
                                 + 0.913521 w_k1_DIS_s102
                                 + 0.844297 w_k1_FDX_s102
                                 + 1.08464 w_k1_IBM_s102
                                 + 1.0482 w_k1_INTC_s102 + 1.09499 w_k1_KO_s102
                                 + 0.936201 w_k1_MAT_s102
                                 + 0.946215 w_k1_MSFT_s102
                                 + 0.960549 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.27202 w_k1_ADM_s102
                                 + 1.48293 w_k1_AXP_s102
                                 + 1.01931 w_k1_DIS_s102
                                 + 1.11954 w_k1_FDX_s102 + 1.1243 w_k1_IBM_s102
                                 + 0.891666 w_k1_INTC_s102
                                 + 1.09582 w_k1_KO_s102 + 1.76712 w_k1_MAT_s102
                                 + 0.91679 w_k1_MSFT_s102
                                 + 1.00455 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.830881 w_k1_ADM_s103
                                 + 1.03257 w_k1_AXP_s103
                                 + 0.789573 w_k1_DIS_s103
                                 + 0.921538 w_k1_FDX_s103
                                 + 1.31378 w_k1_IBM_s103
                                 + 0.91392 w_k1_INTC_s103
                                 + 1.13497 w_k1_KO_s103 + 1.4892 w_k1_MAT_s103
                                 + 1.10969 w_k1_MSFT_s103
                                 + 1.07126 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.03022 w_k1_ADM_s103
                                 + 1.14735 w_k1_AXP_s103
                                 + 1.34256 w_k1_DIS_s103 + 1.198 w_k1_FDX_s103
                                 + 0.905666 w_k1_IBM_s103
                                 + 1.14418 w_k1_INTC_s103
                                 + 1.06782 w_k1_KO_s103
                                 + 0.976563 w_k1_MAT_s103
                                 + 1.10633 w_k1_MSFT_s103 + 1.0262 w_k1_PG_s103
                                 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.644028 w_k1_ADM_s103
                                 + 0.690562 w_k1_AXP_s103
                                 + 0.881065 w_k1_DIS_s103
                                 + 0.811499 w_k1_FDX_s103
                                 + 1.09265 w_k1_IBM_s103
                                 + 0.918477 w_k1_INTC_s103
                                 + 0.988585 w_k1_KO_s103
                                 + 1.13757 w_k1_MAT_s103
                                 + 0.943603 w_k1_MSFT_s103
                                 + 0.942504 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.963705 w_k1_ADM_s103
                                 + 0.710901 w_k1_AXP_s103
                                 + 1.15829 w_k1_DIS_s103
                                 + 0.980539 w_k1_FDX_s103
                                 + 0.775676 w_k1_IBM_s103
                                 + 0.833562 w_k1_INTC_s103
                                 + 0.979395 w_k1_KO_s103
                                 + 0.766748 w_k1_MAT_s103
                                 + 0.984603 w_k1_MSFT_s103
                                 + 0.9199 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.06116 w_k1_ADM_s103
                                 + 1.17197 w_k1_AXP_s103
                                 + 1.61062 w_k1_DIS_s103
                                 + 1.14564 w_k1_FDX_s103
                                 + 1.39837 w_k1_IBM_s103
                                 + 1.14719 w_k1_INTC_s103
                                 + 1.44407 w_k1_KO_s103 + 1.51808 w_k1_MAT_s103
                                 + 1.29997 w_k1_MSFT_s103
                                 + 1.24085 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.977637 w_k1_ADM_s103
                                 + 0.840365 w_k1_AXP_s103
                                 + 0.821573 w_k1_DIS_s103
                                 + 0.934459 w_k1_FDX_s103
                                 + 0.858773 w_k1_IBM_s103
                                 + 0.944593 w_k1_INTC_s103
                                 + 0.97049 w_k1_KO_s103
                                 + 0.937035 w_k1_MAT_s103
                                 + 0.842584 w_k1_MSFT_s103
                                 + 0.911117 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.705825 w_k1_ADM_s103
                                 + 0.564533 w_k1_AXP_s103
                                 + 0.543783 w_k1_DIS_s103
                                 + 0.713729 w_k1_FDX_s103
                                 + 0.894817 w_k1_IBM_s103
                                 + 0.715295 w_k1_INTC_s103
                                 + 0.777851 w_k1_KO_s103
                                 + 0.963753 w_k1_MAT_s103
                                 + 0.766032 w_k1_MSFT_s103
                                 + 0.832476 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.673244 w_k1_ADM_s103
                                 + 0.729542 w_k1_AXP_s103
                                 + 0.65865 w_k1_DIS_s103
                                 + 0.843641 w_k1_FDX_s103
                                 + 0.671383 w_k1_IBM_s103
                                 + 0.796376 w_k1_INTC_s103
                                 + 0.664145 w_k1_KO_s103
                                 + 0.693955 w_k1_MAT_s103
                                 + 0.724285 w_k1_MSFT_s103
                                 + 0.807292 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.11036 w_k1_ADM_s104
                                 + 1.27241 w_k1_AXP_s104 + 1.001 w_k1_DIS_s104
                                 + 1.09948 w_k1_FDX_s104
                                 + 0.878198 w_k1_IBM_s104
                                 + 1.0719 w_k1_INTC_s104
                                 + 0.861753 w_k1_KO_s104
                                 + 0.940149 w_k1_MAT_s104
                                 + 0.966708 w_k1_MSFT_s104
                                 + 1.13593 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.03258 w_k1_ADM_s104
                                 + 1.39369 w_k1_AXP_s104
                                 + 1.22677 w_k1_DIS_s104
                                 + 1.09547 w_k1_FDX_s104
                                 + 1.43708 w_k1_IBM_s104
                                 + 1.18574 w_k1_INTC_s104
                                 + 1.25996 w_k1_KO_s104 + 1.53411 w_k1_MAT_s104
                                 + 1.05919 w_k1_MSFT_s104
                                 + 1.18178 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.847377 w_k1_ADM_s104
                                 + 0.832846 w_k1_AXP_s104
                                 + 0.71097 w_k1_DIS_s104
                                 + 0.81938 w_k1_FDX_s104
                                 + 0.840174 w_k1_IBM_s104
                                 + 0.967453 w_k1_INTC_s104
                                 + 0.805821 w_k1_KO_s104
                                 + 0.684815 w_k1_MAT_s104
                                 + 0.966746 w_k1_MSFT_s104
                                 + 0.8925 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.50623 w_k1_ADM_s104
                                 + 1.15495 w_k1_AXP_s104
                                 + 1.49608 w_k1_DIS_s104
                                 + 1.35782 w_k1_FDX_s104
                                 + 0.890334 w_k1_IBM_s104
                                 + 1.25336 w_k1_INTC_s104
                                 + 1.10548 w_k1_KO_s104
                                 + 0.689016 w_k1_MAT_s104
                                 + 1.4257 w_k1_MSFT_s104 + 1.30993 w_k1_PG_s104
                                 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.942792 w_k1_ADM_s104
                                 + 0.751183 w_k1_AXP_s104
                                 + 1.21975 w_k1_DIS_s104
                                 + 1.03442 w_k1_FDX_s104
                                 + 0.853741 w_k1_IBM_s104
                                 + 1.12563 w_k1_INTC_s104
                                 + 0.971137 w_k1_KO_s104
                                 + 0.527661 w_k1_MAT_s104
                                 + 1.19252 w_k1_MSFT_s104
                                 + 1.01956 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.860174 w_k1_ADM_s104
                                 + 0.874669 w_k1_AXP_s104
                                 + 0.693004 w_k1_DIS_s104
                                 + 0.775801 w_k1_FDX_s104
                                 + 1.30857 w_k1_IBM_s104
                                 + 0.893472 w_k1_INTC_s104
                                 + 0.942011 w_k1_KO_s104 + 1.3032 w_k1_MAT_s104
                                 + 0.822677 w_k1_MSFT_s104
                                 + 0.996678 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.37032 w_k1_ADM_s104
                                 + 1.78671 w_k1_AXP_s104
                                 + 1.63075 w_k1_DIS_s104
                                 + 1.41395 w_k1_FDX_s104
                                 + 1.39061 w_k1_IBM_s104
                                 + 1.0626 w_k1_INTC_s104 + 1.19379 w_k1_KO_s104
                                 + 1.82119 w_k1_MAT_s104
                                 + 1.17059 w_k1_MSFT_s104
                                 + 1.16197 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.951813 w_k1_ADM_s104
                                 + 0.767651 w_k1_AXP_s104
                                 + 1.11153 w_k1_DIS_s104
                                 + 0.820898 w_k1_FDX_s104
                                 + 1.04415 w_k1_IBM_s104
                                 + 0.909234 w_k1_INTC_s104
                                 + 1.17152 w_k1_KO_s104
                                 + 0.710958 w_k1_MAT_s104
                                 + 1.03464 w_k1_MSFT_s104
                                 + 1.03051 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.30885 w_k1_ADM_s105
                                 + 1.26378 w_k1_AXP_s105
                                 + 1.62754 w_k1_DIS_s105
                                 + 1.24398 w_k1_FDX_s105
                                 + 1.26226 w_k1_IBM_s105
                                 + 0.906925 w_k1_INTC_s105
                                 + 1.2697 w_k1_KO_s105 + 1.23907 w_k1_MAT_s105
                                 + 1.29957 w_k1_MSFT_s105
                                 + 1.20397 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.837952 w_k1_ADM_s105
                                 + 1.04595 w_k1_AXP_s105
                                 + 0.972736 w_k1_DIS_s105
                                 + 0.878631 w_k1_FDX_s105
                                 + 1.42123 w_k1_IBM_s105
                                 + 0.746407 w_k1_INTC_s105
                                 + 1.08333 w_k1_KO_s105 + 1.43058 w_k1_MAT_s105
                                 + 0.844351 w_k1_MSFT_s105
                                 + 0.904163 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.703154 w_k1_ADM_s105
                                 + 0.690678 w_k1_AXP_s105
                                 + 0.752972 w_k1_DIS_s105
                                 + 0.766048 w_k1_FDX_s105
                                 + 0.832116 w_k1_IBM_s105
                                 + 0.881219 w_k1_INTC_s105
                                 + 0.790847 w_k1_KO_s105
                                 + 0.619423 w_k1_MAT_s105
                                 + 0.901517 w_k1_MSFT_s105
                                 + 0.888694 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.04607 w_k1_ADM_s105
                                 + 1.24485 w_k1_AXP_s105
                                 + 1.34253 w_k1_DIS_s105
                                 + 1.22797 w_k1_FDX_s105
                                 + 1.03247 w_k1_IBM_s105
                                 + 1.03117 w_k1_INTC_s105
                                 + 1.05255 w_k1_KO_s105 + 1.22348 w_k1_MAT_s105
                                 + 1.06399 w_k1_MSFT_s105
                                 + 1.14885 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.918618 w_k1_ADM_s105
                                 + 0.828146 w_k1_AXP_s105
                                 + 0.80444 w_k1_DIS_s105
                                 + 0.918471 w_k1_FDX_s105
                                 + 0.74622 w_k1_IBM_s105
                                 + 0.825416 w_k1_INTC_s105
                                 + 0.829803 w_k1_KO_s105
                                 + 0.545094 w_k1_MAT_s105
                                 + 0.877397 w_k1_MSFT_s105
                                 + 0.89837 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.22401 w_k1_ADM_s105
                                 + 1.04554 w_k1_AXP_s105
                                 + 1.23748 w_k1_DIS_s105 + 1.1971 w_k1_FDX_s105
                                 + 0.803442 w_k1_IBM_s105
                                 + 1.08552 w_k1_INTC_s105
                                 + 1.00889 w_k1_KO_s105
                                 + 0.839739 w_k1_MAT_s105
                                 + 1.1613 w_k1_MSFT_s105 + 1.0092 w_k1_PG_s105
                                 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.907604 w_k1_ADM_s105
                                 + 0.844293 w_k1_AXP_s105
                                 + 0.882343 w_k1_DIS_s105
                                 + 0.977998 w_k1_FDX_s105
                                 + 1.0331 w_k1_IBM_s105
                                 + 0.890666 w_k1_INTC_s105
                                 + 0.899714 w_k1_KO_s105
                                 + 0.970072 w_k1_MAT_s105
                                 + 0.967213 w_k1_MSFT_s105
                                 + 0.906949 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.573699 w_k1_ADM_s105
                                 + 0.652695 w_k1_AXP_s105
                                 + 0.612894 w_k1_DIS_s105
                                 + 0.706836 w_k1_FDX_s105
                                 + 0.930574 w_k1_IBM_s105
                                 + 0.799048 w_k1_INTC_s105
                                 + 0.808648 w_k1_KO_s105
                                 + 0.966434 w_k1_MAT_s105
                                 + 0.712593 w_k1_MSFT_s105
                                 + 0.786729 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.999371 w_k1_ADM_s106
                                 + 0.918256 w_k1_AXP_s106
                                 + 0.964496 w_k1_DIS_s106
                                 + 0.902138 w_k1_FDX_s106
                                 + 1.22527 w_k1_IBM_s106
                                 + 1.04558 w_k1_INTC_s106 + 1.1631 w_k1_KO_s106
                                 + 1.15359 w_k1_MAT_s106
                                 + 0.953581 w_k1_MSFT_s106
                                 + 1.00039 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.07459 w_k1_ADM_s106
                                 + 1.11366 w_k1_AXP_s106
                                 + 1.06688 w_k1_DIS_s106
                                 + 0.966364 w_k1_FDX_s106
                                 + 1.5976 w_k1_IBM_s106
                                 + 1.04225 w_k1_INTC_s106
                                 + 1.37298 w_k1_KO_s106 + 1.62722 w_k1_MAT_s106
                                 + 1.05251 w_k1_MSFT_s106
                                 + 1.19396 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.927185 w_k1_ADM_s106
                                 + 0.951347 w_k1_AXP_s106
                                 + 1.05857 w_k1_DIS_s106
                                 + 1.01377 w_k1_FDX_s106
                                 + 0.93372 w_k1_IBM_s106
                                 + 1.09993 w_k1_INTC_s106
                                 + 1.06419 w_k1_KO_s106
                                 + 0.979826 w_k1_MAT_s106
                                 + 1.05132 w_k1_MSFT_s106
                                 + 1.05369 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.795816 w_k1_ADM_s106
                                 + 0.590396 w_k1_AXP_s106
                                 + 0.646226 w_k1_DIS_s106
                                 + 0.767264 w_k1_FDX_s106
                                 + 0.985333 w_k1_IBM_s106
                                 + 0.888275 w_k1_INTC_s106
                                 + 1.0091 w_k1_KO_s106 + 0.701117 w_k1_MAT_s106
                                 + 0.865079 w_k1_MSFT_s106
                                 + 0.827025 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.809601 w_k1_ADM_s106
                                 + 0.680263 w_k1_AXP_s106
                                 + 0.749166 w_k1_DIS_s106
                                 + 0.887571 w_k1_FDX_s106
                                 + 0.636953 w_k1_IBM_s106
                                 + 0.852892 w_k1_INTC_s106
                                 + 0.772808 w_k1_KO_s106
                                 + 0.623524 w_k1_MAT_s106
                                 + 0.78893 w_k1_MSFT_s106
                                 + 0.836548 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.24953 w_k1_ADM_s106
                                 + 1.31538 w_k1_AXP_s106
                                 + 1.57271 w_k1_DIS_s106
                                 + 1.35987 w_k1_FDX_s106
                                 + 1.38146 w_k1_IBM_s106
                                 + 1.33253 w_k1_INTC_s106
                                 + 1.46132 w_k1_KO_s106 + 1.35893 w_k1_MAT_s106
                                 + 1.20826 w_k1_MSFT_s106
                                 + 1.23064 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.12283 w_k1_ADM_s106
                                 + 1.23677 w_k1_AXP_s106
                                 + 1.54511 w_k1_DIS_s106 + 1.3359 w_k1_FDX_s106
                                 + 0.861355 w_k1_IBM_s106
                                 + 1.31828 w_k1_INTC_s106
                                 + 1.18435 w_k1_KO_s106 + 1.02088 w_k1_MAT_s106
                                 + 1.2817 w_k1_MSFT_s106 + 1.09677 w_k1_PG_s106
                                 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.08942 w_k1_ADM_s106
                                 + 1.58153 w_k1_AXP_s106
                                 + 1.17768 w_k1_DIS_s106
                                 + 1.30391 w_k1_FDX_s106
                                 + 1.03561 w_k1_IBM_s106
                                 + 1.22433 w_k1_INTC_s106
                                 + 1.00575 w_k1_KO_s106 + 1.31121 w_k1_MAT_s106
                                 + 1.15599 w_k1_MSFT_s106 + 1.0116 w_k1_PG_s106
                                 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.953951 w_k1_ADM_s107
                                 + 0.886056 w_k1_AXP_s107
                                 + 1.05617 w_k1_DIS_s107
                                 + 0.98376 w_k1_FDX_s107
                                 + 0.870182 w_k1_IBM_s107
                                 + 1.05317 w_k1_INTC_s107
                                 + 0.973359 w_k1_KO_s107
                                 + 0.856078 w_k1_MAT_s107
                                 + 1.00138 w_k1_MSFT_s107
                                 + 1.03299 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.927489 w_k1_ADM_s107
                                 + 1.12527 w_k1_AXP_s107
                                 + 1.21397 w_k1_DIS_s107 + 1.2242 w_k1_FDX_s107
                                 + 0.776023 w_k1_IBM_s107
                                 + 1.49219 w_k1_INTC_s107
                                 + 0.952217 w_k1_KO_s107
                                 + 0.74395 w_k1_MAT_s107
                                 + 1.25032 w_k1_MSFT_s107
                                 + 1.13198 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.32421 w_k1_ADM_s107
                                 + 1.09698 w_k1_AXP_s107 + 1.601 w_k1_DIS_s107
                                 + 1.27901 w_k1_FDX_s107
                                 + 1.19088 w_k1_IBM_s107
                                 + 1.20632 w_k1_INTC_s107
                                 + 1.49755 w_k1_KO_s107 + 1.13802 w_k1_MAT_s107
                                 + 1.38528 w_k1_MSFT_s107
                                 + 1.23295 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.946978 w_k1_ADM_s107
                                 + 1.12976 w_k1_AXP_s107
                                 + 0.876491 w_k1_DIS_s107
                                 + 0.97682 w_k1_FDX_s107
                                 + 0.921767 w_k1_IBM_s107
                                 + 1.02745 w_k1_INTC_s107
                                 + 0.827202 w_k1_KO_s107
                                 + 1.01275 w_k1_MAT_s107
                                 + 0.905263 w_k1_MSFT_s107
                                 + 1.07086 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.04584 w_k1_ADM_s107
                                 + 1.28795 w_k1_AXP_s107
                                 + 1.10426 w_k1_DIS_s107
                                 + 1.17689 w_k1_FDX_s107
                                 + 1.17033 w_k1_IBM_s107
                                 + 1.23068 w_k1_INTC_s107
                                 + 1.13901 w_k1_KO_s107 + 1.12209 w_k1_MAT_s107
                                 + 1.11899 w_k1_MSFT_s107
                                 + 1.14521 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.587935 w_k1_ADM_s107
                                 + 0.52436 w_k1_AXP_s107
                                 + 0.605848 w_k1_DIS_s107
                                 + 0.717874 w_k1_FDX_s107
                                 + 0.700006 w_k1_IBM_s107
                                 + 0.798939 w_k1_INTC_s107
                                 + 0.728341 w_k1_KO_s107
                                 + 0.651618 w_k1_MAT_s107
                                 + 0.77922 w_k1_MSFT_s107
                                 + 0.837872 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.957929 w_k1_ADM_s107
                                 + 0.778899 w_k1_AXP_s107
                                 + 0.653277 w_k1_DIS_s107
                                 + 0.771052 w_k1_FDX_s107
                                 + 0.877833 w_k1_IBM_s107
                                 + 0.750985 w_k1_INTC_s107
                                 + 0.801513 w_k1_KO_s107
                                 + 0.854562 w_k1_MAT_s107
                                 + 0.775877 w_k1_MSFT_s107
                                 + 0.918408 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.925453 w_k1_ADM_s107
                                 + 1.08231 w_k1_AXP_s107
                                 + 1.12066 w_k1_DIS_s107
                                 + 0.939241 w_k1_FDX_s107
                                 + 1.39239 w_k1_IBM_s107
                                 + 1.17273 w_k1_INTC_s107
                                 + 1.22899 w_k1_KO_s107
                                 + 0.907691 w_k1_MAT_s107
                                 + 0.94568 w_k1_MSFT_s107
                                 + 1.06911 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.67981 w_k1_ADM_s110
                                 + 0.789648 w_k1_AXP_s110
                                 + 0.842621 w_k1_DIS_s110
                                 + 0.853204 w_k1_FDX_s110
                                 + 0.805509 w_k1_IBM_s110
                                 + 0.885014 w_k1_INTC_s110
                                 + 0.866102 w_k1_KO_s110
                                 + 0.979042 w_k1_MAT_s110
                                 + 0.967556 w_k1_MSFT_s110
                                 + 0.989744 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.727123 w_k1_ADM_s110
                                 + 0.916498 w_k1_AXP_s110
                                 + 0.953977 w_k1_DIS_s110
                                 + 0.840902 w_k1_FDX_s110
                                 + 1.4871 w_k1_IBM_s110
                                 + 1.07732 w_k1_INTC_s110
                                 + 1.07865 w_k1_KO_s110 + 1.27894 w_k1_MAT_s110
                                 + 0.854043 w_k1_MSFT_s110
                                 + 1.00133 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.812703 w_k1_ADM_s110
                                 + 0.762831 w_k1_AXP_s110
                                 + 0.920433 w_k1_DIS_s110
                                 + 0.78016 w_k1_FDX_s110
                                 + 1.08946 w_k1_IBM_s110
                                 + 0.813576 w_k1_INTC_s110
                                 + 0.919189 w_k1_KO_s110
                                 + 0.90899 w_k1_MAT_s110
                                 + 0.855815 w_k1_MSFT_s110
                                 + 1.00236 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.812633 w_k1_ADM_s110
                                 + 0.682211 w_k1_AXP_s110
                                 + 0.633577 w_k1_DIS_s110
                                 + 0.704159 w_k1_FDX_s110
                                 + 0.717662 w_k1_IBM_s110
                                 + 0.601483 w_k1_INTC_s110
                                 + 0.587455 w_k1_KO_s110
                                 + 0.743814 w_k1_MAT_s110
                                 + 0.642247 w_k1_MSFT_s110
                                 + 0.835657 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.33549 w_k1_ADM_s110
                                 + 1.46489 w_k1_AXP_s110
                                 + 1.70712 w_k1_DIS_s110
                                 + 1.31566 w_k1_FDX_s110
                                 + 1.40596 w_k1_IBM_s110
                                 + 1.27249 w_k1_INTC_s110
                                 + 1.43381 w_k1_KO_s110 + 1.33803 w_k1_MAT_s110
                                 + 1.25722 w_k1_MSFT_s110
                                 + 1.36012 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.946544 w_k1_ADM_s110
                                 + 0.832001 w_k1_AXP_s110
                                 + 1.10836 w_k1_DIS_s110
                                 + 0.953753 w_k1_FDX_s110
                                 + 1.01545 w_k1_IBM_s110
                                 + 1.05107 w_k1_INTC_s110
                                 + 0.975422 w_k1_KO_s110
                                 + 0.95205 w_k1_MAT_s110
                                 + 1.02455 w_k1_MSFT_s110
                                 + 1.03672 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.93576 w_k1_ADM_s110
                                 + 1.04253 w_k1_AXP_s110
                                 + 1.20733 w_k1_DIS_s110
                                 + 1.13754 w_k1_FDX_s110
                                 + 0.774311 w_k1_IBM_s110
                                 + 0.970581 w_k1_INTC_s110
                                 + 0.869017 w_k1_KO_s110
                                 + 0.889332 w_k1_MAT_s110
                                 + 1.05009 w_k1_MSFT_s110 + 1.1211 w_k1_PG_s110
                                 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.867458 w_k1_ADM_s110
                                 + 1.27423 w_k1_AXP_s110
                                 + 1.18312 w_k1_DIS_s110
                                 + 1.08353 w_k1_FDX_s110
                                 + 1.26301 w_k1_IBM_s110
                                 + 1.01213 w_k1_INTC_s110
                                 + 1.06019 w_k1_KO_s110 + 1.44617 w_k1_MAT_s110
                                 + 1.10439 w_k1_MSFT_s110
                                 + 1.16968 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.27053 w_k1_ADM_s111
                                 + 1.52709 w_k1_AXP_s111
                                 + 1.64147 w_k1_DIS_s111
                                 + 1.29035 w_k1_FDX_s111
                                 + 1.08433 w_k1_IBM_s111
                                 + 1.06256 w_k1_INTC_s111
                                 + 1.12077 w_k1_KO_s111 + 1.37232 w_k1_MAT_s111
                                 + 1.12691 w_k1_MSFT_s111
                                 + 1.10368 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.86493 w_k1_ADM_s111
                                 + 0.82213 w_k1_AXP_s111
                                 + 0.806076 w_k1_DIS_s111
                                 + 0.789055 w_k1_FDX_s111
                                 + 0.807875 w_k1_IBM_s111
                                 + 0.92016 w_k1_INTC_s111
                                 + 0.844442 w_k1_KO_s111
                                 + 0.537685 w_k1_MAT_s111
                                 + 0.76448 w_k1_MSFT_s111
                                 + 0.890452 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.07581 w_k1_ADM_s111
                                 + 0.995503 w_k1_AXP_s111
                                 + 1.37201 w_k1_DIS_s111
                                 + 1.10933 w_k1_FDX_s111
                                 + 0.764259 w_k1_IBM_s111
                                 + 1.02577 w_k1_INTC_s111
                                 + 1.04279 w_k1_KO_s111
                                 + 0.731138 w_k1_MAT_s111
                                 + 1.08741 w_k1_MSFT_s111
                                 + 1.10548 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.984434 w_k1_ADM_s111
                                 + 1.22994 w_k1_AXP_s111
                                 + 1.07253 w_k1_DIS_s111
                                 + 1.04168 w_k1_FDX_s111
                                 + 1.18107 w_k1_IBM_s111
                                 + 0.922625 w_k1_INTC_s111
                                 + 1.04519 w_k1_KO_s111 + 1.56311 w_k1_MAT_s111
                                 + 1.00877 w_k1_MSFT_s111
                                 + 1.15942 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.18016 w_k1_ADM_s111
                                 + 0.815594 w_k1_AXP_s111
                                 + 1.23643 w_k1_DIS_s111
                                 + 0.972712 w_k1_FDX_s111
                                 + 0.972992 w_k1_IBM_s111
                                 + 0.931714 w_k1_INTC_s111
                                 + 0.977877 w_k1_KO_s111
                                 + 0.89455 w_k1_MAT_s111
                                 + 0.917544 w_k1_MSFT_s111
                                 + 1.01614 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.461 w_k1_ADM_s111
                                 + 1.63953 w_k1_AXP_s111
                                 + 1.93373 w_k1_DIS_s111
                                 + 1.30249 w_k1_FDX_s111
                                 + 1.66605 w_k1_IBM_s111
                                 + 1.26469 w_k1_INTC_s111 + 1.4953 w_k1_KO_s111
                                 + 2.0095 w_k1_MAT_s111
                                 + 1.19431 w_k1_MSFT_s111
                                 + 1.19645 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.870424 w_k1_ADM_s111
                                 + 0.651437 w_k1_AXP_s111
                                 + 0.886583 w_k1_DIS_s111
                                 + 0.799728 w_k1_FDX_s111
                                 + 0.623673 w_k1_IBM_s111
                                 + 0.835709 w_k1_INTC_s111
                                 + 0.715264 w_k1_KO_s111
                                 + 0.464777 w_k1_MAT_s111
                                 + 0.80526 w_k1_MSFT_s111
                                 + 0.758025 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.25295 w_k1_ADM_s111
                                 + 0.96616 w_k1_AXP_s111
                                 + 1.19458 w_k1_DIS_s111
                                 + 0.947379 w_k1_FDX_s111
                                 + 1.32632 w_k1_IBM_s111
                                 + 1.00567 w_k1_INTC_s111 + 1.2039 w_k1_KO_s111
                                 + 1.12152 w_k1_MAT_s111
                                 + 0.882263 w_k1_MSFT_s111
                                 + 1.0712 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.07325 w_k1_ADM_s112
                                 + 1.08545 w_k1_AXP_s112
                                 + 1.16982 w_k1_DIS_s112
                                 + 1.08348 w_k1_FDX_s112
                                 + 0.880515 w_k1_IBM_s112
                                 + 1.1374 w_k1_INTC_s112
                                 + 0.973343 w_k1_KO_s112
                                 + 0.828983 w_k1_MAT_s112
                                 + 1.07327 w_k1_MSFT_s112
                                 + 1.00217 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.17457 w_k1_ADM_s112
                                 + 1.44998 w_k1_AXP_s112 + 1.389 w_k1_DIS_s112
                                 + 1.23655 w_k1_FDX_s112
                                 + 1.41373 w_k1_IBM_s112
                                 + 1.53178 w_k1_INTC_s112
                                 + 1.23182 w_k1_KO_s112 + 1.11587 w_k1_MAT_s112
                                 + 1.23035 w_k1_MSFT_s112
                                 + 1.24195 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.905642 w_k1_ADM_s112
                                 + 0.9806 w_k1_AXP_s112 + 1.13551 w_k1_DIS_s112
                                 + 0.918157 w_k1_FDX_s112
                                 + 1.42454 w_k1_IBM_s112
                                 + 0.930088 w_k1_INTC_s112
                                 + 1.14263 w_k1_KO_s112 + 1.08439 w_k1_MAT_s112
                                 + 1.06468 w_k1_MSFT_s112
                                 + 1.08687 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.855565 w_k1_ADM_s112
                                 + 0.971377 w_k1_AXP_s112
                                 + 0.957625 w_k1_DIS_s112
                                 + 0.867911 w_k1_FDX_s112
                                 + 0.946077 w_k1_IBM_s112
                                 + 0.876305 w_k1_INTC_s112
                                 + 1.02965 w_k1_KO_s112 + 1.06884 w_k1_MAT_s112
                                 + 0.915866 w_k1_MSFT_s112
                                 + 1.03065 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.18471 w_k1_ADM_s112
                                 + 1.19029 w_k1_AXP_s112
                                 + 0.903183 w_k1_DIS_s112
                                 + 1.05378 w_k1_FDX_s112
                                 + 0.878133 w_k1_IBM_s112
                                 + 0.895031 w_k1_INTC_s112
                                 + 0.909879 w_k1_KO_s112
                                 + 0.912439 w_k1_MAT_s112
                                 + 0.889533 w_k1_MSFT_s112
                                 + 1.01425 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.41086 w_k1_ADM_s112
                                 + 1.69084 w_k1_AXP_s112
                                 + 1.49667 w_k1_DIS_s112
                                 + 1.39351 w_k1_FDX_s112 + 1.1414 w_k1_IBM_s112
                                 + 1.11879 w_k1_INTC_s112
                                 + 1.30437 w_k1_KO_s112 + 1.55478 w_k1_MAT_s112
                                 + 1.12285 w_k1_MSFT_s112
                                 + 1.28677 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.776641 w_k1_ADM_s112
                                 + 0.741675 w_k1_AXP_s112
                                 + 0.676789 w_k1_DIS_s112
                                 + 0.764976 w_k1_FDX_s112
                                 + 0.744083 w_k1_IBM_s112
                                 + 0.873289 w_k1_INTC_s112
                                 + 0.727359 w_k1_KO_s112
                                 + 0.543646 w_k1_MAT_s112
                                 + 0.832277 w_k1_MSFT_s112
                                 + 0.885117 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.25671 w_k1_ADM_s112
                                 + 1.22226 w_k1_AXP_s112
                                 + 1.61068 w_k1_DIS_s112
                                 + 1.20839 w_k1_FDX_s112 + 1.1353 w_k1_IBM_s112
                                 + 1.12469 w_k1_INTC_s112
                                 + 1.39826 w_k1_KO_s112 + 1.06144 w_k1_MAT_s112
                                 + 1.27401 w_k1_MSFT_s112
                                 + 1.21333 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.721003 w_k1_ADM_s113
                                 + 0.985758 w_k1_AXP_s113
                                 + 0.792108 w_k1_DIS_s113
                                 + 0.965162 w_k1_FDX_s113
                                 + 0.669688 w_k1_IBM_s113
                                 + 0.77246 w_k1_INTC_s113
                                 + 0.680875 w_k1_KO_s113
                                 + 0.772824 w_k1_MAT_s113
                                 + 0.824124 w_k1_MSFT_s113
                                 + 0.818207 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.939016 w_k1_ADM_s113
                                 + 0.912001 w_k1_AXP_s113
                                 + 1.08934 w_k1_DIS_s113
                                 + 0.917809 w_k1_FDX_s113
                                 + 1.43484 w_k1_IBM_s113
                                 + 0.849675 w_k1_INTC_s113
                                 + 1.11504 w_k1_KO_s113 + 1.2211 w_k1_MAT_s113
                                 + 0.968124 w_k1_MSFT_s113
                                 + 0.989294 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.88922 w_k1_ADM_s113
                                 + 1.04533 w_k1_AXP_s113
                                 + 1.23873 w_k1_DIS_s113
                                 + 1.08555 w_k1_FDX_s113
                                 + 0.949704 w_k1_IBM_s113
                                 + 0.938178 w_k1_INTC_s113
                                 + 0.969871 w_k1_KO_s113
                                 + 0.796464 w_k1_MAT_s113
                                 + 1.0399 w_k1_MSFT_s113 + 1.00836 w_k1_PG_s113
                                 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.975391 w_k1_ADM_s113
                                 + 0.976911 w_k1_AXP_s113
                                 + 1.04313 w_k1_DIS_s113
                                 + 1.09395 w_k1_FDX_s113
                                 + 0.870497 w_k1_IBM_s113
                                 + 0.969568 w_k1_INTC_s113
                                 + 0.894516 w_k1_KO_s113
                                 + 0.660007 w_k1_MAT_s113
                                 + 0.972209 w_k1_MSFT_s113
                                 + 1.02601 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.737956 w_k1_ADM_s113
                                 + 0.551037 w_k1_AXP_s113
                                 + 0.681681 w_k1_DIS_s113
                                 + 0.736473 w_k1_FDX_s113
                                 + 0.698098 w_k1_IBM_s113
                                 + 0.653581 w_k1_INTC_s113
                                 + 0.723239 w_k1_KO_s113
                                 + 0.523806 w_k1_MAT_s113
                                 + 0.750975 w_k1_MSFT_s113
                                 + 0.780026 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.849782 w_k1_ADM_s113
                                 + 0.735888 w_k1_AXP_s113
                                 + 1.04701 w_k1_DIS_s113
                                 + 0.895935 w_k1_FDX_s113
                                 + 0.876925 w_k1_IBM_s113
                                 + 0.724881 w_k1_INTC_s113
                                 + 1.02407 w_k1_KO_s113
                                 + 0.715768 w_k1_MAT_s113
                                 + 1.0734 w_k1_MSFT_s113 + 1.01842 w_k1_PG_s113
                                 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.813531 w_k1_ADM_s113
                                 + 1.03883 w_k1_AXP_s113
                                 + 0.723476 w_k1_DIS_s113
                                 + 0.905172 w_k1_FDX_s113
                                 + 1.0091 w_k1_IBM_s113
                                 + 0.80859 w_k1_INTC_s113
                                 + 0.769094 w_k1_KO_s113
                                 + 1.20851 w_k1_MAT_s113
                                 + 0.836019 w_k1_MSFT_s113
                                 + 1.00115 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.05861 w_k1_ADM_s113
                                 + 1.46205 w_k1_AXP_s113
                                 + 1.50678 w_k1_DIS_s113
                                 + 1.36324 w_k1_FDX_s113
                                 + 1.25909 w_k1_IBM_s113
                                 + 1.12552 w_k1_INTC_s113
                                 + 1.29579 w_k1_KO_s113 + 1.53089 w_k1_MAT_s113
                                 + 1.09057 w_k1_MSFT_s113
                                 + 1.18875 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.27326 w_k1_ADM_s114
                                 + 1.16658 w_k1_AXP_s114 + 1.2553 w_k1_DIS_s114
                                 + 1.23212 w_k1_FDX_s114
                                 + 0.933714 w_k1_IBM_s114
                                 + 0.990482 w_k1_INTC_s114
                                 + 1.1543 w_k1_KO_s114 + 0.685154 w_k1_MAT_s114
                                 + 1.21336 w_k1_MSFT_s114
                                 + 1.09865 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.00304 w_k1_ADM_s114
                                 + 1.50859 w_k1_AXP_s114
                                 + 1.31612 w_k1_DIS_s114
                                 + 1.22615 w_k1_FDX_s114 + 1.6701 w_k1_IBM_s114
                                 + 1.05249 w_k1_INTC_s114
                                 + 1.24593 w_k1_KO_s114 + 1.83135 w_k1_MAT_s114
                                 + 1.1949 w_k1_MSFT_s114 + 1.15294 w_k1_PG_s114
                                 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.961849 w_k1_ADM_s114
                                 + 1.68011 w_k1_AXP_s114
                                 + 1.02704 w_k1_DIS_s114
                                 + 1.15602 w_k1_FDX_s114
                                 + 1.14015 w_k1_IBM_s114
                                 + 0.942002 w_k1_INTC_s114
                                 + 0.922095 w_k1_KO_s114
                                 + 1.26299 w_k1_MAT_s114
                                 + 0.982283 w_k1_MSFT_s114
                                 + 1.15079 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.885741 w_k1_ADM_s114
                                 + 0.978747 w_k1_AXP_s114
                                 + 0.869584 w_k1_DIS_s114
                                 + 0.972054 w_k1_FDX_s114
                                 + 0.79288 w_k1_IBM_s114
                                 + 0.835914 w_k1_INTC_s114
                                 + 0.928085 w_k1_KO_s114
                                 + 0.894417 w_k1_MAT_s114
                                 + 1.04416 w_k1_MSFT_s114
                                 + 1.06985 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.27426 w_k1_ADM_s114
                                 + 1.86955 w_k1_AXP_s114
                                 + 1.88292 w_k1_DIS_s114 + 1.3871 w_k1_FDX_s114
                                 + 1.60162 w_k1_IBM_s114
                                 + 1.44581 w_k1_INTC_s114
                                 + 1.35036 w_k1_KO_s114 + 1.41927 w_k1_MAT_s114
                                 + 1.47069 w_k1_MSFT_s114 + 1.2428 w_k1_PG_s114
                                 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.774699 w_k1_ADM_s114
                                 + 0.864609 w_k1_AXP_s114
                                 + 0.64937 w_k1_DIS_s114
                                 + 0.754668 w_k1_FDX_s114
                                 + 0.680922 w_k1_IBM_s114
                                 + 0.846345 w_k1_INTC_s114
                                 + 0.601805 w_k1_KO_s114
                                 + 0.522104 w_k1_MAT_s114
                                 + 0.794264 w_k1_MSFT_s114
                                 + 0.900792 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.18862 w_k1_ADM_s114
                                 + 1.08477 w_k1_AXP_s114
                                 + 0.971016 w_k1_DIS_s114
                                 + 0.929222 w_k1_FDX_s114
                                 + 1.37303 w_k1_IBM_s114
                                 + 0.789973 w_k1_INTC_s114
                                 + 1.14972 w_k1_KO_s114 + 1.48782 w_k1_MAT_s114
                                 + 0.955595 w_k1_MSFT_s114
                                 + 1.12882 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.854622 w_k1_ADM_s114
                                 + 0.985894 w_k1_AXP_s114
                                 + 0.771125 w_k1_DIS_s114
                                 + 0.810554 w_k1_FDX_s114
                                 + 1.14432 w_k1_IBM_s114
                                 + 0.786473 w_k1_INTC_s114
                                 + 0.90441 w_k1_KO_s114 + 1.09242 w_k1_MAT_s114
                                 + 0.805546 w_k1_MSFT_s114
                                 + 1.03086 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.977167 w_k1_ADM_s115
                                 + 0.908672 w_k1_AXP_s115
                                 + 0.901159 w_k1_DIS_s115
                                 + 0.824558 w_k1_FDX_s115
                                 + 1.10756 w_k1_IBM_s115
                                 + 0.808555 w_k1_INTC_s115
                                 + 0.888682 w_k1_KO_s115
                                 + 0.815913 w_k1_MAT_s115
                                 + 1.03611 w_k1_MSFT_s115
                                 + 0.982972 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.99897 w_k1_ADM_s115
                                 + 1.24944 w_k1_AXP_s115
                                 + 1.17161 w_k1_DIS_s115
                                 + 0.970591 w_k1_FDX_s115
                                 + 1.69392 w_k1_IBM_s115
                                 + 1.00883 w_k1_INTC_s115
                                 + 1.28532 w_k1_KO_s115 + 2.12131 w_k1_MAT_s115
                                 + 1.16176 w_k1_MSFT_s115
                                 + 1.33485 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.01807 w_k1_ADM_s115
                                 + 1.53726 w_k1_AXP_s115
                                 + 1.74813 w_k1_DIS_s115
                                 + 1.42297 w_k1_FDX_s115
                                 + 1.04908 w_k1_IBM_s115
                                 + 1.43325 w_k1_INTC_s115
                                 + 1.32522 w_k1_KO_s115 + 1.55332 w_k1_MAT_s115
                                 + 1.60101 w_k1_MSFT_s115
                                 + 1.38356 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.26131 w_k1_ADM_s115
                                 + 1.66339 w_k1_AXP_s115 + 1.4152 w_k1_DIS_s115
                                 + 1.32553 w_k1_FDX_s115
                                 + 1.50361 w_k1_IBM_s115
                                 + 1.26744 w_k1_INTC_s115 + 1.1691 w_k1_KO_s115
                                 + 1.95994 w_k1_MAT_s115
                                 + 1.31261 w_k1_MSFT_s115
                                 + 1.07604 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.22552 w_k1_ADM_s115
                                 + 1.32901 w_k1_AXP_s115
                                 + 1.30811 w_k1_DIS_s115
                                 + 1.29872 w_k1_FDX_s115
                                 + 0.805892 w_k1_IBM_s115
                                 + 1.21837 w_k1_INTC_s115
                                 + 1.10289 w_k1_KO_s115
                                 + 0.939564 w_k1_MAT_s115
                                 + 1.44045 w_k1_MSFT_s115
                                 + 1.13386 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.10087 w_k1_ADM_s115
                                 + 1.20518 w_k1_AXP_s115
                                 + 1.15752 w_k1_DIS_s115
                                 + 1.07307 w_k1_FDX_s115
                                 + 0.835432 w_k1_IBM_s115
                                 + 0.987111 w_k1_INTC_s115
                                 + 0.885286 w_k1_KO_s115
                                 + 0.970025 w_k1_MAT_s115
                                 + 1.26524 w_k1_MSFT_s115
                                 + 1.14842 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.91152 w_k1_ADM_s115
                                 + 1.20034 w_k1_AXP_s115
                                 + 0.794824 w_k1_DIS_s115
                                 + 0.907505 w_k1_FDX_s115
                                 + 0.923847 w_k1_IBM_s115
                                 + 0.786673 w_k1_INTC_s115
                                 + 0.835969 w_k1_KO_s115
                                 + 1.30581 w_k1_MAT_s115
                                 + 0.787539 w_k1_MSFT_s115
                                 + 0.880546 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.825739 w_k1_ADM_s115
                                 + 0.681785 w_k1_AXP_s115
                                 + 0.783433 w_k1_DIS_s115
                                 + 0.839225 w_k1_FDX_s115
                                 + 0.65572 w_k1_IBM_s115
                                 + 0.708959 w_k1_INTC_s115
                                 + 0.760329 w_k1_KO_s115
                                 + 0.693319 w_k1_MAT_s115
                                 + 0.961812 w_k1_MSFT_s115
                                 + 0.905294 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.833185 w_k1_ADM_s116
                                 + 0.922092 w_k1_AXP_s116
                                 + 0.798256 w_k1_DIS_s116
                                 + 0.863124 w_k1_FDX_s116
                                 + 1.21343 w_k1_IBM_s116
                                 + 1.19739 w_k1_INTC_s116
                                 + 0.995827 w_k1_KO_s116 + 1.0336 w_k1_MAT_s116
                                 + 0.903853 w_k1_MSFT_s116
                                 + 0.915538 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.20523 w_k1_ADM_s116
                                 + 1.4385 w_k1_AXP_s116 + 1.29245 w_k1_DIS_s116
                                 + 1.26684 w_k1_FDX_s116
                                 + 0.996546 w_k1_IBM_s116
                                 + 1.36373 w_k1_INTC_s116
                                 + 1.05303 w_k1_KO_s116 + 1.18345 w_k1_MAT_s116
                                 + 1.07462 w_k1_MSFT_s116
                                 + 0.998627 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.38673 w_k1_ADM_s116
                                 + 1.42536 w_k1_AXP_s116
                                 + 1.72144 w_k1_DIS_s116
                                 + 1.44287 w_k1_FDX_s116
                                 + 1.41178 w_k1_IBM_s116
                                 + 1.72982 w_k1_INTC_s116
                                 + 1.46308 w_k1_KO_s116 + 1.46167 w_k1_MAT_s116
                                 + 1.32688 w_k1_MSFT_s116
                                 + 1.16681 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.884567 w_k1_ADM_s116
                                 + 0.754002 w_k1_AXP_s116
                                 + 0.814636 w_k1_DIS_s116
                                 + 0.949264 w_k1_FDX_s116
                                 + 0.754679 w_k1_IBM_s116
                                 + 0.945263 w_k1_INTC_s116
                                 + 0.799695 w_k1_KO_s116
                                 + 0.775033 w_k1_MAT_s116
                                 + 0.866362 w_k1_MSFT_s116
                                 + 0.86912 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.868932 w_k1_ADM_s116
                                 + 0.992622 w_k1_AXP_s116
                                 + 1.03911 w_k1_DIS_s116
                                 + 1.03309 w_k1_FDX_s116
                                 + 1.00809 w_k1_IBM_s116
                                 + 1.19222 w_k1_INTC_s116
                                 + 1.04019 w_k1_KO_s116
                                 + 0.960115 w_k1_MAT_s116
                                 + 1.0422 w_k1_MSFT_s116
                                 + 0.984575 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.729158 w_k1_ADM_s116
                                 + 0.77416 w_k1_AXP_s116
                                 + 0.694996 w_k1_DIS_s116
                                 + 0.820785 w_k1_FDX_s116
                                 + 0.78018 w_k1_IBM_s116
                                 + 0.949849 w_k1_INTC_s116
                                 + 0.791749 w_k1_KO_s116
                                 + 0.792498 w_k1_MAT_s116
                                 + 0.857779 w_k1_MSFT_s116
                                 + 0.786256 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.673207 w_k1_ADM_s116
                                 + 0.643676 w_k1_AXP_s116
                                 + 0.596171 w_k1_DIS_s116
                                 + 0.736326 w_k1_FDX_s116
                                 + 0.814539 w_k1_IBM_s116
                                 + 0.803007 w_k1_INTC_s116
                                 + 0.805961 w_k1_KO_s116
                                 + 0.778067 w_k1_MAT_s116
                                 + 0.681748 w_k1_MSFT_s116
                                 + 0.802698 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.04872 w_k1_ADM_s116
                                 + 1.20526 w_k1_AXP_s116
                                 + 1.20225 w_k1_DIS_s116
                                 + 1.06685 w_k1_FDX_s116
                                 + 1.36991 w_k1_IBM_s116
                                 + 1.30572 w_k1_INTC_s116 + 1.3227 w_k1_KO_s116
                                 + 1.05159 w_k1_MAT_s116
                                 + 1.26727 w_k1_MSFT_s116
                                 + 1.07427 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.722318 w_k1_ADM_s117
                                 + 0.792215 w_k1_AXP_s117
                                 + 0.723583 w_k1_DIS_s117
                                 + 0.798938 w_k1_FDX_s117
                                 + 0.921382 w_k1_IBM_s117
                                 + 0.874546 w_k1_INTC_s117
                                 + 0.83096 w_k1_KO_s117 + 0.86606 w_k1_MAT_s117
                                 + 0.817993 w_k1_MSFT_s117
                                 + 0.886429 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.999615 w_k1_ADM_s117
                                 + 1.82927 w_k1_AXP_s117
                                 + 1.81309 w_k1_DIS_s117
                                 + 1.46476 w_k1_FDX_s117 + 1.3933 w_k1_IBM_s117
                                 + 1.37673 w_k1_INTC_s117 + 1.3587 w_k1_KO_s117
                                 + 1.70962 w_k1_MAT_s117
                                 + 1.16992 w_k1_MSFT_s117
                                 + 1.22547 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.842279 w_k1_ADM_s117
                                 + 0.75932 w_k1_AXP_s117
                                 + 0.771683 w_k1_DIS_s117
                                 + 0.924405 w_k1_FDX_s117
                                 + 0.809145 w_k1_IBM_s117
                                 + 0.937074 w_k1_INTC_s117
                                 + 0.771034 w_k1_KO_s117
                                 + 0.633352 w_k1_MAT_s117
                                 + 0.883771 w_k1_MSFT_s117
                                 + 0.921815 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.10457 w_k1_ADM_s117
                                 + 1.22887 w_k1_AXP_s117
                                 + 1.30151 w_k1_DIS_s117
                                 + 1.21906 w_k1_FDX_s117
                                 + 0.986713 w_k1_IBM_s117
                                 + 1.12326 w_k1_INTC_s117
                                 + 1.24221 w_k1_KO_s117 + 1.41789 w_k1_MAT_s117
                                 + 1.10896 w_k1_MSFT_s117
                                 + 1.24388 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.00826 w_k1_ADM_s117
                                 + 1.15481 w_k1_AXP_s117
                                 + 1.11756 w_k1_DIS_s117
                                 + 1.16406 w_k1_FDX_s117
                                 + 1.45822 w_k1_IBM_s117
                                 + 1.12956 w_k1_INTC_s117
                                 + 1.21565 w_k1_KO_s117 + 2.1712 w_k1_MAT_s117
                                 + 1.01412 w_k1_MSFT_s117
                                 + 1.20324 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.944087 w_k1_ADM_s117
                                 + 0.947225 w_k1_AXP_s117
                                 + 1.07867 w_k1_DIS_s117
                                 + 1.00788 w_k1_FDX_s117
                                 + 0.976165 w_k1_IBM_s117
                                 + 0.890843 w_k1_INTC_s117
                                 + 0.960986 w_k1_KO_s117
                                 + 1.11489 w_k1_MAT_s117
                                 + 0.954676 w_k1_MSFT_s117
                                 + 0.993562 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.915888 w_k1_ADM_s117
                                 + 0.808374 w_k1_AXP_s117
                                 + 0.797435 w_k1_DIS_s117
                                 + 0.824248 w_k1_FDX_s117
                                 + 1.24287 w_k1_IBM_s117
                                 + 1.01595 w_k1_INTC_s117
                                 + 1.03503 w_k1_KO_s117 + 1.10127 w_k1_MAT_s117
                                 + 0.927337 w_k1_MSFT_s117
                                 + 0.915475 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.17966 w_k1_ADM_s117
                                 + 1.03538 w_k1_AXP_s117
                                 + 1.11642 w_k1_DIS_s117 + 1.1759 w_k1_FDX_s117
                                 + 0.684915 w_k1_IBM_s117
                                 + 1.00814 w_k1_INTC_s117
                                 + 0.826474 w_k1_KO_s117
                                 + 1.12883 w_k1_MAT_s117
                                 + 1.04946 w_k1_MSFT_s117 + 1.0394 w_k1_PG_s117
                                 >= 1
 k1_return_constraint:           0.00552044742725542 w_k1_ADM_s000
                                 + 0.00574212676216509 w_k1_ADM_s001
                                 + 0.0086690251288117 w_k1_ADM_s002
                                 + 0.00360815102433846 w_k1_ADM_s003
                                 + 0.00326975600171763 w_k1_ADM_s004
                                 + 0.0133586061103572 w_k1_ADM_s005
                                 + 0.0110880301589659 w_k1_ADM_s006
                                 + 0.0146029162365669 w_k1_ADM_s007
                                 + 0.0339979924975726 w_k1_ADM_s010
                                 + 0.0563150143420864 w_k1_ADM_s011
                                 + 0.028295431842689 w_k1_ADM_s012
                                 + 0.0301458671951386 w_k1_ADM_s013
                                 + 0.0374640896123276 w_k1_ADM_s014
                                 + 0.0514508492847722 w_k1_ADM_s015
                                 + 0.0188129310153763 w_k1_ADM_s016
                                 + 0.0225206753130439 w_k1_ADM_s017
                                 + 0.0855791198580436 w_k1_ADM_s100
                                 + 0.109579704296608 w_k1_ADM_s101
                                 + 0.0378500424317512 w_k1_ADM_s102
                                 + 0.0211390679504836 w_k1_ADM_s103
                                 + 0.0438659758090396 w_k1_ADM_s104
                                 + 0.0277856594389304 w_k1_ADM_s105
                                 + 0.0859959295072756 w_k1_ADM_s106
                                 + 0.0443742939849754 w_k1_ADM_s107
                                 + 0.0245001746442485 w_k1_ADM_s110
                                 + 0.0380691652528847 w_k1_ADM_s111
                                 + 0.0332996935135076 w_k1_ADM_s112
                                 + 0.0135219777780612 w_k1_ADM_s113
                                 + 0.0136763660635913 w_k1_ADM_s114
                                 + 0.0170883772451343 w_k1_ADM_s115
                                 + 0.0341640275379947 w_k1_ADM_s116
                                 + 0.0142900355377337 w_k1_ADM_s117
                                 + 0.00560341430072938 w_k1_AXP_s000
                                 + 0.00604751228552836 w_k1_AXP_s001
                                 + 0.0073427710715874 w_k1_AXP_s002
                                 + 0.00355017292055751 w_k1_AXP_s003
                                 + 0.00375521763722273 w_k1_AXP_s004
                                 + 0.0138027111440093 w_k1_AXP_s005
                                 + 0.013433953473461 w_k1_AXP_s006
                                 + 0.0126914428480386 w_k1_AXP_s007
                                 + 0.0317204116940579 w_k1_AXP_s010
                                 + 0.0600028842919549 w_k1_AXP_s011
                                 + 0.0257613487668026 w_k1_AXP_s012
                                 + 0.0257714560082134 w_k1_AXP_s013
                                 + 0.0430857359966126 w_k1_AXP_s014
                                 + 0.0537908387802121 w_k1_AXP_s015
                                 + 0.0206864543382192 w_k1_AXP_s016
                                 + 0.0234542926935218 w_k1_AXP_s017
                                 + 0.0941898663841858 w_k1_AXP_s100
                                 + 0.124739758168583 w_k1_AXP_s101
                                 + 0.0357880734208273 w_k1_AXP_s102
                                 + 0.0222364783492217 w_k1_AXP_s103
                                 + 0.0455438180337456 w_k1_AXP_s104
                                 + 0.0276195425186739 w_k1_AXP_s105
                                 + 0.0853569789550419 w_k1_AXP_s106
                                 + 0.0442157638843961 w_k1_AXP_s107
                                 + 0.0277857199737974 w_k1_AXP_s110
                                 + 0.035474961669399 w_k1_AXP_s111
                                 + 0.0352535103564235 w_k1_AXP_s112
                                 + 0.0156639563072038 w_k1_AXP_s113
                                 + 0.0162629958412864 w_k1_AXP_s114
                                 + 0.0189746107152918 w_k1_AXP_s115
                                 + 0.0368576081962026 w_k1_AXP_s116
                                 + 0.0148601453813801 w_k1_AXP_s117
                                 + 0.00596532743186481 w_k1_DIS_s000
                                 + 0.00538240582971318 w_k1_DIS_s001
                                 + 0.00741797543138284 w_k1_DIS_s002
                                 + 0.00366087350307209 w_k1_DIS_s003
                                 + 0.00376135628602172 w_k1_DIS_s004
                                 + 0.0130538746584842 w_k1_DIS_s005
                                 + 0.0129126672671832 w_k1_DIS_s006
                                 + 0.0137647303296228 w_k1_DIS_s007
                                 + 0.0332986873834962 w_k1_DIS_s010
                                 + 0.0539747556376211 w_k1_DIS_s011
                                 + 0.0243469790477198 w_k1_DIS_s012
                                 + 0.0274178500779642 w_k1_DIS_s013
                                 + 0.0428259045456232 w_k1_DIS_s014
                                 + 0.0524895571489991 w_k1_DIS_s015
                                 + 0.0213151577160561 w_k1_DIS_s016
                                 + 0.0228820374518426 w_k1_DIS_s017
                                 + 0.0905873046510937 w_k1_DIS_s100
                                 + 0.113807616559135 w_k1_DIS_s101
                                 + 0.0376014427259276 w_k1_DIS_s102
                                 + 0.0253212031874544 w_k1_DIS_s103
                                 + 0.0446066965136102 w_k1_DIS_s104
                                 + 0.030314486576212 w_k1_DIS_s105
                                 + 0.0924019761582672 w_k1_DIS_s106
                                 + 0.0472488060684214 w_k1_DIS_s107
                                 + 0.0302214660062579 w_k1_DIS_s110
                                 + 0.0420778323636336 w_k1_DIS_s111
                                 + 0.0349000971028744 w_k1_DIS_s112
                                 + 0.0159798439700441 w_k1_DIS_s113
                                 + 0.0140410163477556 w_k1_DIS_s114
                                 + 0.0184707021281979 w_k1_DIS_s115
                                 + 0.0369172564319034 w_k1_DIS_s116
                                 + 0.0151878310396617 w_k1_DIS_s117
                                 + 0.0060399921801577 w_k1_FDX_s000
                                 + 0.00558376591771934 w_k1_FDX_s001
                                 + 0.00758720725226917 w_k1_FDX_s002
                                 + 0.00367482299701612 w_k1_FDX_s003
                                 + 0.00357960180041095 w_k1_FDX_s004
                                 + 0.0134103915823729 w_k1_FDX_s005
                                 + 0.0125843221751597 w_k1_FDX_s006
                                 + 0.0129291622221752 w_k1_FDX_s007
                                 + 0.0311166029154108 w_k1_FDX_s010
                                 + 0.0583378044908637 w_k1_FDX_s011
                                 + 0.0253985586924461 w_k1_FDX_s012
                                 + 0.0268247344003647 w_k1_FDX_s013
                                 + 0.042044065479431 w_k1_FDX_s014
                                 + 0.0511014153081186 w_k1_FDX_s015
                                 + 0.0222401281765429 w_k1_FDX_s016
                                 + 0.0235925903305883 w_k1_FDX_s017
                                 + 0.0907401598414215 w_k1_FDX_s100
                                 + 0.117359445526359 w_k1_FDX_s101
                                 + 0.035866047179333 w_k1_FDX_s102
                                 + 0.0235454834036172 w_k1_FDX_s103
                                 + 0.0431708937449755 w_k1_FDX_s104
                                 + 0.0285598522434274 w_k1_FDX_s105
                                 + 0.0909254921192068 w_k1_FDX_s106
                                 + 0.0451960594875022 w_k1_FDX_s107
                                 + 0.027477958167746 w_k1_FDX_s110
                                 + 0.0346740040052049 w_k1_FDX_s111
                                 + 0.0328783818470374 w_k1_FDX_s112
                                 + 0.015783803018328 w_k1_FDX_s113
                                 + 0.0140989273596004 w_k1_FDX_s114
                                 + 0.0171855449469171 w_k1_FDX_s115
                                 + 0.0368615433481442 w_k1_FDX_s116
                                 + 0.0155055226147234 w_k1_FDX_s117
                                 + 0.0059823656363022 w_k1_IBM_s000
                                 + 0.00521411191745517 w_k1_IBM_s001
                                 + 0.00735469195952613 w_k1_IBM_s002
                                 + 0.00387491587932737 w_k1_IBM_s003
                                 + 0.00356808528868473 w_k1_IBM_s004
                                 + 0.0136095426089491 w_k1_IBM_s005
                                 + 0.0128449875766334 w_k1_IBM_s006
                                 + 0.0134879302884596 w_k1_IBM_s007
                                 + 0.0319857700277463 w_k1_IBM_s010
                                 + 0.0579164288928917 w_k1_IBM_s011
                                 + 0.0270682359932587 w_k1_IBM_s012
                                 + 0.0277990943103748 w_k1_IBM_s013
                                 + 0.0399902282653698 w_k1_IBM_s014
                                 + 0.0506547583249225 w_k1_IBM_s015
                                 + 0.0215290293127197 w_k1_IBM_s016
                                 + 0.0234784260474721 w_k1_IBM_s017
                                 + 0.0910531013331532 w_k1_IBM_s100
                                 + 0.116714032636935 w_k1_IBM_s101
                                 + 0.0375900715332587 w_k1_IBM_s102
                                 + 0.0242348869943511 w_k1_IBM_s103
                                 + 0.0430582983519182 w_k1_IBM_s104
                                 + 0.0286978359757057 w_k1_IBM_s105
                                 + 0.0909985780988614 w_k1_IBM_s106
                                 + 0.0459347499258094 w_k1_IBM_s107
                                 + 0.02912671679522 w_k1_IBM_s110
                                 + 0.0358592791197324 w_k1_IBM_s111
                                 + 0.0325002166552082 w_k1_IBM_s112
                                 + 0.0149184733265605 w_k1_IBM_s113
                                 + 0.01440155413652 w_k1_IBM_s114
                                 + 0.017078568967518 w_k1_IBM_s115
                                 + 0.037955894076457 w_k1_IBM_s116
                                 + 0.0153264294514204 w_k1_IBM_s117
                                 + 0.0059816415685853 w_k1_INTC_s000
                                 + 0.00609020801715358 w_k1_INTC_s001
                                 + 0.00739034950181926 w_k1_INTC_s002
                                 + 0.00331881041221139 w_k1_INTC_s003
                                 + 0.00343022241983241 w_k1_INTC_s004
                                 + 0.01442243261895 w_k1_INTC_s005
                                 + 0.0112545342671748 w_k1_INTC_s006
                                 + 0.0126089796807005 w_k1_INTC_s007
                                 + 0.0342674999508825 w_k1_INTC_s010
                                 + 0.0636713254292665 w_k1_INTC_s011
                                 + 0.0247504454797094 w_k1_INTC_s012
                                 + 0.0250034430675418 w_k1_INTC_s013
                                 + 0.0407080913161634 w_k1_INTC_s014
                                 + 0.0642846169823392 w_k1_INTC_s015
                                 + 0.0219086729304394 w_k1_INTC_s016
                                 + 0.0213130837428489 w_k1_INTC_s017
                                 + 0.0916825450261191 w_k1_INTC_s100
                                 + 0.114412707402937 w_k1_INTC_s101
                                 + 0.0359802890407238 w_k1_INTC_s102
                                 + 0.0230686209242549 w_k1_INTC_s103
                                 + 0.0447488223093194 w_k1_INTC_s104
                                 + 0.0248703821691782 w_k1_INTC_s105
                                 + 0.0949966169938322 w_k1_INTC_s106
                                 + 0.0485876185374449 w_k1_INTC_s107
                                 + 0.0271133389521101 w_k1_INTC_s110
                                 + 0.0340014757589418 w_k1_INTC_s111
                                 + 0.033269366991344 w_k1_INTC_s112
                                 + 0.0135459902074039 w_k1_INTC_s113
                                 + 0.0130117798551475 w_k1_INTC_s114
                                 + 0.0162261677847503 w_k1_INTC_s115
                                 + 0.0428295237641801 w_k1_INTC_s116
                                 + 0.0151164515424017 w_k1_INTC_s117
                                 + 0.0068330812741394 w_k1_KO_s000
                                 + 0.00511691906663502 w_k1_KO_s001
                                 + 0.00769990215144657 w_k1_KO_s002
                                 + 0.00375091095312038 w_k1_KO_s003
                                 + 0.00363387480087268 w_k1_KO_s004
                                 + 0.0150203591731899 w_k1_KO_s005
                                 + 0.0111180111624449 w_k1_KO_s006
                                 + 0.0150711799438428 w_k1_KO_s007
                                 + 0.0302200603748335 w_k1_KO_s010
                                 + 0.0566876581193161 w_k1_KO_s011
                                 + 0.0262276562023979 w_k1_KO_s012
                                 + 0.0297175540303148 w_k1_KO_s013
                                 + 0.0391160113698369 w_k1_KO_s014
                                 + 0.0532046915424687 w_k1_KO_s015
                                 + 0.0199736997973239 w_k1_KO_s016
                                 + 0.0210035380793759 w_k1_KO_s017
                                 + 0.0977329073932559 w_k1_KO_s100
                                 + 0.105867170537539 w_k1_KO_s101
                                 + 0.038242323987473 w_k1_KO_s102
                                 + 0.0248695412643218 w_k1_KO_s103
                                 + 0.0416807861522642 w_k1_KO_s104
                                 + 0.0278512162205851 w_k1_KO_s105
                                 + 0.0973032400660643 w_k1_KO_s106
                                 + 0.0472343310964303 w_k1_KO_s107
                                 + 0.0270706758779563 w_k1_KO_s110
                                 + 0.0357891358391411 w_k1_KO_s111
                                 + 0.0336523417991758 w_k1_KO_s112
                                 + 0.0144251098267504 w_k1_KO_s113
                                 + 0.0133386598263999 w_k1_KO_s114
                                 + 0.0163027870401945 w_k1_KO_s115
                                 + 0.0375940611324743 w_k1_KO_s116
                                 + 0.0148020374695149 w_k1_KO_s117
                                 + 0.00759673697318433 w_k1_MAT_s000
                                 + 0.00575403409300026 w_k1_MAT_s001
                                 + 0.00766473263098958 w_k1_MAT_s002
                                 + 0.00401464022795032 w_k1_MAT_s003
                                 + 0.00443561464808366 w_k1_MAT_s004
                                 + 0.0147394938312694 w_k1_MAT_s005
                                 + 0.0129270219358433 w_k1_MAT_s006
                                 + 0.0132796462942156 w_k1_MAT_s007
                                 + 0.026122580154692 w_k1_MAT_s010
                                 + 0.0538351608387252 w_k1_MAT_s011
                                 + 0.02290529451628 w_k1_MAT_s012
                                 + 0.0314133747016908 w_k1_MAT_s013
                                 + 0.0491145420383186 w_k1_MAT_s014
                                 + 0.0410671677463929 w_k1_MAT_s015
                                 + 0.0175790909559427 w_k1_MAT_s016
                                 + 0.0231135229767622 w_k1_MAT_s017
                                 + 0.0968121031741709 w_k1_MAT_s100
                                 + 0.104057139934228 w_k1_MAT_s101
                                 + 0.0356614719205363 w_k1_MAT_s102
                                 + 0.0261358955258087 w_k1_MAT_s103
                                 + 0.0403600032869425 w_k1_MAT_s104
                                 + 0.0276091204359269 w_k1_MAT_s105
                                 + 0.0889632819955994 w_k1_MAT_s106
                                 + 0.0416135930956169 w_k1_MAT_s107
                                 + 0.0302509341788658 w_k1_MAT_s110
                                 + 0.036134098001361 w_k1_MAT_s111
                                 + 0.0310332214062376 w_k1_MAT_s112
                                 + 0.0150349713692806 w_k1_MAT_s113
                                 + 0.013937485624673 w_k1_MAT_s114
                                 + 0.019100807324544 w_k1_MAT_s115
                                 + 0.0358866546740281 w_k1_MAT_s116
                                 + 0.0181166187509447 w_k1_MAT_s117
                                 + 0.00599138487256088 w_k1_MSFT_s000
                                 + 0.00492444402074313 w_k1_MSFT_s001
                                 + 0.00774394678531186 w_k1_MSFT_s002
                                 + 0.00361305359437232 w_k1_MSFT_s003
                                 + 0.00324215212866653 w_k1_MSFT_s004
                                 + 0.0147803341099939 w_k1_MSFT_s005
                                 + 0.0109366522805583 w_k1_MSFT_s006
                                 + 0.0127653675368696 w_k1_MSFT_s007
                                 + 0.0316534204875106 w_k1_MSFT_s010
                                 + 0.0596192924535031 w_k1_MSFT_s011
                                 + 0.0260612282789548 w_k1_MSFT_s012
                                 + 0.0293696642304379 w_k1_MSFT_s013
                                 + 0.0398309221367593 w_k1_MSFT_s014
                                 + 0.0527550664242694 w_k1_MSFT_s015
                                 + 0.0228250646031585 w_k1_MSFT_s016
                                 + 0.0217258000253041 w_k1_MSFT_s017
                                 + 0.0897975313081044 w_k1_MSFT_s100
                                 + 0.11626544539161 w_k1_MSFT_s101
                                 + 0.0346266426257617 w_k1_MSFT_s102
                                 + 0.0242121843628534 w_k1_MSFT_s103
                                 + 0.0446944146534811 w_k1_MSFT_s104
                                 + 0.0281950296308635 w_k1_MSFT_s105
                                 + 0.0902785789056728 w_k1_MSFT_s106
                                 + 0.045854776247024 w_k1_MSFT_s107
                                 + 0.0277500959406001 w_k1_MSFT_s110
                                 + 0.033040601005416 w_k1_MSFT_s111
                                 + 0.0330816187573366 w_k1_MSFT_s112
                                 + 0.0143923420137322 w_k1_MSFT_s113
                                 + 0.0143149688275361 w_k1_MSFT_s114
                                 + 0.0196101636229446 w_k1_MSFT_s115
                                 + 0.0367307076227038 w_k1_MSFT_s116
                                 + 0.0145679687563823 w_k1_MSFT_s117
                                 + 0.00643877275501942 w_k1_PG_s000
                                 + 0.00536422803918311 w_k1_PG_s001
                                 + 0.00752103071010106 w_k1_PG_s002
                                 + 0.00398869845282639 w_k1_PG_s003
                                 + 0.00338395174592337 w_k1_PG_s004
                                 + 0.0125004937443983 w_k1_PG_s005
                                 + 0.0121869131982276 w_k1_PG_s006
                                 + 0.0143630540167706 w_k1_PG_s007
                                 + 0.0300866622955194 w_k1_PG_s010
                                 + 0.055384777317802 w_k1_PG_s011
                                 + 0.0245380841827854 w_k1_PG_s012
                                 + 0.0268465017218113 w_k1_PG_s013
                                 + 0.0389007201974272 w_k1_PG_s014
                                 + 0.0516880085322229 w_k1_PG_s015
                                 + 0.0202834865920079 w_k1_PG_s016
                                 + 0.0236483234371024 w_k1_PG_s017
                                 + 0.0910620736463374 w_k1_PG_s100
                                 + 0.103363557168333 w_k1_PG_s101
                                 + 0.0345741959642167 w_k1_PG_s102
                                 + 0.0237298503012738 w_k1_PG_s103
                                 + 0.044941172395097 w_k1_PG_s104
                                 + 0.0275676699319094 w_k1_PG_s105
                                 + 0.0887869955062851 w_k1_PG_s106
                                 + 0.0478339019302522 w_k1_PG_s107
                                 + 0.0300125857510107 w_k1_PG_s110
                                 + 0.0354381741438374 w_k1_PG_s111
                                 + 0.0348044676384078 w_k1_PG_s112
                                 + 0.0152149930593848 w_k1_PG_s113
                                 + 0.0148156898222207 w_k1_PG_s114
                                 + 0.0182017762649721 w_k1_PG_s115
                                 + 0.0345161990788249 w_k1_PG_s116
                                 + 0.0155509392868176 w_k1_PG_s117 >= 1.17
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
