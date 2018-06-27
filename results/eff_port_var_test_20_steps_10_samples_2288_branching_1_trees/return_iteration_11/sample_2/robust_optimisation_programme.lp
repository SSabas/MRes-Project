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
 w_k1_ADM_s0_return_constraint:  1.21737 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.919245 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.01366 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.946068 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.10856 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.931313 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.876639 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.34614 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.671395 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.925107 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.41228 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.33815 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.08759 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.87434 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.869335 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.16998 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.860096 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.783221 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.691505 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.658764 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.41083 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.93431 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.07444 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.09503 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.940075 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.74349 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.0841 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.79362 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.777666 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.762769 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.26211 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.17776 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.926164 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.12015 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.579836 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.865707 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.27451 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.951952 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.25584 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.894949 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.2295 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.842702 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.10935 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.833908 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.605957 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.08324 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.657439 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.791855 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.7355 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.42078 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.808548 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.05405 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.793317 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.12628 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.70062 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.867721 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.915774 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.494868 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.5956 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.11361 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.32843 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.72185 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.904799 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.789765 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.23825 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.95101 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.703531 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.847826 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.59879 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.00291 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.708343 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.08956 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.567322 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.801252 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.10073 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.819651 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.31805 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.874095 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.26931 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.855849 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.10716 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.895872 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.621375 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.11465 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.584761 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.760903 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 2.1432 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.53068 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.850956 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.08057 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.956668 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.30452 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.830088 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.669184 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.807729 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.53511 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.49812 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.13538 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.18224 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.44259 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.89858 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.60993 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.48227 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.836546 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.518024 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.852542 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.94763 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.901764 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.655737 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.18048 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.534318 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.633993 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.1979 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.939559 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.21724 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.932799 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.09983 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.918206 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.05878 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.910466 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.698963 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.17635 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.644619 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.913868 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.61815 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.37264 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.888433 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.926726 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.868447 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.06605 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.899616 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.820163 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.813998 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.646548 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.42311 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.12284 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.2408 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.25774 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.906169 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.774587 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.13651 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.818537 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.66374 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.735714 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.47559 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.997231 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.740508 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.05013 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.581616 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.774126 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.25644 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.843732 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.18339 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.963412 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.25776 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.771734 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.18225 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.730562 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.750525 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.835956 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.09423 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.586402 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.20071 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.1962 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.00619 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.5469 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.17472 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.35891 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.628077 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.895042 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.32528 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.805317 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.26072 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.826069 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.727313 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.12674 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.840532 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.512917 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.41948 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.1645 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.790473 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.57526 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.36179 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.891218 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.00676 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.21814 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.903515 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.698927 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.740463 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.52632 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.16398 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.952243 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.99341 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.889825 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.04441 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.897763 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.784587 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.23837 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.845467 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.01993 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.66738 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.48223 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.0493 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.992404 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.00978 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.0275 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.967358 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.956833 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.04168 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.726737 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.24073 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.12849 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.32004 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.13339 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.882358 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.779482 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.22852 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.994999 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.708344 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.720862 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.35696 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.02197 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.817784 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.07095 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.765088 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.824004 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.08459 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.875757 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.21546 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.912289 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.27858 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.807219 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.11162 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.901716 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.687823 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.95663 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.763848 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.690016 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.47155 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.07071 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.869684 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.10624 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.16083 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.23127 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.755419 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.802929 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.0313 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.778916 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.40725 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.958539 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.01875 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.25721 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.959805 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.596827 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.42967 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.01751 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.636128 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.18908 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.35618 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.916777 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.83534 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.24058 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.765242 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.699477 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.903971 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.17351 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.24226 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.93671 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.46235 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.739295 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.057 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.788367 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.580523 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.82663 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.91347 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.529425 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.03771 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.21228 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.830257 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.51076 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.06773 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.43613 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.496508 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.890977 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.28687 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.741866 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.52981 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.879295 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.889935 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.96934 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.04543 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.546418 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.99766 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.16661 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.741689 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.94187 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.25048 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.746573 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.754607 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.04168 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.73772 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.752579 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.579957 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.49518 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.17699 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.947643 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.12334 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.96077 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.04558 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.959844 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.678397 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.13303 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.671474 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.845328 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.55965 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.29885 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.863993 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.0121 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.937073 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.07918 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.916731 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.788295 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.797954 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.643827 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.35368 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.02046 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.13146 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.22996 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.887206 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.754932 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.21786 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.913431 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.652841 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.79636 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.43082 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.967779 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.737089 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.05157 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.700936 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.813811 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.06711 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.991095 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.08105 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.935743 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.24414 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.964919 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.04947 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.917166 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.718894 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.947635 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.804747 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.765979 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.11741 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.17402 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.888219 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.958778 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.00347 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.13525 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.877379 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.871822 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.899213 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.784779 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.23544 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.944185 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.08056 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.25409 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.986434 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.798424 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.13312 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.967005 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.80552 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.948745 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.25757 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.01248 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.833378 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.08349 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.771991 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.852072 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.03196 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.06423 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.170000383021372 z_k1_s0000
                                 - 0.128000751558001 z_k1_s0001
                                 - 0.107000220025183 z_k1_s0002
                                 - 0.283000328840972 z_k1_s0003
                                 - 0.194000002000171 z_k1_s0004
                                 - 0.0810001754807602 z_k1_s0005
                                 - 0.100000162147766 z_k1_s0006
                                 - 0.130000222722114 z_k1_s0007
                                 - 0.390000746240715 z_k1_s0010
                                 - 0.257000591783697 z_k1_s0011
                                 - 0.550001766892183 z_k1_s0012
                                 - 0.31000045026541 z_k1_s0013
                                 - 1.12200017485318 z_k1_s0014
                                 - 0.752001324584945 z_k1_s0015
                                 - 0.365000626954629 z_k1_s0016
                                 - 0.540999048855833 z_k1_s0017
                                 - 0.264000702438524 z_k1_s0020
                                 - 0.0960000899411394 z_k1_s0021
                                 - 0.204999970832452 z_k1_s0022
                                 - 0.118000097910973 z_k1_s0023
                                 - 0.297999735600154 z_k1_s0024
                                 - 0.158999940377333 z_k1_s0025
                                 - 0.169999716812055 z_k1_s0026
                                 - 0.207000895548684 z_k1_s0027
                                 - 0.186999691352982 z_k1_s0030
                                 - 0.251998915753997 z_k1_s0031
                                 - 0.172999590122522 z_k1_s0032
                                 - 0.491998575137394 z_k1_s0033
                                 - 0.229999587075926 z_k1_s0034
                                 - 0.402000000366065 z_k1_s0035
                                 - 0.147999616953435 z_k1_s0036
                                 - 0.191999519907126 z_k1_s0037
                                 - 0.0499999938388003 z_k1_s0040
                                 - 0.111000019370151 z_k1_s0041
                                 - 0.123000120707509 z_k1_s0042
                                 - 0.0870000002955174 z_k1_s0043
                                 - 0.245000538970374 z_k1_s0044
                                 - 0.105999671146116 z_k1_s0045
                                 - 0.0580000001970116 z_k1_s0046
                                 - 0.138000247379206 z_k1_s0047
                                 - 0.314998022571736 z_k1_s0050
                                 - 0.175999291189438 z_k1_s0051
                                 - 0.32799916041948 z_k1_s0052
                                 - 0.239999604574435 z_k1_s0053
                                 - 0.104999580189711 z_k1_s0054
                                 - 0.0939996805839172 z_k1_s0055
                                 - 0.411997675775415 z_k1_s0056
                                 - 0.124999577154138 z_k1_s0057
                                 - 0.197999881969867 z_k1_s0060
                                 - 0.347999109511262 z_k1_s0061
                                 - 0.245999927198863 z_k1_s0062
                                 - 0.416001001184104 z_k1_s0063
                                 - 0.351998504080867 z_k1_s0064
                                 - 0.266999642386809 z_k1_s0065
                                 - 0.573999931663925 z_k1_s0066
                                 - 0.644999084765932 z_k1_s0067
                                 - 0.435000174970669 z_k1_s0070
                                 - 0.372999392283535 z_k1_s0071
                                 - 0.361999394564731 z_k1_s0072
                                 - 0.686999592347358 z_k1_s0073
                                 - 0.235999685876386 z_k1_s0074
                                 - 0.442000969063782 z_k1_s0075
                                 - 0.227999820126235 z_k1_s0076
                                 - 0.153999968063258 z_k1_s0077
                                 - 0.110000058111303 z_k1_s0100
                                 - 0.115999455281296 z_k1_s0101
                                 - 0.104999550470118 z_k1_s0102
                                 - 0.123999661507477 z_k1_s0103
                                 - 0.0649999333385446 z_k1_s0104
                                 - 0.201999904713578 z_k1_s0105
                                 - 0.070999936508252 z_k1_s0106
                                 - 0.216999407638085 z_k1_s0107
                                 - 0.56200171058485 z_k1_s0110
                                 - 0.168000403028752 z_k1_s0111
                                 - 0.260000524563389 z_k1_s0112
                                 - 0.373000656426556 z_k1_s0113
                                 - 0.478002310071737 z_k1_s0114
                                 - 0.25700029423038 z_k1_s0115
                                 - 0.331999003744468 z_k1_s0116
                                 - 0.774001113024149 z_k1_s0117
                                 - 0.167000122693244 z_k1_s0120
                                 - 0.118999820503898 z_k1_s0121
                                 - 0.135999815604429 z_k1_s0122
                                 - 0.080999865620315 z_k1_s0123
                                 - 0.318999651827932 z_k1_s0124
                                 - 0.0939998960383264 z_k1_s0125
                                 - 0.280999406544706 z_k1_s0126
                                 - 0.254999926507969 z_k1_s0127
                                 - 0.250000126250938 z_k1_s0130
                                 - 0.279000903199651 z_k1_s0131
                                 - 0.306000633186899 z_k1_s0132
                                 - 0.314000389035058 z_k1_s0133
                                 - 0.490998865173559 z_k1_s0134
                                 - 0.231999567593155 z_k1_s0135
                                 - 0.144999895945636 z_k1_s0136
                                 - 0.198999799119903 z_k1_s0137
                                 - 0.137999520717522 z_k1_s0140
                                 - 0.403997870346592 z_k1_s0141
                                 - 0.292997934753187 z_k1_s0142
                                 - 0.177999378551598 z_k1_s0143
                                 - 0.235998512203317 z_k1_s0144
                                 - 0.485997903415311 z_k1_s0145
                                 - 0.338999863785052 z_k1_s0146
                                 - 0.163999394739789 z_k1_s0147
                                 - 0.154999665937006 z_k1_s0150
                                 - 0.146999348534332 z_k1_s0151
                                 - 0.07699990575569 z_k1_s0152
                                 - 0.128000138450552 z_k1_s0153
                                 - 0.224000242288467 z_k1_s0154
                                 - 0.111999874144622 z_k1_s0155
                                 - 0.139999379556507 z_k1_s0156
                                 - 0.252000118199768 z_k1_s0157
                                 - 0.179000417820635 z_k1_s0160
                                 - 0.187000617461166 z_k1_s0161
                                 - 0.48200024395926 z_k1_s0162
                                 - 0.145000407100705 z_k1_s0163
                                 - 0.242000956761948 z_k1_s0164
                                 - 0.260000344800193 z_k1_s0165
                                 - 0.434000386519797 z_k1_s0166
                                 - 0.304999931898913 z_k1_s0167
                                 - 0.787997467511508 z_k1_s0170
                                 - 0.835998505041567 z_k1_s0171
                                 - 0.223999444490079 z_k1_s0172
                                 - 0.555999199428968 z_k1_s0173
                                 - 0.310999199819797 z_k1_s0174
                                 - 0.248999411405402 z_k1_s0175
                                 - 0.674997892317411 z_k1_s0176
                                 - 0.409000575979266 z_k1_s0177
                                 - 0.535000589016184 z_k1_s1000
                                 - 0.57100104512387 z_k1_s1001
                                 - 0.961003363336589 z_k1_s1002
                                 - 1.46700429819208 z_k1_s1003
                                 - 2.16000251742442 z_k1_s1004
                                 - 1.23800385873572 z_k1_s1005
                                 - 0.446000450776423 z_k1_s1006
                                 - 0.90500191756799 z_k1_s1007
                                 - 0.300000535538958 z_k1_s1010
                                 - 0.287000577334382 z_k1_s1011
                                 - 0.801002460636802 z_k1_s1012
                                 - 0.903000061207578 z_k1_s1013
                                 - 0.790003174595643 z_k1_s1014
                                 - 0.595001480022989 z_k1_s1015
                                 - 0.494002126180304 z_k1_s1016
                                 - 0.473001800826079 z_k1_s1017
                                 - 1.25399430313547 z_k1_s1020
                                 - 0.982000787413943 z_k1_s1021
                                 - 1.0359947410158 z_k1_s1022
                                 - 1.27300112286531 z_k1_s1023
                                 - 1.10100079928607 z_k1_s1024
                                 - 0.997999333517481 z_k1_s1025
                                 - 1.56499510021143 z_k1_s1026
                                 - 0.906998144723645 z_k1_s1027
                                 - 0.614999588715628 z_k1_s1030
                                 - 0.526998507575032 z_k1_s1031
                                 - 0.18399987897752 z_k1_s1032
                                 - 0.160999987667762 z_k1_s1033
                                 - 0.195999822269567 z_k1_s1034
                                 - 0.181999888428845 z_k1_s1035
                                 - 0.409000063202507 z_k1_s1036
                                 - 0.220999953627819 z_k1_s1037
                                 - 0.554000758867764 z_k1_s1040
                                 - 2.2070036490596 z_k1_s1041
                                 - 0.665000806350366 z_k1_s1042
                                 - 0.494000226019882 z_k1_s1043
                                 - 1.93100070841883 z_k1_s1044
                                 - 0.970000047415143 z_k1_s1045
                                 - 0.471000796867337 z_k1_s1046
                                 - 0.867000723699608 z_k1_s1047
                                 - 0.433000999962232 z_k1_s1050
                                 - 0.725998811253304 z_k1_s1051
                                 - 0.257999952623045 z_k1_s1052
                                 - 0.163999957229378 z_k1_s1053
                                 - 0.298999860305396 z_k1_s1054
                                 - 0.606999365882554 z_k1_s1055
                                 - 0.177999949596521 z_k1_s1056
                                 - 0.599999206448996 z_k1_s1057
                                 - 0.201000107178624 z_k1_s1060
                                 - 0.942001315768233 z_k1_s1061
                                 - 0.708999367577638 z_k1_s1062
                                 - 0.40799973455974 z_k1_s1063
                                 - 0.697000327172654 z_k1_s1064
                                 - 0.420999830464798 z_k1_s1065
                                 - 0.60699900813783 z_k1_s1066
                                 - 0.272999885006866 z_k1_s1067
                                 - 0.329999274597168 z_k1_s1070
                                 - 0.25099969645421 z_k1_s1071
                                 - 0.25600092646326 z_k1_s1072
                                 - 0.25300018845783 z_k1_s1073
                                 - 0.144999929462392 z_k1_s1074
                                 - 0.428001044774512 z_k1_s1075
                                 - 0.679000741228722 z_k1_s1076
                                 - 0.139999940453344 z_k1_s1077
                                 - 0.518999297428269 z_k1_s1100
                                 - 0.389000430264376 z_k1_s1101
                                 - 0.321999786234707 z_k1_s1102
                                 - 0.308999308318777 z_k1_s1103
                                 - 0.5319997753442 z_k1_s1104
                                 - 0.335999391375646 z_k1_s1105
                                 - 0.385000965366638 z_k1_s1106
                                 - 0.163999935190314 z_k1_s1107
                                 - 0.197000671083613 z_k1_s1110
                                 - 0.606002341303777 z_k1_s1111
                                 - 0.439999909521759 z_k1_s1112
                                 - 0.236000656018473 z_k1_s1113
                                 - 0.434001390305748 z_k1_s1114
                                 - 0.423002555071428 z_k1_s1115
                                 - 0.452000047963713 z_k1_s1116
                                 - 0.312000499480871 z_k1_s1117
                                 - 0.374000402643024 z_k1_s1120
                                 - 0.130000027678026 z_k1_s1121
                                 - 0.11299996001242 z_k1_s1122
                                 - 0.296999361510374 z_k1_s1123
                                 - 0.140000099422508 z_k1_s1124
                                 - 0.337999492762708 z_k1_s1125
                                 - 0.22000013037822 z_k1_s1126
                                 - 0.197999574340249 z_k1_s1127
                                 - 0.848999595489265 z_k1_s1130
                                 - 0.406997618809209 z_k1_s1131
                                 - 0.539000259386997 z_k1_s1132
                                 - 0.875996590434962 z_k1_s1133
                                 - 0.24999945261799 z_k1_s1134
                                 - 0.359998929506648 z_k1_s1135
                                 - 0.199999487814587 z_k1_s1136
                                 - 0.232999405160989 z_k1_s1137
                                 - 0.0889999333986169 z_k1_s1140
                                 - 0.153999843186388 z_k1_s1141
                                 - 0.19999972242391 z_k1_s1142
                                 - 0.233999447556068 z_k1_s1143
                                 - 0.330999625991537 z_k1_s1144
                                 - 0.122000001438516 z_k1_s1145
                                 - 0.148999423225964 z_k1_s1146
                                 - 0.14400042626496 z_k1_s1147
                                 - 0.412998779746709 z_k1_s1150
                                 - 0.378997747499831 z_k1_s1151
                                 - 0.165999390257879 z_k1_s1152
                                 - 0.561998717077904 z_k1_s1153
                                 - 0.759997848889539 z_k1_s1154
                                 - 0.297999165864554 z_k1_s1155
                                 - 0.413998184697055 z_k1_s1156
                                 - 0.551997579596073 z_k1_s1157
                                 - 0.203999869799784 z_k1_s1160
                                 - 0.584000621488389 z_k1_s1161
                                 - 0.175999767490154 z_k1_s1162
                                 - 0.516000014822572 z_k1_s1163
                                 - 0.241999749948878 z_k1_s1164
                                 - 0.374000550665327 z_k1_s1165
                                 - 0.450999926567894 z_k1_s1166
                                 - 0.238999788737072 z_k1_s1167
                                 - 0.545998668115482 z_k1_s1170
                                 - 0.31799956450584 z_k1_s1171
                                 - 0.429999622067038 z_k1_s1172
                                 - 0.306000019051528 z_k1_s1173
                                 - 0.210999274550729 z_k1_s1174
                                 - 0.193999445491816 z_k1_s1175
                                 - 0.222999336003555 z_k1_s1176
                                 - 0.351998706625152 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.847821 w_k1_ADM_s000
                                 + 1.24587 w_k1_AXP_s000
                                 + 1.41221 w_k1_DIS_s000 + 1.2499 w_k1_FDX_s000
                                 + 1.45015 w_k1_IBM_s000
                                 + 1.16252 w_k1_INTC_s000
                                 + 1.18076 w_k1_KO_s000 + 1.81679 w_k1_MAT_s000
                                 + 1.32 w_k1_MSFT_s000 + 1.29444 w_k1_PG_s000
                                 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.834589 w_k1_ADM_s000
                                 + 0.813669 w_k1_AXP_s000
                                 + 0.599827 w_k1_DIS_s000
                                 + 0.805641 w_k1_FDX_s000
                                 + 0.974193 w_k1_IBM_s000
                                 + 0.923954 w_k1_INTC_s000
                                 + 0.793012 w_k1_KO_s000
                                 + 0.952688 w_k1_MAT_s000
                                 + 0.847831 w_k1_MSFT_s000
                                 + 0.907876 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.02859 w_k1_ADM_s000
                                 + 1.12031 w_k1_AXP_s000
                                 + 0.750518 w_k1_DIS_s000
                                 + 1.01277 w_k1_FDX_s000
                                 + 0.789113 w_k1_IBM_s000
                                 + 0.980513 w_k1_INTC_s000
                                 + 0.813256 w_k1_KO_s000
                                 + 0.857053 w_k1_MAT_s000
                                 + 0.971494 w_k1_MSFT_s000
                                 + 1.0679 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.15977 w_k1_ADM_s000
                                 + 1.31674 w_k1_AXP_s000
                                 + 1.13681 w_k1_DIS_s000
                                 + 1.23094 w_k1_FDX_s000
                                 + 0.960114 w_k1_IBM_s000
                                 + 1.20878 w_k1_INTC_s000
                                 + 0.911038 w_k1_KO_s000
                                 + 0.686301 w_k1_MAT_s000
                                 + 1.21159 w_k1_MSFT_s000
                                 + 1.01632 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.22688 w_k1_ADM_s000
                                 + 1.30728 w_k1_AXP_s000
                                 + 0.951749 w_k1_DIS_s000
                                 + 1.1007 w_k1_FDX_s000 + 1.1291 w_k1_IBM_s000
                                 + 1.21719 w_k1_INTC_s000
                                 + 0.945895 w_k1_KO_s000
                                 + 1.12469 w_k1_MAT_s000
                                 + 1.05883 w_k1_MSFT_s000 + 1.0046 w_k1_PG_s000
                                 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.76301 w_k1_ADM_s000
                                 + 1.08058 w_k1_AXP_s000
                                 + 1.04879 w_k1_DIS_s000
                                 + 0.975812 w_k1_FDX_s000
                                 + 1.2668 w_k1_IBM_s000
                                 + 0.957842 w_k1_INTC_s000
                                 + 1.05901 w_k1_KO_s000 + 1.29237 w_k1_MAT_s000
                                 + 0.952936 w_k1_MSFT_s000
                                 + 1.03581 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.829452 w_k1_ADM_s000
                                 + 0.719699 w_k1_AXP_s000
                                 + 0.707075 w_k1_DIS_s000
                                 + 0.847124 w_k1_FDX_s000
                                 + 0.868256 w_k1_IBM_s000
                                 + 1.06383 w_k1_INTC_s000
                                 + 0.907337 w_k1_KO_s000
                                 + 0.595394 w_k1_MAT_s000
                                 + 1.07722 w_k1_MSFT_s000
                                 + 0.990884 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.724893 w_k1_ADM_s000
                                 + 0.664354 w_k1_AXP_s000
                                 + 0.685773 w_k1_DIS_s000
                                 + 0.883431 w_k1_FDX_s000
                                 + 0.64582 w_k1_IBM_s000
                                 + 0.887557 w_k1_INTC_s000
                                 + 0.805216 w_k1_KO_s000
                                 + 0.854644 w_k1_MAT_s000
                                 + 0.980363 w_k1_MSFT_s000
                                 + 0.885459 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.805857 w_k1_ADM_s001
                                 + 0.864241 w_k1_AXP_s001
                                 + 0.963342 w_k1_DIS_s001
                                 + 0.879056 w_k1_FDX_s001
                                 + 1.39229 w_k1_IBM_s001
                                 + 0.949421 w_k1_INTC_s001
                                 + 0.994765 w_k1_KO_s001
                                 + 1.26648 w_k1_MAT_s001
                                 + 0.924636 w_k1_MSFT_s001
                                 + 1.00595 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.820662 w_k1_ADM_s001
                                 + 0.96998 w_k1_AXP_s001 + 1.4966 w_k1_DIS_s001
                                 + 1.05761 w_k1_FDX_s001 + 1.7137 w_k1_IBM_s001
                                 + 0.921928 w_k1_INTC_s001
                                 + 1.37245 w_k1_KO_s001 + 1.38732 w_k1_MAT_s001
                                 + 1.10252 w_k1_MSFT_s001
                                 + 1.15462 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.04591 w_k1_ADM_s001
                                 + 0.988524 w_k1_AXP_s001
                                 + 1.02387 w_k1_DIS_s001
                                 + 1.14043 w_k1_FDX_s001
                                 + 0.712165 w_k1_IBM_s001
                                 + 0.944674 w_k1_INTC_s001
                                 + 0.859048 w_k1_KO_s001
                                 + 0.708095 w_k1_MAT_s001
                                 + 1.04595 w_k1_MSFT_s001
                                 + 0.990942 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.13212 w_k1_ADM_s001
                                 + 1.04142 w_k1_AXP_s001
                                 + 1.59166 w_k1_DIS_s001
                                 + 1.33995 w_k1_FDX_s001
                                 + 0.977528 w_k1_IBM_s001
                                 + 1.11023 w_k1_INTC_s001
                                 + 1.22509 w_k1_KO_s001 + 1.03359 w_k1_MAT_s001
                                 + 1.17846 w_k1_MSFT_s001
                                 + 1.23016 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.875628 w_k1_ADM_s001
                                 + 0.809907 w_k1_AXP_s001
                                 + 0.725221 w_k1_DIS_s001
                                 + 0.779839 w_k1_FDX_s001
                                 + 0.861572 w_k1_IBM_s001
                                 + 0.747884 w_k1_INTC_s001
                                 + 0.810597 w_k1_KO_s001
                                 + 0.783925 w_k1_MAT_s001
                                 + 0.799685 w_k1_MSFT_s001
                                 + 0.847466 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.874239 w_k1_ADM_s001
                                 + 0.807736 w_k1_AXP_s001
                                 + 1.07994 w_k1_DIS_s001
                                 + 0.966779 w_k1_FDX_s001
                                 + 0.964959 w_k1_IBM_s001
                                 + 1.08217 w_k1_INTC_s001
                                 + 0.945619 w_k1_KO_s001
                                 + 0.596872 w_k1_MAT_s001
                                 + 1.0043 w_k1_MSFT_s001 + 1.02335 w_k1_PG_s001
                                 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.26789 w_k1_ADM_s001
                                 + 1.53092 w_k1_AXP_s001 + 1.5007 w_k1_DIS_s001
                                 + 1.34223 w_k1_FDX_s001
                                 + 1.24048 w_k1_IBM_s001
                                 + 1.22631 w_k1_INTC_s001
                                 + 1.22749 w_k1_KO_s001 + 1.00222 w_k1_MAT_s001
                                 + 1.23115 w_k1_MSFT_s001
                                 + 1.23086 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.904783 w_k1_ADM_s001
                                 + 1.24777 w_k1_AXP_s001
                                 + 0.88145 w_k1_DIS_s001
                                 + 1.01512 w_k1_FDX_s001
                                 + 1.03355 w_k1_IBM_s001
                                 + 0.813717 w_k1_INTC_s001
                                 + 0.977867 w_k1_KO_s001
                                 + 1.32606 w_k1_MAT_s001
                                 + 0.84636 w_k1_MSFT_s001 + 1.0491 w_k1_PG_s001
                                 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.996691 w_k1_ADM_s002
                                 + 1.04808 w_k1_AXP_s002
                                 + 1.12825 w_k1_DIS_s002
                                 + 1.00869 w_k1_FDX_s002
                                 + 1.18888 w_k1_IBM_s002
                                 + 0.818904 w_k1_INTC_s002
                                 + 1.29118 w_k1_KO_s002 + 1.12352 w_k1_MAT_s002
                                 + 0.96501 w_k1_MSFT_s002
                                 + 1.01514 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.952411 w_k1_ADM_s002
                                 + 0.824536 w_k1_AXP_s002
                                 + 0.668223 w_k1_DIS_s002
                                 + 0.870439 w_k1_FDX_s002
                                 + 0.590128 w_k1_IBM_s002
                                 + 0.790305 w_k1_INTC_s002
                                 + 0.701289 w_k1_KO_s002
                                 + 0.654012 w_k1_MAT_s002
                                 + 0.795511 w_k1_MSFT_s002
                                 + 0.793598 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.982251 w_k1_ADM_s002
                                 + 0.94893 w_k1_AXP_s002
                                 + 0.890404 w_k1_DIS_s002
                                 + 0.90177 w_k1_FDX_s002
                                 + 0.795137 w_k1_IBM_s002
                                 + 1.01806 w_k1_INTC_s002
                                 + 0.907892 w_k1_KO_s002
                                 + 0.631439 w_k1_MAT_s002
                                 + 0.974705 w_k1_MSFT_s002
                                 + 0.92648 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.02225 w_k1_ADM_s002
                                 + 0.683055 w_k1_AXP_s002
                                 + 0.671006 w_k1_DIS_s002
                                 + 0.737523 w_k1_FDX_s002
                                 + 0.949441 w_k1_IBM_s002
                                 + 0.735208 w_k1_INTC_s002
                                 + 0.856807 w_k1_KO_s002
                                 + 0.816573 w_k1_MAT_s002
                                 + 0.790695 w_k1_MSFT_s002
                                 + 0.822632 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.17181 w_k1_ADM_s002
                                 + 0.906003 w_k1_AXP_s002
                                 + 0.861284 w_k1_DIS_s002
                                 + 0.986737 w_k1_FDX_s002
                                 + 1.01623 w_k1_IBM_s002
                                 + 0.892224 w_k1_INTC_s002
                                 + 0.968382 w_k1_KO_s002
                                 + 1.09872 w_k1_MAT_s002
                                 + 0.940461 w_k1_MSFT_s002
                                 + 0.935499 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.58216 w_k1_ADM_s002
                                 + 1.36975 w_k1_AXP_s002
                                 + 1.24824 w_k1_DIS_s002
                                 + 1.19486 w_k1_FDX_s002
                                 + 0.810351 w_k1_IBM_s002
                                 + 0.958111 w_k1_INTC_s002
                                 + 1.00908 w_k1_KO_s002
                                 + 0.857573 w_k1_MAT_s002
                                 + 1.06826 w_k1_MSFT_s002
                                 + 1.02505 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.38118 w_k1_ADM_s002
                                 + 1.14104 w_k1_AXP_s002
                                 + 1.43118 w_k1_DIS_s002
                                 + 1.18641 w_k1_FDX_s002
                                 + 1.52034 w_k1_IBM_s002
                                 + 0.943181 w_k1_INTC_s002
                                 + 1.34975 w_k1_KO_s002 + 1.42912 w_k1_MAT_s002
                                 + 1.11596 w_k1_MSFT_s002
                                 + 1.15412 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.22725 w_k1_ADM_s002
                                 + 1.24204 w_k1_AXP_s002
                                 + 1.58417 w_k1_DIS_s002
                                 + 1.38131 w_k1_FDX_s002
                                 + 1.07709 w_k1_IBM_s002
                                 + 1.2229 w_k1_INTC_s002 + 1.22674 w_k1_KO_s002
                                 + 0.991429 w_k1_MAT_s002
                                 + 1.42196 w_k1_MSFT_s002
                                 + 1.06656 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.20365 w_k1_ADM_s003
                                 + 1.17336 w_k1_AXP_s003
                                 + 1.27493 w_k1_DIS_s003
                                 + 1.20201 w_k1_FDX_s003
                                 + 1.27057 w_k1_IBM_s003
                                 + 1.04574 w_k1_INTC_s003
                                 + 1.16072 w_k1_KO_s003 + 1.21506 w_k1_MAT_s003
                                 + 1.03866 w_k1_MSFT_s003
                                 + 1.21888 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.917263 w_k1_ADM_s003
                                 + 0.655974 w_k1_AXP_s003
                                 + 0.707869 w_k1_DIS_s003
                                 + 0.910133 w_k1_FDX_s003
                                 + 0.659893 w_k1_IBM_s003
                                 + 0.788635 w_k1_INTC_s003
                                 + 0.663629 w_k1_KO_s003
                                 + 0.546611 w_k1_MAT_s003
                                 + 0.840811 w_k1_MSFT_s003
                                 + 0.798454 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.36428 w_k1_ADM_s003
                                 + 1.33171 w_k1_AXP_s003
                                 + 1.47213 w_k1_DIS_s003
                                 + 1.21192 w_k1_FDX_s003
                                 + 1.45774 w_k1_IBM_s003
                                 + 1.16961 w_k1_INTC_s003
                                 + 1.40099 w_k1_KO_s003 + 1.11559 w_k1_MAT_s003
                                 + 1.41428 w_k1_MSFT_s003
                                 + 1.32905 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.24411 w_k1_ADM_s003
                                 + 1.03438 w_k1_AXP_s003
                                 + 0.892745 w_k1_DIS_s003
                                 + 1.04923 w_k1_FDX_s003
                                 + 0.96653 w_k1_IBM_s003
                                 + 0.953091 w_k1_INTC_s003
                                 + 0.854298 w_k1_KO_s003
                                 + 0.79661 w_k1_MAT_s003
                                 + 1.02394 w_k1_MSFT_s003
                                 + 1.11755 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.978957 w_k1_ADM_s003
                                 + 1.05862 w_k1_AXP_s003
                                 + 0.972836 w_k1_DIS_s003
                                 + 0.956108 w_k1_FDX_s003
                                 + 1.4191 w_k1_IBM_s003
                                 + 1.03972 w_k1_INTC_s003
                                 + 1.08931 w_k1_KO_s003 + 1.36854 w_k1_MAT_s003
                                 + 1.0277 w_k1_MSFT_s003 + 1.18915 w_k1_PG_s003
                                 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.998983 w_k1_ADM_s003
                                 + 0.83779 w_k1_AXP_s003
                                 + 1.09592 w_k1_DIS_s003
                                 + 1.02312 w_k1_FDX_s003
                                 + 0.968697 w_k1_IBM_s003
                                 + 0.979147 w_k1_INTC_s003
                                 + 0.981858 w_k1_KO_s003
                                 + 0.994971 w_k1_MAT_s003
                                 + 1.13076 w_k1_MSFT_s003
                                 + 1.03337 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.86704 w_k1_ADM_s003
                                 + 1.47259 w_k1_AXP_s003
                                 + 1.65825 w_k1_DIS_s003
                                 + 1.47854 w_k1_FDX_s003
                                 + 1.15728 w_k1_IBM_s003
                                 + 1.13691 w_k1_INTC_s003
                                 + 1.30101 w_k1_KO_s003
                                 + 0.924123 w_k1_MAT_s003
                                 + 1.27002 w_k1_MSFT_s003
                                 + 1.30813 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.817593 w_k1_ADM_s003
                                 + 0.800486 w_k1_AXP_s003
                                 + 0.743821 w_k1_DIS_s003
                                 + 0.737069 w_k1_FDX_s003
                                 + 1.22984 w_k1_IBM_s003
                                 + 0.692263 w_k1_INTC_s003
                                 + 0.978134 w_k1_KO_s003
                                 + 1.22822 w_k1_MAT_s003
                                 + 0.772433 w_k1_MSFT_s003
                                 + 0.837763 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.29908 w_k1_ADM_s004
                                 + 1.4363 w_k1_AXP_s004 + 1.08823 w_k1_DIS_s004
                                 + 1.34418 w_k1_FDX_s004
                                 + 1.00923 w_k1_IBM_s004
                                 + 1.14129 w_k1_INTC_s004
                                 + 0.956304 w_k1_KO_s004
                                 + 1.29644 w_k1_MAT_s004
                                 + 1.06289 w_k1_MSFT_s004
                                 + 1.12796 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.791869 w_k1_ADM_s004
                                 + 0.593601 w_k1_AXP_s004
                                 + 0.740614 w_k1_DIS_s004
                                 + 0.808887 w_k1_FDX_s004
                                 + 0.851266 w_k1_IBM_s004
                                 + 0.988729 w_k1_INTC_s004
                                 + 0.791545 w_k1_KO_s004
                                 + 0.857214 w_k1_MAT_s004
                                 + 0.887299 w_k1_MSFT_s004
                                 + 0.966709 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.84406 w_k1_ADM_s004
                                 + 1.31443 w_k1_AXP_s004
                                 + 1.75313 w_k1_DIS_s004
                                 + 1.48124 w_k1_FDX_s004
                                 + 1.31833 w_k1_IBM_s004
                                 + 1.31579 w_k1_INTC_s004
                                 + 1.64231 w_k1_KO_s004 + 2.27867 w_k1_MAT_s004
                                 + 1.45262 w_k1_MSFT_s004
                                 + 1.56748 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.78614 w_k1_ADM_s004
                                 + 0.611822 w_k1_AXP_s004
                                 + 0.589594 w_k1_DIS_s004
                                 + 0.720044 w_k1_FDX_s004
                                 + 0.817673 w_k1_IBM_s004
                                 + 0.781641 w_k1_INTC_s004
                                 + 0.778396 w_k1_KO_s004
                                 + 0.747744 w_k1_MAT_s004
                                 + 0.763776 w_k1_MSFT_s004
                                 + 0.854352 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.22878 w_k1_ADM_s004
                                 + 0.721084 w_k1_AXP_s004
                                 + 1.00513 w_k1_DIS_s004
                                 + 1.08718 w_k1_FDX_s004
                                 + 0.655438 w_k1_IBM_s004
                                 + 0.906714 w_k1_INTC_s004
                                 + 0.746516 w_k1_KO_s004
                                 + 0.541328 w_k1_MAT_s004
                                 + 0.925943 w_k1_MSFT_s004
                                 + 1.05684 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.965427 w_k1_ADM_s004
                                 + 1.09347 w_k1_AXP_s004
                                 + 1.15429 w_k1_DIS_s004
                                 + 0.973734 w_k1_FDX_s004
                                 + 2.02277 w_k1_IBM_s004
                                 + 0.962066 w_k1_INTC_s004
                                 + 1.30484 w_k1_KO_s004 + 2.12455 w_k1_MAT_s004
                                 + 1.00904 w_k1_MSFT_s004
                                 + 1.27172 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.997137 w_k1_ADM_s004
                                 + 0.828345 w_k1_AXP_s004
                                 + 0.825829 w_k1_DIS_s004
                                 + 0.816677 w_k1_FDX_s004
                                 + 1.21204 w_k1_IBM_s004
                                 + 1.03208 w_k1_INTC_s004
                                 + 0.965538 w_k1_KO_s004
                                 + 0.984576 w_k1_MAT_s004
                                 + 0.86905 w_k1_MSFT_s004
                                 + 1.06539 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.11589 w_k1_ADM_s004
                                 + 0.804316 w_k1_AXP_s004
                                 + 1.04616 w_k1_DIS_s004
                                 + 0.981806 w_k1_FDX_s004
                                 + 1.11457 w_k1_IBM_s004
                                 + 0.877366 w_k1_INTC_s004
                                 + 1.02845 w_k1_KO_s004 + 1.0251 w_k1_MAT_s004
                                 + 0.887632 w_k1_MSFT_s004
                                 + 1.25723 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.19327 w_k1_ADM_s005
                                 + 1.08352 w_k1_AXP_s005
                                 + 0.954449 w_k1_DIS_s005
                                 + 0.942047 w_k1_FDX_s005
                                 + 1.35796 w_k1_IBM_s005
                                 + 0.818454 w_k1_INTC_s005
                                 + 1.1548 w_k1_KO_s005 + 0.9531 w_k1_MAT_s005
                                 + 0.865347 w_k1_MSFT_s005
                                 + 1.02526 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.2686 w_k1_ADM_s005
                                 + 1.22461 w_k1_AXP_s005 + 1.0063 w_k1_DIS_s005
                                 + 1.00805 w_k1_FDX_s005 + 1.0837 w_k1_IBM_s005
                                 + 1.11908 w_k1_INTC_s005
                                 + 1.06108 w_k1_KO_s005 + 1.2017 w_k1_MAT_s005
                                 + 0.950868 w_k1_MSFT_s005
                                 + 0.876353 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.964853 w_k1_ADM_s005
                                 + 0.882832 w_k1_AXP_s005
                                 + 0.854464 w_k1_DIS_s005
                                 + 1.0527 w_k1_FDX_s005
                                 + 0.713896 w_k1_IBM_s005
                                 + 0.796518 w_k1_INTC_s005
                                 + 0.857386 w_k1_KO_s005
                                 + 0.722611 w_k1_MAT_s005
                                 + 0.963914 w_k1_MSFT_s005
                                 + 0.943096 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.777175 w_k1_ADM_s005
                                 + 1.0392 w_k1_AXP_s005 + 1.24486 w_k1_DIS_s005
                                 + 1.08945 w_k1_FDX_s005
                                 + 1.00511 w_k1_IBM_s005
                                 + 1.00374 w_k1_INTC_s005 + 1.1567 w_k1_KO_s005
                                 + 1.02612 w_k1_MAT_s005
                                 + 1.08643 w_k1_MSFT_s005
                                 + 0.976521 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.41656 w_k1_ADM_s005
                                 + 1.37708 w_k1_AXP_s005
                                 + 1.80266 w_k1_DIS_s005
                                 + 1.40665 w_k1_FDX_s005
                                 + 1.33729 w_k1_IBM_s005
                                 + 1.27087 w_k1_INTC_s005
                                 + 1.71875 w_k1_KO_s005 + 1.48995 w_k1_MAT_s005
                                 + 1.38106 w_k1_MSFT_s005
                                 + 1.20057 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.03019 w_k1_ADM_s005
                                 + 1.43939 w_k1_AXP_s005
                                 + 1.23838 w_k1_DIS_s005
                                 + 1.34306 w_k1_FDX_s005
                                 + 0.801521 w_k1_IBM_s005
                                 + 0.983789 w_k1_INTC_s005
                                 + 1.0302 w_k1_KO_s005 + 1.14739 w_k1_MAT_s005
                                 + 1.13909 w_k1_MSFT_s005
                                 + 0.97186 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.796724 w_k1_ADM_s005
                                 + 0.704455 w_k1_AXP_s005
                                 + 0.692234 w_k1_DIS_s005
                                 + 0.721548 w_k1_FDX_s005
                                 + 0.924574 w_k1_IBM_s005
                                 + 0.738307 w_k1_INTC_s005
                                 + 0.841219 w_k1_KO_s005
                                 + 0.877362 w_k1_MAT_s005
                                 + 0.780256 w_k1_MSFT_s005
                                 + 0.777509 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.56782 w_k1_ADM_s005
                                 + 1.55246 w_k1_AXP_s005
                                 + 1.41107 w_k1_DIS_s005
                                 + 1.15447 w_k1_FDX_s005
                                 + 1.57792 w_k1_IBM_s005
                                 + 0.920073 w_k1_INTC_s005
                                 + 1.42431 w_k1_KO_s005 + 1.21829 w_k1_MAT_s005
                                 + 0.96663 w_k1_MSFT_s005
                                 + 1.10944 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.19452 w_k1_ADM_s006
                                 + 1.86729 w_k1_AXP_s006 + 1.5435 w_k1_DIS_s006
                                 + 1.33124 w_k1_FDX_s006
                                 + 1.38939 w_k1_IBM_s006
                                 + 1.04644 w_k1_INTC_s006
                                 + 1.45215 w_k1_KO_s006 + 2.00059 w_k1_MAT_s006
                                 + 1.1955 w_k1_MSFT_s006 + 1.15094 w_k1_PG_s006
                                 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.842411 w_k1_ADM_s006
                                 + 0.751801 w_k1_AXP_s006
                                 + 0.683547 w_k1_DIS_s006
                                 + 0.769483 w_k1_FDX_s006
                                 + 0.861081 w_k1_IBM_s006
                                 + 0.730105 w_k1_INTC_s006
                                 + 0.799172 w_k1_KO_s006
                                 + 0.741659 w_k1_MAT_s006
                                 + 0.842683 w_k1_MSFT_s006
                                 + 0.843305 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.94383 w_k1_ADM_s006
                                 + 0.797387 w_k1_AXP_s006
                                 + 1.03264 w_k1_DIS_s006
                                 + 0.922372 w_k1_FDX_s006
                                 + 1.13658 w_k1_IBM_s006
                                 + 0.999676 w_k1_INTC_s006
                                 + 1.13706 w_k1_KO_s006
                                 + 0.897306 w_k1_MAT_s006
                                 + 1.19428 w_k1_MSFT_s006
                                 + 1.05711 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.849804 w_k1_ADM_s006
                                 + 1.07776 w_k1_AXP_s006
                                 + 0.873727 w_k1_DIS_s006
                                 + 0.914801 w_k1_FDX_s006
                                 + 1.33307 w_k1_IBM_s006
                                 + 1.06139 w_k1_INTC_s006
                                 + 1.13813 w_k1_KO_s006 + 1.55436 w_k1_MAT_s006
                                 + 1.06481 w_k1_MSFT_s006
                                 + 1.02365 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.0984 w_k1_ADM_s006
                                 + 1.22537 w_k1_AXP_s006
                                 + 1.41836 w_k1_DIS_s006
                                 + 1.24428 w_k1_FDX_s006
                                 + 0.917605 w_k1_IBM_s006
                                 + 0.904666 w_k1_INTC_s006
                                 + 1.12329 w_k1_KO_s006 + 1.07233 w_k1_MAT_s006
                                 + 1.18348 w_k1_MSFT_s006
                                 + 1.12986 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.967866 w_k1_ADM_s006
                                 + 1.22098 w_k1_AXP_s006
                                 + 0.899213 w_k1_DIS_s006
                                 + 0.982959 w_k1_FDX_s006
                                 + 1.04883 w_k1_IBM_s006
                                 + 0.885559 w_k1_INTC_s006
                                 + 0.803138 w_k1_KO_s006
                                 + 1.08848 w_k1_MAT_s006
                                 + 0.979308 w_k1_MSFT_s006
                                 + 1.03335 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.737941 w_k1_ADM_s006
                                 + 0.955267 w_k1_AXP_s006
                                 + 0.855161 w_k1_DIS_s006
                                 + 0.89761 w_k1_FDX_s006
                                 + 0.664318 w_k1_IBM_s006
                                 + 0.796994 w_k1_INTC_s006
                                 + 0.656664 w_k1_KO_s006
                                 + 0.676765 w_k1_MAT_s006
                                 + 0.87874 w_k1_MSFT_s006
                                 + 0.889997 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.79262 w_k1_ADM_s006
                                 + 1.11521 w_k1_AXP_s006
                                 + 1.26965 w_k1_DIS_s006
                                 + 1.09196 w_k1_FDX_s006
                                 + 1.05785 w_k1_IBM_s006
                                 + 1.10678 w_k1_INTC_s006
                                 + 1.08326 w_k1_KO_s006 + 1.38699 w_k1_MAT_s006
                                 + 1.22898 w_k1_MSFT_s006
                                 + 1.01881 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.750232 w_k1_ADM_s007
                                 + 0.854709 w_k1_AXP_s007
                                 + 0.75849 w_k1_DIS_s007
                                 + 0.845265 w_k1_FDX_s007
                                 + 0.98685 w_k1_IBM_s007
                                 + 0.993948 w_k1_INTC_s007
                                 + 0.908238 w_k1_KO_s007
                                 + 0.965227 w_k1_MAT_s007
                                 + 0.851146 w_k1_MSFT_s007
                                 + 0.916216 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.18393 w_k1_ADM_s007
                                 + 0.906475 w_k1_AXP_s007
                                 + 1.06472 w_k1_DIS_s007
                                 + 1.05106 w_k1_FDX_s007
                                 + 0.852388 w_k1_IBM_s007
                                 + 1.15252 w_k1_INTC_s007
                                 + 0.937185 w_k1_KO_s007
                                 + 0.851327 w_k1_MAT_s007
                                 + 1.01732 w_k1_MSFT_s007
                                 + 0.973205 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.6391 w_k1_ADM_s007
                                 + 1.23483 w_k1_AXP_s007
                                 + 1.30288 w_k1_DIS_s007
                                 + 1.40306 w_k1_FDX_s007
                                 + 0.849242 w_k1_IBM_s007
                                 + 0.978394 w_k1_INTC_s007
                                 + 1.0597 w_k1_KO_s007 + 0.861397 w_k1_MAT_s007
                                 + 1.06945 w_k1_MSFT_s007
                                 + 0.956946 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.31446 w_k1_ADM_s007
                                 + 1.20324 w_k1_AXP_s007
                                 + 1.30509 w_k1_DIS_s007
                                 + 1.21428 w_k1_FDX_s007
                                 + 1.25581 w_k1_IBM_s007
                                 + 0.999393 w_k1_INTC_s007
                                 + 1.37012 w_k1_KO_s007 + 1.20841 w_k1_MAT_s007
                                 + 1.12147 w_k1_MSFT_s007
                                 + 1.19176 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.815426 w_k1_ADM_s007
                                 + 0.958601 w_k1_AXP_s007
                                 + 0.665753 w_k1_DIS_s007
                                 + 0.778902 w_k1_FDX_s007
                                 + 0.802711 w_k1_IBM_s007
                                 + 0.954566 w_k1_INTC_s007
                                 + 0.776899 w_k1_KO_s007
                                 + 0.714803 w_k1_MAT_s007
                                 + 0.601455 w_k1_MSFT_s007
                                 + 0.928049 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.09045 w_k1_ADM_s007
                                 + 1.15333 w_k1_AXP_s007
                                 + 1.07094 w_k1_DIS_s007
                                 + 1.04212 w_k1_FDX_s007
                                 + 1.62393 w_k1_IBM_s007
                                 + 1.2333 w_k1_INTC_s007 + 1.1589 w_k1_KO_s007
                                 + 1.31269 w_k1_MAT_s007
                                 + 0.876379 w_k1_MSFT_s007
                                 + 1.13504 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.14072 w_k1_ADM_s007
                                 + 0.846414 w_k1_AXP_s007
                                 + 0.742464 w_k1_DIS_s007
                                 + 0.906159 w_k1_FDX_s007
                                 + 0.613294 w_k1_IBM_s007
                                 + 1.01981 w_k1_INTC_s007
                                 + 0.713637 w_k1_KO_s007
                                 + 0.529808 w_k1_MAT_s007
                                 + 0.877306 w_k1_MSFT_s007
                                 + 0.930231 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.709502 w_k1_ADM_s007
                                 + 0.691596 w_k1_AXP_s007
                                 + 0.720182 w_k1_DIS_s007
                                 + 0.788768 w_k1_FDX_s007
                                 + 0.735216 w_k1_IBM_s007
                                 + 0.989254 w_k1_INTC_s007
                                 + 0.791489 w_k1_KO_s007
                                 + 0.558758 w_k1_MAT_s007
                                 + 0.834818 w_k1_MSFT_s007
                                 + 0.881906 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.14839 w_k1_ADM_s010
                                 + 1.07797 w_k1_AXP_s010
                                 + 1.09654 w_k1_DIS_s010
                                 + 0.895664 w_k1_FDX_s010
                                 + 1.30699 w_k1_IBM_s010
                                 + 0.815496 w_k1_INTC_s010
                                 + 1.14066 w_k1_KO_s010 + 1.16675 w_k1_MAT_s010
                                 + 0.94905 w_k1_MSFT_s010
                                 + 0.969142 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.05088 w_k1_ADM_s010
                                 + 1.08284 w_k1_AXP_s010
                                 + 1.32363 w_k1_DIS_s010
                                 + 1.16653 w_k1_FDX_s010
                                 + 0.808433 w_k1_IBM_s010
                                 + 1.22739 w_k1_INTC_s010
                                 + 1.08448 w_k1_KO_s010
                                 + 0.733948 w_k1_MAT_s010
                                 + 1.27613 w_k1_MSFT_s010 + 1.0755 w_k1_PG_s010
                                 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.04682 w_k1_ADM_s010
                                 + 0.931609 w_k1_AXP_s010
                                 + 0.941427 w_k1_DIS_s010
                                 + 1.01543 w_k1_FDX_s010
                                 + 0.754201 w_k1_IBM_s010
                                 + 0.819928 w_k1_INTC_s010
                                 + 0.744828 w_k1_KO_s010
                                 + 0.698808 w_k1_MAT_s010
                                 + 1.01349 w_k1_MSFT_s010
                                 + 0.884151 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.12986 w_k1_ADM_s010
                                 + 1.45993 w_k1_AXP_s010
                                 + 1.56191 w_k1_DIS_s010
                                 + 1.14732 w_k1_FDX_s010 + 1.609 w_k1_IBM_s010
                                 + 0.934912 w_k1_INTC_s010
                                 + 1.28151 w_k1_KO_s010 + 1.68598 w_k1_MAT_s010
                                 + 1.22652 w_k1_MSFT_s010 + 1.1561 w_k1_PG_s010
                                 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.36557 w_k1_ADM_s010
                                 + 1.53368 w_k1_AXP_s010
                                 + 1.52024 w_k1_DIS_s010
                                 + 1.31942 w_k1_FDX_s010
                                 + 1.11763 w_k1_IBM_s010
                                 + 1.10445 w_k1_INTC_s010
                                 + 1.13652 w_k1_KO_s010 + 1.10641 w_k1_MAT_s010
                                 + 1.20748 w_k1_MSFT_s010
                                 + 1.27585 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.0138 w_k1_ADM_s010
                                 + 1.08169 w_k1_AXP_s010 + 1.0796 w_k1_DIS_s010
                                 + 0.946747 w_k1_FDX_s010
                                 + 1.06237 w_k1_IBM_s010
                                 + 0.913996 w_k1_INTC_s010
                                 + 0.926174 w_k1_KO_s010
                                 + 0.798226 w_k1_MAT_s010
                                 + 0.948396 w_k1_MSFT_s010
                                 + 1.00915 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.777821 w_k1_ADM_s010
                                 + 0.961052 w_k1_AXP_s010
                                 + 0.855314 w_k1_DIS_s010
                                 + 0.820521 w_k1_FDX_s010
                                 + 1.29956 w_k1_IBM_s010
                                 + 0.761173 w_k1_INTC_s010
                                 + 1.02772 w_k1_KO_s010 + 1.34184 w_k1_MAT_s010
                                 + 0.746134 w_k1_MSFT_s010
                                 + 0.893012 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.917262 w_k1_ADM_s010
                                 + 0.859254 w_k1_AXP_s010
                                 + 0.71261 w_k1_DIS_s010
                                 + 0.819656 w_k1_FDX_s010
                                 + 0.820294 w_k1_IBM_s010
                                 + 0.751817 w_k1_INTC_s010
                                 + 0.717755 w_k1_KO_s010
                                 + 0.604169 w_k1_MAT_s010
                                 + 0.728955 w_k1_MSFT_s010
                                 + 0.821899 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.818506 w_k1_ADM_s011
                                 + 0.822229 w_k1_AXP_s011
                                 + 0.918669 w_k1_DIS_s011
                                 + 0.9842 w_k1_FDX_s011
                                 + 0.828939 w_k1_IBM_s011
                                 + 1.16453 w_k1_INTC_s011
                                 + 0.937564 w_k1_KO_s011
                                 + 0.854268 w_k1_MAT_s011
                                 + 1.03189 w_k1_MSFT_s011
                                 + 0.918853 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.947017 w_k1_ADM_s011
                                 + 1.52009 w_k1_AXP_s011
                                 + 1.05339 w_k1_DIS_s011 + 1.1004 w_k1_FDX_s011
                                 + 1.26664 w_k1_IBM_s011
                                 + 1.16858 w_k1_INTC_s011 + 1.2254 w_k1_KO_s011
                                 + 1.52225 w_k1_MAT_s011
                                 + 1.06493 w_k1_MSFT_s011
                                 + 1.17708 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.08921 w_k1_ADM_s011
                                 + 1.43486 w_k1_AXP_s011
                                 + 1.30859 w_k1_DIS_s011 + 1.4129 w_k1_FDX_s011
                                 + 1.05859 w_k1_IBM_s011
                                 + 1.19656 w_k1_INTC_s011
                                 + 1.19175 w_k1_KO_s011 + 1.4598 w_k1_MAT_s011
                                 + 1.14454 w_k1_MSFT_s011
                                 + 1.12821 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.821106 w_k1_ADM_s011
                                 + 0.868543 w_k1_AXP_s011
                                 + 0.95964 w_k1_DIS_s011
                                 + 0.893384 w_k1_FDX_s011
                                 + 1.57171 w_k1_IBM_s011
                                 + 1.17058 w_k1_INTC_s011 + 1.4456 w_k1_KO_s011
                                 + 1.60052 w_k1_MAT_s011
                                 + 1.02438 w_k1_MSFT_s011
                                 + 0.967999 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.00677 w_k1_ADM_s011
                                 + 1.21061 w_k1_AXP_s011
                                 + 1.21973 w_k1_DIS_s011
                                 + 1.13444 w_k1_FDX_s011
                                 + 1.17065 w_k1_IBM_s011
                                 + 1.10479 w_k1_INTC_s011
                                 + 1.25669 w_k1_KO_s011 + 1.58589 w_k1_MAT_s011
                                 + 1.07067 w_k1_MSFT_s011 + 1.17 w_k1_PG_s011
                                 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.898091 w_k1_ADM_s011
                                 + 0.941182 w_k1_AXP_s011
                                 + 1.46183 w_k1_DIS_s011
                                 + 1.27967 w_k1_FDX_s011
                                 + 0.851867 w_k1_IBM_s011
                                 + 1.29511 w_k1_INTC_s011
                                 + 1.00339 w_k1_KO_s011
                                 + 0.682674 w_k1_MAT_s011
                                 + 1.24965 w_k1_MSFT_s011
                                 + 0.958238 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.770709 w_k1_ADM_s011
                                 + 1.06067 w_k1_AXP_s011
                                 + 0.799055 w_k1_DIS_s011
                                 + 0.870783 w_k1_FDX_s011
                                 + 1.15358 w_k1_IBM_s011
                                 + 1.03321 w_k1_INTC_s011
                                 + 0.966142 w_k1_KO_s011
                                 + 1.00874 w_k1_MAT_s011
                                 + 0.762812 w_k1_MSFT_s011
                                 + 0.8646 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.734054 w_k1_ADM_s011
                                 + 0.756558 w_k1_AXP_s011
                                 + 0.701399 w_k1_DIS_s011
                                 + 0.83161 w_k1_FDX_s011
                                 + 0.796213 w_k1_IBM_s011
                                 + 0.765929 w_k1_INTC_s011
                                 + 0.783948 w_k1_KO_s011
                                 + 0.594332 w_k1_MAT_s011
                                 + 0.727407 w_k1_MSFT_s011
                                 + 0.800161 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.29376 w_k1_ADM_s012
                                 + 1.06957 w_k1_AXP_s012
                                 + 1.66583 w_k1_DIS_s012
                                 + 1.25317 w_k1_FDX_s012
                                 + 1.28174 w_k1_IBM_s012
                                 + 1.07843 w_k1_INTC_s012
                                 + 1.62862 w_k1_KO_s012 + 1.33471 w_k1_MAT_s012
                                 + 1.20538 w_k1_MSFT_s012
                                 + 1.17493 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.602701 w_k1_ADM_s012
                                 + 0.691508 w_k1_AXP_s012
                                 + 0.820341 w_k1_DIS_s012
                                 + 0.710393 w_k1_FDX_s012
                                 + 1.16351 w_k1_IBM_s012
                                 + 0.781639 w_k1_INTC_s012
                                 + 1.04627 w_k1_KO_s012 + 1.07724 w_k1_MAT_s012
                                 + 0.841392 w_k1_MSFT_s012
                                 + 0.872848 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.11034 w_k1_ADM_s012
                                 + 1.01102 w_k1_AXP_s012
                                 + 0.775589 w_k1_DIS_s012
                                 + 0.93173 w_k1_FDX_s012
                                 + 1.18096 w_k1_IBM_s012
                                 + 1.06164 w_k1_INTC_s012
                                 + 1.01163 w_k1_KO_s012 + 1.26621 w_k1_MAT_s012
                                 + 0.988748 w_k1_MSFT_s012
                                 + 0.989688 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.36066 w_k1_ADM_s012
                                 + 1.59158 w_k1_AXP_s012
                                 + 1.36659 w_k1_DIS_s012
                                 + 1.32226 w_k1_FDX_s012
                                 + 1.32822 w_k1_IBM_s012
                                 + 0.959013 w_k1_INTC_s012
                                 + 1.28968 w_k1_KO_s012 + 2.06858 w_k1_MAT_s012
                                 + 0.952725 w_k1_MSFT_s012
                                 + 1.15487 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.29606 w_k1_ADM_s012
                                 + 1.19445 w_k1_AXP_s012
                                 + 1.14562 w_k1_DIS_s012
                                 + 1.17579 w_k1_FDX_s012
                                 + 0.902698 w_k1_IBM_s012
                                 + 1.03063 w_k1_INTC_s012
                                 + 1.12557 w_k1_KO_s012 + 1.13149 w_k1_MAT_s012
                                 + 1.15748 w_k1_MSFT_s012
                                 + 1.08178 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.08446 w_k1_ADM_s012
                                 + 1.03363 w_k1_AXP_s012
                                 + 1.12264 w_k1_DIS_s012
                                 + 0.946778 w_k1_FDX_s012
                                 + 1.56909 w_k1_IBM_s012
                                 + 0.852283 w_k1_INTC_s012
                                 + 1.37812 w_k1_KO_s012 + 1.68716 w_k1_MAT_s012
                                 + 0.994023 w_k1_MSFT_s012
                                 + 1.07428 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.915067 w_k1_ADM_s012
                                 + 0.923957 w_k1_AXP_s012
                                 + 0.606633 w_k1_DIS_s012
                                 + 0.849275 w_k1_FDX_s012
                                 + 0.659991 w_k1_IBM_s012
                                 + 0.866216 w_k1_INTC_s012
                                 + 0.651053 w_k1_KO_s012
                                 + 0.910269 w_k1_MAT_s012
                                 + 0.766027 w_k1_MSFT_s012
                                 + 0.850432 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.987135 w_k1_ADM_s012
                                 + 0.921704 w_k1_AXP_s012
                                 + 0.941481 w_k1_DIS_s012
                                 + 0.949067 w_k1_FDX_s012
                                 + 0.907626 w_k1_IBM_s012
                                 + 0.756413 w_k1_INTC_s012
                                 + 1.0377 w_k1_KO_s012 + 1.13347 w_k1_MAT_s012
                                 + 0.898905 w_k1_MSFT_s012
                                 + 1.05443 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.18141 w_k1_ADM_s013
                                 + 1.49263 w_k1_AXP_s013
                                 + 1.54307 w_k1_DIS_s013
                                 + 1.20221 w_k1_FDX_s013
                                 + 1.52745 w_k1_IBM_s013
                                 + 0.891512 w_k1_INTC_s013
                                 + 1.41192 w_k1_KO_s013 + 1.60174 w_k1_MAT_s013
                                 + 1.1243 w_k1_MSFT_s013 + 1.11216 w_k1_PG_s013
                                 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.29734 w_k1_ADM_s013
                                 + 1.10916 w_k1_AXP_s013
                                 + 1.32212 w_k1_DIS_s013
                                 + 1.28017 w_k1_FDX_s013
                                 + 0.95492 w_k1_IBM_s013
                                 + 1.06885 w_k1_INTC_s013
                                 + 1.20317 w_k1_KO_s013 + 1.45013 w_k1_MAT_s013
                                 + 1.27252 w_k1_MSFT_s013
                                 + 1.13978 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.04479 w_k1_ADM_s013
                                 + 0.89587 w_k1_AXP_s013
                                 + 0.934798 w_k1_DIS_s013
                                 + 1.05772 w_k1_FDX_s013
                                 + 0.696249 w_k1_IBM_s013
                                 + 0.901259 w_k1_INTC_s013
                                 + 0.831968 w_k1_KO_s013
                                 + 0.820433 w_k1_MAT_s013
                                 + 1.025 w_k1_MSFT_s013 + 0.890423 w_k1_PG_s013
                                 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.893969 w_k1_ADM_s013
                                 + 0.818983 w_k1_AXP_s013
                                 + 0.955668 w_k1_DIS_s013
                                 + 0.912878 w_k1_FDX_s013
                                 + 0.928641 w_k1_IBM_s013
                                 + 0.902553 w_k1_INTC_s013
                                 + 0.849818 w_k1_KO_s013
                                 + 0.879632 w_k1_MAT_s013
                                 + 1.00325 w_k1_MSFT_s013
                                 + 0.881347 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.01853 w_k1_ADM_s013
                                 + 1.04426 w_k1_AXP_s013
                                 + 1.24225 w_k1_DIS_s013 + 1.0349 w_k1_FDX_s013
                                 + 1.10511 w_k1_IBM_s013
                                 + 0.882648 w_k1_INTC_s013
                                 + 1.13221 w_k1_KO_s013 + 1.17778 w_k1_MAT_s013
                                 + 1.12304 w_k1_MSFT_s013
                                 + 1.02901 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.941412 w_k1_ADM_s013
                                 + 1.20064 w_k1_AXP_s013
                                 + 0.923838 w_k1_DIS_s013
                                 + 0.903945 w_k1_FDX_s013
                                 + 1.1658 w_k1_IBM_s013
                                 + 0.879472 w_k1_INTC_s013
                                 + 1.07969 w_k1_KO_s013 + 1.22289 w_k1_MAT_s013
                                 + 0.955468 w_k1_MSFT_s013
                                 + 0.947277 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.3122 w_k1_ADM_s013
                                 + 0.836726 w_k1_AXP_s013
                                 + 1.09252 w_k1_DIS_s013
                                 + 0.894202 w_k1_FDX_s013
                                 + 1.14568 w_k1_IBM_s013
                                 + 0.793559 w_k1_INTC_s013
                                 + 1.23038 w_k1_KO_s013
                                 + 0.783429 w_k1_MAT_s013
                                 + 0.874677 w_k1_MSFT_s013
                                 + 0.961772 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.912714 w_k1_ADM_s013
                                 + 0.76886 w_k1_AXP_s013
                                 + 0.696826 w_k1_DIS_s013
                                 + 0.715393 w_k1_FDX_s013
                                 + 0.761764 w_k1_IBM_s013
                                 + 0.654429 w_k1_INTC_s013
                                 + 0.793057 w_k1_KO_s013
                                 + 0.90279 w_k1_MAT_s013
                                 + 0.798754 w_k1_MSFT_s013
                                 + 0.791478 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.51878 w_k1_ADM_s014
                                 + 1.59986 w_k1_AXP_s014
                                 + 1.59994 w_k1_DIS_s014
                                 + 1.47598 w_k1_FDX_s014
                                 + 1.22199 w_k1_IBM_s014
                                 + 1.18111 w_k1_INTC_s014 + 1.3077 w_k1_KO_s014
                                 + 1.28878 w_k1_MAT_s014
                                 + 1.35472 w_k1_MSFT_s014
                                 + 1.27862 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.24684 w_k1_ADM_s014
                                 + 1.11365 w_k1_AXP_s014
                                 + 1.09762 w_k1_DIS_s014 + 1.1056 w_k1_FDX_s014
                                 + 1.02423 w_k1_IBM_s014
                                 + 0.939579 w_k1_INTC_s014
                                 + 1.01339 w_k1_KO_s014
                                 + 0.960026 w_k1_MAT_s014
                                 + 1.05371 w_k1_MSFT_s014
                                 + 1.10997 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.12802 w_k1_ADM_s014
                                 + 0.939301 w_k1_AXP_s014
                                 + 0.927293 w_k1_DIS_s014
                                 + 0.976386 w_k1_FDX_s014
                                 + 0.804822 w_k1_IBM_s014
                                 + 0.894013 w_k1_INTC_s014
                                 + 0.990955 w_k1_KO_s014
                                 + 0.894932 w_k1_MAT_s014
                                 + 1.07711 w_k1_MSFT_s014
                                 + 1.03252 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.43191 w_k1_ADM_s014
                                 + 1.26218 w_k1_AXP_s014
                                 + 1.27451 w_k1_DIS_s014
                                 + 1.30768 w_k1_FDX_s014
                                 + 0.78368 w_k1_IBM_s014
                                 + 0.985003 w_k1_INTC_s014
                                 + 1.0054 w_k1_KO_s014 + 0.818902 w_k1_MAT_s014
                                 + 1.19504 w_k1_MSFT_s014
                                 + 1.09861 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.732174 w_k1_ADM_s014
                                 + 0.782 w_k1_AXP_s014 + 0.632466 w_k1_DIS_s014
                                 + 0.728891 w_k1_FDX_s014
                                 + 0.70144 w_k1_IBM_s014
                                 + 0.746559 w_k1_INTC_s014
                                 + 0.690336 w_k1_KO_s014
                                 + 0.627882 w_k1_MAT_s014
                                 + 0.813318 w_k1_MSFT_s014
                                 + 0.944901 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.999166 w_k1_ADM_s014
                                 + 0.819528 w_k1_AXP_s014
                                 + 0.814493 w_k1_DIS_s014
                                 + 0.772745 w_k1_FDX_s014
                                 + 1.18795 w_k1_IBM_s014
                                 + 0.699416 w_k1_INTC_s014
                                 + 1.07657 w_k1_KO_s014
                                 + 0.968665 w_k1_MAT_s014
                                 + 0.792542 w_k1_MSFT_s014
                                 + 0.984595 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.45885 w_k1_ADM_s014
                                 + 1.40322 w_k1_AXP_s014
                                 + 1.10799 w_k1_DIS_s014
                                 + 1.08721 w_k1_FDX_s014
                                 + 1.54247 w_k1_IBM_s014
                                 + 0.852402 w_k1_INTC_s014
                                 + 1.04067 w_k1_KO_s014 + 1.62124 w_k1_MAT_s014
                                 + 0.948742 w_k1_MSFT_s014
                                 + 1.0945 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.15274 w_k1_ADM_s014
                                 + 1.17469 w_k1_AXP_s014
                                 + 0.795208 w_k1_DIS_s014
                                 + 1.00467 w_k1_FDX_s014
                                 + 0.619525 w_k1_IBM_s014
                                 + 0.903787 w_k1_INTC_s014
                                 + 0.828386 w_k1_KO_s014
                                 + 0.953848 w_k1_MAT_s014
                                 + 1.03835 w_k1_MSFT_s014
                                 + 0.948602 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.12382 w_k1_ADM_s015
                                 + 1.22476 w_k1_AXP_s015
                                 + 0.849407 w_k1_DIS_s015
                                 + 1.05289 w_k1_FDX_s015
                                 + 0.618247 w_k1_IBM_s015
                                 + 1.10046 w_k1_INTC_s015
                                 + 0.666044 w_k1_KO_s015
                                 + 0.33891 w_k1_MAT_s015
                                 + 0.923482 w_k1_MSFT_s015
                                 + 0.95213 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.0315 w_k1_ADM_s015
                                 + 1.34655 w_k1_AXP_s015
                                 + 1.46204 w_k1_DIS_s015
                                 + 1.31656 w_k1_FDX_s015
                                 + 1.08947 w_k1_IBM_s015
                                 + 1.2613 w_k1_INTC_s015 + 1.21524 w_k1_KO_s015
                                 + 1.03742 w_k1_MAT_s015
                                 + 1.32274 w_k1_MSFT_s015
                                 + 1.08146 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.831798 w_k1_ADM_s015
                                 + 1.21751 w_k1_AXP_s015
                                 + 0.869861 w_k1_DIS_s015
                                 + 0.725796 w_k1_FDX_s015
                                 + 1.50914 w_k1_IBM_s015
                                 + 0.923272 w_k1_INTC_s015
                                 + 1.11292 w_k1_KO_s015 + 1.4291 w_k1_MAT_s015
                                 + 0.889987 w_k1_MSFT_s015
                                 + 0.97155 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.753977 w_k1_ADM_s015
                                 + 1.35626 w_k1_AXP_s015
                                 + 1.37687 w_k1_DIS_s015
                                 + 1.02807 w_k1_FDX_s015
                                 + 1.76899 w_k1_IBM_s015
                                 + 1.14717 w_k1_INTC_s015
                                 + 1.40319 w_k1_KO_s015 + 1.54678 w_k1_MAT_s015
                                 + 1.16126 w_k1_MSFT_s015
                                 + 1.02681 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.881197 w_k1_ADM_s015
                                 + 1.01696 w_k1_AXP_s015
                                 + 0.657936 w_k1_DIS_s015
                                 + 0.755599 w_k1_FDX_s015
                                 + 0.871495 w_k1_IBM_s015
                                 + 0.906433 w_k1_INTC_s015
                                 + 0.659117 w_k1_KO_s015
                                 + 0.998275 w_k1_MAT_s015
                                 + 0.903104 w_k1_MSFT_s015
                                 + 0.83907 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.0638 w_k1_ADM_s015
                                 + 1.86948 w_k1_AXP_s015
                                 + 1.22572 w_k1_DIS_s015 + 1.1624 w_k1_FDX_s015
                                 + 1.21893 w_k1_IBM_s015
                                 + 1.22769 w_k1_INTC_s015
                                 + 1.17141 w_k1_KO_s015 + 1.34779 w_k1_MAT_s015
                                 + 1.11759 w_k1_MSFT_s015
                                 + 1.11246 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.993894 w_k1_ADM_s015
                                 + 1.26991 w_k1_AXP_s015
                                 + 1.03139 w_k1_DIS_s015
                                 + 0.918459 w_k1_FDX_s015
                                 + 1.15294 w_k1_IBM_s015
                                 + 0.730744 w_k1_INTC_s015
                                 + 0.969573 w_k1_KO_s015
                                 + 1.25982 w_k1_MAT_s015
                                 + 0.954178 w_k1_MSFT_s015
                                 + 0.968779 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.824908 w_k1_ADM_s015
                                 + 1.09173 w_k1_AXP_s015
                                 + 0.918707 w_k1_DIS_s015
                                 + 0.991719 w_k1_FDX_s015
                                 + 0.708037 w_k1_IBM_s015
                                 + 1.07622 w_k1_INTC_s015
                                 + 0.757668 w_k1_KO_s015
                                 + 0.756438 w_k1_MAT_s015
                                 + 1.0835 w_k1_MSFT_s015
                                 + 0.969789 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.972606 w_k1_ADM_s016
                                 + 1.21344 w_k1_AXP_s016
                                 + 1.09473 w_k1_DIS_s016
                                 + 1.24343 w_k1_FDX_s016
                                 + 0.762809 w_k1_IBM_s016
                                 + 1.45992 w_k1_INTC_s016
                                 + 0.925206 w_k1_KO_s016
                                 + 0.626174 w_k1_MAT_s016
                                 + 1.0715 w_k1_MSFT_s016
                                 + 0.977863 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.851142 w_k1_ADM_s016
                                 + 0.868686 w_k1_AXP_s016
                                 + 0.640917 w_k1_DIS_s016
                                 + 0.822317 w_k1_FDX_s016
                                 + 0.657924 w_k1_IBM_s016
                                 + 0.82418 w_k1_INTC_s016
                                 + 0.607449 w_k1_KO_s016
                                 + 0.715565 w_k1_MAT_s016
                                 + 0.744776 w_k1_MSFT_s016
                                 + 0.900481 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.980318 w_k1_ADM_s016
                                 + 1.1347 w_k1_AXP_s016 + 1.45314 w_k1_DIS_s016
                                 + 1.21551 w_k1_FDX_s016
                                 + 0.997666 w_k1_IBM_s016
                                 + 1.11098 w_k1_INTC_s016
                                 + 1.16705 w_k1_KO_s016
                                 + 0.947437 w_k1_MAT_s016
                                 + 1.17363 w_k1_MSFT_s016
                                 + 1.12538 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.728469 w_k1_ADM_s016
                                 + 0.738016 w_k1_AXP_s016
                                 + 0.693465 w_k1_DIS_s016
                                 + 0.693224 w_k1_FDX_s016
                                 + 0.795853 w_k1_IBM_s016
                                 + 0.875834 w_k1_INTC_s016
                                 + 0.735977 w_k1_KO_s016
                                 + 0.818005 w_k1_MAT_s016
                                 + 0.742299 w_k1_MSFT_s016
                                 + 0.791141 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.705749 w_k1_ADM_s016
                                 + 0.834929 w_k1_AXP_s016
                                 + 1.11841 w_k1_DIS_s016
                                 + 1.00443 w_k1_FDX_s016
                                 + 0.804657 w_k1_IBM_s016
                                 + 1.38641 w_k1_INTC_s016
                                 + 0.945634 w_k1_KO_s016
                                 + 0.605874 w_k1_MAT_s016
                                 + 0.995916 w_k1_MSFT_s016
                                 + 0.996995 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.906199 w_k1_ADM_s016
                                 + 0.909329 w_k1_AXP_s016
                                 + 0.830619 w_k1_DIS_s016
                                 + 1.01695 w_k1_FDX_s016
                                 + 0.749981 w_k1_IBM_s016
                                 + 0.800798 w_k1_INTC_s016
                                 + 0.774588 w_k1_KO_s016
                                 + 0.830145 w_k1_MAT_s016
                                 + 0.804611 w_k1_MSFT_s016
                                 + 0.931383 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.794652 w_k1_ADM_s016
                                 + 0.822663 w_k1_AXP_s016
                                 + 0.838276 w_k1_DIS_s016
                                 + 0.84244 w_k1_FDX_s016
                                 + 1.23386 w_k1_IBM_s016
                                 + 1.02199 w_k1_INTC_s016 + 1.0067 w_k1_KO_s016
                                 + 0.982252 w_k1_MAT_s016
                                 + 0.854224 w_k1_MSFT_s016
                                 + 0.988284 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.00326 w_k1_ADM_s016
                                 + 1.36845 w_k1_AXP_s016
                                 + 1.69627 w_k1_DIS_s016
                                 + 1.14565 w_k1_FDX_s016 + 1.8158 w_k1_IBM_s016
                                 + 1.13943 w_k1_INTC_s016
                                 + 1.40212 w_k1_KO_s016 + 1.73818 w_k1_MAT_s016
                                 + 1.18297 w_k1_MSFT_s016
                                 + 1.15909 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.33866 w_k1_ADM_s017
                                 + 1.23609 w_k1_AXP_s017
                                 + 1.01606 w_k1_DIS_s017
                                 + 1.08085 w_k1_FDX_s017
                                 + 1.13144 w_k1_IBM_s017
                                 + 1.10778 w_k1_INTC_s017
                                 + 1.06206 w_k1_KO_s017 + 1.16935 w_k1_MAT_s017
                                 + 1.03362 w_k1_MSFT_s017
                                 + 1.16116 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.209 w_k1_ADM_s017
                                 + 0.957147 w_k1_AXP_s017
                                 + 1.07309 w_k1_DIS_s017
                                 + 1.07238 w_k1_FDX_s017
                                 + 0.955267 w_k1_IBM_s017
                                 + 0.946149 w_k1_INTC_s017
                                 + 0.971152 w_k1_KO_s017
                                 + 0.814991 w_k1_MAT_s017
                                 + 1.15246 w_k1_MSFT_s017
                                 + 1.06785 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.12286 w_k1_ADM_s017
                                 + 1.39738 w_k1_AXP_s017
                                 + 1.21484 w_k1_DIS_s017
                                 + 1.03541 w_k1_FDX_s017
                                 + 1.76297 w_k1_IBM_s017
                                 + 1.47092 w_k1_INTC_s017
                                 + 1.45325 w_k1_KO_s017 + 1.39838 w_k1_MAT_s017
                                 + 1.14826 w_k1_MSFT_s017
                                 + 1.19513 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.757653 w_k1_ADM_s017
                                 + 0.991275 w_k1_AXP_s017
                                 + 0.847039 w_k1_DIS_s017
                                 + 0.910906 w_k1_FDX_s017
                                 + 1.27145 w_k1_IBM_s017
                                 + 1.07874 w_k1_INTC_s017
                                 + 1.09072 w_k1_KO_s017 + 1.14353 w_k1_MAT_s017
                                 + 0.941338 w_k1_MSFT_s017
                                 + 0.99265 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.97159 w_k1_ADM_s017
                                 + 0.800332 w_k1_AXP_s017
                                 + 0.604661 w_k1_DIS_s017
                                 + 0.707126 w_k1_FDX_s017
                                 + 0.989573 w_k1_IBM_s017
                                 + 0.814155 w_k1_INTC_s017
                                 + 0.870304 w_k1_KO_s017
                                 + 0.805556 w_k1_MAT_s017
                                 + 0.759728 w_k1_MSFT_s017
                                 + 0.990365 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.863145 w_k1_ADM_s017
                                 + 0.744137 w_k1_AXP_s017
                                 + 0.56712 w_k1_DIS_s017
                                 + 0.867266 w_k1_FDX_s017
                                 + 0.644349 w_k1_IBM_s017
                                 + 0.880085 w_k1_INTC_s017
                                 + 0.643824 w_k1_KO_s017
                                 + 0.595748 w_k1_MAT_s017
                                 + 0.800195 w_k1_MSFT_s017
                                 + 0.935841 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.985726 w_k1_ADM_s017
                                 + 0.721172 w_k1_AXP_s017
                                 + 0.773237 w_k1_DIS_s017
                                 + 0.852387 w_k1_FDX_s017
                                 + 0.735348 w_k1_IBM_s017
                                 + 0.956563 w_k1_INTC_s017
                                 + 0.767149 w_k1_KO_s017
                                 + 0.659613 w_k1_MAT_s017
                                 + 0.917731 w_k1_MSFT_s017
                                 + 0.868299 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.35414 w_k1_ADM_s017
                                 + 1.5479 w_k1_AXP_s017 + 1.59222 w_k1_DIS_s017
                                 + 1.35707 w_k1_FDX_s017
                                 + 1.19434 w_k1_IBM_s017
                                 + 1.03937 w_k1_INTC_s017 + 1.4669 w_k1_KO_s017
                                 + 1.66221 w_k1_MAT_s017
                                 + 1.43003 w_k1_MSFT_s017 + 1.3564 w_k1_PG_s017
                                 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.21596 w_k1_ADM_s100
                                 + 1.1728 w_k1_AXP_s100 + 1.00914 w_k1_DIS_s100
                                 + 1.09433 w_k1_FDX_s100
                                 + 0.961432 w_k1_IBM_s100
                                 + 0.810895 w_k1_INTC_s100
                                 + 0.912347 w_k1_KO_s100
                                 + 1.21444 w_k1_MAT_s100
                                 + 0.948765 w_k1_MSFT_s100
                                 + 0.912821 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.963548 w_k1_ADM_s100
                                 + 0.952408 w_k1_AXP_s100
                                 + 1.21649 w_k1_DIS_s100
                                 + 1.22832 w_k1_FDX_s100
                                 + 0.69152 w_k1_IBM_s100
                                 + 0.959929 w_k1_INTC_s100
                                 + 0.859271 w_k1_KO_s100
                                 + 0.737194 w_k1_MAT_s100
                                 + 1.12127 w_k1_MSFT_s100
                                 + 0.98746 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.978518 w_k1_ADM_s100
                                 + 0.899553 w_k1_AXP_s100
                                 + 0.990743 w_k1_DIS_s100
                                 + 1.03814 w_k1_FDX_s100
                                 + 0.784036 w_k1_IBM_s100
                                 + 0.770634 w_k1_INTC_s100
                                 + 0.802894 w_k1_KO_s100
                                 + 0.998121 w_k1_MAT_s100
                                 + 0.929694 w_k1_MSFT_s100
                                 + 0.914272 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.833829 w_k1_ADM_s100
                                 + 0.796632 w_k1_AXP_s100
                                 + 0.738043 w_k1_DIS_s100
                                 + 0.840857 w_k1_FDX_s100
                                 + 0.952365 w_k1_IBM_s100
                                 + 0.747192 w_k1_INTC_s100
                                 + 0.778739 w_k1_KO_s100
                                 + 1.18461 w_k1_MAT_s100
                                 + 0.955963 w_k1_MSFT_s100
                                 + 0.967456 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.939252 w_k1_ADM_s100
                                 + 0.880591 w_k1_AXP_s100
                                 + 1.12203 w_k1_DIS_s100
                                 + 0.995928 w_k1_FDX_s100
                                 + 1.34527 w_k1_IBM_s100
                                 + 0.9501 w_k1_INTC_s100 + 1.10948 w_k1_KO_s100
                                 + 1.5229 w_k1_MAT_s100
                                 + 1.09791 w_k1_MSFT_s100
                                 + 1.03716 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.850709 w_k1_ADM_s100
                                 + 0.698035 w_k1_AXP_s100
                                 + 0.591519 w_k1_DIS_s100
                                 + 0.746055 w_k1_FDX_s100
                                 + 0.711197 w_k1_IBM_s100
                                 + 0.740389 w_k1_INTC_s100
                                 + 0.619937 w_k1_KO_s100
                                 + 0.695965 w_k1_MAT_s100
                                 + 0.697999 w_k1_MSFT_s100
                                 + 0.817073 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.23901 w_k1_ADM_s100
                                 + 1.36691 w_k1_AXP_s100
                                 + 1.64809 w_k1_DIS_s100
                                 + 1.32469 w_k1_FDX_s100
                                 + 1.64669 w_k1_IBM_s100
                                 + 1.07902 w_k1_INTC_s100
                                 + 1.42662 w_k1_KO_s100 + 2.14217 w_k1_MAT_s100
                                 + 1.49425 w_k1_MSFT_s100
                                 + 1.38606 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.2999 w_k1_ADM_s100
                                 + 1.55518 w_k1_AXP_s100
                                 + 1.89715 w_k1_DIS_s100
                                 + 1.52674 w_k1_FDX_s100
                                 + 1.13814 w_k1_IBM_s100
                                 + 1.12309 w_k1_INTC_s100 + 1.3137 w_k1_KO_s100
                                 + 1.53802 w_k1_MAT_s100
                                 + 1.54308 w_k1_MSFT_s100 + 1.1282 w_k1_PG_s100
                                 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.39305 w_k1_ADM_s101
                                 + 1.19335 w_k1_AXP_s101
                                 + 1.10317 w_k1_DIS_s101
                                 + 1.04017 w_k1_FDX_s101
                                 + 1.31725 w_k1_IBM_s101
                                 + 1.11736 w_k1_INTC_s101
                                 + 1.19609 w_k1_KO_s101 + 1.19609 w_k1_MAT_s101
                                 + 0.99401 w_k1_MSFT_s101 + 1.1414 w_k1_PG_s101
                                 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.886806 w_k1_ADM_s101
                                 + 0.619114 w_k1_AXP_s101
                                 + 0.527091 w_k1_DIS_s101
                                 + 0.609838 w_k1_FDX_s101
                                 + 0.991674 w_k1_IBM_s101
                                 + 0.736275 w_k1_INTC_s101
                                 + 0.848321 w_k1_KO_s101
                                 + 0.901897 w_k1_MAT_s101
                                 + 0.694576 w_k1_MSFT_s101
                                 + 0.839237 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.919168 w_k1_ADM_s101
                                 + 0.71191 w_k1_AXP_s101
                                 + 0.643391 w_k1_DIS_s101
                                 + 0.880289 w_k1_FDX_s101
                                 + 0.706992 w_k1_IBM_s101
                                 + 0.903588 w_k1_INTC_s101
                                 + 0.714724 w_k1_KO_s101
                                 + 0.721474 w_k1_MAT_s101
                                 + 0.866399 w_k1_MSFT_s101
                                 + 0.789492 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.23455 w_k1_ADM_s101
                                 + 1.34986 w_k1_AXP_s101
                                 + 1.73313 w_k1_DIS_s101
                                 + 1.44122 w_k1_FDX_s101
                                 + 1.30204 w_k1_IBM_s101
                                 + 1.53525 w_k1_INTC_s101
                                 + 1.61367 w_k1_KO_s101 + 1.18501 w_k1_MAT_s101
                                 + 1.45261 w_k1_MSFT_s101
                                 + 1.26474 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.20615 w_k1_ADM_s101
                                 + 0.84135 w_k1_AXP_s101
                                 + 1.04504 w_k1_DIS_s101
                                 + 0.962481 w_k1_FDX_s101
                                 + 1.01626 w_k1_IBM_s101
                                 + 1.25923 w_k1_INTC_s101
                                 + 0.991952 w_k1_KO_s101
                                 + 0.755087 w_k1_MAT_s101
                                 + 1.09217 w_k1_MSFT_s101 + 1.0008 w_k1_PG_s101
                                 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.970302 w_k1_ADM_s101
                                 + 1.05953 w_k1_AXP_s101
                                 + 1.05642 w_k1_DIS_s101
                                 + 0.94208 w_k1_FDX_s101
                                 + 1.62874 w_k1_IBM_s101
                                 + 1.35271 w_k1_INTC_s101
                                 + 1.19457 w_k1_KO_s101 + 1.16655 w_k1_MAT_s101
                                 + 0.933323 w_k1_MSFT_s101
                                 + 0.994306 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.05714 w_k1_ADM_s101
                                 + 1.09172 w_k1_AXP_s101
                                 + 0.961874 w_k1_DIS_s101
                                 + 1.06098 w_k1_FDX_s101
                                 + 0.981257 w_k1_IBM_s101
                                 + 1.03633 w_k1_INTC_s101
                                 + 0.962577 w_k1_KO_s101
                                 + 0.845934 w_k1_MAT_s101
                                 + 0.784894 w_k1_MSFT_s101
                                 + 0.980965 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.736277 w_k1_ADM_s101
                                 + 0.71549 w_k1_AXP_s101
                                 + 0.76385 w_k1_DIS_s101
                                 + 0.799019 w_k1_FDX_s101
                                 + 0.990678 w_k1_IBM_s101
                                 + 0.890511 w_k1_INTC_s101
                                 + 0.894059 w_k1_KO_s101
                                 + 0.761948 w_k1_MAT_s101
                                 + 0.780593 w_k1_MSFT_s101
                                 + 0.876658 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.9272 w_k1_ADM_s102
                                 + 0.765894 w_k1_AXP_s102
                                 + 0.911005 w_k1_DIS_s102
                                 + 0.897171 w_k1_FDX_s102
                                 + 1.06455 w_k1_IBM_s102
                                 + 1.06663 w_k1_INTC_s102
                                 + 0.939689 w_k1_KO_s102
                                 + 0.783104 w_k1_MAT_s102
                                 + 1.02605 w_k1_MSFT_s102
                                 + 0.896467 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.14753 w_k1_ADM_s102
                                 + 1.08869 w_k1_AXP_s102
                                 + 1.30164 w_k1_DIS_s102
                                 + 1.24561 w_k1_FDX_s102
                                 + 0.928925 w_k1_IBM_s102
                                 + 1.17557 w_k1_INTC_s102
                                 + 1.00409 w_k1_KO_s102
                                 + 0.898937 w_k1_MAT_s102
                                 + 1.33847 w_k1_MSFT_s102
                                 + 1.16743 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.20622 w_k1_ADM_s102
                                 + 1.20435 w_k1_AXP_s102
                                 + 0.955962 w_k1_DIS_s102
                                 + 0.985128 w_k1_FDX_s102
                                 + 1.13881 w_k1_IBM_s102
                                 + 1.05144 w_k1_INTC_s102
                                 + 0.989173 w_k1_KO_s102
                                 + 1.03633 w_k1_MAT_s102
                                 + 0.944817 w_k1_MSFT_s102
                                 + 0.978244 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.06537 w_k1_ADM_s102
                                 + 1.14072 w_k1_AXP_s102
                                 + 1.03718 w_k1_DIS_s102
                                 + 1.17304 w_k1_FDX_s102
                                 + 0.802088 w_k1_IBM_s102
                                 + 1.15923 w_k1_INTC_s102
                                 + 0.900587 w_k1_KO_s102
                                 + 0.802335 w_k1_MAT_s102
                                 + 1.04709 w_k1_MSFT_s102
                                 + 0.887885 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.07516 w_k1_ADM_s102
                                 + 1.34475 w_k1_AXP_s102
                                 + 1.57299 w_k1_DIS_s102
                                 + 1.29102 w_k1_FDX_s102
                                 + 1.42609 w_k1_IBM_s102
                                 + 1.39085 w_k1_INTC_s102
                                 + 1.27297 w_k1_KO_s102 + 1.1486 w_k1_MAT_s102
                                 + 1.24729 w_k1_MSFT_s102 + 1.0425 w_k1_PG_s102
                                 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.703202 w_k1_ADM_s102
                                 + 0.821418 w_k1_AXP_s102
                                 + 0.713271 w_k1_DIS_s102
                                 + 0.734656 w_k1_FDX_s102
                                 + 1.05175 w_k1_IBM_s102
                                 + 0.920005 w_k1_INTC_s102
                                 + 0.799777 w_k1_KO_s102
                                 + 0.981444 w_k1_MAT_s102
                                 + 0.897729 w_k1_MSFT_s102
                                 + 0.88928 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.866458 w_k1_ADM_s102
                                 + 0.756466 w_k1_AXP_s102
                                 + 0.709971 w_k1_DIS_s102
                                 + 0.843982 w_k1_FDX_s102
                                 + 0.707424 w_k1_IBM_s102
                                 + 0.817248 w_k1_INTC_s102
                                 + 0.695598 w_k1_KO_s102
                                 + 0.730688 w_k1_MAT_s102
                                 + 0.920044 w_k1_MSFT_s102
                                 + 0.84988 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.05048 w_k1_ADM_s102
                                 + 1.20007 w_k1_AXP_s102 + 1.1352 w_k1_DIS_s102
                                 + 0.990283 w_k1_FDX_s102
                                 + 1.40519 w_k1_IBM_s102
                                 + 1.01854 w_k1_INTC_s102
                                 + 1.07855 w_k1_KO_s102 + 1.36418 w_k1_MAT_s102
                                 + 1.07579 w_k1_MSFT_s102
                                 + 1.06585 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.979054 w_k1_ADM_s103
                                 + 0.976891 w_k1_AXP_s103
                                 + 0.873089 w_k1_DIS_s103
                                 + 1.02602 w_k1_FDX_s103
                                 + 0.937518 w_k1_IBM_s103
                                 + 1.13392 w_k1_INTC_s103
                                 + 0.899311 w_k1_KO_s103
                                 + 1.04224 w_k1_MAT_s103
                                 + 0.937791 w_k1_MSFT_s103
                                 + 0.976972 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.795039 w_k1_ADM_s103
                                 + 0.70233 w_k1_AXP_s103
                                 + 0.649308 w_k1_DIS_s103
                                 + 0.866125 w_k1_FDX_s103
                                 + 0.721602 w_k1_IBM_s103
                                 + 0.876375 w_k1_INTC_s103
                                 + 0.710099 w_k1_KO_s103
                                 + 0.728076 w_k1_MAT_s103
                                 + 0.801017 w_k1_MSFT_s103
                                 + 0.828832 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.79196 w_k1_ADM_s103
                                 + 1.36526 w_k1_AXP_s103
                                 + 1.69195 w_k1_DIS_s103
                                 + 1.53545 w_k1_FDX_s103
                                 + 1.05716 w_k1_IBM_s103
                                 + 1.28395 w_k1_INTC_s103
                                 + 1.35499 w_k1_KO_s103 + 1.21454 w_k1_MAT_s103
                                 + 1.42271 w_k1_MSFT_s103
                                 + 1.11321 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.64523 w_k1_ADM_s103
                                 + 1.39293 w_k1_AXP_s103
                                 + 1.27006 w_k1_DIS_s103
                                 + 1.13027 w_k1_FDX_s103
                                 + 1.60466 w_k1_IBM_s103
                                 + 1.23105 w_k1_INTC_s103
                                 + 1.50506 w_k1_KO_s103 + 2.00069 w_k1_MAT_s103
                                 + 1.28823 w_k1_MSFT_s103
                                 + 1.31519 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.32209 w_k1_ADM_s103
                                 + 0.982173 w_k1_AXP_s103
                                 + 1.20874 w_k1_DIS_s103 + 1.2151 w_k1_FDX_s103
                                 + 0.827557 w_k1_IBM_s103
                                 + 1.10177 w_k1_INTC_s103
                                 + 1.07961 w_k1_KO_s103 + 1.1711 w_k1_MAT_s103
                                 + 1.20986 w_k1_MSFT_s103
                                 + 1.07383 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.33396 w_k1_ADM_s103
                                 + 1.40098 w_k1_AXP_s103 + 1.1743 w_k1_DIS_s103
                                 + 1.32983 w_k1_FDX_s103
                                 + 1.10986 w_k1_IBM_s103
                                 + 1.21824 w_k1_INTC_s103
                                 + 1.16808 w_k1_KO_s103 + 1.35267 w_k1_MAT_s103
                                 + 1.19315 w_k1_MSFT_s103
                                 + 1.16746 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.799724 w_k1_ADM_s103
                                 + 0.837014 w_k1_AXP_s103
                                 + 0.789479 w_k1_DIS_s103
                                 + 0.796579 w_k1_FDX_s103
                                 + 1.2732 w_k1_IBM_s103
                                 + 0.792237 w_k1_INTC_s103
                                 + 1.05957 w_k1_KO_s103 + 1.78254 w_k1_MAT_s103
                                 + 0.885479 w_k1_MSFT_s103
                                 + 0.897799 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.10182 w_k1_ADM_s103
                                 + 0.962211 w_k1_AXP_s103
                                 + 1.13408 w_k1_DIS_s103
                                 + 1.08033 w_k1_FDX_s103
                                 + 1.27432 w_k1_IBM_s103
                                 + 1.03111 w_k1_INTC_s103
                                 + 1.33732 w_k1_KO_s103 + 1.23507 w_k1_MAT_s103
                                 + 1.05279 w_k1_MSFT_s103 + 1.1378 w_k1_PG_s103
                                 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.685905 w_k1_ADM_s104
                                 + 0.658625 w_k1_AXP_s104
                                 + 0.523192 w_k1_DIS_s104
                                 + 0.750549 w_k1_FDX_s104
                                 + 0.763558 w_k1_IBM_s104
                                 + 0.774041 w_k1_INTC_s104
                                 + 0.761614 w_k1_KO_s104
                                 + 0.618671 w_k1_MAT_s104
                                 + 0.71791 w_k1_MSFT_s104
                                 + 0.871744 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.919046 w_k1_ADM_s104
                                 + 1.12548 w_k1_AXP_s104
                                 + 1.08844 w_k1_DIS_s104
                                 + 1.15024 w_k1_FDX_s104
                                 + 0.904914 w_k1_IBM_s104
                                 + 0.976231 w_k1_INTC_s104
                                 + 0.889853 w_k1_KO_s104
                                 + 0.842315 w_k1_MAT_s104
                                 + 1.12868 w_k1_MSFT_s104
                                 + 1.04172 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.14788 w_k1_ADM_s104
                                 + 1.44728 w_k1_AXP_s104
                                 + 1.82223 w_k1_DIS_s104
                                 + 1.44546 w_k1_FDX_s104
                                 + 1.25871 w_k1_IBM_s104
                                 + 1.04084 w_k1_INTC_s104
                                 + 1.11494 w_k1_KO_s104 + 1.03102 w_k1_MAT_s104
                                 + 1.33593 w_k1_MSFT_s104
                                 + 1.31048 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.733474 w_k1_ADM_s104
                                 + 0.616566 w_k1_AXP_s104
                                 + 0.614442 w_k1_DIS_s104
                                 + 0.808316 w_k1_FDX_s104
                                 + 0.575666 w_k1_IBM_s104
                                 + 0.711955 w_k1_INTC_s104
                                 + 0.632841 w_k1_KO_s104
                                 + 0.513435 w_k1_MAT_s104
                                 + 0.752566 w_k1_MSFT_s104
                                 + 0.872809 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.772888 w_k1_ADM_s104
                                 + 0.924792 w_k1_AXP_s104
                                 + 0.844803 w_k1_DIS_s104
                                 + 0.884779 w_k1_FDX_s104
                                 + 1.13326 w_k1_IBM_s104
                                 + 0.970012 w_k1_INTC_s104
                                 + 0.89111 w_k1_KO_s104 + 1.04217 w_k1_MAT_s104
                                 + 0.895179 w_k1_MSFT_s104
                                 + 0.980113 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.27765 w_k1_ADM_s104
                                 + 1.35247 w_k1_AXP_s104
                                 + 1.15046 w_k1_DIS_s104
                                 + 1.12374 w_k1_FDX_s104
                                 + 1.78261 w_k1_IBM_s104
                                 + 1.04317 w_k1_INTC_s104
                                 + 1.22071 w_k1_KO_s104 + 1.13985 w_k1_MAT_s104
                                 + 1.04936 w_k1_MSFT_s104
                                 + 1.24901 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.7339 w_k1_ADM_s104
                                 + 0.590694 w_k1_AXP_s104
                                 + 0.672856 w_k1_DIS_s104
                                 + 0.731569 w_k1_FDX_s104
                                 + 0.933647 w_k1_IBM_s104
                                 + 0.729248 w_k1_INTC_s104
                                 + 0.780703 w_k1_KO_s104
                                 + 0.740838 w_k1_MAT_s104
                                 + 0.744001 w_k1_MSFT_s104
                                 + 0.875029 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.956043 w_k1_ADM_s104
                                 + 0.790334 w_k1_AXP_s104
                                 + 0.77083 w_k1_DIS_s104
                                 + 0.855562 w_k1_FDX_s104
                                 + 0.804025 w_k1_IBM_s104
                                 + 0.754318 w_k1_INTC_s104
                                 + 0.845606 w_k1_KO_s104
                                 + 0.670129 w_k1_MAT_s104
                                 + 0.796551 w_k1_MSFT_s104
                                 + 0.92888 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.859205 w_k1_ADM_s105
                                 + 0.929377 w_k1_AXP_s105
                                 + 1.15352 w_k1_DIS_s105
                                 + 0.966506 w_k1_FDX_s105
                                 + 1.23303 w_k1_IBM_s105
                                 + 1.00327 w_k1_INTC_s105
                                 + 1.05549 w_k1_KO_s105
                                 + 0.857965 w_k1_MAT_s105
                                 + 0.985872 w_k1_MSFT_s105
                                 + 0.964834 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.30687 w_k1_ADM_s105
                                 + 1.08951 w_k1_AXP_s105
                                 + 1.41534 w_k1_DIS_s105
                                 + 1.24942 w_k1_FDX_s105
                                 + 1.00735 w_k1_IBM_s105
                                 + 1.26272 w_k1_INTC_s105
                                 + 1.12541 w_k1_KO_s105
                                 + 0.985857 w_k1_MAT_s105
                                 + 1.1793 w_k1_MSFT_s105 + 1.10405 w_k1_PG_s105
                                 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.996176 w_k1_ADM_s105
                                 + 1.04957 w_k1_AXP_s105 + 1.0344 w_k1_DIS_s105
                                 + 0.98348 w_k1_FDX_s105
                                 + 1.31681 w_k1_IBM_s105
                                 + 0.701548 w_k1_INTC_s105
                                 + 1.05836 w_k1_KO_s105 + 1.6052 w_k1_MAT_s105
                                 + 0.854609 w_k1_MSFT_s105
                                 + 1.06164 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.736441 w_k1_ADM_s105
                                 + 0.614185 w_k1_AXP_s105
                                 + 0.61782 w_k1_DIS_s105
                                 + 0.704493 w_k1_FDX_s105
                                 + 0.677497 w_k1_IBM_s105
                                 + 0.677043 w_k1_INTC_s105
                                 + 0.718259 w_k1_KO_s105
                                 + 0.740678 w_k1_MAT_s105
                                 + 0.772982 w_k1_MSFT_s105
                                 + 0.8432 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.863054 w_k1_ADM_s105
                                 + 0.76726 w_k1_AXP_s105
                                 + 0.642422 w_k1_DIS_s105
                                 + 0.787715 w_k1_FDX_s105
                                 + 0.795498 w_k1_IBM_s105
                                 + 0.707042 w_k1_INTC_s105
                                 + 0.758918 w_k1_KO_s105
                                 + 0.765527 w_k1_MAT_s105
                                 + 0.741466 w_k1_MSFT_s105
                                 + 0.796849 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.855604 w_k1_ADM_s105
                                 + 0.764743 w_k1_AXP_s105
                                 + 0.97314 w_k1_DIS_s105
                                 + 0.939017 w_k1_FDX_s105
                                 + 0.814867 w_k1_IBM_s105
                                 + 0.840019 w_k1_INTC_s105
                                 + 0.889096 w_k1_KO_s105
                                 + 0.740413 w_k1_MAT_s105
                                 + 0.905034 w_k1_MSFT_s105
                                 + 0.962944 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.32705 w_k1_ADM_s105
                                 + 1.33863 w_k1_AXP_s105
                                 + 1.56015 w_k1_DIS_s105
                                 + 1.23165 w_k1_FDX_s105 + 1.5806 w_k1_IBM_s105
                                 + 0.998787 w_k1_INTC_s105
                                 + 1.59775 w_k1_KO_s105 + 2.40203 w_k1_MAT_s105
                                 + 1.09039 w_k1_MSFT_s105
                                 + 1.27566 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.17637 w_k1_ADM_s105
                                 + 0.961961 w_k1_AXP_s105
                                 + 0.920657 w_k1_DIS_s105
                                 + 0.990566 w_k1_FDX_s105
                                 + 0.995126 w_k1_IBM_s105
                                 + 0.763373 w_k1_INTC_s105
                                 + 0.894656 w_k1_KO_s105
                                 + 1.06044 w_k1_MAT_s105
                                 + 0.827903 w_k1_MSFT_s105
                                 + 0.892994 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.18459 w_k1_ADM_s106
                                 + 1.19846 w_k1_AXP_s106 + 1.7087 w_k1_DIS_s106
                                 + 1.48082 w_k1_FDX_s106
                                 + 0.901814 w_k1_IBM_s106
                                 + 1.51347 w_k1_INTC_s106
                                 + 1.46266 w_k1_KO_s106 + 1.1352 w_k1_MAT_s106
                                 + 1.53219 w_k1_MSFT_s106
                                 + 1.22577 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.703987 w_k1_ADM_s106
                                 + 0.847023 w_k1_AXP_s106
                                 + 0.903004 w_k1_DIS_s106
                                 + 0.892396 w_k1_FDX_s106
                                 + 0.958065 w_k1_IBM_s106
                                 + 0.834643 w_k1_INTC_s106
                                 + 0.966549 w_k1_KO_s106
                                 + 1.05271 w_k1_MAT_s106
                                 + 0.854768 w_k1_MSFT_s106
                                 + 0.911831 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.941491 w_k1_ADM_s106
                                 + 1.15971 w_k1_AXP_s106 + 1.543 w_k1_DIS_s106
                                 + 1.24608 w_k1_FDX_s106
                                 + 1.21574 w_k1_IBM_s106
                                 + 1.29173 w_k1_INTC_s106
                                 + 1.33999 w_k1_KO_s106 + 1.49675 w_k1_MAT_s106
                                 + 1.43761 w_k1_MSFT_s106 + 1.0744 w_k1_PG_s106
                                 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.757703 w_k1_ADM_s106
                                 + 0.67246 w_k1_AXP_s106
                                 + 0.670103 w_k1_DIS_s106
                                 + 0.737603 w_k1_FDX_s106
                                 + 1.07095 w_k1_IBM_s106
                                 + 0.886662 w_k1_INTC_s106
                                 + 1.13488 w_k1_KO_s106 + 1.17282 w_k1_MAT_s106
                                 + 0.822752 w_k1_MSFT_s106
                                 + 0.839824 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.827551 w_k1_ADM_s106
                                 + 1.16058 w_k1_AXP_s106
                                 + 0.995426 w_k1_DIS_s106
                                 + 0.973432 w_k1_FDX_s106
                                 + 1.45221 w_k1_IBM_s106
                                 + 1.22609 w_k1_INTC_s106
                                 + 1.20581 w_k1_KO_s106 + 1.57912 w_k1_MAT_s106
                                 + 1.16175 w_k1_MSFT_s106
                                 + 1.08495 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.81463 w_k1_ADM_s106
                                 + 0.611254 w_k1_AXP_s106
                                 + 0.9057 w_k1_DIS_s106
                                 + 0.879023 w_k1_FDX_s106
                                 + 0.710863 w_k1_IBM_s106
                                 + 0.843933 w_k1_INTC_s106
                                 + 0.885117 w_k1_KO_s106
                                 + 0.816356 w_k1_MAT_s106
                                 + 0.948225 w_k1_MSFT_s106
                                 + 0.854856 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.02426 w_k1_ADM_s106
                                 + 1.02039 w_k1_AXP_s106
                                 + 1.05248 w_k1_DIS_s106
                                 + 1.12888 w_k1_FDX_s106
                                 + 0.831779 w_k1_IBM_s106
                                 + 1.16678 w_k1_INTC_s106
                                 + 1.04615 w_k1_KO_s106
                                 + 0.697733 w_k1_MAT_s106
                                 + 1.15694 w_k1_MSFT_s106
                                 + 1.01476 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.12457 w_k1_ADM_s106
                                 + 1.00904 w_k1_AXP_s106
                                 + 1.23362 w_k1_DIS_s106
                                 + 1.12429 w_k1_FDX_s106
                                 + 0.606087 w_k1_IBM_s106
                                 + 1.0285 w_k1_INTC_s106
                                 + 0.836559 w_k1_KO_s106
                                 + 0.515067 w_k1_MAT_s106
                                 + 1.05406 w_k1_MSFT_s106
                                 + 0.917532 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.11631 w_k1_ADM_s107
                                 + 0.91324 w_k1_AXP_s107
                                 + 0.849556 w_k1_DIS_s107
                                 + 0.865839 w_k1_FDX_s107
                                 + 1.0458 w_k1_IBM_s107
                                 + 0.907203 w_k1_INTC_s107
                                 + 0.863477 w_k1_KO_s107
                                 + 0.73124 w_k1_MAT_s107
                                 + 0.884956 w_k1_MSFT_s107
                                 + 0.930684 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.05023 w_k1_ADM_s107
                                 + 1.40463 w_k1_AXP_s107
                                 + 1.03528 w_k1_DIS_s107
                                 + 1.01751 w_k1_FDX_s107
                                 + 1.19192 w_k1_IBM_s107
                                 + 1.07153 w_k1_INTC_s107
                                 + 1.05519 w_k1_KO_s107 + 1.13898 w_k1_MAT_s107
                                 + 0.969375 w_k1_MSFT_s107
                                 + 1.05552 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.59236 w_k1_ADM_s107
                                 + 1.6321 w_k1_AXP_s107 + 1.6314 w_k1_DIS_s107
                                 + 1.35706 w_k1_FDX_s107
                                 + 1.63225 w_k1_IBM_s107
                                 + 1.43224 w_k1_INTC_s107
                                 + 1.25154 w_k1_KO_s107 + 1.52971 w_k1_MAT_s107
                                 + 1.2643 w_k1_MSFT_s107 + 1.29447 w_k1_PG_s107
                                 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.43942 w_k1_ADM_s107
                                 + 1.22817 w_k1_AXP_s107
                                 + 1.05142 w_k1_DIS_s107
                                 + 1.00021 w_k1_FDX_s107
                                 + 1.57651 w_k1_IBM_s107
                                 + 1.27518 w_k1_INTC_s107
                                 + 1.14273 w_k1_KO_s107 + 1.2125 w_k1_MAT_s107
                                 + 1.04521 w_k1_MSFT_s107
                                 + 1.22264 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.31333 w_k1_ADM_s107
                                 + 1.23391 w_k1_AXP_s107
                                 + 1.18211 w_k1_DIS_s107
                                 + 1.17434 w_k1_FDX_s107 + 1.0109 w_k1_IBM_s107
                                 + 1.47872 w_k1_INTC_s107
                                 + 1.01674 w_k1_KO_s107 + 0.66431 w_k1_MAT_s107
                                 + 1.10674 w_k1_MSFT_s107
                                 + 1.20997 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.709797 w_k1_ADM_s107
                                 + 0.82318 w_k1_AXP_s107
                                 + 0.622747 w_k1_DIS_s107
                                 + 0.798686 w_k1_FDX_s107
                                 + 0.609022 w_k1_IBM_s107
                                 + 1.08905 w_k1_INTC_s107
                                 + 0.642276 w_k1_KO_s107
                                 + 0.564133 w_k1_MAT_s107
                                 + 0.820786 w_k1_MSFT_s107
                                 + 0.810417 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.807301 w_k1_ADM_s107
                                 + 1.04755 w_k1_AXP_s107
                                 + 0.892424 w_k1_DIS_s107
                                 + 0.919606 w_k1_FDX_s107
                                 + 0.831685 w_k1_IBM_s107
                                 + 1.01022 w_k1_INTC_s107
                                 + 0.769833 w_k1_KO_s107
                                 + 0.708971 w_k1_MAT_s107
                                 + 0.956769 w_k1_MSFT_s107
                                 + 0.913537 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.4357 w_k1_ADM_s107
                                 + 1.26921 w_k1_AXP_s107
                                 + 1.62797 w_k1_DIS_s107
                                 + 1.38741 w_k1_FDX_s107
                                 + 0.931004 w_k1_IBM_s107
                                 + 1.3066 w_k1_INTC_s107 + 1.07884 w_k1_KO_s107
                                 + 0.879381 w_k1_MAT_s107
                                 + 1.39905 w_k1_MSFT_s107
                                 + 1.15074 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.38372 w_k1_ADM_s110
                                 + 1.25449 w_k1_AXP_s110
                                 + 1.51371 w_k1_DIS_s110
                                 + 1.30153 w_k1_FDX_s110
                                 + 1.18037 w_k1_IBM_s110
                                 + 1.22548 w_k1_INTC_s110 + 1.4687 w_k1_KO_s110
                                 + 1.3883 w_k1_MAT_s110 + 1.114 w_k1_MSFT_s110
                                 + 1.36684 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.873132 w_k1_ADM_s110
                                 + 0.681739 w_k1_AXP_s110
                                 + 0.65811 w_k1_DIS_s110
                                 + 0.786629 w_k1_FDX_s110
                                 + 0.623817 w_k1_IBM_s110
                                 + 0.763187 w_k1_INTC_s110
                                 + 0.752845 w_k1_KO_s110
                                 + 0.673108 w_k1_MAT_s110
                                 + 0.716875 w_k1_MSFT_s110
                                 + 0.842067 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.03689 w_k1_ADM_s110
                                 + 0.617035 w_k1_AXP_s110
                                 + 0.762653 w_k1_DIS_s110
                                 + 0.797229 w_k1_FDX_s110
                                 + 1.00983 w_k1_IBM_s110
                                 + 0.85214 w_k1_INTC_s110 + 0.9597 w_k1_KO_s110
                                 + 0.983312 w_k1_MAT_s110
                                 + 0.795921 w_k1_MSFT_s110
                                 + 0.963072 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.969863 w_k1_ADM_s110
                                 + 0.746014 w_k1_AXP_s110
                                 + 0.818391 w_k1_DIS_s110
                                 + 0.945672 w_k1_FDX_s110
                                 + 0.721791 w_k1_IBM_s110
                                 + 1.04521 w_k1_INTC_s110
                                 + 0.846547 w_k1_KO_s110
                                 + 0.679985 w_k1_MAT_s110
                                 + 0.917147 w_k1_MSFT_s110
                                 + 0.899851 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.35877 w_k1_ADM_s110
                                 + 0.940161 w_k1_AXP_s110
                                 + 1.10883 w_k1_DIS_s110
                                 + 1.20833 w_k1_FDX_s110
                                 + 0.767576 w_k1_IBM_s110
                                 + 0.977548 w_k1_INTC_s110
                                 + 0.942508 w_k1_KO_s110
                                 + 0.824987 w_k1_MAT_s110
                                 + 0.982023 w_k1_MSFT_s110
                                 + 1.11341 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.980473 w_k1_ADM_s110
                                 + 1.14645 w_k1_AXP_s110
                                 + 1.09162 w_k1_DIS_s110
                                 + 0.930396 w_k1_FDX_s110
                                 + 1.63986 w_k1_IBM_s110
                                 + 1.04737 w_k1_INTC_s110
                                 + 1.32728 w_k1_KO_s110 + 1.55602 w_k1_MAT_s110
                                 + 0.986392 w_k1_MSFT_s110
                                 + 1.13121 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.04389 w_k1_ADM_s110
                                 + 0.978933 w_k1_AXP_s110
                                 + 1.08531 w_k1_DIS_s110
                                 + 0.993569 w_k1_FDX_s110
                                 + 1.18994 w_k1_IBM_s110
                                 + 1.02574 w_k1_INTC_s110
                                 + 1.20771 w_k1_KO_s110 + 1.20654 w_k1_MAT_s110
                                 + 0.916827 w_k1_MSFT_s110
                                 + 1.01478 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.707209 w_k1_ADM_s110
                                 + 0.587983 w_k1_AXP_s110
                                 + 0.661931 w_k1_DIS_s110
                                 + 0.67272 w_k1_FDX_s110
                                 + 1.10451 w_k1_IBM_s110
                                 + 0.70792 w_k1_INTC_s110
                                 + 0.983669 w_k1_KO_s110
                                 + 1.06009 w_k1_MAT_s110
                                 + 0.693774 w_k1_MSFT_s110
                                 + 0.901378 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.95054 w_k1_ADM_s111
                                 + 0.810309 w_k1_AXP_s111
                                 + 0.750881 w_k1_DIS_s111
                                 + 0.933981 w_k1_FDX_s111
                                 + 1.29328 w_k1_IBM_s111
                                 + 0.891024 w_k1_INTC_s111
                                 + 0.811447 w_k1_KO_s111
                                 + 1.05494 w_k1_MAT_s111
                                 + 0.867233 w_k1_MSFT_s111
                                 + 0.953161 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.935274 w_k1_ADM_s111
                                 + 1.0907 w_k1_AXP_s111 + 1.21081 w_k1_DIS_s111
                                 + 1.22194 w_k1_FDX_s111
                                 + 1.13892 w_k1_IBM_s111
                                 + 1.15719 w_k1_INTC_s111
                                 + 1.08224 w_k1_KO_s111 + 1.18006 w_k1_MAT_s111
                                 + 1.13353 w_k1_MSFT_s111
                                 + 1.08448 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.07249 w_k1_ADM_s111
                                 + 1.22573 w_k1_AXP_s111
                                 + 1.02398 w_k1_DIS_s111
                                 + 1.04446 w_k1_FDX_s111
                                 + 1.01564 w_k1_IBM_s111
                                 + 0.963149 w_k1_INTC_s111
                                 + 0.84439 w_k1_KO_s111 + 1.27534 w_k1_MAT_s111
                                 + 0.874208 w_k1_MSFT_s111
                                 + 0.997692 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.949197 w_k1_ADM_s111
                                 + 0.630748 w_k1_AXP_s111
                                 + 0.722768 w_k1_DIS_s111
                                 + 0.903465 w_k1_FDX_s111
                                 + 0.645803 w_k1_IBM_s111
                                 + 0.760164 w_k1_INTC_s111
                                 + 0.735203 w_k1_KO_s111
                                 + 0.600985 w_k1_MAT_s111
                                 + 0.863927 w_k1_MSFT_s111
                                 + 0.850413 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.829285 w_k1_ADM_s111
                                 + 0.793465 w_k1_AXP_s111
                                 + 1.11652 w_k1_DIS_s111
                                 + 1.02574 w_k1_FDX_s111
                                 + 0.847707 w_k1_IBM_s111
                                 + 1.07614 w_k1_INTC_s111
                                 + 1.05133 w_k1_KO_s111 + 1.1624 w_k1_MAT_s111
                                 + 1.01025 w_k1_MSFT_s111
                                 + 1.03349 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.816868 w_k1_ADM_s111
                                 + 0.693756 w_k1_AXP_s111
                                 + 0.797125 w_k1_DIS_s111
                                 + 0.911244 w_k1_FDX_s111
                                 + 0.684735 w_k1_IBM_s111
                                 + 0.832901 w_k1_INTC_s111
                                 + 0.732585 w_k1_KO_s111
                                 + 0.770329 w_k1_MAT_s111
                                 + 0.88635 w_k1_MSFT_s111
                                 + 0.829304 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.796554 w_k1_ADM_s111
                                 + 0.936413 w_k1_AXP_s111
                                 + 0.778484 w_k1_DIS_s111
                                 + 0.818774 w_k1_FDX_s111
                                 + 1.02189 w_k1_IBM_s111
                                 + 0.724471 w_k1_INTC_s111
                                 + 0.826993 w_k1_KO_s111
                                 + 1.06226 w_k1_MAT_s111
                                 + 0.732002 w_k1_MSFT_s111
                                 + 0.940917 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.17588 w_k1_ADM_s111
                                 + 1.61974 w_k1_AXP_s111
                                 + 1.46895 w_k1_DIS_s111
                                 + 1.34983 w_k1_FDX_s111
                                 + 1.39819 w_k1_IBM_s111
                                 + 1.14315 w_k1_INTC_s111
                                 + 1.27475 w_k1_KO_s111 + 1.86568 w_k1_MAT_s111
                                 + 1.12839 w_k1_MSFT_s111
                                 + 1.23264 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.802238 w_k1_ADM_s112
                                 + 1.12001 w_k1_AXP_s112
                                 + 0.855834 w_k1_DIS_s112
                                 + 0.940141 w_k1_FDX_s112
                                 + 1.10914 w_k1_IBM_s112
                                 + 0.878613 w_k1_INTC_s112
                                 + 0.806538 w_k1_KO_s112
                                 + 1.21745 w_k1_MAT_s112
                                 + 1.03864 w_k1_MSFT_s112
                                 + 1.02745 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.936154 w_k1_ADM_s112
                                 + 1.13065 w_k1_AXP_s112
                                 + 0.944769 w_k1_DIS_s112
                                 + 1.0366 w_k1_FDX_s112
                                 + 0.704484 w_k1_IBM_s112
                                 + 0.948176 w_k1_INTC_s112
                                 + 0.727809 w_k1_KO_s112
                                 + 0.595496 w_k1_MAT_s112
                                 + 0.945966 w_k1_MSFT_s112
                                 + 0.954879 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.761184 w_k1_ADM_s112
                                 + 0.777945 w_k1_AXP_s112
                                 + 0.533958 w_k1_DIS_s112
                                 + 0.738953 w_k1_FDX_s112
                                 + 1.17849 w_k1_IBM_s112
                                 + 0.753391 w_k1_INTC_s112
                                 + 0.803017 w_k1_KO_s112
                                 + 1.23671 w_k1_MAT_s112
                                 + 0.743346 w_k1_MSFT_s112
                                 + 0.893521 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.855115 w_k1_ADM_s112
                                 + 1.10397 w_k1_AXP_s112
                                 + 0.852339 w_k1_DIS_s112
                                 + 0.903366 w_k1_FDX_s112
                                 + 0.884195 w_k1_IBM_s112
                                 + 0.948766 w_k1_INTC_s112
                                 + 0.889169 w_k1_KO_s112
                                 + 0.863774 w_k1_MAT_s112
                                 + 0.881043 w_k1_MSFT_s112
                                 + 0.925284 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.70191 w_k1_ADM_s112
                                 + 0.870791 w_k1_AXP_s112
                                 + 0.50829 w_k1_DIS_s112
                                 + 0.744404 w_k1_FDX_s112
                                 + 0.74953 w_k1_IBM_s112
                                 + 0.722937 w_k1_INTC_s112
                                 + 0.638683 w_k1_KO_s112
                                 + 0.712464 w_k1_MAT_s112
                                 + 0.669062 w_k1_MSFT_s112
                                 + 0.79425 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.00999 w_k1_ADM_s112
                                 + 1.364 w_k1_AXP_s112 + 1.17489 w_k1_DIS_s112
                                 + 1.14217 w_k1_FDX_s112
                                 + 0.939436 w_k1_IBM_s112
                                 + 0.975593 w_k1_INTC_s112
                                 + 0.919002 w_k1_KO_s112
                                 + 1.01665 w_k1_MAT_s112
                                 + 1.12069 w_k1_MSFT_s112
                                 + 1.03456 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.07319 w_k1_ADM_s112
                                 + 1.31598 w_k1_AXP_s112
                                 + 1.12146 w_k1_DIS_s112
                                 + 1.00244 w_k1_FDX_s112
                                 + 1.58517 w_k1_IBM_s112
                                 + 0.956496 w_k1_INTC_s112
                                 + 1.13443 w_k1_KO_s112 + 1.86285 w_k1_MAT_s112
                                 + 0.982399 w_k1_MSFT_s112
                                 + 1.19693 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.26987 w_k1_ADM_s112
                                 + 1.64308 w_k1_AXP_s112
                                 + 1.47735 w_k1_DIS_s112
                                 + 1.41098 w_k1_FDX_s112
                                 + 1.16797 w_k1_IBM_s112
                                 + 1.28176 w_k1_INTC_s112
                                 + 1.12786 w_k1_KO_s112 + 1.68053 w_k1_MAT_s112
                                 + 1.35238 w_k1_MSFT_s112
                                 + 1.33473 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.950634 w_k1_ADM_s113
                                 + 0.810699 w_k1_AXP_s113
                                 + 1.01859 w_k1_DIS_s113
                                 + 1.07357 w_k1_FDX_s113
                                 + 0.877761 w_k1_IBM_s113
                                 + 1.13088 w_k1_INTC_s113
                                 + 0.870156 w_k1_KO_s113
                                 + 0.62183 w_k1_MAT_s113
                                 + 1.00537 w_k1_MSFT_s113
                                 + 0.969842 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.809216 w_k1_ADM_s113
                                 + 0.674253 w_k1_AXP_s113
                                 + 0.552917 w_k1_DIS_s113
                                 + 0.756887 w_k1_FDX_s113
                                 + 0.962554 w_k1_IBM_s113
                                 + 0.831618 w_k1_INTC_s113
                                 + 0.765755 w_k1_KO_s113
                                 + 0.940365 w_k1_MAT_s113
                                 + 0.740476 w_k1_MSFT_s113
                                 + 0.884548 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.95544 w_k1_ADM_s113
                                 + 0.82328 w_k1_AXP_s113
                                 + 0.800123 w_k1_DIS_s113
                                 + 0.950269 w_k1_FDX_s113
                                 + 0.71882 w_k1_IBM_s113
                                 + 1.03914 w_k1_INTC_s113
                                 + 0.769336 w_k1_KO_s113
                                 + 0.549216 w_k1_MAT_s113
                                 + 0.912414 w_k1_MSFT_s113
                                 + 0.849675 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.954007 w_k1_ADM_s113
                                 + 0.976352 w_k1_AXP_s113
                                 + 1.05544 w_k1_DIS_s113 + 1.0121 w_k1_FDX_s113
                                 + 1.33369 w_k1_IBM_s113
                                 + 1.12939 w_k1_INTC_s113
                                 + 1.04037 w_k1_KO_s113
                                 + 0.959443 w_k1_MAT_s113
                                 + 1.0386 w_k1_MSFT_s113
                                 + 0.936402 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.05357 w_k1_ADM_s113
                                 + 1.33496 w_k1_AXP_s113
                                 + 0.868499 w_k1_DIS_s113
                                 + 1.0871 w_k1_FDX_s113 + 1.0674 w_k1_IBM_s113
                                 + 1.002 w_k1_INTC_s113 + 0.790047 w_k1_KO_s113
                                 + 1.14929 w_k1_MAT_s113
                                 + 0.95137 w_k1_MSFT_s113
                                 + 1.01114 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.04774 w_k1_ADM_s113
                                 + 1.25827 w_k1_AXP_s113
                                 + 1.52646 w_k1_DIS_s113
                                 + 1.22381 w_k1_FDX_s113
                                 + 1.20161 w_k1_IBM_s113
                                 + 1.28085 w_k1_INTC_s113
                                 + 1.32893 w_k1_KO_s113 + 1.71585 w_k1_MAT_s113
                                 + 1.2828 w_k1_MSFT_s113 + 1.19889 w_k1_PG_s113
                                 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.643166 w_k1_ADM_s113
                                 + 0.583287 w_k1_AXP_s113
                                 + 0.620415 w_k1_DIS_s113
                                 + 0.701234 w_k1_FDX_s113
                                 + 0.666489 w_k1_IBM_s113
                                 + 0.871522 w_k1_INTC_s113
                                 + 0.6951 w_k1_KO_s113 + 0.522423 w_k1_MAT_s113
                                 + 0.77971 w_k1_MSFT_s113
                                 + 0.710206 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.34243 w_k1_ADM_s113
                                 + 1.2558 w_k1_AXP_s113 + 1.41597 w_k1_DIS_s113
                                 + 1.42565 w_k1_FDX_s113
                                 + 0.928849 w_k1_IBM_s113
                                 + 1.50034 w_k1_INTC_s113
                                 + 1.05322 w_k1_KO_s113 + 1.12247 w_k1_MAT_s113
                                 + 1.34685 w_k1_MSFT_s113
                                 + 1.11521 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.69387 w_k1_ADM_s114
                                 + 1.43742 w_k1_AXP_s114
                                 + 1.93927 w_k1_DIS_s114
                                 + 1.61621 w_k1_FDX_s114
                                 + 0.986254 w_k1_IBM_s114
                                 + 1.31477 w_k1_INTC_s114
                                 + 1.29137 w_k1_KO_s114
                                 + 0.964087 w_k1_MAT_s114
                                 + 1.39655 w_k1_MSFT_s114
                                 + 1.32285 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.828818 w_k1_ADM_s114
                                 + 0.7564 w_k1_AXP_s114
                                 + 0.556289 w_k1_DIS_s114
                                 + 0.835679 w_k1_FDX_s114
                                 + 0.733112 w_k1_IBM_s114
                                 + 0.871203 w_k1_INTC_s114
                                 + 0.656241 w_k1_KO_s114
                                 + 0.74926 w_k1_MAT_s114
                                 + 0.778834 w_k1_MSFT_s114
                                 + 0.844076 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.889776 w_k1_ADM_s114
                                 + 1.07184 w_k1_AXP_s114
                                 + 0.951844 w_k1_DIS_s114
                                 + 1.02961 w_k1_FDX_s114 + 1.5466 w_k1_IBM_s114
                                 + 0.861502 w_k1_INTC_s114
                                 + 1.10093 w_k1_KO_s114 + 1.45437 w_k1_MAT_s114
                                 + 0.886977 w_k1_MSFT_s114
                                 + 1.08951 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.17181 w_k1_ADM_s114
                                 + 0.864446 w_k1_AXP_s114
                                 + 0.733635 w_k1_DIS_s114
                                 + 1.02742 w_k1_FDX_s114
                                 + 0.664936 w_k1_IBM_s114
                                 + 0.979838 w_k1_INTC_s114
                                 + 0.739564 w_k1_KO_s114
                                 + 0.808788 w_k1_MAT_s114
                                 + 0.943067 w_k1_MSFT_s114
                                 + 0.867416 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.26965 w_k1_ADM_s114
                                 + 1.20416 w_k1_AXP_s114
                                 + 1.45091 w_k1_DIS_s114 + 1.3424 w_k1_FDX_s114
                                 + 1.16931 w_k1_IBM_s114
                                 + 1.28879 w_k1_INTC_s114
                                 + 1.27909 w_k1_KO_s114 + 1.04471 w_k1_MAT_s114
                                 + 1.14536 w_k1_MSFT_s114
                                 + 1.15657 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.16189 w_k1_ADM_s114
                                 + 1.17645 w_k1_AXP_s114
                                 + 1.04082 w_k1_DIS_s114
                                 + 0.979282 w_k1_FDX_s114
                                 + 1.13928 w_k1_IBM_s114
                                 + 0.955925 w_k1_INTC_s114
                                 + 0.878468 w_k1_KO_s114
                                 + 0.910871 w_k1_MAT_s114
                                 + 0.909584 w_k1_MSFT_s114
                                 + 0.991009 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.770139 w_k1_ADM_s114
                                 + 0.5361 w_k1_AXP_s114
                                 + 0.466501 w_k1_DIS_s114
                                 + 0.672755 w_k1_FDX_s114
                                 + 0.851238 w_k1_IBM_s114
                                 + 0.6803 w_k1_INTC_s114
                                 + 0.665264 w_k1_KO_s114
                                 + 0.675228 w_k1_MAT_s114
                                 + 0.648696 w_k1_MSFT_s114
                                 + 0.742558 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.19413 w_k1_ADM_s114
                                 + 0.810178 w_k1_AXP_s114
                                 + 1.1649 w_k1_DIS_s114 + 1.00303 w_k1_FDX_s114
                                 + 0.965889 w_k1_IBM_s114
                                 + 0.947441 w_k1_INTC_s114
                                 + 0.859682 w_k1_KO_s114
                                 + 0.835124 w_k1_MAT_s114
                                 + 1.08735 w_k1_MSFT_s114
                                 + 0.943732 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.937397 w_k1_ADM_s115
                                 + 1.10469 w_k1_AXP_s115
                                 + 1.24311 w_k1_DIS_s115
                                 + 1.29078 w_k1_FDX_s115
                                 + 0.860872 w_k1_IBM_s115
                                 + 1.22647 w_k1_INTC_s115
                                 + 0.994219 w_k1_KO_s115
                                 + 0.614776 w_k1_MAT_s115
                                 + 1.25045 w_k1_MSFT_s115
                                 + 1.06095 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.829195 w_k1_ADM_s115
                                 + 0.73431 w_k1_AXP_s115
                                 + 0.770753 w_k1_DIS_s115
                                 + 1.00425 w_k1_FDX_s115
                                 + 0.679296 w_k1_IBM_s115
                                 + 0.980513 w_k1_INTC_s115
                                 + 0.823413 w_k1_KO_s115
                                 + 0.490394 w_k1_MAT_s115
                                 + 0.981219 w_k1_MSFT_s115
                                 + 0.935864 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.674252 w_k1_ADM_s115
                                 + 0.699685 w_k1_AXP_s115
                                 + 0.771092 w_k1_DIS_s115
                                 + 0.801145 w_k1_FDX_s115
                                 + 1.30004 w_k1_IBM_s115
                                 + 0.930647 w_k1_INTC_s115
                                 + 1.04928 w_k1_KO_s115 + 1.22106 w_k1_MAT_s115
                                 + 0.845086 w_k1_MSFT_s115
                                 + 0.912233 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.22053 w_k1_ADM_s115
                                 + 1.24751 w_k1_AXP_s115
                                 + 1.75517 w_k1_DIS_s115
                                 + 1.30191 w_k1_FDX_s115
                                 + 1.30882 w_k1_IBM_s115
                                 + 1.22357 w_k1_INTC_s115 + 1.4078 w_k1_KO_s115
                                 + 0.839792 w_k1_MAT_s115
                                 + 1.12222 w_k1_MSFT_s115
                                 + 1.10703 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.869022 w_k1_ADM_s115
                                 + 0.822986 w_k1_AXP_s115
                                 + 0.917559 w_k1_DIS_s115
                                 + 0.932719 w_k1_FDX_s115
                                 + 0.8719 w_k1_IBM_s115
                                 + 0.969194 w_k1_INTC_s115
                                 + 0.899457 w_k1_KO_s115
                                 + 0.520238 w_k1_MAT_s115
                                 + 0.839633 w_k1_MSFT_s115
                                 + 0.89584 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.893696 w_k1_ADM_s115
                                 + 0.824219 w_k1_AXP_s115
                                 + 1.18473 w_k1_DIS_s115
                                 + 0.997675 w_k1_FDX_s115
                                 + 1.43115 w_k1_IBM_s115
                                 + 1.10828 w_k1_INTC_s115
                                 + 1.46016 w_k1_KO_s115 + 1.33998 w_k1_MAT_s115
                                 + 1.15706 w_k1_MSFT_s115
                                 + 1.18102 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.747863 w_k1_ADM_s115
                                 + 0.534561 w_k1_AXP_s115
                                 + 0.627855 w_k1_DIS_s115
                                 + 0.724195 w_k1_FDX_s115
                                 + 0.725698 w_k1_IBM_s115
                                 + 0.71828 w_k1_INTC_s115
                                 + 0.646424 w_k1_KO_s115
                                 + 0.645277 w_k1_MAT_s115
                                 + 0.686886 w_k1_MSFT_s115
                                 + 0.744522 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.98821 w_k1_ADM_s115
                                 + 1.06693 w_k1_AXP_s115
                                 + 0.930331 w_k1_DIS_s115
                                 + 1.04182 w_k1_FDX_s115
                                 + 1.29006 w_k1_IBM_s115
                                 + 0.882991 w_k1_INTC_s115
                                 + 1.07009 w_k1_KO_s115 + 1.07093 w_k1_MAT_s115
                                 + 0.915594 w_k1_MSFT_s115
                                 + 0.966387 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.30327 w_k1_ADM_s116
                                 + 1.7526 w_k1_AXP_s116 + 1.69112 w_k1_DIS_s116
                                 + 1.46146 w_k1_FDX_s116
                                 + 1.22001 w_k1_IBM_s116
                                 + 1.45432 w_k1_INTC_s116
                                 + 1.48105 w_k1_KO_s116 + 2.21758 w_k1_MAT_s116
                                 + 1.37397 w_k1_MSFT_s116
                                 + 1.33821 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.850335 w_k1_ADM_s116
                                 + 0.729489 w_k1_AXP_s116
                                 + 0.749226 w_k1_DIS_s116
                                 + 0.831542 w_k1_FDX_s116
                                 + 0.828141 w_k1_IBM_s116
                                 + 0.850133 w_k1_INTC_s116
                                 + 0.776565 w_k1_KO_s116
                                 + 0.806672 w_k1_MAT_s116
                                 + 0.834819 w_k1_MSFT_s116
                                 + 0.959835 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.25225 w_k1_ADM_s116
                                 + 1.43256 w_k1_AXP_s116
                                 + 1.39631 w_k1_DIS_s116
                                 + 1.21143 w_k1_FDX_s116
                                 + 1.55955 w_k1_IBM_s116
                                 + 1.26637 w_k1_INTC_s116
                                 + 1.27802 w_k1_KO_s116 + 1.42105 w_k1_MAT_s116
                                 + 1.10353 w_k1_MSFT_s116
                                 + 1.18587 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.893849 w_k1_ADM_s116
                                 + 0.976024 w_k1_AXP_s116
                                 + 1.01589 w_k1_DIS_s116
                                 + 1.01801 w_k1_FDX_s116
                                 + 0.815842 w_k1_IBM_s116
                                 + 0.972834 w_k1_INTC_s116
                                 + 0.909995 w_k1_KO_s116
                                 + 0.952751 w_k1_MAT_s116
                                 + 1.09677 w_k1_MSFT_s116
                                 + 1.00493 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.685789 w_k1_ADM_s116
                                 + 0.570893 w_k1_AXP_s116
                                 + 0.59717 w_k1_DIS_s116
                                 + 0.763026 w_k1_FDX_s116
                                 + 0.621863 w_k1_IBM_s116
                                 + 0.996689 w_k1_INTC_s116
                                 + 0.76592 w_k1_KO_s116
                                 + 0.493885 w_k1_MAT_s116
                                 + 0.818041 w_k1_MSFT_s116
                                 + 0.817878 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.07874 w_k1_ADM_s116
                                 + 1.08173 w_k1_AXP_s116
                                 + 1.07712 w_k1_DIS_s116
                                 + 1.04431 w_k1_FDX_s116
                                 + 1.44536 w_k1_IBM_s116
                                 + 1.02843 w_k1_INTC_s116
                                 + 1.15335 w_k1_KO_s116 + 1.20732 w_k1_MAT_s116
                                 + 1.07982 w_k1_MSFT_s116
                                 + 1.12484 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.916548 w_k1_ADM_s116
                                 + 1.21745 w_k1_AXP_s116
                                 + 1.29852 w_k1_DIS_s116
                                 + 1.17405 w_k1_FDX_s116
                                 + 0.938717 w_k1_IBM_s116
                                 + 1.01873 w_k1_INTC_s116
                                 + 1.05837 w_k1_KO_s116 + 1.14949 w_k1_MAT_s116
                                 + 1.10065 w_k1_MSFT_s116
                                 + 1.06061 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.820434 w_k1_ADM_s116
                                 + 0.832795 w_k1_AXP_s116
                                 + 0.792247 w_k1_DIS_s116
                                 + 0.827782 w_k1_FDX_s116
                                 + 1.21975 w_k1_IBM_s116
                                 + 1.08461 w_k1_INTC_s116
                                 + 0.979027 w_k1_KO_s116
                                 + 1.25717 w_k1_MAT_s116
                                 + 0.947938 w_k1_MSFT_s116
                                 + 1.04475 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.27529 w_k1_ADM_s117
                                 + 1.26584 w_k1_AXP_s117
                                 + 1.52997 w_k1_DIS_s117
                                 + 1.26825 w_k1_FDX_s117
                                 + 1.24492 w_k1_IBM_s117
                                 + 1.22295 w_k1_INTC_s117
                                 + 1.29082 w_k1_KO_s117 + 1.30163 w_k1_MAT_s117
                                 + 1.16538 w_k1_MSFT_s117
                                 + 1.22227 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.985102 w_k1_ADM_s117
                                 + 1.10244 w_k1_AXP_s117
                                 + 0.946248 w_k1_DIS_s117
                                 + 1.09069 w_k1_FDX_s117
                                 + 0.978588 w_k1_IBM_s117
                                 + 1.36273 w_k1_INTC_s117
                                 + 1.06131 w_k1_KO_s117 + 0.82414 w_k1_MAT_s117
                                 + 1.06509 w_k1_MSFT_s117
                                 + 0.981598 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.00747 w_k1_ADM_s117
                                 + 1.42634 w_k1_AXP_s117
                                 + 0.997091 w_k1_DIS_s117
                                 + 1.01668 w_k1_FDX_s117
                                 + 1.28819 w_k1_IBM_s117
                                 + 0.98749 w_k1_INTC_s117
                                 + 0.971289 w_k1_KO_s117
                                 + 1.03284 w_k1_MAT_s117
                                 + 0.829239 w_k1_MSFT_s117
                                 + 0.964102 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.997996 w_k1_ADM_s117
                                 + 1.27781 w_k1_AXP_s117
                                 + 1.24373 w_k1_DIS_s117
                                 + 1.13309 w_k1_FDX_s117
                                 + 0.740944 w_k1_IBM_s117
                                 + 1.16124 w_k1_INTC_s117
                                 + 1.00924 w_k1_KO_s117
                                 + 0.720981 w_k1_MAT_s117
                                 + 1.02318 w_k1_MSFT_s117
                                 + 1.01269 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.710808 w_k1_ADM_s117
                                 + 0.769307 w_k1_AXP_s117
                                 + 0.700943 w_k1_DIS_s117
                                 + 0.727025 w_k1_FDX_s117
                                 + 0.659878 w_k1_IBM_s117
                                 + 0.817495 w_k1_INTC_s117
                                 + 0.640553 w_k1_KO_s117
                                 + 0.450542 w_k1_MAT_s117
                                 + 0.707029 w_k1_MSFT_s117
                                 + 0.828414 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.732074 w_k1_ADM_s117
                                 + 0.747638 w_k1_AXP_s117
                                 + 0.557403 w_k1_DIS_s117
                                 + 0.676206 w_k1_FDX_s117
                                 + 1.03784 w_k1_IBM_s117
                                 + 0.893496 w_k1_INTC_s117
                                 + 0.789484 w_k1_KO_s117
                                 + 0.814993 w_k1_MAT_s117
                                 + 0.730253 w_k1_MSFT_s117
                                 + 0.740233 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.817355 w_k1_ADM_s117
                                 + 0.956641 w_k1_AXP_s117
                                 + 0.765941 w_k1_DIS_s117
                                 + 0.91154 w_k1_FDX_s117
                                 + 0.644358 w_k1_IBM_s117
                                 + 0.992552 w_k1_INTC_s117
                                 + 0.750731 w_k1_KO_s117
                                 + 0.448052 w_k1_MAT_s117
                                 + 0.760247 w_k1_MSFT_s117
                                 + 0.86423 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.06435 w_k1_ADM_s117
                                 + 1.03539 w_k1_AXP_s117
                                 + 0.812211 w_k1_DIS_s117
                                 + 0.881657 w_k1_FDX_s117
                                 + 0.917213 w_k1_IBM_s117
                                 + 0.959161 w_k1_INTC_s117
                                 + 0.819028 w_k1_KO_s117
                                 + 0.750574 w_k1_MAT_s117
                                 + 0.872596 w_k1_MSFT_s117
                                 + 0.860704 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0116623281396224 w_k1_ADM_s000
                                 + 0.0404355655122297 w_k1_ADM_s001
                                 + 0.0176615016598434 w_k1_ADM_s002
                                 + 0.0236440122853176 w_k1_ADM_s003
                                 + 0.0106327899819497 w_k1_ADM_s004
                                 + 0.0187194406735303 w_k1_ADM_s005
                                 + 0.0269524681385146 w_k1_ADM_s006
                                 + 0.0330811039028906 w_k1_ADM_s007
                                 + 0.0104606237590587 w_k1_ADM_s010
                                 + 0.0274465052830782 w_k1_ADM_s011
                                 + 0.0157323315437421 w_k1_ADM_s012
                                 + 0.023481277134298 w_k1_ADM_s013
                                 + 0.0264068140665339 w_k1_ADM_s014
                                 + 0.0114993434505058 w_k1_ADM_s015
                                 + 0.019686802691569 w_k1_ADM_s016
                                 + 0.0447465068200877 w_k1_ADM_s017
                                 + 0.0817529476760999 w_k1_ADM_s100
                                 + 0.0492416734500344 w_k1_ADM_s101
                                 + 0.090897667465203 w_k1_ADM_s102
                                 + 0.0268820391828121 w_k1_ADM_s103
                                 + 0.0744033250735524 w_k1_ADM_s104
                                 + 0.0341805235442639 w_k1_ADM_s105
                                 + 0.0372641323810264 w_k1_ADM_s106
                                 + 0.0264719057480789 w_k1_ADM_s107
                                 + 0.032615483399352 w_k1_ADM_s110
                                 + 0.0288230755695717 w_k1_ADM_s111
                                 + 0.0168889898837558 w_k1_ADM_s112
                                 + 0.0356912161738651 w_k1_ADM_s113
                                 + 0.0157925975750725 w_k1_ADM_s114
                                 + 0.032811475351205 w_k1_ADM_s115
                                 + 0.0262294163878705 w_k1_ADM_s116
                                 + 0.0259708853079155 w_k1_ADM_s117
                                 + 0.0130793556710845 w_k1_AXP_s000
                                 + 0.0420283416501889 w_k1_AXP_s001
                                 + 0.0156983925753541 w_k1_AXP_s002
                                 + 0.0207593089172027 w_k1_AXP_s003
                                 + 0.0080422152267649 w_k1_AXP_s004
                                 + 0.0185999858162604 w_k1_AXP_s005
                                 + 0.0330081791346833 w_k1_AXP_s006
                                 + 0.0301903680475175 w_k1_AXP_s007
                                 + 0.0109591757952164 w_k1_AXP_s010
                                 + 0.0317277784383142 w_k1_AXP_s011
                                 + 0.0150018001640529 w_k1_AXP_s012
                                 + 0.0227951870237263 w_k1_AXP_s013
                                 + 0.0242175221445896 w_k1_AXP_s014
                                 + 0.0154521249890187 w_k1_AXP_s015
                                 + 0.0224648206979526 w_k1_AXP_s016
                                 + 0.0419244355902862 w_k1_AXP_s017
                                 + 0.0798774283944746 w_k1_AXP_s100
                                 + 0.0449768818763624 w_k1_AXP_s101
                                 + 0.0930203154566688 w_k1_AXP_s102
                                 + 0.0244885511505065 w_k1_AXP_s103
                                 + 0.081769507930606 w_k1_AXP_s104
                                 + 0.030739805172403 w_k1_AXP_s105
                                 + 0.0409649051897282 w_k1_AXP_s106
                                 + 0.028026919085999 w_k1_AXP_s107
                                 + 0.0270417040800651 w_k1_AXP_s110
                                 + 0.0307522008502364 w_k1_AXP_s111
                                 + 0.0217944190816351 w_k1_AXP_s112
                                 + 0.0345770452169832 w_k1_AXP_s113
                                 + 0.0139971133404516 w_k1_AXP_s114
                                 + 0.0323311798624364 w_k1_AXP_s115
                                 + 0.0283014209514307 w_k1_AXP_s116
                                 + 0.0293120876420626 w_k1_AXP_s117
                                 + 0.0114832748200753 w_k1_DIS_s000
                                 + 0.044673112620501 w_k1_DIS_s001
                                 + 0.0165007712161663 w_k1_DIS_s002
                                 + 0.0216324624996427 w_k1_DIS_s003
                                 + 0.00964429714534665 w_k1_DIS_s004
                                 + 0.0182405483682261 w_k1_DIS_s005
                                 + 0.0321012282792623 w_k1_DIS_s006
                                 + 0.0300598494496311 w_k1_DIS_s007
                                 + 0.0109894332917989 w_k1_DIS_s010
                                 + 0.0315833749876898 w_k1_DIS_s011
                                 + 0.0147350933669563 w_k1_DIS_s012
                                 + 0.0246212614977059 w_k1_DIS_s013
                                 + 0.0221390903262857 w_k1_DIS_s014
                                 + 0.0125036203624324 w_k1_DIS_s015
                                 + 0.0248456740290135 w_k1_DIS_s016
                                 + 0.0394324470935026 w_k1_DIS_s017
                                 + 0.0887719086421074 w_k1_DIS_s100
                                 + 0.0485322789011441 w_k1_DIS_s101
                                 + 0.0931574215435065 w_k1_DIS_s102
                                 + 0.024190958449014 w_k1_DIS_s103
                                 + 0.0793984604909783 w_k1_DIS_s104
                                 + 0.0350808761751121 w_k1_DIS_s105
                                 + 0.0461221193793528 w_k1_DIS_s106
                                 + 0.024956716928524 w_k1_DIS_s107
                                 + 0.0302315886090792 w_k1_DIS_s110
                                 + 0.0313474766834611 w_k1_DIS_s111
                                 + 0.0176389252013661 w_k1_DIS_s112
                                 + 0.0366629783681101 w_k1_DIS_s113
                                 + 0.0146478722544191 w_k1_DIS_s114
                                 + 0.0374378336253614 w_k1_DIS_s115
                                 + 0.0287482395393748 w_k1_DIS_s116
                                 + 0.0265833408437408 w_k1_DIS_s117
                                 + 0.0126446467939947 w_k1_FDX_s000
                                 + 0.0429835483914381 w_k1_FDX_s001
                                 + 0.0159339950565966 w_k1_FDX_s002
                                 + 0.021715482898111 w_k1_FDX_s003
                                 + 0.00954263898176967 w_k1_FDX_s004
                                 + 0.0179644054430678 w_k1_FDX_s005
                                 + 0.0305880201779261 w_k1_FDX_s006
                                 + 0.0307436491270798 w_k1_FDX_s007
                                 + 0.00995853742996222 w_k1_FDX_s010
                                 + 0.0324248394788225 w_k1_FDX_s011
                                 + 0.0147236559418948 w_k1_FDX_s012
                                 + 0.0225790010984751 w_k1_FDX_s013
                                 + 0.0225008884216214 w_k1_FDX_s014
                                 + 0.0122215039465717 w_k1_FDX_s015
                                 + 0.0228526498427111 w_k1_FDX_s016
                                 + 0.0405287559495744 w_k1_FDX_s017
                                 + 0.085653769838731 w_k1_FDX_s100
                                 + 0.0471657665993418 w_k1_FDX_s101
                                 + 0.0923571877681931 w_k1_FDX_s102
                                 + 0.0259668656754232 w_k1_FDX_s103
                                 + 0.0819980812931417 w_k1_FDX_s104
                                 + 0.0331393196486902 w_k1_FDX_s105
                                 + 0.0436340598253616 w_k1_FDX_s106
                                 + 0.024723510114551 w_k1_FDX_s107
                                 + 0.0297870657535638 w_k1_FDX_s110
                                 + 0.0321913867077234 w_k1_FDX_s111
                                 + 0.0182835181065174 w_k1_FDX_s112
                                 + 0.0380306982677019 w_k1_FDX_s113
                                 + 0.0152735760179395 w_k1_FDX_s114
                                 + 0.0365942866927099 w_k1_FDX_s115
                                 + 0.028248240711086 w_k1_FDX_s116
                                 + 0.0262139960824472 w_k1_FDX_s117
                                 + 0.0121981024313213 w_k1_IBM_s000
                                 + 0.0438240210707742 w_k1_IBM_s001
                                 + 0.0155865268499862 w_k1_IBM_s002
                                 + 0.0225482261833327 w_k1_IBM_s003
                                 + 0.00977349210671173 w_k1_IBM_s004
                                 + 0.0188778771621278 w_k1_IBM_s005
                                 + 0.030755751198158 w_k1_IBM_s006
                                 + 0.0307765851630962 w_k1_IBM_s007
                                 + 0.0107376948875742 w_k1_IBM_s010
                                 + 0.0331790461634705 w_k1_IBM_s011
                                 + 0.0147306048936787 w_k1_IBM_s012
                                 + 0.0228371972101088 w_k1_IBM_s013
                                 + 0.0232510536057475 w_k1_IBM_s014
                                 + 0.0127018545150402 w_k1_IBM_s015
                                 + 0.0233488726405419 w_k1_IBM_s016
                                 + 0.0424504615226124 w_k1_IBM_s017
                                 + 0.0861050469213442 w_k1_IBM_s100
                                 + 0.0514711810646674 w_k1_IBM_s101
                                 + 0.0944939665814144 w_k1_IBM_s102
                                 + 0.0257628287621349 w_k1_IBM_s103
                                 + 0.0859587889580871 w_k1_IBM_s104
                                 + 0.033269835959369 w_k1_IBM_s105
                                 + 0.043644841650134 w_k1_IBM_s106
                                 + 0.0256329735591156 w_k1_IBM_s107
                                 + 0.0300208408464205 w_k1_IBM_s110
                                 + 0.0309993763272364 w_k1_IBM_s111
                                 + 0.0190463475492892 w_k1_IBM_s112
                                 + 0.0374187443158866 w_k1_IBM_s113
                                 + 0.0145754563377284 w_k1_IBM_s114
                                 + 0.0366602303044635 w_k1_IBM_s115
                                 + 0.0283390907810877 w_k1_IBM_s116
                                 + 0.0257868867065719 w_k1_IBM_s117
                                 + 0.0129838190860263 w_k1_INTC_s000
                                 + 0.0401169840318838 w_k1_INTC_s001
                                 + 0.0141922147464124 w_k1_INTC_s002
                                 + 0.0199947835993361 w_k1_INTC_s003
                                 + 0.00901719990735117 w_k1_INTC_s004
                                 + 0.0160202928890208 w_k1_INTC_s005
                                 + 0.0287496201578656 w_k1_INTC_s006
                                 + 0.0305827608501096 w_k1_INTC_s007
                                 + 0.00907705381372772 w_k1_INTC_s010
                                 + 0.0339531351143124 w_k1_INTC_s011
                                 + 0.013403521772892 w_k1_INTC_s012
                                 + 0.0196298954775228 w_k1_INTC_s013
                                 + 0.0193314061920193 w_k1_INTC_s014
                                 + 0.0128796459742705 w_k1_INTC_s015
                                 + 0.0241272780647293 w_k1_INTC_s016
                                 + 0.0413629396142823 w_k1_INTC_s017
                                 + 0.0728513034711989 w_k1_INTC_s100
                                 + 0.0538944968350535 w_k1_INTC_s101
                                 + 0.0970924035587061 w_k1_INTC_s102
                                 + 0.0258322084366212 w_k1_INTC_s103
                                 + 0.0750966950056794 w_k1_INTC_s104
                                 + 0.0300028819542472 w_k1_INTC_s105
                                 + 0.0456693151564504 w_k1_INTC_s106
                                 + 0.0280699846839608 w_k1_INTC_s107
                                 + 0.0291324211854287 w_k1_INTC_s110
                                 + 0.0298346665583758 w_k1_INTC_s111
                                 + 0.0171395894123917 w_k1_INTC_s112
                                 + 0.0408350808865673 w_k1_INTC_s113
                                 + 0.0143376935421837 w_k1_INTC_s114
                                 + 0.0357190181820092 w_k1_INTC_s115
                                 + 0.0286252306612359 w_k1_INTC_s116
                                 + 0.0278582685552154 w_k1_INTC_s117
                                 + 0.0111177414296416 w_k1_KO_s000
                                 + 0.0419059237676473 w_k1_KO_s001
                                 + 0.016278320136275 w_k1_KO_s002
                                 + 0.0207257017133315 w_k1_KO_s003
                                 + 0.00924538476765219 w_k1_KO_s004
                                 + 0.0191126339507094 w_k1_KO_s005
                                 + 0.0300427732777324 w_k1_KO_s006
                                 + 0.0304971699562301 w_k1_KO_s007
                                 + 0.00978066415508354 w_k1_KO_s010
                                 + 0.0336795269403347 w_k1_KO_s011
                                 + 0.0157468958737518 w_k1_KO_s012
                                 + 0.0235271620629622 w_k1_KO_s013
                                 + 0.0223394854135827 w_k1_KO_s014
                                 + 0.0115269151945739 w_k1_KO_s015
                                 + 0.0224323312142404 w_k1_KO_s016
                                 + 0.0412951139766581 w_k1_KO_s017
                                 + 0.0788188628940518 w_k1_KO_s100
                                 + 0.0502475879745727 w_k1_KO_s101
                                 + 0.0860217632268463 w_k1_KO_s102
                                 + 0.025720352985624 w_k1_KO_s103
                                 + 0.0744557869639857 w_k1_KO_s104
                                 + 0.032527152164651 w_k1_KO_s105
                                 + 0.0469404464245229 w_k1_KO_s106
                                 + 0.0225538219889131 w_k1_KO_s107
                                 + 0.0319938741394091 w_k1_KO_s110
                                 + 0.0289842301216525 w_k1_KO_s111
                                 + 0.0162401268222307 w_k1_KO_s112
                                 + 0.0343679940236571 w_k1_KO_s113
                                 + 0.0136270605254448 w_k1_KO_s114
                                 + 0.0366506183986713 w_k1_KO_s115
                                 + 0.0277815512816958 w_k1_KO_s116
                                 + 0.0251278846847098 w_k1_KO_s117
                                 + 0.0121024351005239 w_k1_MAT_s000
                                 + 0.0397195772136133 w_k1_MAT_s001
                                 + 0.0149714496209144 w_k1_MAT_s002
                                 + 0.0203721829961885 w_k1_MAT_s003
                                 + 0.0106169989076592 w_k1_MAT_s004
                                 + 0.017730624167649 w_k1_MAT_s005
                                 + 0.0347271845326501 w_k1_MAT_s006
                                 + 0.0283516911578969 w_k1_MAT_s007
                                 + 0.00955448633281985 w_k1_MAT_s010
                                 + 0.0344080327411605 w_k1_MAT_s011
                                 + 0.0175520472508272 w_k1_MAT_s012
                                 + 0.0248753133643956 w_k1_MAT_s013
                                 + 0.0229865576741456 w_k1_MAT_s014
                                 + 0.0125462225641478 w_k1_MAT_s015
                                 + 0.021400750308077 w_k1_MAT_s016
                                 + 0.040757646088656 w_k1_MAT_s017
                                 + 0.102660852837104 w_k1_MAT_s100
                                 + 0.0433455522642677 w_k1_MAT_s101
                                 + 0.0858469242726628 w_k1_MAT_s102
                                 + 0.0304798898141359 w_k1_MAT_s103
                                 + 0.0718902579821769 w_k1_MAT_s104
                                 + 0.0336499046223691 w_k1_MAT_s105
                                 + 0.047680039438594 w_k1_MAT_s106
                                 + 0.021678417482275 w_k1_MAT_s107
                                 + 0.0310919560238281 w_k1_MAT_s110
                                 + 0.0351849436385705 w_k1_MAT_s111
                                 + 0.0211497347640938 w_k1_MAT_s112
                                 + 0.0331820130627772 w_k1_MAT_s113
                                 + 0.0135911081989799 w_k1_MAT_s114
                                 + 0.0276740415901494 w_k1_MAT_s115
                                 + 0.0305514864165187 w_k1_MAT_s116
                                 + 0.0225479401781839 w_k1_MAT_s117
                                 + 0.0129752453931307 w_k1_MSFT_s000
                                 + 0.0414428661812799 w_k1_MSFT_s001
                                 + 0.0155841931245506 w_k1_MSFT_s002
                                 + 0.0218176477585487 w_k1_MSFT_s003
                                 + 0.00903468425188325 w_k1_MSFT_s004
                                 + 0.0171121725711495 w_k1_MSFT_s005
                                 + 0.0324186236216332 w_k1_MSFT_s006
                                 + 0.0276518919990074 w_k1_MSFT_s007
                                 + 0.00992150290216596 w_k1_MSFT_s010
                                 + 0.0308771782982221 w_k1_MSFT_s011
                                 + 0.0142021186451013 w_k1_MSFT_s012
                                 + 0.0232364039852541 w_k1_MSFT_s013
                                 + 0.0220998245812763 w_k1_MSFT_s014
                                 + 0.0128884358589291 w_k1_MSFT_s015
                                 + 0.0218614720097093 w_k1_MSFT_s016
                                 + 0.0419841002448392 w_k1_MSFT_s017
                                 + 0.0874221741832304 w_k1_MSFT_s100
                                 + 0.0467834942821371 w_k1_MSFT_s101
                                 + 0.0959761575433748 w_k1_MSFT_s102
                                 + 0.0251717234477106 w_k1_MSFT_s103
                                 + 0.0793639175499283 w_k1_MSFT_s104
                                 + 0.0309219585897674 w_k1_MSFT_s105
                                 + 0.0466707249892702 w_k1_MSFT_s106
                                 + 0.0248073623145164 w_k1_MSFT_s107
                                 + 0.0271733537941879 w_k1_MSFT_s110
                                 + 0.0294260749854189 w_k1_MSFT_s111
                                 + 0.0181345463161596 w_k1_MSFT_s112
                                 + 0.0372593758908979 w_k1_MSFT_s113
                                 + 0.0138562265594313 w_k1_MSFT_s114
                                 + 0.0343198149363137 w_k1_MSFT_s115
                                 + 0.0285274791859465 w_k1_MSFT_s116
                                 + 0.0241219816286438 w_k1_MSFT_s117
                                 + 0.0123114015694723 w_k1_PG_s000
                                 + 0.0435267360829647 w_k1_PG_s001
                                 + 0.0148992247383363 w_k1_PG_s002
                                 + 0.0225226942104377 w_k1_PG_s003
                                 + 0.0105985050685264 w_k1_PG_s004
                                 + 0.0169731703684024 w_k1_PG_s005
                                 + 0.0304884662611679 w_k1_PG_s006
                                 + 0.0299532554959995 w_k1_PG_s007
                                 + 0.00996088316920769 w_k1_PG_s010
                                 + 0.0308044832037366 w_k1_PG_s011
                                 + 0.0148214402128291 w_k1_PG_s012
                                 + 0.0216722449211364 w_k1_PG_s013
                                 + 0.0235106442884469 w_k1_PG_s014
                                 + 0.0120535749283801 w_k1_PG_s015
                                 + 0.0226644924428828 w_k1_PG_s016
                                 + 0.0430922816124907 w_k1_PG_s017
                                 + 0.0824109532804461 w_k1_PG_s100
                                 + 0.046392357739974 w_k1_PG_s101
                                 + 0.0874639212049268 w_k1_PG_s102
                                 + 0.0249580835814819 w_k1_PG_s103
                                 + 0.0840628242387962 w_k1_PG_s104
                                 + 0.0321712815213123 w_k1_PG_s105
                                 + 0.0419227197276277 w_k1_PG_s106
                                 + 0.025164746331473 w_k1_PG_s107
                                 + 0.0313605269432764 w_k1_PG_s110
                                 + 0.0309386582397309 w_k1_PG_s111
                                 + 0.0187265416672244 w_k1_PG_s112
                                 + 0.0354793206607089 w_k1_PG_s113
                                 + 0.0141886355676345 w_k1_PG_s114
                                 + 0.0344089445411702 w_k1_PG_s115
                                 + 0.0290744275375958 w_k1_PG_s116
                                 + 0.0251804113902854 w_k1_PG_s117 >= 1.14
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
