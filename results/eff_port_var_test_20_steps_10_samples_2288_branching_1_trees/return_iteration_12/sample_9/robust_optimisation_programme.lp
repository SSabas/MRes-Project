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
 w_k1_ADM_s0_return_constraint:  1.0994 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.974274 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.13408 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.780139 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.09816 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.801691 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.13542 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.931106 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.916903 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.663024 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.27057 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.05608 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.746119 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.18066 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.03212 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.784048 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.34107 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.07908 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.843774 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.23698 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.10413 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.0627 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.894057 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.19844 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.710786 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.13059 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.914784 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.649076 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.792983 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.17633 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.965963 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.856531 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.00558 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.969359 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.872312 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.966005 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.18584 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.891871 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.13238 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.91241 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.14465 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.750819 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.15497 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.781533 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.09213 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.95748 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.05551 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.681406 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.66102 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.10168 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.839054 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.51998 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.774727 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.951546 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.41575 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.925873 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.841247 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.33963 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.993826 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.864505 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.931806 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.11137 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.722662 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.07763 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.758443 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.577019 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.853476 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.60765 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.06469 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.642509 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.76704 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.860857 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.730615 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.03913 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.37026 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.04406 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.20342 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.914352 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.15477 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.662233 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.25472 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.662483 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.05847 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.08005 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.17901 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.663984 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.68844 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.963797 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.769142 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.99578 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.15939 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.894803 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.31658 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.959318 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.697142 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.43848 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.925833 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.963472 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.05518 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.2578 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.604446 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.81526 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.841898 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.46392 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.748833 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.43087 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.10416 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.73447 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.936733 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.961019 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.698556 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.996011 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.72145 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.13071 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.08639 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.971578 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.09123 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.768786 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.14126 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.772413 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.18539 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.06976 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.00611 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.776746 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.33942 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.880868 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.811468 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.57811 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.04929 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.816104 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.12754 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.15733 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.779221 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.22811 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.02022 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.901993 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.946477 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.28399 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.775705 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.990517 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.92362 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.655086 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.821726 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.27974 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.12152 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.755791 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.06238 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.921957 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.755627 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.996407 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.3417 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.994394 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.23414 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.848459 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.12296 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.864549 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.05908 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.949462 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.703556 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.906734 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.34072 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.751769 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.67729 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.79093 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.15892 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.07237 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.905887 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.48262 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.74996 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.591724 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.851509 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.14463 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.789775 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.15028 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.35557 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.815867 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.736829 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.874215 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.687902 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.670699 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.16433 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.45128 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.764399 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.803004 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.561409 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.884551 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.634807 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.03085 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.13045 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.45592 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.04056 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.932828 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.08104 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.938707 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.02246 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.80724 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.19594 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.00911 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.11636 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.869355 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.40743 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.913999 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.884628 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.58219 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.889478 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.80317 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.986307 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.941037 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.636532 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.12568 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.992884 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.876362 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.08488 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.23369 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.91498 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.04572 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.13775 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.871678 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.880677 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.21648 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.25542 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.903053 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.07655 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.01827 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.804286 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.08636 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.39639 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.19133 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.15917 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.891247 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.15066 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.835924 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.09118 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.867735 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.777443 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.959839 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.21067 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.783352 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.55724 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.18604 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.899255 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.27931 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.967017 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.941301 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.41136 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.771518 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.774609 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.08737 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.958415 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.06823 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.16739 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.950854 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.832965 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.928296 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.810814 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.675447 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.949848 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.45316 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.910134 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.744568 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.741184 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.839234 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.695617 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.00256 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.28875 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.13267 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.23884 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.920056 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.30349 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.711347 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.01343 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.801834 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.558834 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.79116 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.09024 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.583257 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.83319 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.38376 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.16309 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.763646 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.830489 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.79612 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 2.21549 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.860741 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.0726 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.24647 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.859786 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.23088 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.27179 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.576057 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.704473 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.811312 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.535458 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.590426 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.10338 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 2.04126 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.804361 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.64083 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.529161 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.862754 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.691375 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.989105 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.29043 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.33375 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.11475 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.986796 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.07869 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.826734 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.04694 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.730453 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.04951 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.00887 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.11463 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.737901 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.30618 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.880104 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.753794 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.6024 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.0477 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.845684 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.0181 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.08105 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.739108 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.19273 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.942776 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.88228 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.02785 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.18387 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.838058 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.913981 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.05986 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.758824 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.868151 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.31715 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.16709 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.811908 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.12903 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.01905 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.803739 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.08951 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.493 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.05067 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.09675 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.980403 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.0437 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.852888 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.01281 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.811676 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.02815 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.983739 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.12765 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.795389 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.46108 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.05489 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.946487 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.26326 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.996787 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.928839 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.22438 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.899293 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.831344 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.05368 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.00436 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.02976 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.07504 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.06091 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.86342 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.03311 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.908782 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.773866 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.922233 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.22587 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.06504 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.839095 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.94301 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.05168 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.840368 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.07564 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.174 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.1026 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.500000272074817 z_k1_s0000
                                 - 0.872998761397546 z_k1_s0001
                                 - 0.662000934255954 z_k1_s0002
                                 - 0.431000204982888 z_k1_s0003
                                 - 0.740998495412961 z_k1_s0004
                                 - 0.511000206306851 z_k1_s0005
                                 - 1.13800173353367 z_k1_s0006
                                 - 0.420000270750854 z_k1_s0007
                                 - 0.794999941301294 z_k1_s0010
                                 - 0.604999198965915 z_k1_s0011
                                 - 0.51999933049961 z_k1_s0012
                                 - 1.05799811046321 z_k1_s0013
                                 - 0.36399976114949 z_k1_s0014
                                 - 1.43100142634075 z_k1_s0015
                                 - 1.66199614846013 z_k1_s0016
                                 - 1.22500125848174 z_k1_s0017
                                 - 0.544997940367568 z_k1_s0020
                                 - 0.282998939188995 z_k1_s0021
                                 - 0.206999298788818 z_k1_s0022
                                 - 0.250999123820364 z_k1_s0023
                                 - 0.358999841984004 z_k1_s0024
                                 - 0.211999013516217 z_k1_s0025
                                 - 0.956996273335795 z_k1_s0026
                                 - 0.341998539600153 z_k1_s0027
                                 - 1.15900046845 z_k1_s0030
                                 - 0.845999402781551 z_k1_s0031
                                 - 1.28899666257864 z_k1_s0032
                                 - 0.387999818633996 z_k1_s0033
                                 - 0.379999722872573 z_k1_s0034
                                 - 0.489999430993799 z_k1_s0035
                                 - 0.485999383113088 z_k1_s0036
                                 - 0.323999588742058 z_k1_s0037
                                 - 0.130999841797698 z_k1_s0040
                                 - 0.231999759035591 z_k1_s0041
                                 - 0.193999952433103 z_k1_s0042
                                 - 0.4280007826786 z_k1_s0043
                                 - 0.418999508931559 z_k1_s0044
                                 - 0.607999351638112 z_k1_s0045
                                 - 0.21399990213964 z_k1_s0046
                                 - 0.219999935971567 z_k1_s0047
                                 - 0.306000212995837 z_k1_s0050
                                 - 0.101999908598805 z_k1_s0051
                                 - 0.136999827878846 z_k1_s0052
                                 - 0.174000025585981 z_k1_s0053
                                 - 0.307999354624281 z_k1_s0054
                                 - 0.400999228299713 z_k1_s0055
                                 - 0.101999908598805 z_k1_s0056
                                 - 0.0939999316890759 z_k1_s0057
                                 - 0.671002465983793 z_k1_s0060
                                 - 0.227000680218145 z_k1_s0061
                                 - 0.395001642152051 z_k1_s0062
                                 - 0.231000729207135 z_k1_s0063
                                 - 0.302000959659565 z_k1_s0064
                                 - 0.594003440252156 z_k1_s0065
                                 - 0.586001881469287 z_k1_s0066
                                 - 0.646000790298023 z_k1_s0067
                                 - 0.317000096093232 z_k1_s0070
                                 - 0.143999912692344 z_k1_s0071
                                 - 0.133999934172025 z_k1_s0072
                                 - 0.244998918748462 z_k1_s0073
                                 - 0.421000316704044 z_k1_s0074
                                 - 0.414999663592617 z_k1_s0075
                                 - 0.297000139052594 z_k1_s0076
                                 - 0.246999136452271 z_k1_s0077
                                 - 0.110000119847219 z_k1_s0100
                                 - 0.143999578824603 z_k1_s0101
                                 - 0.219000534630805 z_k1_s0102
                                 - 0.0990001591425718 z_k1_s0103
                                 - 0.138999771504565 z_k1_s0104
                                 - 0.210000881454736 z_k1_s0105
                                 - 0.268999889833062 z_k1_s0106
                                 - 0.0920001724941431 z_k1_s0107
                                 - 0.203000073918023 z_k1_s0110
                                 - 0.212999713125472 z_k1_s0111
                                 - 0.121000073214218 z_k1_s0112
                                 - 0.145999664433294 z_k1_s0113
                                 - 0.137999953067335 z_k1_s0114
                                 - 0.100000091077894 z_k1_s0115
                                 - 0.126000386018396 z_k1_s0116
                                 - 0.18600068726536 z_k1_s0117
                                 - 0.14599958618369 z_k1_s0120
                                 - 0.0830000650130308 z_k1_s0121
                                 - 0.139999656491278 z_k1_s0122
                                 - 0.237999805135672 z_k1_s0123
                                 - 0.0880001361235405 z_k1_s0124
                                 - 0.131999750234729 z_k1_s0125
                                 - 0.131999750234729 z_k1_s0126
                                 - 0.337999930344203 z_k1_s0127
                                 - 0.216001082200917 z_k1_s0130
                                 - 0.226001663397804 z_k1_s0131
                                 - 0.158000747872695 z_k1_s0132
                                 - 0.260001615058071 z_k1_s0133
                                 - 0.168000822967295 z_k1_s0134
                                 - 0.156000631633318 z_k1_s0135
                                 - 0.17400134038619 z_k1_s0136
                                 - 0.329001913899819 z_k1_s0137
                                 - 0.30200062533857 z_k1_s0140
                                 - 0.417999567783338 z_k1_s0141
                                 - 0.423002079963909 z_k1_s0142
                                 - 0.342000712734418 z_k1_s0143
                                 - 0.491001745535675 z_k1_s0144
                                 - 0.354001911956029 z_k1_s0145
                                 - 0.860000844008216 z_k1_s0146
                                 - 0.260000568073013 z_k1_s0147
                                 - 0.273000453984317 z_k1_s0150
                                 - 0.46400108032576 z_k1_s0151
                                 - 0.280999849850456 z_k1_s0152
                                 - 0.182000463189943 z_k1_s0153
                                 - 0.393001024000321 z_k1_s0154
                                 - 0.181000418306377 z_k1_s0155
                                 - 0.434002141824749 z_k1_s0156
                                 - 0.200000548692344 z_k1_s0157
                                 - 0.158001113285348 z_k1_s0160
                                 - 0.369001517414965 z_k1_s0161
                                 - 0.118000433831667 z_k1_s0162
                                 - 0.22700142231427 z_k1_s0163
                                 - 0.126000476422005 z_k1_s0164
                                 - 0.126000476422005 z_k1_s0165
                                 - 0.175000445724087 z_k1_s0166
                                 - 0.256000740888186 z_k1_s0167
                                 - 0.274999648752695 z_k1_s0170
                                 - 0.124999608988218 z_k1_s0171
                                 - 0.123999615709103 z_k1_s0172
                                 - 0.104999743405908 z_k1_s0173
                                 - 0.102999756847677 z_k1_s0174
                                 - 0.345999171569897 z_k1_s0175
                                 - 0.147999604107468 z_k1_s0176
                                 - 0.270999675636233 z_k1_s0177
                                 - 1.11300270506639 z_k1_s1000
                                 - 1.02500238299458 z_k1_s1001
                                 - 0.486002082805907 z_k1_s1002
                                 - 0.281000268206047 z_k1_s1003
                                 - 0.238000039875452 z_k1_s1004
                                 - 0.421000212956258 z_k1_s1005
                                 - 0.251000057045131 z_k1_s1006
                                 - 0.64500119495735 z_k1_s1007
                                 - 1.32599555493998 z_k1_s1010
                                 - 0.443999087240982 z_k1_s1011
                                 - 0.848996974869233 z_k1_s1012
                                 - 0.38899880172739 z_k1_s1013
                                 - 0.630996213276302 z_k1_s1014
                                 - 0.309999039264515 z_k1_s1015
                                 - 1.00999989748044 z_k1_s1016
                                 - 0.694999341663676 z_k1_s1017
                                 - 0.190000122669376 z_k1_s1020
                                 - 0.394999716412339 z_k1_s1021
                                 - 0.307999090103401 z_k1_s1022
                                 - 0.422001460026354 z_k1_s1023
                                 - 0.360999184751209 z_k1_s1024
                                 - 0.126000149660784 z_k1_s1025
                                 - 0.244000115495385 z_k1_s1026
                                 - 0.44999910821626 z_k1_s1027
                                 - 0.317001777588429 z_k1_s1030
                                 - 0.752004115482446 z_k1_s1031
                                 - 0.675001648791021 z_k1_s1032
                                 - 0.229001035494481 z_k1_s1033
                                 - 0.741004309672159 z_k1_s1034
                                 - 0.319001637935225 z_k1_s1035
                                 - 0.437004111250519 z_k1_s1036
                                 - 0.356001923865502 z_k1_s1037
                                 - 0.77999858580422 z_k1_s1040
                                 - 0.50100022415195 z_k1_s1041
                                 - 0.799001948858195 z_k1_s1042
                                 - 0.511002471022791 z_k1_s1043
                                 - 0.380000217033447 z_k1_s1044
                                 - 0.374000227911877 z_k1_s1045
                                 - 0.342000285930165 z_k1_s1046
                                 - 0.843001869083049 z_k1_s1047
                                 - 0.108999967455976 z_k1_s1050
                                 - 0.127000110248795 z_k1_s1051
                                 - 0.370999680450031 z_k1_s1052
                                 - 0.434000180224896 z_k1_s1053
                                 - 0.130000134047598 z_k1_s1054
                                 - 0.456001170416608 z_k1_s1055
                                 - 0.423000908629774 z_k1_s1056
                                 - 0.396999071039169 z_k1_s1057
                                 - 0.326999691937977 z_k1_s1060
                                 - 0.31499983952722 z_k1_s1061
                                 - 0.970998703638984 z_k1_s1062
                                 - 1.38999825439139 z_k1_s1063
                                 - 0.658996598967533 z_k1_s1064
                                 - 0.845997022448055 z_k1_s1065
                                 - 0.929000705699283 z_k1_s1066
                                 - 0.504999186261868 z_k1_s1067
                                 - 2.13600162726593 z_k1_s1070
                                 - 0.483001102747995 z_k1_s1071
                                 - 0.528001322939683 z_k1_s1072
                                 - 0.614001586170116 z_k1_s1073
                                 - 1.73800094043557 z_k1_s1074
                                 - 0.789004806176357 z_k1_s1075
                                 - 0.493001151679481 z_k1_s1076
                                 - 1.0980041120344 z_k1_s1077
                                 - 0.35199924920009 z_k1_s1100
                                 - 0.146999316784453 z_k1_s1101
                                 - 0.115999490469247 z_k1_s1102
                                 - 0.344999084858751 z_k1_s1103
                                 - 0.169999328308083 z_k1_s1104
                                 - 0.123999522859963 z_k1_s1105
                                 - 0.568997815807389 z_k1_s1106
                                 - 0.352998029253765 z_k1_s1107
                                 - 0.283997856871556 z_k1_s1110
                                 - 0.190999063444474 z_k1_s1111
                                 - 0.200999073838162 z_k1_s1112
                                 - 0.278999323068178 z_k1_s1113
                                 - 0.448997818168335 z_k1_s1114
                                 - 0.265998510799931 z_k1_s1115
                                 - 0.192999107563025 z_k1_s1116
                                 - 0.238998440697161 z_k1_s1117
                                 - 0.273999638214976 z_k1_s1120
                                 - 0.149999673840466 z_k1_s1121
                                 - 0.12399996437451 z_k1_s1122
                                 - 0.0669999713152708 z_k1_s1123
                                 - 0.158000124444741 z_k1_s1124
                                 - 0.0869999278273329 z_k1_s1125
                                 - 0.155000247967795 z_k1_s1126
                                 - 0.155000247967795 z_k1_s1127
                                 - 0.122999992625729 z_k1_s1130
                                 - 0.108999808637812 z_k1_s1131
                                 - 0.061999908091198 z_k1_s1132
                                 - 0.112999795263066 z_k1_s1133
                                 - 0.145999165622471 z_k1_s1134
                                 - 0.197000091392217 z_k1_s1135
                                 - 0.208999358869393 z_k1_s1136
                                 - 0.194999059481712 z_k1_s1137
                                 - 0.187000177082456 z_k1_s1140
                                 - 0.168000174594956 z_k1_s1141
                                 - 0.14300026942207 z_k1_s1142
                                 - 0.280999895088614 z_k1_s1143
                                 - 0.335999520788726 z_k1_s1144
                                 - 0.258999630607976 z_k1_s1145
                                 - 0.135000192074592 z_k1_s1146
                                 - 0.561999790177228 z_k1_s1147
                                 - 0.124999843646626 z_k1_s1150
                                 - 0.286000310685096 z_k1_s1151
                                 - 0.219999560066254 z_k1_s1152
                                 - 0.119999920508678 z_k1_s1153
                                 - 0.254999022031889 z_k1_s1154
                                 - 0.142999861142893 z_k1_s1155
                                 - 0.231999963996639 z_k1_s1156
                                 - 0.0899999403815083 z_k1_s1157
                                 - 0.194999349198921 z_k1_s1160
                                 - 0.179999323876182 z_k1_s1161
                                 - 0.258999404976118 z_k1_s1162
                                 - 0.311999585916154 z_k1_s1163
                                 - 0.319998318892608 z_k1_s1164
                                 - 0.123999569070141 z_k1_s1165
                                 - 0.149999469230041 z_k1_s1166
                                 - 0.419999141041996 z_k1_s1167
                                 - 0.34799986726211 z_k1_s1170
                                 - 0.45399943045493 z_k1_s1171
                                 - 0.235999971392643 z_k1_s1172
                                 - 0.576000852184646 z_k1_s1173
                                 - 0.620000377615878 z_k1_s1174
                                 - 0.787999007171116 z_k1_s1175
                                 - 1.12200004098606 z_k1_s1176
                                 - 0.712997277210956 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.02929 w_k1_ADM_s000
                                 + 1.06857 w_k1_AXP_s000
                                 + 1.10302 w_k1_DIS_s000
                                 + 1.13789 w_k1_FDX_s000
                                 + 0.843161 w_k1_IBM_s000
                                 + 1.09729 w_k1_INTC_s000
                                 + 1.16482 w_k1_KO_s000 + 1.43274 w_k1_MAT_s000
                                 + 1.18745 w_k1_MSFT_s000
                                 + 0.96416 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.82278 w_k1_ADM_s000
                                 + 0.773628 w_k1_AXP_s000
                                 + 0.89789 w_k1_DIS_s000
                                 + 0.915582 w_k1_FDX_s000
                                 + 0.777966 w_k1_IBM_s000
                                 + 0.927095 w_k1_INTC_s000
                                 + 0.907622 w_k1_KO_s000
                                 + 0.795986 w_k1_MAT_s000
                                 + 1.03221 w_k1_MSFT_s000
                                 + 0.97342 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.996871 w_k1_ADM_s000
                                 + 0.984828 w_k1_AXP_s000
                                 + 1.37303 w_k1_DIS_s000
                                 + 1.25586 w_k1_FDX_s000
                                 + 0.934593 w_k1_IBM_s000
                                 + 1.10094 w_k1_INTC_s000
                                 + 1.21507 w_k1_KO_s000
                                 + 0.981864 w_k1_MAT_s000
                                 + 1.26886 w_k1_MSFT_s000
                                 + 1.07665 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.873392 w_k1_ADM_s000
                                 + 1.36099 w_k1_AXP_s000
                                 + 1.26554 w_k1_DIS_s000
                                 + 1.12563 w_k1_FDX_s000
                                 + 1.74098 w_k1_IBM_s000
                                 + 1.12007 w_k1_INTC_s000
                                 + 1.39663 w_k1_KO_s000 + 1.95395 w_k1_MAT_s000
                                 + 1.18357 w_k1_MSFT_s000
                                 + 1.12714 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.10547 w_k1_ADM_s000
                                 + 1.12524 w_k1_AXP_s000
                                 + 1.61049 w_k1_DIS_s000
                                 + 1.20402 w_k1_FDX_s000 + 1.3777 w_k1_IBM_s000
                                 + 0.875045 w_k1_INTC_s000
                                 + 1.32801 w_k1_KO_s000 + 1.29787 w_k1_MAT_s000
                                 + 1.18627 w_k1_MSFT_s000
                                 + 1.14326 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.981199 w_k1_ADM_s000
                                 + 1.47128 w_k1_AXP_s000
                                 + 1.03996 w_k1_DIS_s000
                                 + 1.12159 w_k1_FDX_s000
                                 + 0.982824 w_k1_IBM_s000
                                 + 0.868264 w_k1_INTC_s000
                                 + 1.21841 w_k1_KO_s000 + 1.9195 w_k1_MAT_s000
                                 + 1.07306 w_k1_MSFT_s000 + 1.1161 w_k1_PG_s000
                                 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.771462 w_k1_ADM_s000
                                 + 0.673716 w_k1_AXP_s000
                                 + 0.701425 w_k1_DIS_s000
                                 + 0.80178 w_k1_FDX_s000
                                 + 0.725557 w_k1_IBM_s000
                                 + 0.773729 w_k1_INTC_s000
                                 + 0.820135 w_k1_KO_s000
                                 + 0.893703 w_k1_MAT_s000
                                 + 0.84152 w_k1_MSFT_s000
                                 + 0.889022 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.779094 w_k1_ADM_s000
                                 + 0.685938 w_k1_AXP_s000
                                 + 0.575069 w_k1_DIS_s000
                                 + 0.713275 w_k1_FDX_s000
                                 + 0.903238 w_k1_IBM_s000
                                 + 0.647786 w_k1_INTC_s000
                                 + 0.829239 w_k1_KO_s000
                                 + 0.994676 w_k1_MAT_s000
                                 + 0.741067 w_k1_MSFT_s000
                                 + 0.858569 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.860238 w_k1_ADM_s001
                                 + 0.628916 w_k1_AXP_s001
                                 + 0.601241 w_k1_DIS_s001
                                 + 0.717309 w_k1_FDX_s001
                                 + 0.835985 w_k1_IBM_s001
                                 + 0.803701 w_k1_INTC_s001
                                 + 0.758705 w_k1_KO_s001
                                 + 0.830805 w_k1_MAT_s001
                                 + 0.810861 w_k1_MSFT_s001
                                 + 0.945182 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.37176 w_k1_ADM_s001
                                 + 1.8205 w_k1_AXP_s001 + 1.3183 w_k1_DIS_s001
                                 + 1.29614 w_k1_FDX_s001 + 1.2617 w_k1_IBM_s001
                                 + 1.25837 w_k1_INTC_s001
                                 + 1.26203 w_k1_KO_s001 + 1.73582 w_k1_MAT_s001
                                 + 1.20584 w_k1_MSFT_s001
                                 + 1.22629 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.50142 w_k1_ADM_s001
                                 + 1.75486 w_k1_AXP_s001 + 1.7195 w_k1_DIS_s001
                                 + 1.32363 w_k1_FDX_s001
                                 + 1.70106 w_k1_IBM_s001
                                 + 1.24969 w_k1_INTC_s001
                                 + 1.57055 w_k1_KO_s001 + 1.8263 w_k1_MAT_s001
                                 + 1.23497 w_k1_MSFT_s001
                                 + 1.32573 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.971859 w_k1_ADM_s001
                                 + 0.789387 w_k1_AXP_s001
                                 + 0.698522 w_k1_DIS_s001
                                 + 0.84099 w_k1_FDX_s001
                                 + 0.688955 w_k1_IBM_s001
                                 + 0.986308 w_k1_INTC_s001
                                 + 0.767399 w_k1_KO_s001
                                 + 0.559046 w_k1_MAT_s001
                                 + 0.875904 w_k1_MSFT_s001
                                 + 0.82718 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.40401 w_k1_ADM_s001
                                 + 1.57184 w_k1_AXP_s001
                                 + 1.09145 w_k1_DIS_s001
                                 + 1.21537 w_k1_FDX_s001
                                 + 1.08673 w_k1_IBM_s001
                                 + 1.17547 w_k1_INTC_s001
                                 + 0.945826 w_k1_KO_s001
                                 + 1.27368 w_k1_MAT_s001
                                 + 1.11242 w_k1_MSFT_s001
                                 + 1.13253 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.2671 w_k1_ADM_s001
                                 + 1.22627 w_k1_AXP_s001
                                 + 1.11447 w_k1_DIS_s001
                                 + 1.14667 w_k1_FDX_s001
                                 + 0.888775 w_k1_IBM_s001
                                 + 1.26596 w_k1_INTC_s001
                                 + 0.970454 w_k1_KO_s001
                                 + 0.745376 w_k1_MAT_s001
                                 + 1.01059 w_k1_MSFT_s001
                                 + 1.13014 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.05113 w_k1_ADM_s001
                                 + 1.00527 w_k1_AXP_s001
                                 + 0.852941 w_k1_DIS_s001
                                 + 0.858079 w_k1_FDX_s001
                                 + 1.10639 w_k1_IBM_s001
                                 + 0.836929 w_k1_INTC_s001
                                 + 1.04231 w_k1_KO_s001 + 1.03817 w_k1_MAT_s001
                                 + 1.00091 w_k1_MSFT_s001
                                 + 0.947321 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.17529 w_k1_ADM_s001
                                 + 1.00667 w_k1_AXP_s001
                                 + 1.22532 w_k1_DIS_s001
                                 + 1.02327 w_k1_FDX_s001
                                 + 1.05649 w_k1_IBM_s001
                                 + 1.01322 w_k1_INTC_s001
                                 + 1.14752 w_k1_KO_s001 + 1.19091 w_k1_MAT_s001
                                 + 1.11123 w_k1_MSFT_s001
                                 + 1.04488 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.24366 w_k1_ADM_s002
                                 + 0.98247 w_k1_AXP_s002
                                 + 1.08784 w_k1_DIS_s002
                                 + 1.10951 w_k1_FDX_s002
                                 + 0.765479 w_k1_IBM_s002
                                 + 1.10666 w_k1_INTC_s002
                                 + 0.990177 w_k1_KO_s002
                                 + 0.743658 w_k1_MAT_s002
                                 + 1.01683 w_k1_MSFT_s002
                                 + 0.957707 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.840167 w_k1_ADM_s002
                                 + 0.810544 w_k1_AXP_s002
                                 + 0.718096 w_k1_DIS_s002
                                 + 0.889674 w_k1_FDX_s002
                                 + 0.656997 w_k1_IBM_s002
                                 + 0.737541 w_k1_INTC_s002
                                 + 0.763154 w_k1_KO_s002
                                 + 0.832233 w_k1_MAT_s002
                                 + 0.804291 w_k1_MSFT_s002
                                 + 0.821537 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.11939 w_k1_ADM_s002
                                 + 0.893303 w_k1_AXP_s002
                                 + 1.33465 w_k1_DIS_s002 + 1.0493 w_k1_FDX_s002
                                 + 0.943552 w_k1_IBM_s002
                                 + 0.988564 w_k1_INTC_s002
                                 + 1.1113 w_k1_KO_s002 + 1.10874 w_k1_MAT_s002
                                 + 0.98883 w_k1_MSFT_s002
                                 + 1.09419 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.826052 w_k1_ADM_s002
                                 + 0.758151 w_k1_AXP_s002
                                 + 0.600207 w_k1_DIS_s002
                                 + 0.696565 w_k1_FDX_s002
                                 + 0.864879 w_k1_IBM_s002
                                 + 0.679532 w_k1_INTC_s002
                                 + 0.815918 w_k1_KO_s002
                                 + 1.02719 w_k1_MAT_s002
                                 + 0.666346 w_k1_MSFT_s002
                                 + 0.769194 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.05643 w_k1_ADM_s002
                                 + 0.909869 w_k1_AXP_s002
                                 + 0.875605 w_k1_DIS_s002
                                 + 0.917544 w_k1_FDX_s002
                                 + 1.08202 w_k1_IBM_s002
                                 + 0.79054 w_k1_INTC_s002
                                 + 0.986262 w_k1_KO_s002
                                 + 1.17419 w_k1_MAT_s002
                                 + 0.840788 w_k1_MSFT_s002
                                 + 0.900084 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.02892 w_k1_ADM_s002
                                 + 0.964798 w_k1_AXP_s002
                                 + 0.91887 w_k1_DIS_s002
                                 + 0.950722 w_k1_FDX_s002
                                 + 0.949205 w_k1_IBM_s002
                                 + 1.00855 w_k1_INTC_s002
                                 + 1.01108 w_k1_KO_s002
                                 + 0.970991 w_k1_MAT_s002
                                 + 0.937651 w_k1_MSFT_s002
                                 + 0.965413 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.1979 w_k1_ADM_s002
                                 + 1.3437 w_k1_AXP_s002 + 1.40412 w_k1_DIS_s002
                                 + 1.28542 w_k1_FDX_s002
                                 + 1.15294 w_k1_IBM_s002
                                 + 1.13049 w_k1_INTC_s002
                                 + 1.03157 w_k1_KO_s002 + 1.52468 w_k1_MAT_s002
                                 + 1.17336 w_k1_MSFT_s002 + 1.0304 w_k1_PG_s002
                                 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.977539 w_k1_ADM_s002
                                 + 1.04596 w_k1_AXP_s002 + 1.1732 w_k1_DIS_s002
                                 + 0.942672 w_k1_FDX_s002
                                 + 1.49789 w_k1_IBM_s002
                                 + 1.01734 w_k1_INTC_s002
                                 + 1.23542 w_k1_KO_s002 + 1.53194 w_k1_MAT_s002
                                 + 0.950627 w_k1_MSFT_s002
                                 + 1.0987 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.26093 w_k1_ADM_s003
                                 + 1.32742 w_k1_AXP_s003 + 0.9613 w_k1_DIS_s003
                                 + 1.03423 w_k1_FDX_s003 + 1.1649 w_k1_IBM_s003
                                 + 1.08648 w_k1_INTC_s003
                                 + 0.982347 w_k1_KO_s003
                                 + 1.26994 w_k1_MAT_s003
                                 + 1.05466 w_k1_MSFT_s003
                                 + 1.14141 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.51833 w_k1_ADM_s003
                                 + 1.07922 w_k1_AXP_s003
                                 + 1.20004 w_k1_DIS_s003
                                 + 1.20273 w_k1_FDX_s003
                                 + 0.91327 w_k1_IBM_s003
                                 + 0.947545 w_k1_INTC_s003
                                 + 0.96398 w_k1_KO_s003 + 1.08342 w_k1_MAT_s003
                                 + 1.17699 w_k1_MSFT_s003
                                 + 1.19872 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.07855 w_k1_ADM_s003
                                 + 0.844057 w_k1_AXP_s003
                                 + 0.854855 w_k1_DIS_s003
                                 + 0.939352 w_k1_FDX_s003
                                 + 0.753033 w_k1_IBM_s003
                                 + 0.830541 w_k1_INTC_s003
                                 + 0.792078 w_k1_KO_s003
                                 + 0.873541 w_k1_MAT_s003
                                 + 1.03453 w_k1_MSFT_s003
                                 + 0.882575 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.1888 w_k1_ADM_s003
                                 + 1.36054 w_k1_AXP_s003
                                 + 0.973426 w_k1_DIS_s003
                                 + 0.976369 w_k1_FDX_s003
                                 + 1.53934 w_k1_IBM_s003
                                 + 0.84381 w_k1_INTC_s003
                                 + 1.09019 w_k1_KO_s003 + 2.12391 w_k1_MAT_s003
                                 + 0.888469 w_k1_MSFT_s003
                                 + 1.16303 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.92873 w_k1_ADM_s003
                                 + 0.769509 w_k1_AXP_s003
                                 + 0.523485 w_k1_DIS_s003
                                 + 0.709096 w_k1_FDX_s003
                                 + 0.629487 w_k1_IBM_s003
                                 + 0.699563 w_k1_INTC_s003
                                 + 0.581122 w_k1_KO_s003
                                 + 0.833139 w_k1_MAT_s003
                                 + 0.75887 w_k1_MSFT_s003
                                 + 0.889133 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.0114 w_k1_ADM_s003
                                 + 0.948499 w_k1_AXP_s003
                                 + 1.04296 w_k1_DIS_s003
                                 + 0.914679 w_k1_FDX_s003
                                 + 1.48574 w_k1_IBM_s003
                                 + 1.0374 w_k1_INTC_s003 + 1.11782 w_k1_KO_s003
                                 + 1.23691 w_k1_MAT_s003
                                 + 0.996102 w_k1_MSFT_s003
                                 + 1.11086 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.826721 w_k1_ADM_s003
                                 + 0.792548 w_k1_AXP_s003
                                 + 0.712296 w_k1_DIS_s003
                                 + 0.696504 w_k1_FDX_s003
                                 + 1.04535 w_k1_IBM_s003
                                 + 0.690727 w_k1_INTC_s003
                                 + 0.731205 w_k1_KO_s003
                                 + 1.08045 w_k1_MAT_s003
                                 + 0.821714 w_k1_MSFT_s003
                                 + 0.894442 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.78603 w_k1_ADM_s003
                                 + 1.67832 w_k1_AXP_s003
                                 + 1.68773 w_k1_DIS_s003 + 1.2854 w_k1_FDX_s003
                                 + 1.53535 w_k1_IBM_s003
                                 + 1.16584 w_k1_INTC_s003 + 1.5153 w_k1_KO_s003
                                 + 1.56689 w_k1_MAT_s003
                                 + 1.3206 w_k1_MSFT_s003 + 1.13691 w_k1_PG_s003
                                 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.00309 w_k1_ADM_s004
                                 + 1.54237 w_k1_AXP_s004
                                 + 1.84809 w_k1_DIS_s004 + 1.4113 w_k1_FDX_s004
                                 + 1.68238 w_k1_IBM_s004
                                 + 1.64004 w_k1_INTC_s004
                                 + 1.58439 w_k1_KO_s004 + 1.56895 w_k1_MAT_s004
                                 + 1.35689 w_k1_MSFT_s004
                                 + 1.24557 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.872709 w_k1_ADM_s004
                                 + 0.762978 w_k1_AXP_s004
                                 + 0.776871 w_k1_DIS_s004
                                 + 0.83234 w_k1_FDX_s004
                                 + 1.27777 w_k1_IBM_s004
                                 + 0.815536 w_k1_INTC_s004
                                 + 0.972167 w_k1_KO_s004 + 0.9685 w_k1_MAT_s004
                                 + 0.899407 w_k1_MSFT_s004
                                 + 0.791641 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.28932 w_k1_ADM_s004
                                 + 1.16384 w_k1_AXP_s004
                                 + 1.73119 w_k1_DIS_s004
                                 + 1.44714 w_k1_FDX_s004
                                 + 1.01909 w_k1_IBM_s004
                                 + 1.28264 w_k1_INTC_s004
                                 + 1.18981 w_k1_KO_s004
                                 + 0.728473 w_k1_MAT_s004
                                 + 1.40501 w_k1_MSFT_s004
                                 + 1.17763 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.01885 w_k1_ADM_s004
                                 + 1.06282 w_k1_AXP_s004
                                 + 1.20121 w_k1_DIS_s004
                                 + 1.14462 w_k1_FDX_s004
                                 + 0.883396 w_k1_IBM_s004
                                 + 1.05488 w_k1_INTC_s004
                                 + 0.990875 w_k1_KO_s004
                                 + 0.886139 w_k1_MAT_s004
                                 + 1.2391 w_k1_MSFT_s004
                                 + 0.905008 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.819649 w_k1_ADM_s004
                                 + 0.864039 w_k1_AXP_s004
                                 + 0.813843 w_k1_DIS_s004
                                 + 0.90372 w_k1_FDX_s004
                                 + 0.999237 w_k1_IBM_s004
                                 + 0.926474 w_k1_INTC_s004
                                 + 0.863928 w_k1_KO_s004
                                 + 0.974717 w_k1_MAT_s004
                                 + 0.951843 w_k1_MSFT_s004
                                 + 0.813879 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.899666 w_k1_ADM_s004
                                 + 0.717607 w_k1_AXP_s004
                                 + 0.757494 w_k1_DIS_s004
                                 + 0.967171 w_k1_FDX_s004
                                 + 0.70207 w_k1_IBM_s004
                                 + 0.784027 w_k1_INTC_s004
                                 + 0.783796 w_k1_KO_s004
                                 + 0.549339 w_k1_MAT_s004
                                 + 0.916177 w_k1_MSFT_s004
                                 + 0.840957 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.14588 w_k1_ADM_s004
                                 + 1.54417 w_k1_AXP_s004
                                 + 1.14545 w_k1_DIS_s004
                                 + 1.20684 w_k1_FDX_s004
                                 + 1.30979 w_k1_IBM_s004
                                 + 1.06415 w_k1_INTC_s004
                                 + 1.22309 w_k1_KO_s004 + 1.49256 w_k1_MAT_s004
                                 + 1.02502 w_k1_MSFT_s004
                                 + 1.08041 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.791401 w_k1_ADM_s004
                                 + 0.915223 w_k1_AXP_s004
                                 + 1.19053 w_k1_DIS_s004
                                 + 0.849192 w_k1_FDX_s004
                                 + 1.69382 w_k1_IBM_s004
                                 + 1.0936 w_k1_INTC_s004 + 1.32507 w_k1_KO_s004
                                 + 1.17737 w_k1_MAT_s004
                                 + 1.00707 w_k1_MSFT_s004
                                 + 0.926857 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.722181 w_k1_ADM_s005
                                 + 0.667578 w_k1_AXP_s005
                                 + 0.635545 w_k1_DIS_s005
                                 + 0.68109 w_k1_FDX_s005
                                 + 0.773543 w_k1_IBM_s005
                                 + 0.831385 w_k1_INTC_s005
                                 + 0.699696 w_k1_KO_s005
                                 + 0.707918 w_k1_MAT_s005
                                 + 0.828026 w_k1_MSFT_s005
                                 + 0.73049 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.987047 w_k1_ADM_s005
                                 + 0.909301 w_k1_AXP_s005
                                 + 0.811164 w_k1_DIS_s005
                                 + 1.06149 w_k1_FDX_s005
                                 + 0.575534 w_k1_IBM_s005
                                 + 0.912308 w_k1_INTC_s005
                                 + 0.704293 w_k1_KO_s005
                                 + 0.785627 w_k1_MAT_s005
                                 + 0.921958 w_k1_MSFT_s005
                                 + 0.990542 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.909935 w_k1_ADM_s005
                                 + 1.61781 w_k1_AXP_s005
                                 + 1.21684 w_k1_DIS_s005
                                 + 1.24898 w_k1_FDX_s005
                                 + 1.15949 w_k1_IBM_s005
                                 + 0.977454 w_k1_INTC_s005
                                 + 1.03612 w_k1_KO_s005 + 1.55662 w_k1_MAT_s005
                                 + 0.860776 w_k1_MSFT_s005
                                 + 1.13892 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.856932 w_k1_ADM_s005
                                 + 1.08633 w_k1_AXP_s005
                                 + 0.873071 w_k1_DIS_s005
                                 + 1.02867 w_k1_FDX_s005
                                 + 0.785415 w_k1_IBM_s005
                                 + 1.07607 w_k1_INTC_s005
                                 + 0.802132 w_k1_KO_s005
                                 + 0.871951 w_k1_MAT_s005
                                 + 0.971247 w_k1_MSFT_s005
                                 + 0.961765 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.889452 w_k1_ADM_s005
                                 + 1.13156 w_k1_AXP_s005
                                 + 1.29499 w_k1_DIS_s005
                                 + 0.965947 w_k1_FDX_s005
                                 + 1.41043 w_k1_IBM_s005
                                 + 1.15687 w_k1_INTC_s005
                                 + 1.29523 w_k1_KO_s005 + 1.73393 w_k1_MAT_s005
                                 + 1.18575 w_k1_MSFT_s005
                                 + 1.05127 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.978678 w_k1_ADM_s005
                                 + 1.02876 w_k1_AXP_s005
                                 + 1.01521 w_k1_DIS_s005 + 1.0662 w_k1_FDX_s005
                                 + 0.983338 w_k1_IBM_s005
                                 + 0.977001 w_k1_INTC_s005
                                 + 0.939187 w_k1_KO_s005
                                 + 1.06403 w_k1_MAT_s005
                                 + 0.974498 w_k1_MSFT_s005
                                 + 0.987798 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.951589 w_k1_ADM_s005
                                 + 1.36968 w_k1_AXP_s005
                                 + 1.58468 w_k1_DIS_s005
                                 + 1.46089 w_k1_FDX_s005
                                 + 1.17052 w_k1_IBM_s005
                                 + 1.29771 w_k1_INTC_s005
                                 + 1.29183 w_k1_KO_s005 + 1.20318 w_k1_MAT_s005
                                 + 1.40078 w_k1_MSFT_s005 + 1.1253 w_k1_PG_s005
                                 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.04828 w_k1_ADM_s005
                                 + 1.18186 w_k1_AXP_s005
                                 + 1.45663 w_k1_DIS_s005 + 1.3904 w_k1_FDX_s005
                                 + 0.848536 w_k1_IBM_s005
                                 + 1.11298 w_k1_INTC_s005
                                 + 1.06704 w_k1_KO_s005 + 1.14342 w_k1_MAT_s005
                                 + 1.26242 w_k1_MSFT_s005
                                 + 1.00786 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.1356 w_k1_ADM_s006
                                 + 1.27373 w_k1_AXP_s006
                                 + 1.22751 w_k1_DIS_s006
                                 + 1.16565 w_k1_FDX_s006
                                 + 1.20086 w_k1_IBM_s006
                                 + 1.19461 w_k1_INTC_s006
                                 + 1.19065 w_k1_KO_s006
                                 + 0.915984 w_k1_MAT_s006
                                 + 1.16325 w_k1_MSFT_s006
                                 + 1.06686 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.06315 w_k1_ADM_s006
                                 + 1.10985 w_k1_AXP_s006
                                 + 0.779766 w_k1_DIS_s006
                                 + 0.769186 w_k1_FDX_s006
                                 + 1.39439 w_k1_IBM_s006
                                 + 0.946341 w_k1_INTC_s006
                                 + 1.00666 w_k1_KO_s006 + 1.19682 w_k1_MAT_s006
                                 + 0.742574 w_k1_MSFT_s006
                                 + 0.932349 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.61113 w_k1_ADM_s006
                                 + 1.44814 w_k1_AXP_s006 + 1.4625 w_k1_DIS_s006
                                 + 1.20552 w_k1_FDX_s006
                                 + 1.62043 w_k1_IBM_s006
                                 + 1.0099 w_k1_INTC_s006 + 1.56064 w_k1_KO_s006
                                 + 1.60443 w_k1_MAT_s006
                                 + 1.24205 w_k1_MSFT_s006
                                 + 1.29079 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.949444 w_k1_ADM_s006
                                 + 0.835489 w_k1_AXP_s006
                                 + 0.815264 w_k1_DIS_s006
                                 + 0.830133 w_k1_FDX_s006
                                 + 1.0948 w_k1_IBM_s006
                                 + 0.799636 w_k1_INTC_s006
                                 + 0.988998 w_k1_KO_s006
                                 + 1.04145 w_k1_MAT_s006
                                 + 0.938805 w_k1_MSFT_s006
                                 + 0.894405 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.950646 w_k1_ADM_s006
                                 + 0.723071 w_k1_AXP_s006
                                 + 0.667457 w_k1_DIS_s006
                                 + 0.724595 w_k1_FDX_s006
                                 + 0.721257 w_k1_IBM_s006
                                 + 0.81795 w_k1_INTC_s006
                                 + 0.71046 w_k1_KO_s006
                                 + 0.606151 w_k1_MAT_s006
                                 + 0.801718 w_k1_MSFT_s006
                                 + 0.797061 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.1309 w_k1_ADM_s006
                                 + 1.3276 w_k1_AXP_s006 + 1.23348 w_k1_DIS_s006
                                 + 1.25512 w_k1_FDX_s006
                                 + 0.85333 w_k1_IBM_s006
                                 + 0.956329 w_k1_INTC_s006
                                 + 0.872936 w_k1_KO_s006
                                 + 0.837227 w_k1_MAT_s006
                                 + 1.24201 w_k1_MSFT_s006
                                 + 1.05167 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.908745 w_k1_ADM_s006
                                 + 0.865561 w_k1_AXP_s006
                                 + 0.792709 w_k1_DIS_s006
                                 + 0.784512 w_k1_FDX_s006
                                 + 0.917865 w_k1_IBM_s006
                                 + 0.842614 w_k1_INTC_s006
                                 + 0.928129 w_k1_KO_s006
                                 + 0.679476 w_k1_MAT_s006
                                 + 0.981654 w_k1_MSFT_s006
                                 + 0.923013 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.932275 w_k1_ADM_s006
                                 + 1.17185 w_k1_AXP_s006
                                 + 0.90352 w_k1_DIS_s006
                                 + 1.00633 w_k1_FDX_s006
                                 + 0.881533 w_k1_IBM_s006
                                 + 1.0595 w_k1_INTC_s006
                                 + 0.957806 w_k1_KO_s006
                                 + 0.76978 w_k1_MAT_s006
                                 + 1.02884 w_k1_MSFT_s006
                                 + 0.950305 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.14527 w_k1_ADM_s007
                                 + 1.53372 w_k1_AXP_s007
                                 + 1.13597 w_k1_DIS_s007
                                 + 1.26107 w_k1_FDX_s007
                                 + 0.995938 w_k1_IBM_s007
                                 + 0.964186 w_k1_INTC_s007
                                 + 1.22525 w_k1_KO_s007 + 2.06612 w_k1_MAT_s007
                                 + 1.07794 w_k1_MSFT_s007
                                 + 1.25759 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.25486 w_k1_ADM_s007
                                 + 1.53644 w_k1_AXP_s007
                                 + 1.77373 w_k1_DIS_s007
                                 + 1.30408 w_k1_FDX_s007 + 1.3695 w_k1_IBM_s007
                                 + 1.23061 w_k1_INTC_s007
                                 + 1.67383 w_k1_KO_s007 + 2.68497 w_k1_MAT_s007
                                 + 1.31319 w_k1_MSFT_s007
                                 + 1.33371 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.3156 w_k1_ADM_s007
                                 + 0.75582 w_k1_AXP_s007
                                 + 0.778936 w_k1_DIS_s007
                                 + 1.04457 w_k1_FDX_s007
                                 + 0.591227 w_k1_IBM_s007
                                 + 0.800112 w_k1_INTC_s007
                                 + 0.846972 w_k1_KO_s007
                                 + 0.845413 w_k1_MAT_s007
                                 + 0.972413 w_k1_MSFT_s007
                                 + 0.928067 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.28187 w_k1_ADM_s007
                                 + 1.41053 w_k1_AXP_s007 + 1.5352 w_k1_DIS_s007
                                 + 1.38421 w_k1_FDX_s007
                                 + 1.56396 w_k1_IBM_s007
                                 + 1.26089 w_k1_INTC_s007
                                 + 1.46305 w_k1_KO_s007 + 1.87632 w_k1_MAT_s007
                                 + 1.26456 w_k1_MSFT_s007
                                 + 1.24438 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.064 w_k1_ADM_s007
                                 + 0.972501 w_k1_AXP_s007
                                 + 0.669133 w_k1_DIS_s007
                                 + 0.817498 w_k1_FDX_s007
                                 + 0.841976 w_k1_IBM_s007
                                 + 0.77815 w_k1_INTC_s007
                                 + 0.74705 w_k1_KO_s007 + 1.34674 w_k1_MAT_s007
                                 + 0.726656 w_k1_MSFT_s007
                                 + 0.887515 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.878425 w_k1_ADM_s007
                                 + 0.868622 w_k1_AXP_s007
                                 + 0.896536 w_k1_DIS_s007
                                 + 0.839034 w_k1_FDX_s007
                                 + 1.13026 w_k1_IBM_s007
                                 + 0.924679 w_k1_INTC_s007
                                 + 0.943257 w_k1_KO_s007
                                 + 1.51894 w_k1_MAT_s007
                                 + 0.923901 w_k1_MSFT_s007
                                 + 0.870062 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.24024 w_k1_ADM_s007
                                 + 0.986825 w_k1_AXP_s007
                                 + 1.14371 w_k1_DIS_s007
                                 + 1.08035 w_k1_FDX_s007
                                 + 1.09245 w_k1_IBM_s007
                                 + 1.01578 w_k1_INTC_s007 + 1.2053 w_k1_KO_s007
                                 + 1.55225 w_k1_MAT_s007
                                 + 0.965915 w_k1_MSFT_s007
                                 + 1.05239 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.14679 w_k1_ADM_s007
                                 + 0.944195 w_k1_AXP_s007
                                 + 0.993059 w_k1_DIS_s007
                                 + 1.12109 w_k1_FDX_s007
                                 + 0.804537 w_k1_IBM_s007
                                 + 0.925449 w_k1_INTC_s007
                                 + 0.986652 w_k1_KO_s007
                                 + 1.34637 w_k1_MAT_s007
                                 + 1.00621 w_k1_MSFT_s007
                                 + 1.07449 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.05041 w_k1_ADM_s010
                                 + 1.33978 w_k1_AXP_s010
                                 + 1.15403 w_k1_DIS_s010
                                 + 1.23183 w_k1_FDX_s010
                                 + 0.875711 w_k1_IBM_s010
                                 + 1.05896 w_k1_INTC_s010
                                 + 0.94616 w_k1_KO_s010 + 1.05815 w_k1_MAT_s010
                                 + 1.11073 w_k1_MSFT_s010
                                 + 1.11045 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.764443 w_k1_ADM_s010
                                 + 0.809211 w_k1_AXP_s010
                                 + 0.780566 w_k1_DIS_s010
                                 + 0.924682 w_k1_FDX_s010
                                 + 0.635133 w_k1_IBM_s010
                                 + 0.851267 w_k1_INTC_s010
                                 + 0.686382 w_k1_KO_s010
                                 + 0.565781 w_k1_MAT_s010
                                 + 0.832705 w_k1_MSFT_s010
                                 + 0.766784 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.819786 w_k1_ADM_s010
                                 + 1.01346 w_k1_AXP_s010
                                 + 0.789335 w_k1_DIS_s010
                                 + 0.869458 w_k1_FDX_s010
                                 + 0.85707 w_k1_IBM_s010
                                 + 0.935173 w_k1_INTC_s010
                                 + 0.818607 w_k1_KO_s010
                                 + 0.911107 w_k1_MAT_s010
                                 + 0.863406 w_k1_MSFT_s010
                                 + 0.795153 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.083 w_k1_ADM_s010
                                 + 1.39411 w_k1_AXP_s010
                                 + 1.37793 w_k1_DIS_s010 + 1.3525 w_k1_FDX_s010
                                 + 0.810628 w_k1_IBM_s010
                                 + 1.25081 w_k1_INTC_s010
                                 + 1.09176 w_k1_KO_s010
                                 + 0.850619 w_k1_MAT_s010
                                 + 1.24665 w_k1_MSFT_s010
                                 + 1.12429 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.978846 w_k1_ADM_s010
                                 + 1.45981 w_k1_AXP_s010
                                 + 1.50185 w_k1_DIS_s010
                                 + 1.18799 w_k1_FDX_s010
                                 + 1.17309 w_k1_IBM_s010
                                 + 1.30559 w_k1_INTC_s010
                                 + 1.39599 w_k1_KO_s010 + 1.45544 w_k1_MAT_s010
                                 + 1.23151 w_k1_MSFT_s010
                                 + 1.23454 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.17051 w_k1_ADM_s010
                                 + 1.10566 w_k1_AXP_s010
                                 + 0.966009 w_k1_DIS_s010
                                 + 1.05027 w_k1_FDX_s010
                                 + 0.958208 w_k1_IBM_s010
                                 + 1.11364 w_k1_INTC_s010
                                 + 0.950246 w_k1_KO_s010
                                 + 1.05958 w_k1_MAT_s010
                                 + 0.911856 w_k1_MSFT_s010
                                 + 1.00965 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.14932 w_k1_ADM_s010
                                 + 1.37041 w_k1_AXP_s010
                                 + 1.13204 w_k1_DIS_s010
                                 + 1.09562 w_k1_FDX_s010
                                 + 1.38314 w_k1_IBM_s010 + 1.337 w_k1_INTC_s010
                                 + 1.20769 w_k1_KO_s010 + 1.08413 w_k1_MAT_s010
                                 + 1.09073 w_k1_MSFT_s010
                                 + 1.13358 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.893202 w_k1_ADM_s010
                                 + 0.975274 w_k1_AXP_s010
                                 + 0.688989 w_k1_DIS_s010
                                 + 0.911448 w_k1_FDX_s010
                                 + 1.15548 w_k1_IBM_s010
                                 + 0.932068 w_k1_INTC_s010
                                 + 0.928362 w_k1_KO_s010
                                 + 1.36908 w_k1_MAT_s010
                                 + 0.861283 w_k1_MSFT_s010
                                 + 0.920819 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.24919 w_k1_ADM_s011
                                 + 0.828126 w_k1_AXP_s011
                                 + 0.905754 w_k1_DIS_s011
                                 + 0.992553 w_k1_FDX_s011
                                 + 0.721546 w_k1_IBM_s011
                                 + 1.08044 w_k1_INTC_s011
                                 + 0.960861 w_k1_KO_s011
                                 + 0.606758 w_k1_MAT_s011
                                 + 1.10764 w_k1_MSFT_s011
                                 + 0.968049 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.992933 w_k1_ADM_s011
                                 + 0.918576 w_k1_AXP_s011
                                 + 0.737225 w_k1_DIS_s011
                                 + 0.790788 w_k1_FDX_s011
                                 + 1.05716 w_k1_IBM_s011
                                 + 0.943252 w_k1_INTC_s011
                                 + 0.895203 w_k1_KO_s011
                                 + 0.863688 w_k1_MAT_s011
                                 + 0.732721 w_k1_MSFT_s011
                                 + 0.947257 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.13544 w_k1_ADM_s011
                                 + 1.12096 w_k1_AXP_s011
                                 + 0.778679 w_k1_DIS_s011
                                 + 1.02047 w_k1_FDX_s011
                                 + 0.919164 w_k1_IBM_s011
                                 + 1.14099 w_k1_INTC_s011
                                 + 1.06483 w_k1_KO_s011
                                 + 0.981124 w_k1_MAT_s011
                                 + 1.05045 w_k1_MSFT_s011
                                 + 0.925843 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.33391 w_k1_ADM_s011
                                 + 1.34528 w_k1_AXP_s011
                                 + 1.41141 w_k1_DIS_s011
                                 + 1.18365 w_k1_FDX_s011
                                 + 1.51851 w_k1_IBM_s011
                                 + 1.30152 w_k1_INTC_s011
                                 + 1.67492 w_k1_KO_s011 + 1.08796 w_k1_MAT_s011
                                 + 1.15188 w_k1_MSFT_s011
                                 + 1.18744 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.88958 w_k1_ADM_s011
                                 + 0.933227 w_k1_AXP_s011
                                 + 0.954739 w_k1_DIS_s011
                                 + 0.919931 w_k1_FDX_s011
                                 + 1.3592 w_k1_IBM_s011
                                 + 0.867969 w_k1_INTC_s011
                                 + 1.22298 w_k1_KO_s011 + 1.08357 w_k1_MAT_s011
                                 + 0.816008 w_k1_MSFT_s011
                                 + 0.922684 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.51498 w_k1_ADM_s011
                                 + 1.67061 w_k1_AXP_s011
                                 + 1.52175 w_k1_DIS_s011
                                 + 1.31328 w_k1_FDX_s011
                                 + 1.08149 w_k1_IBM_s011
                                 + 1.21661 w_k1_INTC_s011
                                 + 1.30429 w_k1_KO_s011 + 1.04497 w_k1_MAT_s011
                                 + 1.19528 w_k1_MSFT_s011
                                 + 1.30008 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.976598 w_k1_ADM_s011
                                 + 0.798902 w_k1_AXP_s011
                                 + 0.526917 w_k1_DIS_s011
                                 + 0.717626 w_k1_FDX_s011
                                 + 0.710938 w_k1_IBM_s011
                                 + 0.864429 w_k1_INTC_s011
                                 + 0.722433 w_k1_KO_s011
                                 + 0.564286 w_k1_MAT_s011
                                 + 0.752557 w_k1_MSFT_s011
                                 + 0.85358 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.36664 w_k1_ADM_s011
                                 + 1.2227 w_k1_AXP_s011 + 1.21508 w_k1_DIS_s011
                                 + 1.15749 w_k1_FDX_s011
                                 + 1.05449 w_k1_IBM_s011
                                 + 1.27716 w_k1_INTC_s011
                                 + 1.30065 w_k1_KO_s011
                                 + 0.985672 w_k1_MAT_s011
                                 + 1.1787 w_k1_MSFT_s011 + 1.00627 w_k1_PG_s011
                                 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.815711 w_k1_ADM_s012
                                 + 0.712729 w_k1_AXP_s012
                                 + 0.898015 w_k1_DIS_s012
                                 + 0.882897 w_k1_FDX_s012
                                 + 1.14225 w_k1_IBM_s012
                                 + 0.900726 w_k1_INTC_s012
                                 + 1.07198 w_k1_KO_s012
                                 + 0.888259 w_k1_MAT_s012
                                 + 0.939263 w_k1_MSFT_s012
                                 + 0.880045 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.707432 w_k1_ADM_s012
                                 + 0.698015 w_k1_AXP_s012
                                 + 0.827329 w_k1_DIS_s012
                                 + 0.8803 w_k1_FDX_s012
                                 + 0.614506 w_k1_IBM_s012
                                 + 0.927142 w_k1_INTC_s012
                                 + 0.683231 w_k1_KO_s012
                                 + 0.446481 w_k1_MAT_s012
                                 + 0.806752 w_k1_MSFT_s012
                                 + 0.790467 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.782502 w_k1_ADM_s012
                                 + 1.25244 w_k1_AXP_s012
                                 + 1.19671 w_k1_DIS_s012
                                 + 1.06277 w_k1_FDX_s012
                                 + 1.59735 w_k1_IBM_s012
                                 + 1.20765 w_k1_INTC_s012 + 1.1837 w_k1_KO_s012
                                 + 1.54359 w_k1_MAT_s012
                                 + 1.17678 w_k1_MSFT_s012
                                 + 1.25173 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.72143 w_k1_ADM_s012
                                 + 0.883245 w_k1_AXP_s012
                                 + 0.794462 w_k1_DIS_s012
                                 + 0.88301 w_k1_FDX_s012
                                 + 0.760026 w_k1_IBM_s012
                                 + 0.940916 w_k1_INTC_s012
                                 + 0.705405 w_k1_KO_s012
                                 + 0.765321 w_k1_MAT_s012
                                 + 0.807923 w_k1_MSFT_s012
                                 + 0.769136 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.829621 w_k1_ADM_s012
                                 + 1.34558 w_k1_AXP_s012
                                 + 1.49188 w_k1_DIS_s012
                                 + 1.40899 w_k1_FDX_s012
                                 + 0.746437 w_k1_IBM_s012
                                 + 1.35446 w_k1_INTC_s012
                                 + 0.927914 w_k1_KO_s012
                                 + 0.679703 w_k1_MAT_s012
                                 + 1.32101 w_k1_MSFT_s012
                                 + 1.05315 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.30537 w_k1_ADM_s012
                                 + 1.13586 w_k1_AXP_s012
                                 + 1.70784 w_k1_DIS_s012
                                 + 1.20564 w_k1_FDX_s012
                                 + 1.30648 w_k1_IBM_s012
                                 + 1.26678 w_k1_INTC_s012 + 1.4246 w_k1_KO_s012
                                 + 1.03377 w_k1_MAT_s012
                                 + 1.36813 w_k1_MSFT_s012
                                 + 1.12281 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.114 w_k1_ADM_s012
                                 + 1.19044 w_k1_AXP_s012 + 1.2851 w_k1_DIS_s012
                                 + 1.14999 w_k1_FDX_s012
                                 + 1.07062 w_k1_IBM_s012
                                 + 1.18827 w_k1_INTC_s012
                                 + 1.06018 w_k1_KO_s012
                                 + 0.993607 w_k1_MAT_s012
                                 + 1.2899 w_k1_MSFT_s012 + 1.04387 w_k1_PG_s012
                                 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.862551 w_k1_ADM_s012
                                 + 1.06869 w_k1_AXP_s012
                                 + 1.26727 w_k1_DIS_s012
                                 + 1.06684 w_k1_FDX_s012
                                 + 0.916369 w_k1_IBM_s012
                                 + 1.08331 w_k1_INTC_s012
                                 + 0.942598 w_k1_KO_s012
                                 + 0.751143 w_k1_MAT_s012
                                 + 1.03825 w_k1_MSFT_s012
                                 + 0.896505 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.922606 w_k1_ADM_s013
                                 + 0.914781 w_k1_AXP_s013
                                 + 0.721859 w_k1_DIS_s013
                                 + 0.902706 w_k1_FDX_s013
                                 + 0.751704 w_k1_IBM_s013
                                 + 0.934686 w_k1_INTC_s013
                                 + 0.785404 w_k1_KO_s013
                                 + 0.838445 w_k1_MAT_s013
                                 + 0.822192 w_k1_MSFT_s013
                                 + 0.975826 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.21707 w_k1_ADM_s013
                                 + 1.2456 w_k1_AXP_s013 + 1.27543 w_k1_DIS_s013
                                 + 1.19303 w_k1_FDX_s013
                                 + 1.45438 w_k1_IBM_s013
                                 + 1.10869 w_k1_INTC_s013
                                 + 1.14817 w_k1_KO_s013 + 1.37113 w_k1_MAT_s013
                                 + 1.12636 w_k1_MSFT_s013
                                 + 1.14698 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.09004 w_k1_ADM_s013
                                 + 1.83628 w_k1_AXP_s013
                                 + 1.95614 w_k1_DIS_s013
                                 + 1.52344 w_k1_FDX_s013
                                 + 1.27554 w_k1_IBM_s013
                                 + 1.44569 w_k1_INTC_s013
                                 + 1.40812 w_k1_KO_s013 + 1.62367 w_k1_MAT_s013
                                 + 1.31495 w_k1_MSFT_s013 + 1.4133 w_k1_PG_s013
                                 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.05948 w_k1_ADM_s013
                                 + 1.31811 w_k1_AXP_s013
                                 + 1.05052 w_k1_DIS_s013
                                 + 0.973836 w_k1_FDX_s013
                                 + 1.24843 w_k1_IBM_s013
                                 + 1.00064 w_k1_INTC_s013
                                 + 1.14008 w_k1_KO_s013 + 1.00931 w_k1_MAT_s013
                                 + 0.916264 w_k1_MSFT_s013
                                 + 1.09536 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.776119 w_k1_ADM_s013
                                 + 0.636221 w_k1_AXP_s013
                                 + 0.523866 w_k1_DIS_s013
                                 + 0.703852 w_k1_FDX_s013
                                 + 0.678489 w_k1_IBM_s013
                                 + 0.630536 w_k1_INTC_s013
                                 + 0.676111 w_k1_KO_s013
                                 + 0.552725 w_k1_MAT_s013
                                 + 0.639421 w_k1_MSFT_s013
                                 + 0.784585 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.988187 w_k1_ADM_s013
                                 + 1.41206 w_k1_AXP_s013
                                 + 1.54462 w_k1_DIS_s013
                                 + 1.18583 w_k1_FDX_s013
                                 + 1.30191 w_k1_IBM_s013
                                 + 1.13257 w_k1_INTC_s013
                                 + 1.18624 w_k1_KO_s013 + 1.17775 w_k1_MAT_s013
                                 + 1.03499 w_k1_MSFT_s013
                                 + 1.25409 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.827071 w_k1_ADM_s013
                                 + 0.825925 w_k1_AXP_s013
                                 + 0.7851 w_k1_DIS_s013
                                 + 0.754324 w_k1_FDX_s013
                                 + 0.973932 w_k1_IBM_s013
                                 + 0.835693 w_k1_INTC_s013
                                 + 0.844675 w_k1_KO_s013
                                 + 0.748487 w_k1_MAT_s013
                                 + 0.713124 w_k1_MSFT_s013
                                 + 0.862334 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.0003 w_k1_ADM_s013
                                 + 0.823806 w_k1_AXP_s013
                                 + 1.09827 w_k1_DIS_s013
                                 + 1.02002 w_k1_FDX_s013
                                 + 0.900673 w_k1_IBM_s013
                                 + 1.11059 w_k1_INTC_s013
                                 + 1.07807 w_k1_KO_s013
                                 + 0.594725 w_k1_MAT_s013
                                 + 0.992422 w_k1_MSFT_s013
                                 + 1.10007 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.45827 w_k1_ADM_s014
                                 + 1.15725 w_k1_AXP_s014 + 1.3524 w_k1_DIS_s014
                                 + 1.35343 w_k1_FDX_s014
                                 + 0.942918 w_k1_IBM_s014
                                 + 1.37202 w_k1_INTC_s014 + 1.0357 w_k1_KO_s014
                                 + 0.840783 w_k1_MAT_s014
                                 + 1.36478 w_k1_MSFT_s014
                                 + 1.25984 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.798181 w_k1_ADM_s014
                                 + 0.630783 w_k1_AXP_s014
                                 + 0.597422 w_k1_DIS_s014
                                 + 0.734868 w_k1_FDX_s014
                                 + 0.689124 w_k1_IBM_s014
                                 + 0.915498 w_k1_INTC_s014
                                 + 0.678869 w_k1_KO_s014
                                 + 0.600863 w_k1_MAT_s014
                                 + 0.762713 w_k1_MSFT_s014
                                 + 0.726222 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.909311 w_k1_ADM_s014
                                 + 0.655144 w_k1_AXP_s014
                                 + 0.734732 w_k1_DIS_s014
                                 + 0.776069 w_k1_FDX_s014
                                 + 1.16449 w_k1_IBM_s014
                                 + 1.0089 w_k1_INTC_s014
                                 + 0.902919 w_k1_KO_s014 + 1.0034 w_k1_MAT_s014
                                 + 0.907293 w_k1_MSFT_s014
                                 + 0.912005 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.18842 w_k1_ADM_s014
                                 + 1.53195 w_k1_AXP_s014
                                 + 1.50956 w_k1_DIS_s014
                                 + 1.30264 w_k1_FDX_s014
                                 + 1.63597 w_k1_IBM_s014
                                 + 1.28147 w_k1_INTC_s014 + 1.402 w_k1_KO_s014
                                 + 1.81866 w_k1_MAT_s014
                                 + 1.12276 w_k1_MSFT_s014
                                 + 1.27952 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.884887 w_k1_ADM_s014
                                 + 0.859934 w_k1_AXP_s014
                                 + 1.11809 w_k1_DIS_s014
                                 + 1.11252 w_k1_FDX_s014
                                 + 0.847142 w_k1_IBM_s014
                                 + 1.34248 w_k1_INTC_s014
                                 + 0.982306 w_k1_KO_s014
                                 + 0.783202 w_k1_MAT_s014
                                 + 1.09393 w_k1_MSFT_s014
                                 + 1.05637 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.880456 w_k1_ADM_s014
                                 + 0.671646 w_k1_AXP_s014
                                 + 0.93782 w_k1_DIS_s014
                                 + 0.89868 w_k1_FDX_s014
                                 + 0.706918 w_k1_IBM_s014
                                 + 1.05092 w_k1_INTC_s014
                                 + 0.788996 w_k1_KO_s014
                                 + 0.572667 w_k1_MAT_s014
                                 + 0.946524 w_k1_MSFT_s014
                                 + 0.982069 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.08592 w_k1_ADM_s014
                                 + 1.14057 w_k1_AXP_s014
                                 + 1.13582 w_k1_DIS_s014
                                 + 1.06393 w_k1_FDX_s014 + 1.283 w_k1_IBM_s014
                                 + 1.34743 w_k1_INTC_s014
                                 + 1.05923 w_k1_KO_s014 + 1.06412 w_k1_MAT_s014
                                 + 0.962018 w_k1_MSFT_s014
                                 + 1.01116 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.88072 w_k1_ADM_s014
                                 + 0.885295 w_k1_AXP_s014
                                 + 0.756871 w_k1_DIS_s014
                                 + 0.839758 w_k1_FDX_s014
                                 + 0.909873 w_k1_IBM_s014
                                 + 1.05405 w_k1_INTC_s014
                                 + 0.858417 w_k1_KO_s014
                                 + 0.762168 w_k1_MAT_s014
                                 + 0.785796 w_k1_MSFT_s014
                                 + 0.919076 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.890765 w_k1_ADM_s015
                                 + 1.04466 w_k1_AXP_s015
                                 + 1.02387 w_k1_DIS_s015
                                 + 0.990699 w_k1_FDX_s015
                                 + 1.23376 w_k1_IBM_s015
                                 + 0.941501 w_k1_INTC_s015
                                 + 1.21272 w_k1_KO_s015 + 1.28342 w_k1_MAT_s015
                                 + 0.917095 w_k1_MSFT_s015
                                 + 1.05362 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.946057 w_k1_ADM_s015
                                 + 0.867394 w_k1_AXP_s015
                                 + 0.779951 w_k1_DIS_s015
                                 + 0.883334 w_k1_FDX_s015
                                 + 0.726032 w_k1_IBM_s015
                                 + 0.799294 w_k1_INTC_s015
                                 + 0.667287 w_k1_KO_s015
                                 + 0.751049 w_k1_MAT_s015
                                 + 0.801964 w_k1_MSFT_s015
                                 + 0.85459 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.986861 w_k1_ADM_s015
                                 + 1.16115 w_k1_AXP_s015
                                 + 1.27711 w_k1_DIS_s015
                                 + 1.11873 w_k1_FDX_s015
                                 + 1.46913 w_k1_IBM_s015
                                 + 0.893201 w_k1_INTC_s015
                                 + 1.21989 w_k1_KO_s015 + 2.17632 w_k1_MAT_s015
                                 + 0.966431 w_k1_MSFT_s015
                                 + 1.15778 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.947253 w_k1_ADM_s015
                                 + 1.20819 w_k1_AXP_s015
                                 + 0.767204 w_k1_DIS_s015
                                 + 0.947936 w_k1_FDX_s015
                                 + 0.937666 w_k1_IBM_s015
                                 + 0.874815 w_k1_INTC_s015
                                 + 0.923795 w_k1_KO_s015
                                 + 1.41846 w_k1_MAT_s015
                                 + 0.82525 w_k1_MSFT_s015
                                 + 1.02896 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.20139 w_k1_ADM_s015
                                 + 1.22737 w_k1_AXP_s015
                                 + 1.54909 w_k1_DIS_s015
                                 + 1.33527 w_k1_FDX_s015
                                 + 0.882257 w_k1_IBM_s015
                                 + 1.15093 w_k1_INTC_s015 + 1.3407 w_k1_KO_s015
                                 + 0.966036 w_k1_MAT_s015
                                 + 1.27868 w_k1_MSFT_s015 + 1.139 w_k1_PG_s015
                                 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.846433 w_k1_ADM_s015
                                 + 0.601512 w_k1_AXP_s015
                                 + 0.700154 w_k1_DIS_s015
                                 + 0.755011 w_k1_FDX_s015
                                 + 1.22251 w_k1_IBM_s015
                                 + 0.649693 w_k1_INTC_s015
                                 + 0.994006 w_k1_KO_s015
                                 + 1.18723 w_k1_MAT_s015
                                 + 0.760052 w_k1_MSFT_s015
                                 + 0.950655 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.06852 w_k1_ADM_s015
                                 + 0.872291 w_k1_AXP_s015
                                 + 0.928401 w_k1_DIS_s015
                                 + 1.03691 w_k1_FDX_s015
                                 + 0.883965 w_k1_IBM_s015
                                 + 0.885001 w_k1_INTC_s015
                                 + 1.04451 w_k1_KO_s015
                                 + 0.804413 w_k1_MAT_s015
                                 + 0.972521 w_k1_MSFT_s015
                                 + 1.17276 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.06743 w_k1_ADM_s015
                                 + 1.45212 w_k1_AXP_s015
                                 + 1.32097 w_k1_DIS_s015
                                 + 1.33395 w_k1_FDX_s015
                                 + 1.28776 w_k1_IBM_s015
                                 + 1.13555 w_k1_INTC_s015
                                 + 1.35163 w_k1_KO_s015 + 1.87895 w_k1_MAT_s015
                                 + 1.16153 w_k1_MSFT_s015
                                 + 1.25068 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.07691 w_k1_ADM_s016
                                 + 1.21524 w_k1_AXP_s016
                                 + 1.17406 w_k1_DIS_s016
                                 + 1.06581 w_k1_FDX_s016 + 1.0512 w_k1_IBM_s016
                                 + 0.893918 w_k1_INTC_s016
                                 + 1.08901 w_k1_KO_s016 + 1.33563 w_k1_MAT_s016
                                 + 1.10925 w_k1_MSFT_s016
                                 + 1.13547 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.789683 w_k1_ADM_s016
                                 + 0.74661 w_k1_AXP_s016
                                 + 0.783376 w_k1_DIS_s016
                                 + 0.823818 w_k1_FDX_s016
                                 + 0.725616 w_k1_IBM_s016
                                 + 0.6823 w_k1_INTC_s016 + 0.70524 w_k1_KO_s016
                                 + 0.756147 w_k1_MAT_s016
                                 + 0.782118 w_k1_MSFT_s016
                                 + 0.737411 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.517 w_k1_ADM_s016
                                 + 1.50767 w_k1_AXP_s016
                                 + 1.39628 w_k1_DIS_s016
                                 + 1.18967 w_k1_FDX_s016
                                 + 1.64029 w_k1_IBM_s016
                                 + 1.05579 w_k1_INTC_s016
                                 + 1.20689 w_k1_KO_s016 + 1.65398 w_k1_MAT_s016
                                 + 1.10646 w_k1_MSFT_s016
                                 + 1.04418 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.08695 w_k1_ADM_s016
                                 + 0.96573 w_k1_AXP_s016
                                 + 1.00185 w_k1_DIS_s016
                                 + 1.11725 w_k1_FDX_s016
                                 + 0.740383 w_k1_IBM_s016
                                 + 0.875187 w_k1_INTC_s016
                                 + 0.795179 w_k1_KO_s016
                                 + 0.971212 w_k1_MAT_s016
                                 + 1.00223 w_k1_MSFT_s016 + 1.0166 w_k1_PG_s016
                                 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.12682 w_k1_ADM_s016
                                 + 0.944908 w_k1_AXP_s016
                                 + 1.0393 w_k1_DIS_s016
                                 + 0.990053 w_k1_FDX_s016
                                 + 1.03256 w_k1_IBM_s016
                                 + 1.01043 w_k1_INTC_s016 + 0.9478 w_k1_KO_s016
                                 + 0.811158 w_k1_MAT_s016
                                 + 0.991028 w_k1_MSFT_s016
                                 + 0.962048 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.02255 w_k1_ADM_s016
                                 + 1.23213 w_k1_AXP_s016
                                 + 1.06159 w_k1_DIS_s016
                                 + 0.982204 w_k1_FDX_s016
                                 + 1.26265 w_k1_IBM_s016
                                 + 0.787393 w_k1_INTC_s016
                                 + 0.957374 w_k1_KO_s016
                                 + 1.71901 w_k1_MAT_s016
                                 + 0.857941 w_k1_MSFT_s016
                                 + 0.889084 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.897982 w_k1_ADM_s016
                                 + 0.819153 w_k1_AXP_s016
                                 + 0.853208 w_k1_DIS_s016
                                 + 0.857249 w_k1_FDX_s016
                                 + 1.08209 w_k1_IBM_s016
                                 + 0.727684 w_k1_INTC_s016
                                 + 0.898757 w_k1_KO_s016
                                 + 0.931398 w_k1_MAT_s016
                                 + 0.822972 w_k1_MSFT_s016
                                 + 0.86115 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.16279 w_k1_ADM_s016
                                 + 1.46552 w_k1_AXP_s016
                                 + 1.71993 w_k1_DIS_s016
                                 + 1.30311 w_k1_FDX_s016
                                 + 1.12058 w_k1_IBM_s016
                                 + 1.15702 w_k1_INTC_s016
                                 + 1.04868 w_k1_KO_s016 + 1.56835 w_k1_MAT_s016
                                 + 1.25735 w_k1_MSFT_s016
                                 + 1.10428 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.909847 w_k1_ADM_s017
                                 + 1.68932 w_k1_AXP_s017
                                 + 1.46572 w_k1_DIS_s017
                                 + 1.25198 w_k1_FDX_s017
                                 + 1.45258 w_k1_IBM_s017
                                 + 1.26882 w_k1_INTC_s017
                                 + 1.33851 w_k1_KO_s017 + 1.47278 w_k1_MAT_s017
                                 + 1.40376 w_k1_MSFT_s017
                                 + 1.12007 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.01842 w_k1_ADM_s017
                                 + 1.1076 w_k1_AXP_s017 + 1.12243 w_k1_DIS_s017
                                 + 1.07632 w_k1_FDX_s017
                                 + 0.718665 w_k1_IBM_s017
                                 + 1.11221 w_k1_INTC_s017
                                 + 0.799945 w_k1_KO_s017
                                 + 0.602207 w_k1_MAT_s017
                                 + 1.18778 w_k1_MSFT_s017
                                 + 0.923266 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.753065 w_k1_ADM_s017
                                 + 0.925019 w_k1_AXP_s017
                                 + 1.25491 w_k1_DIS_s017
                                 + 0.932128 w_k1_FDX_s017
                                 + 0.984792 w_k1_IBM_s017
                                 + 0.974024 w_k1_INTC_s017
                                 + 1.01934 w_k1_KO_s017 + 1.00736 w_k1_MAT_s017
                                 + 1.07969 w_k1_MSFT_s017
                                 + 0.946559 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.649844 w_k1_ADM_s017
                                 + 0.750336 w_k1_AXP_s017
                                 + 0.774626 w_k1_DIS_s017
                                 + 0.87053 w_k1_FDX_s017
                                 + 0.675183 w_k1_IBM_s017
                                 + 0.986174 w_k1_INTC_s017
                                 + 0.783184 w_k1_KO_s017
                                 + 0.581501 w_k1_MAT_s017
                                 + 0.906074 w_k1_MSFT_s017
                                 + 0.806959 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.928777 w_k1_ADM_s017
                                 + 0.795575 w_k1_AXP_s017
                                 + 0.678442 w_k1_DIS_s017
                                 + 0.780768 w_k1_FDX_s017
                                 + 0.855282 w_k1_IBM_s017
                                 + 0.822509 w_k1_INTC_s017
                                 + 0.890622 w_k1_KO_s017
                                 + 0.88595 w_k1_MAT_s017
                                 + 0.80367 w_k1_MSFT_s017
                                 + 0.914807 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.988626 w_k1_ADM_s017
                                 + 1.01162 w_k1_AXP_s017
                                 + 1.09814 w_k1_DIS_s017
                                 + 1.05313 w_k1_FDX_s017
                                 + 1.10082 w_k1_IBM_s017
                                 + 0.947405 w_k1_INTC_s017
                                 + 1.0335 w_k1_KO_s017 + 0.988854 w_k1_MAT_s017
                                 + 1.08327 w_k1_MSFT_s017
                                 + 0.965606 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.918894 w_k1_ADM_s017
                                 + 0.834017 w_k1_AXP_s017
                                 + 0.788696 w_k1_DIS_s017
                                 + 0.801271 w_k1_FDX_s017
                                 + 1.08384 w_k1_IBM_s017
                                 + 0.888126 w_k1_INTC_s017
                                 + 0.857363 w_k1_KO_s017
                                 + 0.817171 w_k1_MAT_s017
                                 + 0.884493 w_k1_MSFT_s017
                                 + 0.806893 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.961385 w_k1_ADM_s017
                                 + 1.04121 w_k1_AXP_s017
                                 + 1.13206 w_k1_DIS_s017 + 1.1422 w_k1_FDX_s017
                                 + 0.882996 w_k1_IBM_s017
                                 + 1.01208 w_k1_INTC_s017
                                 + 0.922878 w_k1_KO_s017
                                 + 0.904585 w_k1_MAT_s017
                                 + 1.05627 w_k1_MSFT_s017
                                 + 0.901912 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.09546 w_k1_ADM_s100
                                 + 1.01117 w_k1_AXP_s100
                                 + 0.971441 w_k1_DIS_s100
                                 + 0.963393 w_k1_FDX_s100
                                 + 0.753379 w_k1_IBM_s100
                                 + 0.996501 w_k1_INTC_s100
                                 + 0.940676 w_k1_KO_s100
                                 + 0.54487 w_k1_MAT_s100
                                 + 1.04729 w_k1_MSFT_s100
                                 + 0.907347 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.26556 w_k1_ADM_s100
                                 + 1.14753 w_k1_AXP_s100
                                 + 1.33593 w_k1_DIS_s100
                                 + 1.19355 w_k1_FDX_s100
                                 + 0.984666 w_k1_IBM_s100
                                 + 1.13606 w_k1_INTC_s100
                                 + 1.12711 w_k1_KO_s100 + 0.94123 w_k1_MAT_s100
                                 + 1.21152 w_k1_MSFT_s100
                                 + 0.977044 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.0976 w_k1_ADM_s100
                                 + 1.16012 w_k1_AXP_s100
                                 + 1.04459 w_k1_DIS_s100
                                 + 0.941635 w_k1_FDX_s100
                                 + 1.34697 w_k1_IBM_s100
                                 + 0.923746 w_k1_INTC_s100
                                 + 1.12585 w_k1_KO_s100 + 1.52519 w_k1_MAT_s100
                                 + 1.03534 w_k1_MSFT_s100
                                 + 1.00131 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.30219 w_k1_ADM_s100
                                 + 1.27371 w_k1_AXP_s100
                                 + 1.76356 w_k1_DIS_s100
                                 + 1.28741 w_k1_FDX_s100 + 1.3491 w_k1_IBM_s100
                                 + 1.1851 w_k1_INTC_s100 + 1.5444 w_k1_KO_s100
                                 + 1.71317 w_k1_MAT_s100
                                 + 1.47949 w_k1_MSFT_s100
                                 + 1.20341 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.12009 w_k1_ADM_s100
                                 + 1.51491 w_k1_AXP_s100
                                 + 1.37487 w_k1_DIS_s100
                                 + 1.11438 w_k1_FDX_s100
                                 + 1.99815 w_k1_IBM_s100
                                 + 1.15275 w_k1_INTC_s100
                                 + 1.50404 w_k1_KO_s100 + 2.12108 w_k1_MAT_s100
                                 + 1.22142 w_k1_MSFT_s100
                                 + 1.03558 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.895925 w_k1_ADM_s100
                                 + 0.683449 w_k1_AXP_s100
                                 + 0.64164 w_k1_DIS_s100
                                 + 0.782304 w_k1_FDX_s100
                                 + 0.729925 w_k1_IBM_s100
                                 + 0.779756 w_k1_INTC_s100
                                 + 0.738526 w_k1_KO_s100
                                 + 0.714242 w_k1_MAT_s100
                                 + 0.710658 w_k1_MSFT_s100
                                 + 0.849911 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.805436 w_k1_ADM_s100
                                 + 0.923711 w_k1_AXP_s100
                                 + 0.824284 w_k1_DIS_s100
                                 + 0.741632 w_k1_FDX_s100
                                 + 1.47054 w_k1_IBM_s100
                                 + 0.862169 w_k1_INTC_s100
                                 + 0.942714 w_k1_KO_s100
                                 + 1.32003 w_k1_MAT_s100
                                 + 0.669624 w_k1_MSFT_s100
                                 + 0.868229 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.959667 w_k1_ADM_s100
                                 + 0.935703 w_k1_AXP_s100
                                 + 0.803556 w_k1_DIS_s100
                                 + 0.850465 w_k1_FDX_s100
                                 + 1.0694 w_k1_IBM_s100
                                 + 0.867826 w_k1_INTC_s100
                                 + 0.892244 w_k1_KO_s100
                                 + 1.11838 w_k1_MAT_s100
                                 + 0.827465 w_k1_MSFT_s100
                                 + 0.87166 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.997102 w_k1_ADM_s101
                                 + 1.02936 w_k1_AXP_s101
                                 + 0.937814 w_k1_DIS_s101
                                 + 0.953057 w_k1_FDX_s101
                                 + 0.791597 w_k1_IBM_s101
                                 + 1.19006 w_k1_INTC_s101
                                 + 1.04737 w_k1_KO_s101 + 1.12119 w_k1_MAT_s101
                                 + 1.12899 w_k1_MSFT_s101
                                 + 1.01612 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.843266 w_k1_ADM_s101
                                 + 0.74253 w_k1_AXP_s101
                                 + 0.585334 w_k1_DIS_s101
                                 + 0.783735 w_k1_FDX_s101
                                 + 0.655551 w_k1_IBM_s101
                                 + 0.873959 w_k1_INTC_s101
                                 + 0.85242 w_k1_KO_s101
                                 + 0.729225 w_k1_MAT_s101
                                 + 0.942574 w_k1_MSFT_s101
                                 + 0.825346 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.894837 w_k1_ADM_s101
                                 + 0.99522 w_k1_AXP_s101
                                 + 1.14704 w_k1_DIS_s101
                                 + 0.928157 w_k1_FDX_s101
                                 + 1.48244 w_k1_IBM_s101
                                 + 1.14069 w_k1_INTC_s101
                                 + 1.29777 w_k1_KO_s101 + 1.40569 w_k1_MAT_s101
                                 + 0.981543 w_k1_MSFT_s101
                                 + 1.04576 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.771209 w_k1_ADM_s101
                                 + 0.747252 w_k1_AXP_s101
                                 + 0.583132 w_k1_DIS_s101
                                 + 0.661173 w_k1_FDX_s101
                                 + 0.898303 w_k1_IBM_s101
                                 + 0.756627 w_k1_INTC_s101
                                 + 0.869287 w_k1_KO_s101
                                 + 0.857378 w_k1_MAT_s101
                                 + 0.779635 w_k1_MSFT_s101
                                 + 0.943652 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.998879 w_k1_ADM_s101
                                 + 1.20701 w_k1_AXP_s101
                                 + 1.05112 w_k1_DIS_s101
                                 + 1.09585 w_k1_FDX_s101
                                 + 0.892936 w_k1_IBM_s101
                                 + 1.21927 w_k1_INTC_s101
                                 + 0.988506 w_k1_KO_s101
                                 + 1.53921 w_k1_MAT_s101
                                 + 1.14307 w_k1_MSFT_s101
                                 + 1.09807 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.849502 w_k1_ADM_s101
                                 + 0.765425 w_k1_AXP_s101
                                 + 0.802188 w_k1_DIS_s101
                                 + 0.795006 w_k1_FDX_s101
                                 + 0.805682 w_k1_IBM_s101
                                 + 0.895641 w_k1_INTC_s101
                                 + 0.88272 w_k1_KO_s101
                                 + 0.831534 w_k1_MAT_s101
                                 + 0.852055 w_k1_MSFT_s101
                                 + 0.868604 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.3953 w_k1_ADM_s101
                                 + 1.5025 w_k1_AXP_s101 + 1.63216 w_k1_DIS_s101
                                 + 1.35705 w_k1_FDX_s101 + 1.141 w_k1_IBM_s101
                                 + 1.20028 w_k1_INTC_s101
                                 + 1.36174 w_k1_KO_s101 + 1.62297 w_k1_MAT_s101
                                 + 1.48792 w_k1_MSFT_s101
                                 + 1.17862 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.905899 w_k1_ADM_s101
                                 + 1.09508 w_k1_AXP_s101
                                 + 0.776758 w_k1_DIS_s101
                                 + 0.868334 w_k1_FDX_s101
                                 + 1.08384 w_k1_IBM_s101
                                 + 1.15143 w_k1_INTC_s101
                                 + 1.03824 w_k1_KO_s101 + 1.8037 w_k1_MAT_s101
                                 + 0.943453 w_k1_MSFT_s101
                                 + 1.03122 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.976169 w_k1_ADM_s102
                                 + 0.688232 w_k1_AXP_s102
                                 + 0.932486 w_k1_DIS_s102
                                 + 0.819451 w_k1_FDX_s102
                                 + 0.907276 w_k1_IBM_s102
                                 + 0.949468 w_k1_INTC_s102
                                 + 0.818891 w_k1_KO_s102
                                 + 0.853181 w_k1_MAT_s102
                                 + 0.974782 w_k1_MSFT_s102
                                 + 0.861614 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.08737 w_k1_ADM_s102
                                 + 0.99991 w_k1_AXP_s102
                                 + 0.931949 w_k1_DIS_s102
                                 + 0.87105 w_k1_FDX_s102
                                 + 1.32182 w_k1_IBM_s102
                                 + 0.864474 w_k1_INTC_s102
                                 + 1.01929 w_k1_KO_s102 + 1.43672 w_k1_MAT_s102
                                 + 0.955619 w_k1_MSFT_s102
                                 + 0.985733 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.12446 w_k1_ADM_s102
                                 + 0.881239 w_k1_AXP_s102
                                 + 1.21006 w_k1_DIS_s102
                                 + 1.04132 w_k1_FDX_s102
                                 + 1.07015 w_k1_IBM_s102
                                 + 1.03656 w_k1_INTC_s102
                                 + 1.02981 w_k1_KO_s102 + 1.05349 w_k1_MAT_s102
                                 + 1.07751 w_k1_MSFT_s102
                                 + 0.95388 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.32144 w_k1_ADM_s102
                                 + 1.1176 w_k1_AXP_s102 + 1.11644 w_k1_DIS_s102
                                 + 1.10621 w_k1_FDX_s102
                                 + 0.890136 w_k1_IBM_s102
                                 + 1.00854 w_k1_INTC_s102
                                 + 0.91864 w_k1_KO_s102
                                 + 0.946378 w_k1_MAT_s102
                                 + 1.06655 w_k1_MSFT_s102
                                 + 1.09264 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.2966 w_k1_ADM_s102
                                 + 1.36621 w_k1_AXP_s102
                                 + 1.70365 w_k1_DIS_s102
                                 + 1.21034 w_k1_FDX_s102
                                 + 1.38165 w_k1_IBM_s102
                                 + 0.995783 w_k1_INTC_s102
                                 + 1.35578 w_k1_KO_s102 + 1.69652 w_k1_MAT_s102
                                 + 1.21841 w_k1_MSFT_s102 + 1.1426 w_k1_PG_s102
                                 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.39641 w_k1_ADM_s102
                                 + 1.35018 w_k1_AXP_s102
                                 + 1.06575 w_k1_DIS_s102
                                 + 1.25282 w_k1_FDX_s102
                                 + 0.864473 w_k1_IBM_s102
                                 + 1.19045 w_k1_INTC_s102
                                 + 0.934817 w_k1_KO_s102
                                 + 0.943996 w_k1_MAT_s102
                                 + 1.04779 w_k1_MSFT_s102 + 1.0443 w_k1_PG_s102
                                 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.04404 w_k1_ADM_s102
                                 + 0.65329 w_k1_AXP_s102
                                 + 0.76347 w_k1_DIS_s102
                                 + 0.872687 w_k1_FDX_s102
                                 + 0.615998 w_k1_IBM_s102
                                 + 0.935463 w_k1_INTC_s102
                                 + 0.605505 w_k1_KO_s102
                                 + 0.346318 w_k1_MAT_s102
                                 + 0.865369 w_k1_MSFT_s102
                                 + 0.751978 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.950336 w_k1_ADM_s102
                                 + 0.930343 w_k1_AXP_s102
                                 + 0.649598 w_k1_DIS_s102
                                 + 0.848787 w_k1_FDX_s102
                                 + 0.966156 w_k1_IBM_s102
                                 + 0.951529 w_k1_INTC_s102
                                 + 0.651405 w_k1_KO_s102
                                 + 0.862509 w_k1_MAT_s102
                                 + 0.835986 w_k1_MSFT_s102
                                 + 0.843114 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.14002 w_k1_ADM_s103
                                 + 1.29063 w_k1_AXP_s103
                                 + 1.20746 w_k1_DIS_s103
                                 + 1.11341 w_k1_FDX_s103
                                 + 1.46742 w_k1_IBM_s103
                                 + 0.915208 w_k1_INTC_s103 + 1.214 w_k1_KO_s103
                                 + 1.4727 w_k1_MAT_s103 + 1.04 w_k1_MSFT_s103
                                 + 1.04904 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.14608 w_k1_ADM_s103
                                 + 0.965737 w_k1_AXP_s103
                                 + 0.88045 w_k1_DIS_s103
                                 + 1.00754 w_k1_FDX_s103
                                 + 0.698676 w_k1_IBM_s103
                                 + 1.00414 w_k1_INTC_s103
                                 + 0.805216 w_k1_KO_s103
                                 + 0.669716 w_k1_MAT_s103
                                 + 1.06106 w_k1_MSFT_s103
                                 + 0.867498 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.785367 w_k1_ADM_s103
                                 + 0.799034 w_k1_AXP_s103
                                 + 0.632162 w_k1_DIS_s103
                                 + 0.758587 w_k1_FDX_s103
                                 + 0.893121 w_k1_IBM_s103
                                 + 0.796131 w_k1_INTC_s103
                                 + 0.710319 w_k1_KO_s103
                                 + 0.848261 w_k1_MAT_s103
                                 + 0.860852 w_k1_MSFT_s103
                                 + 0.743203 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.14514 w_k1_ADM_s103
                                 + 0.89837 w_k1_AXP_s103
                                 + 1.22613 w_k1_DIS_s103 + 1.0962 w_k1_FDX_s103
                                 + 0.874025 w_k1_IBM_s103
                                 + 1.02727 w_k1_INTC_s103 + 1.0002 w_k1_KO_s103
                                 + 0.861699 w_k1_MAT_s103
                                 + 1.26077 w_k1_MSFT_s103
                                 + 1.10926 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.788852 w_k1_ADM_s103
                                 + 1.0169 w_k1_AXP_s103
                                 + 0.919379 w_k1_DIS_s103
                                 + 0.943833 w_k1_FDX_s103
                                 + 0.936775 w_k1_IBM_s103
                                 + 1.06002 w_k1_INTC_s103
                                 + 0.855651 w_k1_KO_s103
                                 + 1.05862 w_k1_MAT_s103
                                 + 0.905119 w_k1_MSFT_s103
                                 + 0.897504 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.06037 w_k1_ADM_s103
                                 + 1.56832 w_k1_AXP_s103
                                 + 1.40004 w_k1_DIS_s103
                                 + 1.34623 w_k1_FDX_s103
                                 + 1.06916 w_k1_IBM_s103
                                 + 1.32628 w_k1_INTC_s103
                                 + 0.880963 w_k1_KO_s103
                                 + 1.34744 w_k1_MAT_s103
                                 + 1.24038 w_k1_MSFT_s103
                                 + 1.00357 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.772128 w_k1_ADM_s103
                                 + 1.08485 w_k1_AXP_s103
                                 + 0.941054 w_k1_DIS_s103
                                 + 0.919939 w_k1_FDX_s103
                                 + 1.25929 w_k1_IBM_s103
                                 + 0.959901 w_k1_INTC_s103
                                 + 1.03618 w_k1_KO_s103 + 1.75955 w_k1_MAT_s103
                                 + 0.988141 w_k1_MSFT_s103
                                 + 0.988552 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.12841 w_k1_ADM_s103
                                 + 1.34125 w_k1_AXP_s103
                                 + 1.55129 w_k1_DIS_s103
                                 + 1.24524 w_k1_FDX_s103
                                 + 1.24747 w_k1_IBM_s103
                                 + 1.12844 w_k1_INTC_s103
                                 + 1.19808 w_k1_KO_s103 + 1.69691 w_k1_MAT_s103
                                 + 1.22635 w_k1_MSFT_s103
                                 + 1.09736 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.903262 w_k1_ADM_s104
                                 + 0.851213 w_k1_AXP_s104
                                 + 0.854864 w_k1_DIS_s104
                                 + 0.929854 w_k1_FDX_s104
                                 + 0.864966 w_k1_IBM_s104
                                 + 0.789765 w_k1_INTC_s104
                                 + 1.02129 w_k1_KO_s104
                                 + 0.887888 w_k1_MAT_s104
                                 + 0.8194 w_k1_MSFT_s104 + 0.99951 w_k1_PG_s104
                                 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.684789 w_k1_ADM_s104
                                 + 0.793561 w_k1_AXP_s104
                                 + 0.741039 w_k1_DIS_s104
                                 + 0.763285 w_k1_FDX_s104
                                 + 0.997747 w_k1_IBM_s104
                                 + 0.735176 w_k1_INTC_s104
                                 + 0.933159 w_k1_KO_s104
                                 + 1.33184 w_k1_MAT_s104
                                 + 0.755803 w_k1_MSFT_s104
                                 + 0.956301 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.888866 w_k1_ADM_s104
                                 + 1.05538 w_k1_AXP_s104
                                 + 0.888483 w_k1_DIS_s104
                                 + 0.957406 w_k1_FDX_s104
                                 + 1.33206 w_k1_IBM_s104
                                 + 0.934037 w_k1_INTC_s104
                                 + 1.08984 w_k1_KO_s104 + 1.30884 w_k1_MAT_s104
                                 + 0.937428 w_k1_MSFT_s104
                                 + 1.1003 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.04745 w_k1_ADM_s104
                                 + 0.937354 w_k1_AXP_s104
                                 + 0.925694 w_k1_DIS_s104
                                 + 1.05101 w_k1_FDX_s104
                                 + 0.597894 w_k1_IBM_s104
                                 + 0.824612 w_k1_INTC_s104
                                 + 0.837929 w_k1_KO_s104
                                 + 0.918631 w_k1_MAT_s104
                                 + 0.952099 w_k1_MSFT_s104
                                 + 1.00666 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.0608 w_k1_ADM_s104
                                 + 1.37244 w_k1_AXP_s104
                                 + 0.868246 w_k1_DIS_s104
                                 + 1.11061 w_k1_FDX_s104
                                 + 0.750032 w_k1_IBM_s104
                                 + 0.969657 w_k1_INTC_s104
                                 + 0.881964 w_k1_KO_s104
                                 + 1.54904 w_k1_MAT_s104
                                 + 0.92974 w_k1_MSFT_s104
                                 + 0.929747 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.19263 w_k1_ADM_s104
                                 + 1.39818 w_k1_AXP_s104
                                 + 1.68909 w_k1_DIS_s104
                                 + 1.25951 w_k1_FDX_s104
                                 + 1.51916 w_k1_IBM_s104
                                 + 1.02907 w_k1_INTC_s104
                                 + 1.43707 w_k1_KO_s104 + 1.5385 w_k1_MAT_s104
                                 + 1.14145 w_k1_MSFT_s104
                                 + 1.21773 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.793542 w_k1_ADM_s104
                                 + 0.767041 w_k1_AXP_s104
                                 + 0.572243 w_k1_DIS_s104
                                 + 0.718393 w_k1_FDX_s104
                                 + 0.683481 w_k1_IBM_s104
                                 + 0.750168 w_k1_INTC_s104
                                 + 0.696971 w_k1_KO_s104
                                 + 0.677848 w_k1_MAT_s104
                                 + 0.768983 w_k1_MSFT_s104
                                 + 0.800396 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.20688 w_k1_ADM_s104
                                 + 1.22973 w_k1_AXP_s104 + 1.3208 w_k1_DIS_s104
                                 + 1.14244 w_k1_FDX_s104
                                 + 1.11283 w_k1_IBM_s104
                                 + 1.02967 w_k1_INTC_s104
                                 + 1.25844 w_k1_KO_s104 + 1.50568 w_k1_MAT_s104
                                 + 1.08768 w_k1_MSFT_s104
                                 + 1.18408 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.04662 w_k1_ADM_s105
                                 + 0.791232 w_k1_AXP_s105
                                 + 1.18161 w_k1_DIS_s105
                                 + 0.941276 w_k1_FDX_s105
                                 + 1.14688 w_k1_IBM_s105
                                 + 0.927553 w_k1_INTC_s105
                                 + 1.1536 w_k1_KO_s105 + 1.15631 w_k1_MAT_s105
                                 + 1.07214 w_k1_MSFT_s105
                                 + 1.04386 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.05298 w_k1_ADM_s105
                                 + 1.29901 w_k1_AXP_s105
                                 + 1.63276 w_k1_DIS_s105
                                 + 1.23646 w_k1_FDX_s105
                                 + 1.65982 w_k1_IBM_s105
                                 + 1.23115 w_k1_INTC_s105
                                 + 1.43985 w_k1_KO_s105 + 1.80784 w_k1_MAT_s105
                                 + 1.32901 w_k1_MSFT_s105
                                 + 1.23092 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.10486 w_k1_ADM_s105
                                 + 0.931549 w_k1_AXP_s105
                                 + 1.04965 w_k1_DIS_s105
                                 + 0.871522 w_k1_FDX_s105
                                 + 1.47329 w_k1_IBM_s105
                                 + 0.946261 w_k1_INTC_s105
                                 + 1.17353 w_k1_KO_s105 + 1.22341 w_k1_MAT_s105
                                 + 0.851916 w_k1_MSFT_s105
                                 + 1.10336 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.29444 w_k1_ADM_s105
                                 + 1.18032 w_k1_AXP_s105
                                 + 1.08771 w_k1_DIS_s105
                                 + 1.05903 w_k1_FDX_s105
                                 + 0.878928 w_k1_IBM_s105
                                 + 0.978484 w_k1_INTC_s105
                                 + 0.967246 w_k1_KO_s105
                                 + 0.827108 w_k1_MAT_s105
                                 + 0.881278 w_k1_MSFT_s105
                                 + 1.01563 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.23229 w_k1_ADM_s105
                                 + 0.792821 w_k1_AXP_s105
                                 + 1.24809 w_k1_DIS_s105
                                 + 1.02019 w_k1_FDX_s105
                                 + 0.897143 w_k1_IBM_s105
                                 + 1.00253 w_k1_INTC_s105
                                 + 0.937678 w_k1_KO_s105
                                 + 0.586964 w_k1_MAT_s105
                                 + 0.987282 w_k1_MSFT_s105
                                 + 0.996091 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.930008 w_k1_ADM_s105
                                 + 0.807306 w_k1_AXP_s105
                                 + 0.698354 w_k1_DIS_s105
                                 + 0.7725 w_k1_FDX_s105
                                 + 0.892525 w_k1_IBM_s105
                                 + 0.837043 w_k1_INTC_s105
                                 + 0.783274 w_k1_KO_s105
                                 + 0.91922 w_k1_MAT_s105
                                 + 0.763313 w_k1_MSFT_s105
                                 + 0.941673 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.38858 w_k1_ADM_s105
                                 + 1.24905 w_k1_AXP_s105 + 1.4023 w_k1_DIS_s105
                                 + 1.20955 w_k1_FDX_s105
                                 + 1.22601 w_k1_IBM_s105
                                 + 0.952488 w_k1_INTC_s105
                                 + 1.08807 w_k1_KO_s105 + 1.72221 w_k1_MAT_s105
                                 + 1.07344 w_k1_MSFT_s105
                                 + 1.11721 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.07482 w_k1_ADM_s105
                                 + 0.96295 w_k1_AXP_s105
                                 + 0.998685 w_k1_DIS_s105
                                 + 1.08025 w_k1_FDX_s105
                                 + 0.640293 w_k1_IBM_s105
                                 + 0.991537 w_k1_INTC_s105
                                 + 0.769622 w_k1_KO_s105
                                 + 0.446274 w_k1_MAT_s105
                                 + 0.94766 w_k1_MSFT_s105
                                 + 0.911712 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.1158 w_k1_ADM_s106
                                 + 1.47129 w_k1_AXP_s106
                                 + 1.02928 w_k1_DIS_s106 + 1.1764 w_k1_FDX_s106
                                 + 1.14015 w_k1_IBM_s106
                                 + 1.28601 w_k1_INTC_s106
                                 + 1.01152 w_k1_KO_s106 + 1.21396 w_k1_MAT_s106
                                 + 1.06935 w_k1_MSFT_s106
                                 + 1.13916 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.29155 w_k1_ADM_s106
                                 + 0.802656 w_k1_AXP_s106
                                 + 0.783776 w_k1_DIS_s106
                                 + 0.909394 w_k1_FDX_s106
                                 + 0.691879 w_k1_IBM_s106
                                 + 1.02891 w_k1_INTC_s106
                                 + 0.784842 w_k1_KO_s106
                                 + 0.591508 w_k1_MAT_s106
                                 + 0.880227 w_k1_MSFT_s106
                                 + 0.933547 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.12529 w_k1_ADM_s106
                                 + 1.07379 w_k1_AXP_s106
                                 + 0.865671 w_k1_DIS_s106
                                 + 0.894693 w_k1_FDX_s106
                                 + 1.19506 w_k1_IBM_s106
                                 + 1.00833 w_k1_INTC_s106
                                 + 1.07177 w_k1_KO_s106 + 1.40862 w_k1_MAT_s106
                                 + 1.01136 w_k1_MSFT_s106
                                 + 1.04271 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.19426 w_k1_ADM_s106
                                 + 1.03825 w_k1_AXP_s106
                                 + 1.01858 w_k1_DIS_s106
                                 + 1.11131 w_k1_FDX_s106
                                 + 0.822149 w_k1_IBM_s106
                                 + 1.03439 w_k1_INTC_s106
                                 + 0.991501 w_k1_KO_s106
                                 + 0.774711 w_k1_MAT_s106
                                 + 1.02102 w_k1_MSFT_s106
                                 + 0.991975 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.972188 w_k1_ADM_s106
                                 + 0.848087 w_k1_AXP_s106
                                 + 0.806397 w_k1_DIS_s106
                                 + 0.880245 w_k1_FDX_s106
                                 + 0.74695 w_k1_IBM_s106
                                 + 0.953321 w_k1_INTC_s106
                                 + 0.82819 w_k1_KO_s106
                                 + 0.808267 w_k1_MAT_s106
                                 + 0.968048 w_k1_MSFT_s106
                                 + 0.936774 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.865159 w_k1_ADM_s106
                                 + 0.764737 w_k1_AXP_s106
                                 + 0.653373 w_k1_DIS_s106
                                 + 0.696605 w_k1_FDX_s106
                                 + 0.905539 w_k1_IBM_s106
                                 + 1.00598 w_k1_INTC_s106
                                 + 0.815477 w_k1_KO_s106
                                 + 0.861596 w_k1_MAT_s106
                                 + 0.850061 w_k1_MSFT_s106
                                 + 0.930045 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.33962 w_k1_ADM_s106
                                 + 1.04473 w_k1_AXP_s106
                                 + 1.30083 w_k1_DIS_s106
                                 + 1.03551 w_k1_FDX_s106 + 1.4172 w_k1_IBM_s106
                                 + 1.37618 w_k1_INTC_s106
                                 + 1.19343 w_k1_KO_s106 + 1.02031 w_k1_MAT_s106
                                 + 1.22902 w_k1_MSFT_s106
                                 + 1.15281 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.28592 w_k1_ADM_s106
                                 + 1.79042 w_k1_AXP_s106
                                 + 1.63747 w_k1_DIS_s106
                                 + 1.40972 w_k1_FDX_s106
                                 + 1.19345 w_k1_IBM_s106
                                 + 1.41245 w_k1_INTC_s106
                                 + 1.33345 w_k1_KO_s106 + 1.57788 w_k1_MAT_s106
                                 + 1.36134 w_k1_MSFT_s106
                                 + 1.31247 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.982984 w_k1_ADM_s107
                                 + 0.770705 w_k1_AXP_s107
                                 + 0.68097 w_k1_DIS_s107
                                 + 0.834458 w_k1_FDX_s107
                                 + 0.763257 w_k1_IBM_s107
                                 + 0.958479 w_k1_INTC_s107
                                 + 0.695615 w_k1_KO_s107
                                 + 0.563263 w_k1_MAT_s107
                                 + 0.78744 w_k1_MSFT_s107
                                 + 0.933167 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 2.01253 w_k1_ADM_s107
                                 + 1.7506 w_k1_AXP_s107 + 1.72309 w_k1_DIS_s107
                                 + 1.46878 w_k1_FDX_s107
                                 + 1.21987 w_k1_IBM_s107
                                 + 1.20732 w_k1_INTC_s107
                                 + 1.41171 w_k1_KO_s107 + 1.32482 w_k1_MAT_s107
                                 + 1.31388 w_k1_MSFT_s107
                                 + 1.33075 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.835595 w_k1_ADM_s107
                                 + 0.731662 w_k1_AXP_s107
                                 + 1.02451 w_k1_DIS_s107
                                 + 0.832956 w_k1_FDX_s107
                                 + 1.63032 w_k1_IBM_s107
                                 + 0.897711 w_k1_INTC_s107
                                 + 1.09349 w_k1_KO_s107 + 1.43278 w_k1_MAT_s107
                                 + 0.821069 w_k1_MSFT_s107
                                 + 0.97801 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.07817 w_k1_ADM_s107
                                 + 1.06305 w_k1_AXP_s107
                                 + 1.12953 w_k1_DIS_s107
                                 + 1.03863 w_k1_FDX_s107
                                 + 1.38055 w_k1_IBM_s107
                                 + 0.988263 w_k1_INTC_s107
                                 + 1.12506 w_k1_KO_s107 + 1.25547 w_k1_MAT_s107
                                 + 0.957447 w_k1_MSFT_s107
                                 + 1.07302 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.14444 w_k1_ADM_s107
                                 + 1.064 w_k1_AXP_s107 + 1.0172 w_k1_DIS_s107
                                 + 1.08629 w_k1_FDX_s107
                                 + 0.832708 w_k1_IBM_s107
                                 + 1.13801 w_k1_INTC_s107
                                 + 0.763806 w_k1_KO_s107
                                 + 0.707144 w_k1_MAT_s107
                                 + 1.0832 w_k1_MSFT_s107
                                 + 0.939064 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.2861 w_k1_ADM_s107
                                 + 1.0646 w_k1_AXP_s107 + 1.78074 w_k1_DIS_s107
                                 + 1.38108 w_k1_FDX_s107
                                 + 0.991436 w_k1_IBM_s107
                                 + 1.55731 w_k1_INTC_s107
                                 + 1.42286 w_k1_KO_s107
                                 + 0.646272 w_k1_MAT_s107
                                 + 1.41377 w_k1_MSFT_s107 + 1.2505 w_k1_PG_s107
                                 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.22582 w_k1_ADM_s107
                                 + 1.32653 w_k1_AXP_s107
                                 + 1.52496 w_k1_DIS_s107
                                 + 1.18704 w_k1_FDX_s107 + 1.6798 w_k1_IBM_s107
                                 + 1.17316 w_k1_INTC_s107
                                 + 1.25715 w_k1_KO_s107 + 1.54296 w_k1_MAT_s107
                                 + 1.17535 w_k1_MSFT_s107 + 1.1441 w_k1_PG_s107
                                 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.908988 w_k1_ADM_s107
                                 + 0.931322 w_k1_AXP_s107
                                 + 0.770562 w_k1_DIS_s107
                                 + 0.857924 w_k1_FDX_s107
                                 + 1.15432 w_k1_IBM_s107
                                 + 0.811405 w_k1_INTC_s107
                                 + 0.845788 w_k1_KO_s107
                                 + 0.886848 w_k1_MAT_s107
                                 + 0.709223 w_k1_MSFT_s107
                                 + 0.986895 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.02976 w_k1_ADM_s110
                                 + 0.707116 w_k1_AXP_s110
                                 + 0.75063 w_k1_DIS_s110
                                 + 0.922292 w_k1_FDX_s110
                                 + 0.779853 w_k1_IBM_s110
                                 + 0.798868 w_k1_INTC_s110
                                 + 0.865912 w_k1_KO_s110
                                 + 0.957742 w_k1_MAT_s110
                                 + 0.961357 w_k1_MSFT_s110
                                 + 0.840672 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.877942 w_k1_ADM_s110
                                 + 0.66542 w_k1_AXP_s110
                                 + 0.779759 w_k1_DIS_s110
                                 + 0.733029 w_k1_FDX_s110
                                 + 1.01399 w_k1_IBM_s110
                                 + 0.674765 w_k1_INTC_s110
                                 + 0.84681 w_k1_KO_s110
                                 + 0.996021 w_k1_MAT_s110
                                 + 0.82969 w_k1_MSFT_s110
                                 + 0.802157 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.758036 w_k1_ADM_s110
                                 + 0.678607 w_k1_AXP_s110
                                 + 0.643801 w_k1_DIS_s110
                                 + 0.758009 w_k1_FDX_s110
                                 + 0.680108 w_k1_IBM_s110
                                 + 0.725421 w_k1_INTC_s110
                                 + 0.735638 w_k1_KO_s110
                                 + 0.592293 w_k1_MAT_s110
                                 + 0.781041 w_k1_MSFT_s110
                                 + 0.787257 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.41235 w_k1_ADM_s110
                                 + 1.22675 w_k1_AXP_s110
                                 + 0.953005 w_k1_DIS_s110
                                 + 1.04328 w_k1_FDX_s110
                                 + 0.979709 w_k1_IBM_s110
                                 + 0.836537 w_k1_INTC_s110
                                 + 0.898846 w_k1_KO_s110
                                 + 1.23125 w_k1_MAT_s110
                                 + 0.874246 w_k1_MSFT_s110
                                 + 1.08188 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.5739 w_k1_ADM_s110
                                 + 1.30303 w_k1_AXP_s110
                                 + 1.70914 w_k1_DIS_s110 + 1.321 w_k1_FDX_s110
                                 + 1.37106 w_k1_IBM_s110
                                 + 0.894506 w_k1_INTC_s110
                                 + 1.34815 w_k1_KO_s110 + 1.54107 w_k1_MAT_s110
                                 + 1.31608 w_k1_MSFT_s110 + 1.1371 w_k1_PG_s110
                                 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.17002 w_k1_ADM_s110
                                 + 0.820693 w_k1_AXP_s110
                                 + 1.20744 w_k1_DIS_s110
                                 + 1.02554 w_k1_FDX_s110
                                 + 1.22994 w_k1_IBM_s110
                                 + 0.903447 w_k1_INTC_s110
                                 + 1.25094 w_k1_KO_s110
                                 + 0.813534 w_k1_MAT_s110
                                 + 1.14486 w_k1_MSFT_s110
                                 + 1.03302 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.07746 w_k1_ADM_s110
                                 + 0.975683 w_k1_AXP_s110
                                 + 1.12911 w_k1_DIS_s110
                                 + 1.07821 w_k1_FDX_s110
                                 + 1.04213 w_k1_IBM_s110
                                 + 0.958706 w_k1_INTC_s110
                                 + 1.11057 w_k1_KO_s110 + 1.05675 w_k1_MAT_s110
                                 + 1.14743 w_k1_MSFT_s110
                                 + 0.998838 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.94295 w_k1_ADM_s110
                                 + 0.941671 w_k1_AXP_s110
                                 + 0.935014 w_k1_DIS_s110
                                 + 0.905789 w_k1_FDX_s110
                                 + 1.45845 w_k1_IBM_s110
                                 + 0.906812 w_k1_INTC_s110
                                 + 1.05347 w_k1_KO_s110 + 1.13347 w_k1_MAT_s110
                                 + 0.837611 w_k1_MSFT_s110
                                 + 1.03799 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.843306 w_k1_ADM_s111
                                 + 0.822876 w_k1_AXP_s111
                                 + 0.970128 w_k1_DIS_s111
                                 + 0.981562 w_k1_FDX_s111
                                 + 0.986542 w_k1_IBM_s111
                                 + 0.866406 w_k1_INTC_s111
                                 + 0.982774 w_k1_KO_s111
                                 + 1.41453 w_k1_MAT_s111
                                 + 1.00925 w_k1_MSFT_s111
                                 + 1.06071 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.854102 w_k1_ADM_s111
                                 + 1.04459 w_k1_AXP_s111
                                 + 0.931705 w_k1_DIS_s111
                                 + 0.817399 w_k1_FDX_s111
                                 + 1.62238 w_k1_IBM_s111
                                 + 1.14332 w_k1_INTC_s111
                                 + 1.26254 w_k1_KO_s111 + 1.76828 w_k1_MAT_s111
                                 + 1.00514 w_k1_MSFT_s111
                                 + 1.09669 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.636133 w_k1_ADM_s111
                                 + 0.88021 w_k1_AXP_s111
                                 + 0.641999 w_k1_DIS_s111
                                 + 0.719127 w_k1_FDX_s111
                                 + 1.09903 w_k1_IBM_s111
                                 + 0.859251 w_k1_INTC_s111
                                 + 0.858722 w_k1_KO_s111
                                 + 1.35258 w_k1_MAT_s111
                                 + 0.83444 w_k1_MSFT_s111
                                 + 0.79939 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.02034 w_k1_ADM_s111
                                 + 1.15177 w_k1_AXP_s111
                                 + 1.08064 w_k1_DIS_s111
                                 + 1.02315 w_k1_FDX_s111 + 1.1803 w_k1_IBM_s111
                                 + 0.970346 w_k1_INTC_s111
                                 + 1.1957 w_k1_KO_s111 + 1.51196 w_k1_MAT_s111
                                 + 1.18822 w_k1_MSFT_s111
                                 + 1.23735 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.914344 w_k1_ADM_s111
                                 + 0.729455 w_k1_AXP_s111
                                 + 0.905351 w_k1_DIS_s111
                                 + 0.912314 w_k1_FDX_s111
                                 + 0.673684 w_k1_IBM_s111
                                 + 0.940789 w_k1_INTC_s111
                                 + 0.823758 w_k1_KO_s111
                                 + 0.643759 w_k1_MAT_s111
                                 + 1.16395 w_k1_MSFT_s111
                                 + 0.895715 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.07268 w_k1_ADM_s111
                                 + 1.04937 w_k1_AXP_s111
                                 + 1.36073 w_k1_DIS_s111
                                 + 1.28104 w_k1_FDX_s111
                                 + 0.878951 w_k1_IBM_s111
                                 + 1.22055 w_k1_INTC_s111
                                 + 1.15088 w_k1_KO_s111
                                 + 0.795826 w_k1_MAT_s111
                                 + 1.44891 w_k1_MSFT_s111
                                 + 1.09503 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.868039 w_k1_ADM_s111
                                 + 0.84134 w_k1_AXP_s111
                                 + 0.677169 w_k1_DIS_s111
                                 + 0.796797 w_k1_FDX_s111
                                 + 0.813718 w_k1_IBM_s111
                                 + 0.768934 w_k1_INTC_s111
                                 + 0.836729 w_k1_KO_s111 + 1.0888 w_k1_MAT_s111
                                 + 0.817954 w_k1_MSFT_s111
                                 + 0.879419 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.949447 w_k1_ADM_s111
                                 + 1.60819 w_k1_AXP_s111 + 1.5586 w_k1_DIS_s111
                                 + 1.29459 w_k1_FDX_s111
                                 + 1.38746 w_k1_IBM_s111
                                 + 1.28303 w_k1_INTC_s111
                                 + 1.49615 w_k1_KO_s111 + 1.80717 w_k1_MAT_s111
                                 + 1.31792 w_k1_MSFT_s111
                                 + 1.25908 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.851572 w_k1_ADM_s112
                                 + 0.913921 w_k1_AXP_s112
                                 + 0.869932 w_k1_DIS_s112
                                 + 0.831664 w_k1_FDX_s112
                                 + 1.08562 w_k1_IBM_s112
                                 + 0.912667 w_k1_INTC_s112
                                 + 0.884537 w_k1_KO_s112
                                 + 0.804545 w_k1_MAT_s112
                                 + 0.864563 w_k1_MSFT_s112
                                 + 0.988637 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.26542 w_k1_ADM_s112
                                 + 1.31017 w_k1_AXP_s112
                                 + 1.10102 w_k1_DIS_s112
                                 + 1.06866 w_k1_FDX_s112
                                 + 0.972066 w_k1_IBM_s112
                                 + 0.977893 w_k1_INTC_s112
                                 + 1.02275 w_k1_KO_s112
                                 + 0.927021 w_k1_MAT_s112
                                 + 0.913343 w_k1_MSFT_s112
                                 + 0.991628 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.753967 w_k1_ADM_s112
                                 + 0.698329 w_k1_AXP_s112
                                 + 0.636236 w_k1_DIS_s112
                                 + 0.744746 w_k1_FDX_s112
                                 + 0.690397 w_k1_IBM_s112
                                 + 0.702595 w_k1_INTC_s112
                                 + 0.686234 w_k1_KO_s112
                                 + 0.757856 w_k1_MAT_s112
                                 + 0.786413 w_k1_MSFT_s112
                                 + 0.801379 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.02596 w_k1_ADM_s112
                                 + 1.02452 w_k1_AXP_s112
                                 + 0.956113 w_k1_DIS_s112
                                 + 1.01886 w_k1_FDX_s112
                                 + 0.559314 w_k1_IBM_s112
                                 + 0.897603 w_k1_INTC_s112
                                 + 0.665015 w_k1_KO_s112
                                 + 0.679098 w_k1_MAT_s112
                                 + 0.901982 w_k1_MSFT_s112
                                 + 0.837196 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.21016 w_k1_ADM_s112
                                 + 1.52445 w_k1_AXP_s112
                                 + 1.60222 w_k1_DIS_s112
                                 + 1.25852 w_k1_FDX_s112
                                 + 1.19099 w_k1_IBM_s112
                                 + 1.07985 w_k1_INTC_s112
                                 + 1.35248 w_k1_KO_s112 + 1.8622 w_k1_MAT_s112
                                 + 1.49595 w_k1_MSFT_s112 + 1.1856 w_k1_PG_s112
                                 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.773352 w_k1_ADM_s112
                                 + 0.831191 w_k1_AXP_s112
                                 + 0.81075 w_k1_DIS_s112
                                 + 0.86523 w_k1_FDX_s112
                                 + 0.698268 w_k1_IBM_s112
                                 + 0.959555 w_k1_INTC_s112
                                 + 0.858244 w_k1_KO_s112
                                 + 0.872113 w_k1_MAT_s112
                                 + 0.927648 w_k1_MSFT_s112
                                 + 0.883913 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.27162 w_k1_ADM_s112
                                 + 1.37125 w_k1_AXP_s112
                                 + 1.41751 w_k1_DIS_s112 + 1.1534 w_k1_FDX_s112
                                 + 1.46347 w_k1_IBM_s112
                                 + 1.14313 w_k1_INTC_s112
                                 + 1.44269 w_k1_KO_s112 + 1.77637 w_k1_MAT_s112
                                 + 1.27464 w_k1_MSFT_s112
                                 + 1.19225 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.17223 w_k1_ADM_s112
                                 + 1.22765 w_k1_AXP_s112
                                 + 1.23661 w_k1_DIS_s112
                                 + 1.20423 w_k1_FDX_s112
                                 + 0.825789 w_k1_IBM_s112
                                 + 1.35902 w_k1_INTC_s112
                                 + 1.07482 w_k1_KO_s112
                                 + 0.700331 w_k1_MAT_s112
                                 + 1.20325 w_k1_MSFT_s112
                                 + 0.984126 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.850923 w_k1_ADM_s113
                                 + 0.77141 w_k1_AXP_s113
                                 + 0.742427 w_k1_DIS_s113
                                 + 0.820583 w_k1_FDX_s113
                                 + 1.2978 w_k1_IBM_s113
                                 + 0.887018 w_k1_INTC_s113
                                 + 0.995833 w_k1_KO_s113
                                 + 1.36967 w_k1_MAT_s113
                                 + 0.937352 w_k1_MSFT_s113
                                 + 0.871885 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.892904 w_k1_ADM_s113
                                 + 0.857431 w_k1_AXP_s113
                                 + 0.75735 w_k1_DIS_s113
                                 + 0.871607 w_k1_FDX_s113
                                 + 0.808999 w_k1_IBM_s113
                                 + 0.951573 w_k1_INTC_s113
                                 + 0.961056 w_k1_KO_s113
                                 + 0.717012 w_k1_MAT_s113
                                 + 0.952639 w_k1_MSFT_s113
                                 + 0.804822 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.884071 w_k1_ADM_s113
                                 + 0.74765 w_k1_AXP_s113
                                 + 0.718627 w_k1_DIS_s113
                                 + 0.816004 w_k1_FDX_s113
                                 + 0.957255 w_k1_IBM_s113
                                 + 0.905033 w_k1_INTC_s113
                                 + 1.17385 w_k1_KO_s113 + 1.04035 w_k1_MAT_s113
                                 + 0.888336 w_k1_MSFT_s113
                                 + 0.888453 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.998133 w_k1_ADM_s113
                                 + 0.802674 w_k1_AXP_s113
                                 + 0.729768 w_k1_DIS_s113
                                 + 0.869998 w_k1_FDX_s113
                                 + 0.707171 w_k1_IBM_s113
                                 + 0.889423 w_k1_INTC_s113
                                 + 0.862145 w_k1_KO_s113
                                 + 0.407833 w_k1_MAT_s113
                                 + 0.883905 w_k1_MSFT_s113
                                 + 0.950977 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.28705 w_k1_ADM_s113
                                 + 1.04743 w_k1_AXP_s113
                                 + 0.849156 w_k1_DIS_s113
                                 + 1.05391 w_k1_FDX_s113
                                 + 0.858704 w_k1_IBM_s113
                                 + 0.787092 w_k1_INTC_s113
                                 + 0.977324 w_k1_KO_s113
                                 + 1.01612 w_k1_MAT_s113
                                 + 0.915415 w_k1_MSFT_s113
                                 + 0.933922 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.15101 w_k1_ADM_s113
                                 + 1.20268 w_k1_AXP_s113 + 1.3697 w_k1_DIS_s113
                                 + 1.26642 w_k1_FDX_s113 + 1.4963 w_k1_IBM_s113
                                 + 1.07054 w_k1_INTC_s113
                                 + 1.56789 w_k1_KO_s113 + 1.84707 w_k1_MAT_s113
                                 + 1.24724 w_k1_MSFT_s113
                                 + 1.13011 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.19767 w_k1_ADM_s113
                                 + 1.20163 w_k1_AXP_s113
                                 + 1.00251 w_k1_DIS_s113
                                 + 1.02692 w_k1_FDX_s113
                                 + 1.03981 w_k1_IBM_s113
                                 + 0.954616 w_k1_INTC_s113
                                 + 1.10881 w_k1_KO_s113 + 1.15013 w_k1_MAT_s113
                                 + 1.10262 w_k1_MSFT_s113
                                 + 1.00363 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.11407 w_k1_ADM_s113
                                 + 0.992331 w_k1_AXP_s113
                                 + 1.25436 w_k1_DIS_s113
                                 + 1.24612 w_k1_FDX_s113
                                 + 0.999043 w_k1_IBM_s113
                                 + 1.14011 w_k1_INTC_s113
                                 + 1.27561 w_k1_KO_s113 + 1.05811 w_k1_MAT_s113
                                 + 1.22024 w_k1_MSFT_s113
                                 + 1.07137 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.36927 w_k1_ADM_s114
                                 + 1.28511 w_k1_AXP_s114
                                 + 1.23001 w_k1_DIS_s114
                                 + 1.29508 w_k1_FDX_s114
                                 + 1.32494 w_k1_IBM_s114
                                 + 1.14878 w_k1_INTC_s114
                                 + 1.14673 w_k1_KO_s114 + 1.45009 w_k1_MAT_s114
                                 + 1.20201 w_k1_MSFT_s114
                                 + 1.06334 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.896842 w_k1_ADM_s114
                                 + 0.885947 w_k1_AXP_s114
                                 + 1.12697 w_k1_DIS_s114
                                 + 1.21449 w_k1_FDX_s114
                                 + 1.03932 w_k1_IBM_s114
                                 + 1.12289 w_k1_INTC_s114
                                 + 1.07594 w_k1_KO_s114
                                 + 0.778365 w_k1_MAT_s114
                                 + 1.22785 w_k1_MSFT_s114
                                 + 1.03016 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.29086 w_k1_ADM_s114
                                 + 1.84222 w_k1_AXP_s114 + 1.5518 w_k1_DIS_s114
                                 + 1.40759 w_k1_FDX_s114
                                 + 1.46408 w_k1_IBM_s114
                                 + 1.29995 w_k1_INTC_s114
                                 + 1.27182 w_k1_KO_s114 + 1.7828 w_k1_MAT_s114
                                 + 1.38757 w_k1_MSFT_s114
                                 + 1.23171 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.834377 w_k1_ADM_s114
                                 + 0.807025 w_k1_AXP_s114
                                 + 0.684324 w_k1_DIS_s114
                                 + 0.733549 w_k1_FDX_s114
                                 + 1.11597 w_k1_IBM_s114
                                 + 0.774812 w_k1_INTC_s114
                                 + 0.93736 w_k1_KO_s114
                                 + 0.932713 w_k1_MAT_s114
                                 + 0.738992 w_k1_MSFT_s114
                                 + 0.922163 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.03269 w_k1_ADM_s114
                                 + 1.11809 w_k1_AXP_s114
                                 + 0.800973 w_k1_DIS_s114
                                 + 0.976804 w_k1_FDX_s114
                                 + 1.21718 w_k1_IBM_s114
                                 + 1.03453 w_k1_INTC_s114
                                 + 0.969874 w_k1_KO_s114
                                 + 1.30709 w_k1_MAT_s114
                                 + 0.909566 w_k1_MSFT_s114
                                 + 1.0155 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.05467 w_k1_ADM_s114
                                 + 1.16041 w_k1_AXP_s114
                                 + 0.847602 w_k1_DIS_s114
                                 + 1.05669 w_k1_FDX_s114
                                 + 0.944555 w_k1_IBM_s114
                                 + 1.00917 w_k1_INTC_s114
                                 + 0.834078 w_k1_KO_s114
                                 + 0.892807 w_k1_MAT_s114
                                 + 1.07116 w_k1_MSFT_s114
                                 + 1.05612 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.12211 w_k1_ADM_s114
                                 + 1.4776 w_k1_AXP_s114 + 1.19548 w_k1_DIS_s114
                                 + 1.42381 w_k1_FDX_s114
                                 + 0.766075 w_k1_IBM_s114
                                 + 1.31938 w_k1_INTC_s114
                                 + 0.841682 w_k1_KO_s114
                                 + 0.617454 w_k1_MAT_s114
                                 + 1.30348 w_k1_MSFT_s114
                                 + 0.98599 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.913496 w_k1_ADM_s114
                                 + 0.867051 w_k1_AXP_s114
                                 + 0.7491 w_k1_DIS_s114
                                 + 0.955716 w_k1_FDX_s114
                                 + 0.67825 w_k1_IBM_s114
                                 + 1.05753 w_k1_INTC_s114
                                 + 0.84834 w_k1_KO_s114
                                 + 0.652155 w_k1_MAT_s114
                                 + 1.02863 w_k1_MSFT_s114
                                 + 0.894375 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.903577 w_k1_ADM_s115
                                 + 0.704037 w_k1_AXP_s115
                                 + 0.565014 w_k1_DIS_s115
                                 + 0.685194 w_k1_FDX_s115
                                 + 0.740466 w_k1_IBM_s115
                                 + 0.635544 w_k1_INTC_s115
                                 + 0.686488 w_k1_KO_s115
                                 + 0.961204 w_k1_MAT_s115
                                 + 0.716996 w_k1_MSFT_s115
                                 + 0.804543 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.38304 w_k1_ADM_s115
                                 + 1.61885 w_k1_AXP_s115
                                 + 1.42449 w_k1_DIS_s115
                                 + 1.21791 w_k1_FDX_s115
                                 + 1.36805 w_k1_IBM_s115
                                 + 1.09737 w_k1_INTC_s115
                                 + 1.25415 w_k1_KO_s115 + 1.71578 w_k1_MAT_s115
                                 + 1.10229 w_k1_MSFT_s115
                                 + 1.24317 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.732636 w_k1_ADM_s115
                                 + 0.877334 w_k1_AXP_s115
                                 + 0.722495 w_k1_DIS_s115
                                 + 0.746506 w_k1_FDX_s115
                                 + 1.05734 w_k1_IBM_s115
                                 + 0.832069 w_k1_INTC_s115
                                 + 0.784952 w_k1_KO_s115
                                 + 1.11717 w_k1_MAT_s115
                                 + 0.767751 w_k1_MSFT_s115
                                 + 0.888752 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.837575 w_k1_ADM_s115
                                 + 0.999049 w_k1_AXP_s115
                                 + 1.11373 w_k1_DIS_s115
                                 + 0.927938 w_k1_FDX_s115
                                 + 1.2047 w_k1_IBM_s115
                                 + 0.907728 w_k1_INTC_s115
                                 + 1.14784 w_k1_KO_s115 + 1.61099 w_k1_MAT_s115
                                 + 1.05623 w_k1_MSFT_s115
                                 + 0.993589 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.03981 w_k1_ADM_s115
                                 + 1.23712 w_k1_AXP_s115
                                 + 0.985118 w_k1_DIS_s115
                                 + 1.08513 w_k1_FDX_s115
                                 + 0.754222 w_k1_IBM_s115
                                 + 1.0383 w_k1_INTC_s115
                                 + 0.883607 w_k1_KO_s115
                                 + 0.975666 w_k1_MAT_s115
                                 + 0.953407 w_k1_MSFT_s115
                                 + 0.900584 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.12481 w_k1_ADM_s115
                                 + 0.811385 w_k1_AXP_s115
                                 + 0.991844 w_k1_DIS_s115
                                 + 1.0258 w_k1_FDX_s115
                                 + 0.801188 w_k1_IBM_s115
                                 + 1.1097 w_k1_INTC_s115 + 1.03922 w_k1_KO_s115
                                 + 0.65605 w_k1_MAT_s115
                                 + 0.971031 w_k1_MSFT_s115
                                 + 0.857119 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.16228 w_k1_ADM_s115
                                 + 1.09302 w_k1_AXP_s115
                                 + 0.919895 w_k1_DIS_s115
                                 + 1.04581 w_k1_FDX_s115
                                 + 1.02914 w_k1_IBM_s115
                                 + 0.952486 w_k1_INTC_s115
                                 + 0.982748 w_k1_KO_s115
                                 + 1.03271 w_k1_MAT_s115
                                 + 0.995921 w_k1_MSFT_s115
                                 + 0.934016 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.07376 w_k1_ADM_s115
                                 + 1.00252 w_k1_AXP_s115
                                 + 0.760114 w_k1_DIS_s115
                                 + 0.866626 w_k1_FDX_s115
                                 + 1.30977 w_k1_IBM_s115
                                 + 0.873225 w_k1_INTC_s115
                                 + 0.992965 w_k1_KO_s115
                                 + 1.61877 w_k1_MAT_s115
                                 + 0.797559 w_k1_MSFT_s115
                                 + 0.874987 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.876317 w_k1_ADM_s116
                                 + 0.715728 w_k1_AXP_s116
                                 + 0.742755 w_k1_DIS_s116
                                 + 0.899747 w_k1_FDX_s116
                                 + 0.567045 w_k1_IBM_s116
                                 + 0.930808 w_k1_INTC_s116
                                 + 0.760022 w_k1_KO_s116
                                 + 0.758735 w_k1_MAT_s116
                                 + 1.00515 w_k1_MSFT_s116
                                 + 0.947069 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.17801 w_k1_ADM_s116
                                 + 1.24354 w_k1_AXP_s116
                                 + 0.856487 w_k1_DIS_s116
                                 + 0.931999 w_k1_FDX_s116
                                 + 1.35775 w_k1_IBM_s116
                                 + 0.786082 w_k1_INTC_s116
                                 + 1.13187 w_k1_KO_s116 + 1.26973 w_k1_MAT_s116
                                 + 0.782585 w_k1_MSFT_s116
                                 + 1.05025 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.30506 w_k1_ADM_s116
                                 + 1.37251 w_k1_AXP_s116
                                 + 1.23845 w_k1_DIS_s116
                                 + 1.23281 w_k1_FDX_s116
                                 + 1.11334 w_k1_IBM_s116
                                 + 1.16548 w_k1_INTC_s116
                                 + 1.07508 w_k1_KO_s116 + 1.35198 w_k1_MAT_s116
                                 + 1.12271 w_k1_MSFT_s116
                                 + 1.36448 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.18902 w_k1_ADM_s116
                                 + 1.0649 w_k1_AXP_s116 + 1.04934 w_k1_DIS_s116
                                 + 0.998191 w_k1_FDX_s116
                                 + 1.15776 w_k1_IBM_s116
                                 + 1.07268 w_k1_INTC_s116
                                 + 1.19541 w_k1_KO_s116 + 0.91102 w_k1_MAT_s116
                                 + 1.10494 w_k1_MSFT_s116
                                 + 1.16873 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.14617 w_k1_ADM_s116
                                 + 0.967133 w_k1_AXP_s116
                                 + 0.875812 w_k1_DIS_s116
                                 + 1.06843 w_k1_FDX_s116
                                 + 0.717909 w_k1_IBM_s116
                                 + 0.855509 w_k1_INTC_s116
                                 + 0.761318 w_k1_KO_s116
                                 + 0.968308 w_k1_MAT_s116
                                 + 0.947757 w_k1_MSFT_s116
                                 + 0.990061 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.19287 w_k1_ADM_s116
                                 + 1.08836 w_k1_AXP_s116
                                 + 1.45627 w_k1_DIS_s116
                                 + 1.15145 w_k1_FDX_s116
                                 + 2.01106 w_k1_IBM_s116
                                 + 1.10806 w_k1_INTC_s116
                                 + 1.43039 w_k1_KO_s116 + 1.73051 w_k1_MAT_s116
                                 + 0.946041 w_k1_MSFT_s116
                                 + 1.35199 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.41692 w_k1_ADM_s116
                                 + 1.55455 w_k1_AXP_s116
                                 + 1.69063 w_k1_DIS_s116
                                 + 1.33579 w_k1_FDX_s116
                                 + 1.34203 w_k1_IBM_s116
                                 + 1.18379 w_k1_INTC_s116
                                 + 1.18768 w_k1_KO_s116 + 1.48498 w_k1_MAT_s116
                                 + 1.32976 w_k1_MSFT_s116
                                 + 1.29784 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.851363 w_k1_ADM_s116
                                 + 0.842889 w_k1_AXP_s116
                                 + 0.73823 w_k1_DIS_s116
                                 + 0.805485 w_k1_FDX_s116
                                 + 0.99748 w_k1_IBM_s116
                                 + 0.845554 w_k1_INTC_s116
                                 + 0.859178 w_k1_KO_s116
                                 + 0.887191 w_k1_MAT_s116
                                 + 0.70523 w_k1_MSFT_s116
                                 + 1.01447 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.11876 w_k1_ADM_s117
                                 + 0.766625 w_k1_AXP_s117
                                 + 0.994106 w_k1_DIS_s117
                                 + 0.933856 w_k1_FDX_s117
                                 + 0.855251 w_k1_IBM_s117
                                 + 0.846784 w_k1_INTC_s117
                                 + 0.95313 w_k1_KO_s117 + 1.11829 w_k1_MAT_s117
                                 + 1.01577 w_k1_MSFT_s117
                                 + 0.926492 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.758606 w_k1_ADM_s117
                                 + 0.922531 w_k1_AXP_s117
                                 + 0.717967 w_k1_DIS_s117
                                 + 0.810651 w_k1_FDX_s117
                                 + 0.892988 w_k1_IBM_s117
                                 + 0.756384 w_k1_INTC_s117
                                 + 0.824542 w_k1_KO_s117
                                 + 0.914318 w_k1_MAT_s117
                                 + 0.797271 w_k1_MSFT_s117
                                 + 0.913809 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.28434 w_k1_ADM_s117
                                 + 0.854495 w_k1_AXP_s117
                                 + 1.16871 w_k1_DIS_s117
                                 + 1.05496 w_k1_FDX_s117
                                 + 0.684631 w_k1_IBM_s117
                                 + 0.989933 w_k1_INTC_s117
                                 + 0.930921 w_k1_KO_s117
                                 + 0.678892 w_k1_MAT_s117
                                 + 1.12647 w_k1_MSFT_s117
                                 + 1.01303 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.29995 w_k1_ADM_s117
                                 + 1.38972 w_k1_AXP_s117
                                 + 1.54773 w_k1_DIS_s117
                                 + 1.05847 w_k1_FDX_s117 + 1.7178 w_k1_IBM_s117
                                 + 1.16498 w_k1_INTC_s117
                                 + 1.64629 w_k1_KO_s117 + 2.03139 w_k1_MAT_s117
                                 + 1.35233 w_k1_MSFT_s117
                                 + 1.31399 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.12151 w_k1_ADM_s117
                                 + 1.12729 w_k1_AXP_s117
                                 + 1.19856 w_k1_DIS_s117
                                 + 1.16684 w_k1_FDX_s117
                                 + 0.802916 w_k1_IBM_s117
                                 + 0.970154 w_k1_INTC_s117
                                 + 1.12278 w_k1_KO_s117
                                 + 0.976263 w_k1_MAT_s117
                                 + 1.17123 w_k1_MSFT_s117
                                 + 1.05357 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.09006 w_k1_ADM_s117
                                 + 0.910978 w_k1_AXP_s117
                                 + 0.970715 w_k1_DIS_s117
                                 + 0.868014 w_k1_FDX_s117
                                 + 1.17247 w_k1_IBM_s117
                                 + 0.745914 w_k1_INTC_s117
                                 + 1.05527 w_k1_KO_s117 + 1.30881 w_k1_MAT_s117
                                 + 0.972501 w_k1_MSFT_s117
                                 + 0.953593 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.07272 w_k1_ADM_s117
                                 + 1.31798 w_k1_AXP_s117
                                 + 1.36979 w_k1_DIS_s117
                                 + 1.12829 w_k1_FDX_s117
                                 + 1.09556 w_k1_IBM_s117
                                 + 0.995963 w_k1_INTC_s117
                                 + 1.30191 w_k1_KO_s117 + 1.22895 w_k1_MAT_s117
                                 + 1.2773 w_k1_MSFT_s117 + 1.17379 w_k1_PG_s117
                                 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.944609 w_k1_ADM_s117
                                 + 0.811318 w_k1_AXP_s117
                                 + 0.677667 w_k1_DIS_s117
                                 + 0.762155 w_k1_FDX_s117
                                 + 0.634787 w_k1_IBM_s117
                                 + 0.656736 w_k1_INTC_s117
                                 + 0.739377 w_k1_KO_s117
                                 + 0.613693 w_k1_MAT_s117
                                 + 0.79952 w_k1_MSFT_s117
                                 + 0.741108 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0479498214775761 w_k1_ADM_s000
                                 + 0.0883375219369378 w_k1_ADM_s001
                                 + 0.0343270075584582 w_k1_ADM_s002
                                 + 0.0642638771734245 w_k1_ADM_s003
                                 + 0.0232982477549889 w_k1_ADM_s004
                                 + 0.0145743337604849 w_k1_ADM_s005
                                 + 0.0395267858703548 w_k1_ADM_s006
                                 + 0.0249819551648176 w_k1_ADM_s007
                                 + 0.0128558463410318 w_k1_ADM_s010
                                 + 0.0144872668246499 w_k1_ADM_s011
                                 + 0.0114296575563522 w_k1_ADM_s012
                                 + 0.0167959588921043 w_k1_ADM_s013
                                 + 0.034741626821209 w_k1_ADM_s014
                                 + 0.0243443777858974 w_k1_ADM_s015
                                 + 0.0161293705404881 w_k1_ADM_s016
                                 + 0.0137338148459876 w_k1_ADM_s017
                                 + 0.0488072050862192 w_k1_ADM_s100
                                 + 0.056887611729483 w_k1_ADM_s101
                                 + 0.0284538098260007 w_k1_ADM_s102
                                 + 0.0367754924868228 w_k1_ADM_s103
                                 + 0.0443101953835426 w_k1_ADM_s104
                                 + 0.0281785576718266 w_k1_ADM_s105
                                 + 0.0679086574798108 w_k1_ADM_s106
                                 + 0.0878108309952978 w_k1_ADM_s107
                                 + 0.0242529861115564 w_k1_ADM_s110
                                 + 0.0190548294038493 w_k1_ADM_s111
                                 + 0.0122265853367465 w_k1_ADM_s112
                                 + 0.0125180319889006 w_k1_ADM_s113
                                 + 0.0211078863370149 w_k1_ADM_s114
                                 + 0.0156247084557054 w_k1_ADM_s115
                                 + 0.021767025750937 w_k1_ADM_s116
                                 + 0.0521700988474617 w_k1_ADM_s117
                                 + 0.0508861525371583 w_k1_AXP_s000
                                 + 0.0857995559302808 w_k1_AXP_s001
                                 + 0.0331484728639846 w_k1_AXP_s002
                                 + 0.0575350581248146 w_k1_AXP_s003
                                 + 0.0238987137234454 w_k1_AXP_s004
                                 + 0.0171954247831556 w_k1_AXP_s005
                                 + 0.041428324260662 w_k1_AXP_s006
                                 + 0.0245049834699258 w_k1_AXP_s007
                                 + 0.0151733574372283 w_k1_AXP_s010
                                 + 0.0131974416940345 w_k1_AXP_s011
                                 + 0.0133424612479989 w_k1_AXP_s012
                                 + 0.0185385941126685 w_k1_AXP_s013
                                 + 0.0328527394035325 w_k1_AXP_s014
                                 + 0.0249407107210781 w_k1_AXP_s015
                                 + 0.0165747146456411 w_k1_AXP_s016
                                 + 0.0163406453312382 w_k1_AXP_s017
                                 + 0.0470705139953213 w_k1_AXP_s100
                                 + 0.0610773280764891 w_k1_AXP_s101
                                 + 0.0251015790680711 w_k1_AXP_s102
                                 + 0.0408584255216681 w_k1_AXP_s103
                                 + 0.0472720012842035 w_k1_AXP_s104
                                 + 0.024909210070523 w_k1_AXP_s105
                                 + 0.0630032949746598 w_k1_AXP_s106
                                 + 0.0789658547537975 w_k1_AXP_s107
                                 + 0.0205951554835703 w_k1_AXP_s110
                                 + 0.0208486250949978 w_k1_AXP_s111
                                 + 0.0131818166893026 w_k1_AXP_s112
                                 + 0.0115989495543522 w_k1_AXP_s113
                                 + 0.0224234917435577 w_k1_AXP_s114
                                 + 0.0163919468583784 w_k1_AXP_s115
                                 + 0.0208276192180637 w_k1_AXP_s116
                                 + 0.0516176577087841 w_k1_AXP_s117
                                 + 0.0555430474322348 w_k1_DIS_s000
                                 + 0.0781942970644293 w_k1_DIS_s001
                                 + 0.0347712589508987 w_k1_DIS_s002
                                 + 0.0521194783105604 w_k1_DIS_s003
                                 + 0.025809013701389 w_k1_DIS_s004
                                 + 0.0170035165258322 w_k1_DIS_s005
                                 + 0.0374915374709588 w_k1_DIS_s006
                                 + 0.0233475336488348 w_k1_DIS_s007
                                 + 0.0132814998030275 w_k1_DIS_s010
                                 + 0.0121750898112003 w_k1_DIS_s011
                                 + 0.015110891165561 w_k1_DIS_s012
                                 + 0.0185329310541243 w_k1_DIS_s013
                                 + 0.0353977880695954 w_k1_DIS_s014
                                 + 0.0254255915345168 w_k1_DIS_s015
                                 + 0.017210810159477 w_k1_DIS_s016
                                 + 0.0165366916513642 w_k1_DIS_s017
                                 + 0.0477632124570339 w_k1_DIS_s100
                                 + 0.0580435407670668 w_k1_DIS_s101
                                 + 0.0261703494357929 w_k1_DIS_s102
                                 + 0.038437499675763 w_k1_DIS_s103
                                 + 0.0449178283780715 w_k1_DIS_s104
                                 + 0.0266799560227315 w_k1_DIS_s105
                                 + 0.0595941089636765 w_k1_DIS_s106
                                 + 0.0829207750124913 w_k1_DIS_s107
                                 + 0.0219510553651047 w_k1_DIS_s110
                                 + 0.021556568478003 w_k1_DIS_s111
                                 + 0.0128154188808155 w_k1_DIS_s112
                                 + 0.0114881802331325 w_k1_DIS_s113
                                 + 0.0190458525638264 w_k1_DIS_s114
                                 + 0.0144549084026943 w_k1_DIS_s115
                                 + 0.0197164006977248 w_k1_DIS_s116
                                 + 0.0536732396148735 w_k1_DIS_s117
                                 + 0.0536208696844993 w_k1_FDX_s000
                                 + 0.0769538864548176 w_k1_FDX_s001
                                 + 0.0333198390607908 w_k1_FDX_s002
                                 + 0.052784525628358 w_k1_FDX_s003
                                 + 0.0256040950785898 w_k1_FDX_s004
                                 + 0.0167154880069935 w_k1_FDX_s005
                                 + 0.0369889294430398 w_k1_FDX_s006
                                 + 0.0235678708864144 w_k1_FDX_s007
                                 + 0.0135722774509097 w_k1_FDX_s010
                                 + 0.0123020903998559 w_k1_FDX_s011
                                 + 0.0135643692586643 w_k1_FDX_s012
                                 + 0.017285956930921 w_k1_FDX_s013
                                 + 0.0348739577755117 w_k1_FDX_s014
                                 + 0.0254544863709839 w_k1_FDX_s015
                                 + 0.0159850957800352 w_k1_FDX_s016
                                 + 0.0155874722181479 w_k1_FDX_s017
                                 + 0.0444432298640813 w_k1_FDX_s100
                                 + 0.0556896517592983 w_k1_FDX_s101
                                 + 0.0247698452373142 w_k1_FDX_s102
                                 + 0.0384786444298071 w_k1_FDX_s103
                                 + 0.0451158746283669 w_k1_FDX_s104
                                 + 0.0246796794047812 w_k1_FDX_s105
                                 + 0.0592790729560178 w_k1_FDX_s106
                                 + 0.080742163091761 w_k1_FDX_s107
                                 + 0.0216523660701919 w_k1_FDX_s110
                                 + 0.0207845379522684 w_k1_FDX_s111
                                 + 0.0118834085359166 w_k1_FDX_s112
                                 + 0.0120581115403112 w_k1_FDX_s113
                                 + 0.0218484247317634 w_k1_FDX_s114
                                 + 0.0145357577609613 w_k1_FDX_s115
                                 + 0.0199728741119605 w_k1_FDX_s116
                                 + 0.0476845828505673 w_k1_FDX_s117
                                 + 0.0519795048205617 w_k1_IBM_s000
                                 + 0.0784182391599607 w_k1_IBM_s001
                                 + 0.0322082341453604 w_k1_IBM_s002
                                 + 0.0566337504877235 w_k1_IBM_s003
                                 + 0.0259110457736837 w_k1_IBM_s004
                                 + 0.016188054666291 w_k1_IBM_s005
                                 + 0.0384732250255924 w_k1_IBM_s006
                                 + 0.0232202063876294 w_k1_IBM_s007
                                 + 0.0129839092217909 w_k1_IBM_s010
                                 + 0.01286002491656 w_k1_IBM_s011
                                 + 0.0131148237852381 w_k1_IBM_s012
                                 + 0.0180006537388263 w_k1_IBM_s013
                                 + 0.0363104706195097 w_k1_IBM_s014
                                 + 0.0246637576388437 w_k1_IBM_s015
                                 + 0.015609001970398 w_k1_IBM_s016
                                 + 0.0155097587167184 w_k1_IBM_s017
                                 + 0.0472325381060158 w_k1_IBM_s100
                                 + 0.0566762186510174 w_k1_IBM_s101
                                 + 0.0259251616687125 w_k1_IBM_s102
                                 + 0.038232274841047 w_k1_IBM_s103
                                 + 0.0456943455123758 w_k1_IBM_s104
                                 + 0.0256027246035202 w_k1_IBM_s105
                                 + 0.0607154974462641 w_k1_IBM_s106
                                 + 0.0825307425315272 w_k1_IBM_s107
                                 + 0.0233384531937362 w_k1_IBM_s110
                                 + 0.0216518571494123 w_k1_IBM_s111
                                 + 0.0117011384202993 w_k1_IBM_s112
                                 + 0.0121934339984829 w_k1_IBM_s113
                                 + 0.0208352916609495 w_k1_IBM_s114
                                 + 0.0152453440834092 w_k1_IBM_s115
                                 + 0.0210388730347522 w_k1_IBM_s116
                                 + 0.0495760362481952 w_k1_IBM_s117
                                 + 0.048142371436191 w_k1_INTC_s000
                                 + 0.0796523496807552 w_k1_INTC_s001
                                 + 0.0311446358419535 w_k1_INTC_s002
                                 + 0.0494639976944995 w_k1_INTC_s003
                                 + 0.0243757068519258 w_k1_INTC_s004
                                 + 0.016536847627786 w_k1_INTC_s005
                                 + 0.0359333059175642 w_k1_INTC_s006
                                 + 0.0214060120465294 w_k1_INTC_s007
                                 + 0.0142844695519508 w_k1_INTC_s010
                                 + 0.0133623497956243 w_k1_INTC_s011
                                 + 0.0141088423508077 w_k1_INTC_s012
                                 + 0.0173445722898315 w_k1_INTC_s013
                                 + 0.0412609136678581 w_k1_INTC_s014
                                 + 0.0221922388792443 w_k1_INTC_s015
                                 + 0.0136633170243531 w_k1_INTC_s016
                                 + 0.0153051270834988 w_k1_INTC_s017
                                 + 0.0443431367687875 w_k1_INTC_s100
                                 + 0.0628834899542303 w_k1_INTC_s101
                                 + 0.0243264492148541 w_k1_INTC_s102
                                 + 0.0384764798182944 w_k1_INTC_s103
                                 + 0.0402992876686919 w_k1_INTC_s104
                                 + 0.0234174837295169 w_k1_INTC_s105
                                 + 0.0663256844908876 w_k1_INTC_s106
                                 + 0.0838712002280184 w_k1_INTC_s107
                                 + 0.0188284078102813 w_k1_INTC_s110
                                 + 0.021099874645906 w_k1_INTC_s111
                                 + 0.0118594670966446 w_k1_INTC_s112
                                 + 0.0111708682486487 w_k1_INTC_s113
                                 + 0.0218850750341604 w_k1_INTC_s114
                                 + 0.0140829250948825 w_k1_INTC_s115
                                 + 0.019033957547583 w_k1_INTC_s116
                                 + 0.0431772910171121 w_k1_INTC_s117
                                 + 0.0566934531971549 w_k1_KO_s000
                                 + 0.0786631935594498 w_k1_KO_s001
                                 + 0.0316858556810909 w_k1_KO_s002
                                 + 0.0501292567691106 w_k1_KO_s003
                                 + 0.0247980503106388 w_k1_KO_s004
                                 + 0.0157507569219811 w_k1_KO_s005
                                 + 0.0376807157498266 w_k1_KO_s006
                                 + 0.0240901168577468 w_k1_KO_s007
                                 + 0.0129414894641644 w_k1_KO_s010
                                 + 0.0139126403428571 w_k1_KO_s011
                                 + 0.0127506565578834 w_k1_KO_s012
                                 + 0.0174834616741292 w_k1_KO_s013
                                 + 0.0335371826535543 w_k1_KO_s014
                                 + 0.0258207331763251 w_k1_KO_s015
                                 + 0.0142101829432123 w_k1_KO_s016
                                 + 0.0150302749586947 w_k1_KO_s017
                                 + 0.0466440852485129 w_k1_KO_s100
                                 + 0.0620155534338107 w_k1_KO_s101
                                 + 0.0231115413124086 w_k1_KO_s102
                                 + 0.034932826591898 w_k1_KO_s103
                                 + 0.0473492826970559 w_k1_KO_s104
                                 + 0.0240863360867376 w_k1_KO_s105
                                 + 0.0601461775656718 w_k1_KO_s106
                                 + 0.0743443907443847 w_k1_KO_s107
                                 + 0.0221280084168252 w_k1_KO_s110
                                 + 0.0222151486275665 w_k1_KO_s111
                                 + 0.0120399781051357 w_k1_KO_s112
                                 + 0.0132948946355581 w_k1_KO_s113
                                 + 0.0197276271202446 w_k1_KO_s114
                                 + 0.0144621951371695 w_k1_KO_s115
                                 + 0.0196334575075659 w_k1_KO_s116
                                 + 0.0538958508165116 w_k1_KO_s117
                                 + 0.0628079728356211 w_k1_MAT_s000
                                 + 0.0796635814986168 w_k1_MAT_s001
                                 + 0.037385628064091 w_k1_MAT_s002
                                 + 0.0629394939067898 w_k1_MAT_s003
                                 + 0.0227164860005446 w_k1_MAT_s004
                                 + 0.0185266329113009 w_k1_MAT_s005
                                 + 0.033364612757906 w_k1_MAT_s006
                                 + 0.0360548559042101 w_k1_MAT_s007
                                 + 0.0132401819713228 w_k1_MAT_s010
                                 + 0.0109316071339118 w_k1_MAT_s011
                                 + 0.0114630521291438 w_k1_MAT_s012
                                 + 0.0161243739577975 w_k1_MAT_s013
                                 + 0.0325208774244265 w_k1_MAT_s014
                                 + 0.0288801801999729 w_k1_MAT_s015
                                 + 0.0178898348078989 w_k1_MAT_s016
                                 + 0.0146573785634725 w_k1_MAT_s017
                                 + 0.0465204872341183 w_k1_MAT_s100
                                 + 0.0745919403159322 w_k1_MAT_s101
                                 + 0.026574712334129 w_k1_MAT_s102
                                 + 0.0432769660085064 w_k1_MAT_s103
                                 + 0.0554014209548284 w_k1_MAT_s104
                                 + 0.0256962092463771 w_k1_MAT_s105
                                 + 0.0603415428334361 w_k1_MAT_s106
                                 + 0.0684376297125724 w_k1_MAT_s107
                                 + 0.0234128449165002 w_k1_MAT_s110
                                 + 0.0257594961385771 w_k1_MAT_s111
                                 + 0.0125296241903256 w_k1_MAT_s112
                                 + 0.0131614273899967 w_k1_MAT_s113
                                 + 0.0203925135674976 w_k1_MAT_s114
                                 + 0.0177784636428218 w_k1_MAT_s115
                                 + 0.0213070834721994 w_k1_MAT_s116
                                 + 0.0558763328415458 w_k1_MAT_s117
                                 + 0.0554120651860995 w_k1_MSFT_s000
                                 + 0.0781889809078035 w_k1_MSFT_s001
                                 + 0.0310236117343607 w_k1_MSFT_s002
                                 + 0.0550000195623099 w_k1_MSFT_s003
                                 + 0.0258608851237264 w_k1_MSFT_s004
                                 + 0.0165186893724589 w_k1_MSFT_s005
                                 + 0.0387634521313222 w_k1_MSFT_s006
                                 + 0.0219567634034218 w_k1_MSFT_s007
                                 + 0.0128990913317084 w_k1_MSFT_s010
                                 + 0.0122239764219755 w_k1_MSFT_s011
                                 + 0.0137916357371432 w_k1_MSFT_s012
                                 + 0.0159762241581104 w_k1_MSFT_s013
                                 + 0.034025862077054 w_k1_MSFT_s014
                                 + 0.0233871747546294 w_k1_MSFT_s015
                                 + 0.0152080983247647 w_k1_MSFT_s016
                                 + 0.0163826600348124 w_k1_MSFT_s017
                                 + 0.0461803872728293 w_k1_MSFT_s100
                                 + 0.0619604916997756 w_k1_MSFT_s101
                                 + 0.0250384549427577 w_k1_MSFT_s102
                                 + 0.039321824606071 w_k1_MSFT_s103
                                 + 0.0421343269496964 w_k1_MSFT_s104
                                 + 0.0229088766498602 w_k1_MSFT_s105
                                 + 0.0621451920501323 w_k1_MSFT_s106
                                 + 0.0769423072263502 w_k1_MSFT_s107
                                 + 0.0216683023914746 w_k1_MSFT_s110
                                 + 0.0235870400823916 w_k1_MSFT_s111
                                 + 0.0123297821307827 w_k1_MSFT_s112
                                 + 0.0122183833777959 w_k1_MSFT_s113
                                 + 0.0217423833452776 w_k1_MSFT_s114
                                 + 0.0138534111799767 w_k1_MSFT_s115
                                 + 0.0188863902262764 w_k1_MSFT_s116
                                 + 0.0525591762547176 w_k1_MSFT_s117
                                 + 0.053202046532871 w_k1_PG_s000
                                 + 0.0794175346143233 w_k1_PG_s001
                                 + 0.0306364539768169 w_k1_PG_s002
                                 + 0.0561115048151486 w_k1_PG_s003
                                 + 0.0225006661751557 w_k1_PG_s004
                                 + 0.0157736025318034 w_k1_PG_s005
                                 + 0.036641758455021 w_k1_PG_s006
                                 + 0.0233262055767071 w_k1_PG_s007
                                 + 0.0129128659699635 w_k1_PG_s010
                                 + 0.0123572918830853 w_k1_PG_s011
                                 + 0.0123408819610666 w_k1_PG_s012
                                 + 0.0181751885105948 w_k1_PG_s013
                                 + 0.0348230226697061 w_k1_PG_s014
                                 + 0.0257559078141268 w_k1_PG_s015
                                 + 0.0147213628688696 w_k1_PG_s016
                                 + 0.0141769166439062 w_k1_PG_s017
                                 + 0.042205774354335 w_k1_PG_s100
                                 + 0.0583799776532295 w_k1_PG_s101
                                 + 0.0241490376026509 w_k1_PG_s102
                                 + 0.0354845291504183 w_k1_PG_s103
                                 + 0.0473292366698493 w_k1_PG_s104
                                 + 0.0251366310805806 w_k1_PG_s105
                                 + 0.0619578946720471 w_k1_PG_s106
                                 + 0.0807762851486867 w_k1_PG_s107
                                 + 0.0213454675262719 w_k1_PG_s110
                                 + 0.021806982605539 w_k1_PG_s111
                                 + 0.0117665726675124 w_k1_PG_s112
                                 + 0.0113516927813996 w_k1_PG_s113
                                 + 0.0205766400122609 w_k1_PG_s114
                                 + 0.0141852679891506 w_k1_PG_s115
                                 + 0.021969807854437 w_k1_PG_s116
                                 + 0.0498326681866436 w_k1_PG_s117 >= 1.15
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
