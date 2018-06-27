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
 w_k1_ADM_s0_return_constraint:  1.24606 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.950611 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.28085 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.859879 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.218 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.946284 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.25338 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.1339 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.678474 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.21948 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.803015 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.15205 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.815929 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.962519 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.954472 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.992878 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.15714 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.03922 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.20854 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.24645 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.3022 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.05775 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.98002 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.22438 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.1732 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.737913 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.996679 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.00948 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.679422 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.811822 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.945998 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.02049 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.74555 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.38623 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.13051 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.807541 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.20652 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.861971 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.27126 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.935817 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.51132 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.956608 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.36558 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.923257 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.03111 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.28074 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.754735 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.47942 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.597441 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.947615 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.01957 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.871113 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.748879 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.895371 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.43997 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.07193 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.01256 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.3124 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.54262 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.11133 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.931231 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.40209 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.991388 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.631224 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.819226 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.09111 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.765292 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.05496 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.817946 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.23278 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.532452 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.31866 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.05927 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.618576 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.18459 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.909043 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.41587 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.904282 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.48998 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.905036 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.35921 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.884404 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.17309 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.10072 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.812021 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.68768 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.659428 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.50124 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.867956 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.845353 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.710124 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.939547 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.60095 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.20474 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.20501 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.31438 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.60311 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.911173 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.05735 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.74899 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.19085 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.697505 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.750697 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.04629 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.709881 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.00474 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.968352 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.41518 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.533302 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.48139 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.0547 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.703571 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.17618 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.800423 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.27571 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.941382 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.31901 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.925331 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.21452 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.906642 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.16582 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.02744 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.794567 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.24178 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.72103 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.13621 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.808085 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.965954 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.725805 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.03435 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.19784 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.972969 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.19446 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.21338 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.47646 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.960848 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.984347 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.28748 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.25211 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.70324 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.855765 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.06452 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.683531 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.929537 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.933574 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.09809 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.715945 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.4561 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.13906 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.730876 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.09085 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.831753 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.19539 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.977788 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.22932 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.995348 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.31909 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.858092 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.779482 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.18341 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.09271 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.64735 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.712147 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.23588 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.56752 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.631213 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.02607 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.578827 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.86357 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.41796 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.942755 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.22539 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.961725 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.958337 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.186 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.51557 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.579968 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.790981 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.90329 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.876949 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.41947 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.62082 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.994916 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.87943 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.560565 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.07919 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.843244 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.876772 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.34602 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.49184 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.21116 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.908217 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.31705 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.839398 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.29878 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.893305 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.00305 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.01785 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.784858 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.18238 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.612779 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.07901 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.870207 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.914191 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.10136 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.01499 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.5665 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.19898 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.201 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.30833 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.55895 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.09075 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.06115 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.16218 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.94254 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.615429 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.869121 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.939051 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.71121 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.01985 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.997736 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.19286 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.721307 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.39335 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.17344 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.871396 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.19854 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.89498 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.20939 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.944104 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.31932 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.978705 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.35918 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.898324 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.947477 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.10308 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.944536 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.52895 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.687008 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.37856 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.04376 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.771117 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.86797 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.752806 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.59605 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.28587 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.07024 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.21887 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.06824 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.918943 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.0484 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.50618 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.726898 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.690869 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.80148 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.908168 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.948962 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.08282 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.01499 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.43149 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.541489 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.29654 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.01091 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.760201 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.18236 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.04487 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.27476 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.965414 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.23823 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.965895 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.43493 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.829595 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.991085 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.32851 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.04253 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.98844 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.778817 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.29875 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.64581 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.845122 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.849586 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.602697 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.75671 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.51483 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.898635 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.27722 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.18995 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.13841 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.01403 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.63953 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.579674 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.698134 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.974035 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.89535 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.24388 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.64747 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.944169 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 2.14503 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.482506 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.04472 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.04574 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.803799 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.3117 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.44032 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.21084 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.945663 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.29431 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.953389 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.16797 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.963522 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.10644 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.973029 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.816659 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.17334 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.752948 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.25692 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.782274 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.903635 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.814717 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.986211 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.33466 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.0591 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.07546 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.13469 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.35663 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.909046 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.04629 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.29645 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.16052 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.741649 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.926799 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.05117 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.654619 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.917622 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.965998 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.10904 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.677053 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.26904 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.09342 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.784073 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.07718 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.786206 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.16751 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.943991 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.17544 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.979995 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.22639 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.922572 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.0321 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.08485 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.853197 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.16355 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.799592 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.20702 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.918645 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.870384 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.82633 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.916151 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.2487 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.10524 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.03808 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.11032 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.12276 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.929349 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.02929 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.33544 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.915045 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.777156 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.912354 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.987693 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.844103 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.952938 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.01247 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.13407 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.771299 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.27287 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.11028 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.888111 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.05182 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.02079 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.282998819705591 z_k1_s0000
                                 - 0.152999134527171 z_k1_s0001
                                 - 0.088999722599297 z_k1_s0002
                                 - 0.101999576217557 z_k1_s0003
                                 - 0.111999631546377 z_k1_s0004
                                 - 0.148999480074306 z_k1_s0005
                                 - 0.0849997234476854 z_k1_s0006
                                 - 0.175999732871743 z_k1_s0007
                                 - 0.0199999777358786 z_k1_s0010
                                 - 0.073999934922722 z_k1_s0011
                                 - 0.0579998835341342 z_k1_s0012
                                 - 0.067000020565035 z_k1_s0013
                                 - 0.018999970199099 z_k1_s0014
                                 - 0.0440000548187602 z_k1_s0015
                                 - 0.0229999657462747 z_k1_s0016
                                 - 0.0409998938086518 z_k1_s0017
                                 - 0.025999992913972 z_k1_s0020
                                 - 0.0619999742717882 z_k1_s0021
                                 - 0.0939998684121582 z_k1_s0022
                                 - 0.0359999813578162 z_k1_s0023
                                 - 0.0939998684121582 z_k1_s0024
                                 - 0.0899996663948262 z_k1_s0025
                                 - 0.0660001762891202 z_k1_s0026
                                 - 0.105999900464725 z_k1_s0027
                                 - 0.0619996342762451 z_k1_s0030
                                 - 0.0449997306714899 z_k1_s0031
                                 - 0.157998834963343 z_k1_s0032
                                 - 0.0559997143331336 z_k1_s0033
                                 - 0.0709994840910801 z_k1_s0034
                                 - 0.0319997937620304 z_k1_s0035
                                 - 0.0999994683807156 z_k1_s0036
                                 - 0.0779995010574281 z_k1_s0037
                                 - 0.0259999355083763 z_k1_s0040
                                 - 0.118999545227259 z_k1_s0041
                                 - 0.0809999586833279 z_k1_s0042
                                 - 0.106999787792011 z_k1_s0043
                                 - 0.0269999330279292 z_k1_s0044
                                 - 0.0309999231061409 z_k1_s0045
                                 - 0.0239999404692704 z_k1_s0046
                                 - 0.11699976298754 z_k1_s0047
                                 - 0.0379999014858726 z_k1_s0050
                                 - 0.0359999444707665 z_k1_s0051
                                 - 0.0610000308585998 z_k1_s0052
                                 - 0.0269999678030598 z_k1_s0053
                                 - 0.0530000515984161 z_k1_s0054
                                 - 0.0239999755804909 z_k1_s0055
                                 - 0.0690000101187835 z_k1_s0056
                                 - 0.0699998185266074 z_k1_s0057
                                 - 0.0470001326733384 z_k1_s0060
                                 - 0.0470001326733384 z_k1_s0061
                                 - 0.0309999287210008 z_k1_s0062
                                 - 0.0509998544120205 z_k1_s0063
                                 - 0.11099963148508 z_k1_s0064
                                 - 0.0550000151499057 z_k1_s0065
                                 - 0.0599997770730591 z_k1_s0066
                                 - 0.036999950328227 z_k1_s0067
                                 - 0.140000270382277 z_k1_s0070
                                 - 0.0669999855258631 z_k1_s0071
                                 - 0.207999634396841 z_k1_s0072
                                 - 0.090999951948572 z_k1_s0073
                                 - 0.291999572826299 z_k1_s0074
                                 - 0.0799999626756657 z_k1_s0075
                                 - 0.175999604566593 z_k1_s0076
                                 - 0.0649999976489668 z_k1_s0077
                                 - 0.3099998563452 z_k1_s0100
                                 - 0.459001922826418 z_k1_s0101
                                 - 0.689001220346922 z_k1_s0102
                                 - 1.24300061162698 z_k1_s0103
                                 - 0.355000036434063 z_k1_s0104
                                 - 0.458000954610902 z_k1_s0105
                                 - 0.432001557013045 z_k1_s0106
                                 - 0.349999920957502 z_k1_s0107
                                 - 0.226999471602157 z_k1_s0110
                                 - 0.734997039044402 z_k1_s0111
                                 - 0.593999348392038 z_k1_s0112
                                 - 0.238999774720638 z_k1_s0113
                                 - 0.378999397109723 z_k1_s0114
                                 - 1.02299648590156 z_k1_s0115
                                 - 0.289999497376905 z_k1_s0116
                                 - 0.427001000982965 z_k1_s0117
                                 - 0.0960000013237264 z_k1_s0120
                                 - 0.257999251201238 z_k1_s0121
                                 - 0.264999256450885 z_k1_s0122
                                 - 0.340000726125723 z_k1_s0123
                                 - 0.22999923020265 z_k1_s0124
                                 - 0.144000084435593 z_k1_s0125
                                 - 0.179000605383845 z_k1_s0126
                                 - 0.137000079185946 z_k1_s0127
                                 - 0.193000140362626 z_k1_s0130
                                 - 0.183999603039939 z_k1_s0131
                                 - 0.204999806126055 z_k1_s0132
                                 - 0.173000697423578 z_k1_s0133
                                 - 0.283000785588806 z_k1_s0134
                                 - 0.264000526796595 z_k1_s0135
                                 - 0.200999917538245 z_k1_s0136
                                 - 0.0730000155278271 z_k1_s0137
                                 - 0.221000150934 z_k1_s0140
                                 - 0.318999665406863 z_k1_s0141
                                 - 0.0769999547430444 z_k1_s0142
                                 - 0.148000287449058 z_k1_s0143
                                 - 0.180000429113738 z_k1_s0144
                                 - 0.174000218676567 z_k1_s0145
                                 - 0.120000089942445 z_k1_s0146
                                 - 0.232000291568756 z_k1_s0147
                                 - 0.0950000134407871 z_k1_s0150
                                 - 0.198000404089239 z_k1_s0151
                                 - 0.0760000610726322 z_k1_s0152
                                 - 0.103999997499387 z_k1_s0153
                                 - 0.128000248522335 z_k1_s0154
                                 - 0.0790000138254968 z_k1_s0155
                                 - 0.280000622267613 z_k1_s0156
                                 - 0.298000087184787 z_k1_s0157
                                 - 0.10400007914701 z_k1_s0160
                                 - 0.101000120326469 z_k1_s0161
                                 - 0.226000606618007 z_k1_s0162
                                 - 0.227999883764287 z_k1_s0163
                                 - 0.128000097411704 z_k1_s0164
                                 - 0.0540000700705938 z_k1_s0165
                                 - 0.0920001279647246 z_k1_s0166
                                 - 0.226000606618007 z_k1_s0167
                                 - 0.306000538927251 z_k1_s0170
                                 - 0.386999376225843 z_k1_s0171
                                 - 0.590002860285123 z_k1_s0172
                                 - 0.344999806078062 z_k1_s0173
                                 - 0.653002215506795 z_k1_s0174
                                 - 0.317000648814042 z_k1_s0175
                                 - 0.164000212500066 z_k1_s0176
                                 - 0.57499967679677 z_k1_s0177
                                 - 0.239999959001162 z_k1_s1000
                                 - 0.324999814689133 z_k1_s1001
                                 - 0.355999835351771 z_k1_s1002
                                 - 1.10899733709401 z_k1_s1003
                                 - 0.377999735176922 z_k1_s1004
                                 - 1.19000064963037 z_k1_s1005
                                 - 0.496001220601797 z_k1_s1006
                                 - 0.355999835351771 z_k1_s1007
                                 - 0.419001305108474 z_k1_s1010
                                 - 0.524002581891749 z_k1_s1011
                                 - 0.275000527164802 z_k1_s1012
                                 - 0.77599991016004 z_k1_s1013
                                 - 0.443999560528931 z_k1_s1014
                                 - 0.303000509139619 z_k1_s1015
                                 - 0.555002139416587 z_k1_s1016
                                 - 0.289000697402572 z_k1_s1017
                                 - 0.443997671395131 z_k1_s1020
                                 - 0.483998769933746 z_k1_s1021
                                 - 0.357999746932066 z_k1_s1022
                                 - 0.401999341726376 z_k1_s1023
                                 - 0.761997529987206 z_k1_s1024
                                 - 0.503999319203053 z_k1_s1025
                                 - 0.28299970416987 z_k1_s1026
                                 - 0.797000508206202 z_k1_s1027
                                 - 1.12200333526333 z_k1_s1030
                                 - 0.254000463250883 z_k1_s1031
                                 - 0.374002183756392 z_k1_s1032
                                 - 0.259000387538345 z_k1_s1033
                                 - 0.240000546605757 z_k1_s1034
                                 - 0.363000227760119 z_k1_s1035
                                 - 0.304000670927253 z_k1_s1036
                                 - 0.30000060285705 z_k1_s1037
                                 - 0.65799776544904 z_k1_s1040
                                 - 0.149999858533791 z_k1_s1041
                                 - 0.293998694243597 z_k1_s1042
                                 - 0.301998324611213 z_k1_s1043
                                 - 0.221999709738118 z_k1_s1044
                                 - 0.520998678535803 z_k1_s1045
                                 - 0.543997976967221 z_k1_s1046
                                 - 0.197999663036803 z_k1_s1047
                                 - 0.871002568818295 z_k1_s1050
                                 - 1.30599789845569 z_k1_s1051
                                 - 0.631002779913633 z_k1_s1052
                                 - 0.588999958353781 z_k1_s1053
                                 - 0.561000364115109 z_k1_s1054
                                 - 0.603000898873732 z_k1_s1055
                                 - 0.76799932449492 z_k1_s1056
                                 - 0.38800042099628 z_k1_s1057
                                 - 0.40499965861332 z_k1_s1060
                                 - 0.301000693833647 z_k1_s1061
                                 - 0.179000432226655 z_k1_s1062
                                 - 0.0919999698282132 z_k1_s1063
                                 - 0.131000024420611 z_k1_s1064
                                 - 0.126000074190819 z_k1_s1065
                                 - 0.0929999598741716 z_k1_s1066
                                 - 0.149000730847903 z_k1_s1067
                                 - 0.151999792005886 z_k1_s1070
                                 - 0.269999432477759 z_k1_s1071
                                 - 0.332999099356104 z_k1_s1072
                                 - 0.119999814667826 z_k1_s1073
                                 - 0.342999970335868 z_k1_s1074
                                 - 0.296999576425334 z_k1_s1075
                                 - 0.30599895540853 z_k1_s1076
                                 - 0.1859997428401 z_k1_s1077
                                 - 0.664999682886077 z_k1_s1100
                                 - 1.64500029922259 z_k1_s1101
                                 - 1.41799566493354 z_k1_s1102
                                 - 1.93900164788261 z_k1_s1103
                                 - 0.650999661292709 z_k1_s1104
                                 - 1.71299963679099 z_k1_s1105
                                 - 0.420999690201348 z_k1_s1106
                                 - 0.499999236564388 z_k1_s1107
                                 - 0.318998748648795 z_k1_s1110
                                 - 0.289999869426192 z_k1_s1111
                                 - 0.333999123867396 z_k1_s1112
                                 - 0.299000094557353 z_k1_s1113
                                 - 0.612999948933298 z_k1_s1114
                                 - 0.758000499046403 z_k1_s1115
                                 - 0.20499989708748 z_k1_s1116
                                 - 0.259000017074428 z_k1_s1117
                                 - 0.220000032237345 z_k1_s1120
                                 - 0.998000283076685 z_k1_s1121
                                 - 0.425000746458509 z_k1_s1122
                                 - 0.228999999347055 z_k1_s1123
                                 - 0.782000319843655 z_k1_s1124
                                 - 0.583998785630043 z_k1_s1125
                                 - 0.246000104465395 z_k1_s1126
                                 - 0.279000109300997 z_k1_s1127
                                 - 1.23499400126256 z_k1_s1130
                                 - 1.11199910006457 z_k1_s1131
                                 - 0.945000267235316 z_k1_s1132
                                 - 1.26099395341522 z_k1_s1133
                                 - 0.999995866801576 z_k1_s1134
                                 - 0.721997582181086 z_k1_s1135
                                 - 0.570998146678222 z_k1_s1136
                                 - 0.605998598174538 z_k1_s1137
                                 - 0.500000563345613 z_k1_s1140
                                 - 1.08100432618472 z_k1_s1141
                                 - 1.5680015850388 z_k1_s1142
                                 - 0.779000758508907 z_k1_s1143
                                 - 1.23000269117393 z_k1_s1144
                                 - 1.45800272861044 z_k1_s1145
                                 - 0.780000576130755 z_k1_s1146
                                 - 2.06300116847267 z_k1_s1147
                                 - 2.11300124847612 z_k1_s1150
                                 - 1.40400410261477 z_k1_s1151
                                 - 0.469000516036387 z_k1_s1152
                                 - 0.67400030798369 z_k1_s1153
                                 - 0.615000167142198 z_k1_s1154
                                 - 0.972001622886702 z_k1_s1155
                                 - 0.881999128025966 z_k1_s1156
                                 - 0.784000543745785 z_k1_s1157
                                 - 0.242000047444693 z_k1_s1160
                                 - 0.701001920958301 z_k1_s1161
                                 - 1.04900209539947 z_k1_s1162
                                 - 0.21200025399296 z_k1_s1163
                                 - 0.260999952330806 z_k1_s1164
                                 - 0.254000250425505 z_k1_s1165
                                 - 0.641000906054242 z_k1_s1166
                                 - 0.209999982162726 z_k1_s1167
                                 - 0.826000968742273 z_k1_s1170
                                 - 0.941998183091968 z_k1_s1171
                                 - 0.911998743484205 z_k1_s1172
                                 - 0.446000332039917 z_k1_s1173
                                 - 0.203000230214622 z_k1_s1174
                                 - 0.579999994755719 z_k1_s1175
                                 - 0.450000412021033 z_k1_s1176
                                 - 0.28200030151587 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.713173 w_k1_ADM_s000
                                 + 1.10351 w_k1_AXP_s000
                                 + 1.06544 w_k1_DIS_s000 + 1.0221 w_k1_FDX_s000
                                 + 1.1312 w_k1_IBM_s000
                                 + 1.10731 w_k1_INTC_s000
                                 + 1.07029 w_k1_KO_s000 + 1.08782 w_k1_MAT_s000
                                 + 0.89273 w_k1_MSFT_s000
                                 + 1.01518 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.690846 w_k1_ADM_s000
                                 + 1.43843 w_k1_AXP_s000
                                 + 1.38713 w_k1_DIS_s000
                                 + 1.18481 w_k1_FDX_s000
                                 + 1.75998 w_k1_IBM_s000
                                 + 1.33844 w_k1_INTC_s000
                                 + 1.48637 w_k1_KO_s000 + 1.69971 w_k1_MAT_s000
                                 + 1.09233 w_k1_MSFT_s000 + 1.1952 w_k1_PG_s000
                                 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.776522 w_k1_ADM_s000
                                 + 0.823648 w_k1_AXP_s000
                                 + 0.981599 w_k1_DIS_s000
                                 + 0.839968 w_k1_FDX_s000
                                 + 0.972423 w_k1_IBM_s000
                                 + 0.793739 w_k1_INTC_s000
                                 + 0.94615 w_k1_KO_s000
                                 + 0.791862 w_k1_MAT_s000
                                 + 0.879931 w_k1_MSFT_s000
                                 + 1.03026 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.530542 w_k1_ADM_s000
                                 + 0.560676 w_k1_AXP_s000
                                 + 0.637299 w_k1_DIS_s000
                                 + 0.717105 w_k1_FDX_s000
                                 + 0.801971 w_k1_IBM_s000
                                 + 0.790936 w_k1_INTC_s000
                                 + 0.762951 w_k1_KO_s000
                                 + 0.541788 w_k1_MAT_s000
                                 + 0.797928 w_k1_MSFT_s000
                                 + 0.801632 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.658278 w_k1_ADM_s000
                                 + 1.04632 w_k1_AXP_s000 + 0.7421 w_k1_DIS_s000
                                 + 0.770543 w_k1_FDX_s000
                                 + 1.4477 w_k1_IBM_s000
                                 + 0.754044 w_k1_INTC_s000
                                 + 1.0949 w_k1_KO_s000 + 1.35234 w_k1_MAT_s000
                                 + 0.750818 w_k1_MSFT_s000
                                 + 0.934021 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.812044 w_k1_ADM_s000
                                 + 0.896659 w_k1_AXP_s000
                                 + 0.755731 w_k1_DIS_s000
                                 + 0.842121 w_k1_FDX_s000
                                 + 0.888876 w_k1_IBM_s000
                                 + 0.870028 w_k1_INTC_s000
                                 + 1.0143 w_k1_KO_s000 + 0.825108 w_k1_MAT_s000
                                 + 0.796333 w_k1_MSFT_s000
                                 + 1.05919 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.907505 w_k1_ADM_s000
                                 + 1.31993 w_k1_AXP_s000
                                 + 1.71352 w_k1_DIS_s000 + 1.3254 w_k1_FDX_s000
                                 + 0.9106 w_k1_IBM_s000
                                 + 1.08049 w_k1_INTC_s000
                                 + 1.15169 w_k1_KO_s000
                                 + 0.904277 w_k1_MAT_s000
                                 + 1.20705 w_k1_MSFT_s000
                                 + 1.13254 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.09297 w_k1_ADM_s000
                                 + 1.04647 w_k1_AXP_s000
                                 + 1.09385 w_k1_DIS_s000
                                 + 1.20256 w_k1_FDX_s000
                                 + 0.740858 w_k1_IBM_s000
                                 + 1.11104 w_k1_INTC_s000
                                 + 0.980601 w_k1_KO_s000
                                 + 0.977162 w_k1_MAT_s000
                                 + 1.25657 w_k1_MSFT_s000
                                 + 0.998036 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.41452 w_k1_ADM_s001
                                 + 1.55092 w_k1_AXP_s001
                                 + 1.42025 w_k1_DIS_s001
                                 + 1.35708 w_k1_FDX_s001
                                 + 1.37998 w_k1_IBM_s001
                                 + 0.901308 w_k1_INTC_s001
                                 + 1.29022 w_k1_KO_s001 + 2.03876 w_k1_MAT_s001
                                 + 1.24681 w_k1_MSFT_s001
                                 + 1.40805 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.02622 w_k1_ADM_s001
                                 + 0.88838 w_k1_AXP_s001
                                 + 0.80946 w_k1_DIS_s001
                                 + 0.964116 w_k1_FDX_s001
                                 + 0.740528 w_k1_IBM_s001
                                 + 0.83682 w_k1_INTC_s001
                                 + 0.826089 w_k1_KO_s001
                                 + 0.63921 w_k1_MAT_s001
                                 + 0.943173 w_k1_MSFT_s001
                                 + 0.955365 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.895118 w_k1_ADM_s001
                                 + 1.10468 w_k1_AXP_s001
                                 + 0.756953 w_k1_DIS_s001
                                 + 0.950225 w_k1_FDX_s001
                                 + 1.01846 w_k1_IBM_s001
                                 + 0.683599 w_k1_INTC_s001
                                 + 0.824304 w_k1_KO_s001
                                 + 1.49215 w_k1_MAT_s001
                                 + 0.912961 w_k1_MSFT_s001
                                 + 1.05544 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.14966 w_k1_ADM_s001
                                 + 1.14331 w_k1_AXP_s001
                                 + 1.18938 w_k1_DIS_s001
                                 + 1.17962 w_k1_FDX_s001
                                 + 0.846714 w_k1_IBM_s001
                                 + 0.934908 w_k1_INTC_s001
                                 + 0.92247 w_k1_KO_s001 + 1.09254 w_k1_MAT_s001
                                 + 1.14865 w_k1_MSFT_s001
                                 + 0.931836 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.11207 w_k1_ADM_s001
                                 + 0.90649 w_k1_AXP_s001
                                 + 1.04188 w_k1_DIS_s001
                                 + 0.912571 w_k1_FDX_s001
                                 + 1.32321 w_k1_IBM_s001
                                 + 0.825851 w_k1_INTC_s001
                                 + 1.21952 w_k1_KO_s001 + 1.53395 w_k1_MAT_s001
                                 + 1.05147 w_k1_MSFT_s001
                                 + 1.00664 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.485238 w_k1_ADM_s001
                                 + 0.600802 w_k1_AXP_s001
                                 + 0.675456 w_k1_DIS_s001
                                 + 0.726778 w_k1_FDX_s001
                                 + 0.823324 w_k1_IBM_s001
                                 + 0.830943 w_k1_INTC_s001
                                 + 0.885127 w_k1_KO_s001
                                 + 0.850422 w_k1_MAT_s001
                                 + 0.873629 w_k1_MSFT_s001
                                 + 0.871326 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.925415 w_k1_ADM_s001
                                 + 1.49961 w_k1_AXP_s001
                                 + 1.02386 w_k1_DIS_s001
                                 + 1.07246 w_k1_FDX_s001
                                 + 1.22321 w_k1_IBM_s001
                                 + 1.18155 w_k1_INTC_s001
                                 + 1.02805 w_k1_KO_s001 + 1.20735 w_k1_MAT_s001
                                 + 1.01663 w_k1_MSFT_s001
                                 + 1.08875 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.978931 w_k1_ADM_s001
                                 + 0.997491 w_k1_AXP_s001
                                 + 1.04724 w_k1_DIS_s001
                                 + 1.04076 w_k1_FDX_s001
                                 + 1.26649 w_k1_IBM_s001
                                 + 0.922532 w_k1_INTC_s001
                                 + 1.18371 w_k1_KO_s001 + 1.60968 w_k1_MAT_s001
                                 + 1.03629 w_k1_MSFT_s001
                                 + 1.00305 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.08319 w_k1_ADM_s002
                                 + 0.811306 w_k1_AXP_s002
                                 + 0.858345 w_k1_DIS_s002
                                 + 0.810776 w_k1_FDX_s002
                                 + 1.09592 w_k1_IBM_s002
                                 + 1.00303 w_k1_INTC_s002
                                 + 1.11673 w_k1_KO_s002
                                 + 0.814406 w_k1_MAT_s002
                                 + 0.93035 w_k1_MSFT_s002
                                 + 1.09266 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.05031 w_k1_ADM_s002
                                 + 1.21608 w_k1_AXP_s002
                                 + 1.16757 w_k1_DIS_s002 + 1.1641 w_k1_FDX_s002
                                 + 1.06513 w_k1_IBM_s002
                                 + 1.07356 w_k1_INTC_s002
                                 + 1.04357 w_k1_KO_s002 + 1.0569 w_k1_MAT_s002
                                 + 1.07624 w_k1_MSFT_s002
                                 + 1.00552 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.32894 w_k1_ADM_s002
                                 + 1.67751 w_k1_AXP_s002
                                 + 1.45164 w_k1_DIS_s002
                                 + 1.35739 w_k1_FDX_s002 + 1.1298 w_k1_IBM_s002
                                 + 1.37607 w_k1_INTC_s002
                                 + 1.09983 w_k1_KO_s002 + 1.1899 w_k1_MAT_s002
                                 + 1.32996 w_k1_MSFT_s002
                                 + 1.24311 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.34557 w_k1_ADM_s002
                                 + 1.23003 w_k1_AXP_s002
                                 + 0.981129 w_k1_DIS_s002
                                 + 1.00875 w_k1_FDX_s002
                                 + 1.12542 w_k1_IBM_s002
                                 + 1.05009 w_k1_INTC_s002
                                 + 1.02062 w_k1_KO_s002 + 1.04847 w_k1_MAT_s002
                                 + 0.96206 w_k1_MSFT_s002
                                 + 0.983081 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.86115 w_k1_ADM_s002
                                 + 0.753738 w_k1_AXP_s002
                                 + 0.634746 w_k1_DIS_s002
                                 + 0.809647 w_k1_FDX_s002
                                 + 0.788175 w_k1_IBM_s002
                                 + 0.877804 w_k1_INTC_s002
                                 + 0.812547 w_k1_KO_s002 + 1.0435 w_k1_MAT_s002
                                 + 0.981175 w_k1_MSFT_s002
                                 + 1.06592 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.16478 w_k1_ADM_s002
                                 + 0.794194 w_k1_AXP_s002
                                 + 1.09044 w_k1_DIS_s002
                                 + 0.979827 w_k1_FDX_s002
                                 + 0.809292 w_k1_IBM_s002
                                 + 1.14046 w_k1_INTC_s002
                                 + 0.973408 w_k1_KO_s002
                                 + 0.56004 w_k1_MAT_s002
                                 + 0.973935 w_k1_MSFT_s002
                                 + 1.08815 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.890616 w_k1_ADM_s002
                                 + 1.08388 w_k1_AXP_s002 + 1.2159 w_k1_DIS_s002
                                 + 0.99062 w_k1_FDX_s002
                                 + 1.39642 w_k1_IBM_s002
                                 + 1.24116 w_k1_INTC_s002
                                 + 1.30298 w_k1_KO_s002 + 1.31346 w_k1_MAT_s002
                                 + 1.06831 w_k1_MSFT_s002
                                 + 1.14035 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.21519 w_k1_ADM_s002
                                 + 1.01033 w_k1_AXP_s002
                                 + 0.903614 w_k1_DIS_s002
                                 + 1.02337 w_k1_FDX_s002
                                 + 0.846636 w_k1_IBM_s002
                                 + 0.956395 w_k1_INTC_s002
                                 + 0.943345 w_k1_KO_s002
                                 + 0.755672 w_k1_MAT_s002
                                 + 0.981866 w_k1_MSFT_s002
                                 + 0.990662 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.99457 w_k1_ADM_s003
                                 + 1.22364 w_k1_AXP_s003
                                 + 1.46002 w_k1_DIS_s003 + 1.3184 w_k1_FDX_s003
                                 + 1.6793 w_k1_IBM_s003
                                 + 1.35938 w_k1_INTC_s003
                                 + 1.48185 w_k1_KO_s003 + 1.30826 w_k1_MAT_s003
                                 + 1.33946 w_k1_MSFT_s003
                                 + 1.29218 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.703413 w_k1_ADM_s003
                                 + 0.995754 w_k1_AXP_s003
                                 + 0.754747 w_k1_DIS_s003
                                 + 0.785128 w_k1_FDX_s003
                                 + 1.61017 w_k1_IBM_s003
                                 + 0.70119 w_k1_INTC_s003
                                 + 0.96195 w_k1_KO_s003 + 1.51864 w_k1_MAT_s003
                                 + 0.783488 w_k1_MSFT_s003
                                 + 0.945456 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.91932 w_k1_ADM_s003
                                 + 1.08081 w_k1_AXP_s003
                                 + 1.22479 w_k1_DIS_s003
                                 + 1.18271 w_k1_FDX_s003
                                 + 0.846179 w_k1_IBM_s003
                                 + 0.813221 w_k1_INTC_s003
                                 + 1.01631 w_k1_KO_s003 + 1.07095 w_k1_MAT_s003
                                 + 0.990488 w_k1_MSFT_s003
                                 + 1.24445 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.700563 w_k1_ADM_s003
                                 + 0.882442 w_k1_AXP_s003
                                 + 0.667715 w_k1_DIS_s003
                                 + 0.739532 w_k1_FDX_s003
                                 + 0.946834 w_k1_IBM_s003
                                 + 0.701659 w_k1_INTC_s003
                                 + 0.681554 w_k1_KO_s003
                                 + 1.00312 w_k1_MAT_s003
                                 + 0.720392 w_k1_MSFT_s003
                                 + 0.865843 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.652214 w_k1_ADM_s003
                                 + 0.584954 w_k1_AXP_s003
                                 + 0.629108 w_k1_DIS_s003
                                 + 0.673349 w_k1_FDX_s003
                                 + 0.854439 w_k1_IBM_s003
                                 + 0.849319 w_k1_INTC_s003
                                 + 0.736512 w_k1_KO_s003
                                 + 0.489231 w_k1_MAT_s003
                                 + 0.808005 w_k1_MSFT_s003
                                 + 0.821396 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.851611 w_k1_ADM_s003
                                 + 0.836839 w_k1_AXP_s003
                                 + 0.665252 w_k1_DIS_s003
                                 + 1.00593 w_k1_FDX_s003
                                 + 0.478373 w_k1_IBM_s003
                                 + 0.930064 w_k1_INTC_s003
                                 + 0.674113 w_k1_KO_s003
                                 + 0.45881 w_k1_MAT_s003
                                 + 0.978778 w_k1_MSFT_s003
                                 + 0.907456 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.798355 w_k1_ADM_s003
                                 + 1.0162 w_k1_AXP_s003
                                 + 0.919909 w_k1_DIS_s003
                                 + 0.9749 w_k1_FDX_s003 + 1.08747 w_k1_IBM_s003
                                 + 1.04186 w_k1_INTC_s003
                                 + 0.863375 w_k1_KO_s003
                                 + 0.838457 w_k1_MAT_s003
                                 + 0.983736 w_k1_MSFT_s003
                                 + 1.04006 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.868954 w_k1_ADM_s003
                                 + 0.999129 w_k1_AXP_s003
                                 + 0.898568 w_k1_DIS_s003
                                 + 1.0094 w_k1_FDX_s003
                                 + 0.664347 w_k1_IBM_s003
                                 + 0.894845 w_k1_INTC_s003
                                 + 0.747915 w_k1_KO_s003
                                 + 0.672921 w_k1_MAT_s003
                                 + 0.927331 w_k1_MSFT_s003
                                 + 0.916125 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.771476 w_k1_ADM_s004
                                 + 0.6739 w_k1_AXP_s004
                                 + 0.542322 w_k1_DIS_s004
                                 + 0.662333 w_k1_FDX_s004
                                 + 0.991783 w_k1_IBM_s004
                                 + 0.887992 w_k1_INTC_s004
                                 + 0.946996 w_k1_KO_s004
                                 + 1.17684 w_k1_MAT_s004
                                 + 0.77595 w_k1_MSFT_s004
                                 + 0.889298 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.854204 w_k1_ADM_s004
                                 + 1.02955 w_k1_AXP_s004
                                 + 0.874124 w_k1_DIS_s004
                                 + 0.982385 w_k1_FDX_s004
                                 + 1.09066 w_k1_IBM_s004
                                 + 0.957103 w_k1_INTC_s004
                                 + 0.930815 w_k1_KO_s004
                                 + 0.914082 w_k1_MAT_s004
                                 + 0.888367 w_k1_MSFT_s004
                                 + 1.01038 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.923685 w_k1_ADM_s004
                                 + 1.332 w_k1_AXP_s004 + 1.23472 w_k1_DIS_s004
                                 + 1.13172 w_k1_FDX_s004
                                 + 1.44583 w_k1_IBM_s004
                                 + 1.49624 w_k1_INTC_s004
                                 + 1.33268 w_k1_KO_s004 + 1.33238 w_k1_MAT_s004
                                 + 1.14195 w_k1_MSFT_s004
                                 + 1.16697 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.67352 w_k1_ADM_s004
                                 + 0.989079 w_k1_AXP_s004
                                 + 0.707414 w_k1_DIS_s004
                                 + 0.89518 w_k1_FDX_s004
                                 + 0.648713 w_k1_IBM_s004
                                 + 1.10066 w_k1_INTC_s004
                                 + 0.789424 w_k1_KO_s004
                                 + 0.664173 w_k1_MAT_s004
                                 + 0.853395 w_k1_MSFT_s004
                                 + 0.863341 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.655708 w_k1_ADM_s004
                                 + 0.727778 w_k1_AXP_s004
                                 + 1.08464 w_k1_DIS_s004
                                 + 0.922102 w_k1_FDX_s004
                                 + 0.950599 w_k1_IBM_s004
                                 + 0.994788 w_k1_INTC_s004
                                 + 1.18873 w_k1_KO_s004
                                 + 0.934477 w_k1_MAT_s004
                                 + 0.9838 w_k1_MSFT_s004
                                 + 0.954749 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.1881 w_k1_ADM_s004
                                 + 1.82397 w_k1_AXP_s004 + 1.6953 w_k1_DIS_s004
                                 + 1.22201 w_k1_FDX_s004
                                 + 1.48142 w_k1_IBM_s004
                                 + 1.36341 w_k1_INTC_s004
                                 + 1.55893 w_k1_KO_s004 + 1.43873 w_k1_MAT_s004
                                 + 1.06662 w_k1_MSFT_s004 + 1.3669 w_k1_PG_s004
                                 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.851006 w_k1_ADM_s004
                                 + 1.0205 w_k1_AXP_s004
                                 + 0.804653 w_k1_DIS_s004
                                 + 0.899202 w_k1_FDX_s004
                                 + 1.55731 w_k1_IBM_s004
                                 + 0.859729 w_k1_INTC_s004
                                 + 1.23957 w_k1_KO_s004 + 1.44764 w_k1_MAT_s004
                                 + 0.740297 w_k1_MSFT_s004
                                 + 1.04057 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.91582 w_k1_ADM_s004
                                 + 1.39262 w_k1_AXP_s004
                                 + 1.22752 w_k1_DIS_s004 + 1.1583 w_k1_FDX_s004
                                 + 0.975531 w_k1_IBM_s004
                                 + 1.48561 w_k1_INTC_s004
                                 + 1.11777 w_k1_KO_s004
                                 + 0.814682 w_k1_MAT_s004
                                 + 1.15728 w_k1_MSFT_s004 + 1.0073 w_k1_PG_s004
                                 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.686035 w_k1_ADM_s005
                                 + 1.00027 w_k1_AXP_s005
                                 + 1.20218 w_k1_DIS_s005
                                 + 1.01742 w_k1_FDX_s005
                                 + 1.20095 w_k1_IBM_s005
                                 + 1.22048 w_k1_INTC_s005
                                 + 1.25265 w_k1_KO_s005 + 1.00546 w_k1_MAT_s005
                                 + 1.06322 w_k1_MSFT_s005
                                 + 0.917924 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.913019 w_k1_ADM_s005
                                 + 1.23201 w_k1_AXP_s005
                                 + 0.863352 w_k1_DIS_s005
                                 + 0.937118 w_k1_FDX_s005
                                 + 0.995138 w_k1_IBM_s005
                                 + 0.886861 w_k1_INTC_s005
                                 + 0.817402 w_k1_KO_s005
                                 + 1.04149 w_k1_MAT_s005
                                 + 0.870579 w_k1_MSFT_s005
                                 + 0.846681 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.00294 w_k1_ADM_s005
                                 + 1.13509 w_k1_AXP_s005
                                 + 1.11621 w_k1_DIS_s005
                                 + 1.13883 w_k1_FDX_s005
                                 + 0.697334 w_k1_IBM_s005
                                 + 0.904351 w_k1_INTC_s005
                                 + 0.853747 w_k1_KO_s005
                                 + 0.572861 w_k1_MAT_s005
                                 + 1.05344 w_k1_MSFT_s005
                                 + 0.976297 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.00276 w_k1_ADM_s005
                                 + 0.950816 w_k1_AXP_s005
                                 + 0.897741 w_k1_DIS_s005
                                 + 0.83512 w_k1_FDX_s005
                                 + 1.34457 w_k1_IBM_s005
                                 + 0.864849 w_k1_INTC_s005
                                 + 1.12642 w_k1_KO_s005 + 1.42471 w_k1_MAT_s005
                                 + 1.00717 w_k1_MSFT_s005
                                 + 0.954917 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.10172 w_k1_ADM_s005
                                 + 1.67804 w_k1_AXP_s005
                                 + 1.86073 w_k1_DIS_s005
                                 + 1.52908 w_k1_FDX_s005
                                 + 0.988763 w_k1_IBM_s005
                                 + 1.15883 w_k1_INTC_s005
                                 + 1.07715 w_k1_KO_s005 + 1.1169 w_k1_MAT_s005
                                 + 1.24835 w_k1_MSFT_s005
                                 + 1.07027 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.591115 w_k1_ADM_s005
                                 + 0.729505 w_k1_AXP_s005
                                 + 0.58406 w_k1_DIS_s005
                                 + 0.587627 w_k1_FDX_s005
                                 + 1.07815 w_k1_IBM_s005
                                 + 0.505836 w_k1_INTC_s005
                                 + 0.715004 w_k1_KO_s005
                                 + 0.984246 w_k1_MAT_s005
                                 + 0.616494 w_k1_MSFT_s005
                                 + 0.746838 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.715283 w_k1_ADM_s005
                                 + 0.591141 w_k1_AXP_s005
                                 + 0.778364 w_k1_DIS_s005
                                 + 0.802752 w_k1_FDX_s005
                                 + 0.751912 w_k1_IBM_s005
                                 + 0.845354 w_k1_INTC_s005
                                 + 0.768907 w_k1_KO_s005
                                 + 0.659939 w_k1_MAT_s005
                                 + 0.927984 w_k1_MSFT_s005
                                 + 0.866449 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.831652 w_k1_ADM_s005
                                 + 1.42436 w_k1_AXP_s005
                                 + 1.23199 w_k1_DIS_s005
                                 + 1.08273 w_k1_FDX_s005
                                 + 1.11647 w_k1_IBM_s005
                                 + 0.988838 w_k1_INTC_s005
                                 + 0.928408 w_k1_KO_s005
                                 + 1.13379 w_k1_MAT_s005
                                 + 0.92941 w_k1_MSFT_s005
                                 + 0.911339 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.00966 w_k1_ADM_s006
                                 + 0.870114 w_k1_AXP_s006
                                 + 0.930126 w_k1_DIS_s006
                                 + 1.01599 w_k1_FDX_s006
                                 + 0.774713 w_k1_IBM_s006
                                 + 1.07078 w_k1_INTC_s006
                                 + 0.930082 w_k1_KO_s006
                                 + 0.77128 w_k1_MAT_s006
                                 + 1.03774 w_k1_MSFT_s006
                                 + 1.09496 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.926037 w_k1_ADM_s006
                                 + 0.870199 w_k1_AXP_s006
                                 + 1.28884 w_k1_DIS_s006
                                 + 1.03491 w_k1_FDX_s006
                                 + 1.61397 w_k1_IBM_s006
                                 + 1.10266 w_k1_INTC_s006
                                 + 1.49377 w_k1_KO_s006 + 1.63272 w_k1_MAT_s006
                                 + 1.31351 w_k1_MSFT_s006
                                 + 1.35571 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.867158 w_k1_ADM_s006
                                 + 0.714661 w_k1_AXP_s006
                                 + 1.1011 w_k1_DIS_s006
                                 + 0.971198 w_k1_FDX_s006
                                 + 0.758379 w_k1_IBM_s006
                                 + 0.919477 w_k1_INTC_s006
                                 + 0.940523 w_k1_KO_s006
                                 + 0.651151 w_k1_MAT_s006
                                 + 1.08411 w_k1_MSFT_s006
                                 + 0.975877 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.04434 w_k1_ADM_s006
                                 + 0.684983 w_k1_AXP_s006
                                 + 0.682431 w_k1_DIS_s006
                                 + 0.753217 w_k1_FDX_s006
                                 + 0.810301 w_k1_IBM_s006
                                 + 0.601381 w_k1_INTC_s006
                                 + 0.695165 w_k1_KO_s006
                                 + 0.722016 w_k1_MAT_s006
                                 + 0.772409 w_k1_MSFT_s006
                                 + 0.921933 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.935697 w_k1_ADM_s006
                                 + 0.891107 w_k1_AXP_s006
                                 + 0.997561 w_k1_DIS_s006
                                 + 1.00035 w_k1_FDX_s006
                                 + 1.07577 w_k1_IBM_s006
                                 + 1.09476 w_k1_INTC_s006
                                 + 1.18784 w_k1_KO_s006 + 1.05144 w_k1_MAT_s006
                                 + 1.22636 w_k1_MSFT_s006
                                 + 0.992315 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.926559 w_k1_ADM_s006
                                 + 1.09073 w_k1_AXP_s006
                                 + 1.57294 w_k1_DIS_s006
                                 + 1.25888 w_k1_FDX_s006
                                 + 0.958067 w_k1_IBM_s006
                                 + 1.29029 w_k1_INTC_s006
                                 + 1.11599 w_k1_KO_s006
                                 + 0.885859 w_k1_MAT_s006
                                 + 1.38859 w_k1_MSFT_s006
                                 + 1.21688 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.12091 w_k1_ADM_s006
                                 + 1.02218 w_k1_AXP_s006
                                 + 1.13087 w_k1_DIS_s006
                                 + 1.16367 w_k1_FDX_s006
                                 + 1.08795 w_k1_IBM_s006
                                 + 1.05293 w_k1_INTC_s006
                                 + 1.16904 w_k1_KO_s006 + 1.25716 w_k1_MAT_s006
                                 + 1.16884 w_k1_MSFT_s006
                                 + 1.23751 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.07938 w_k1_ADM_s006
                                 + 0.831689 w_k1_AXP_s006
                                 + 0.937633 w_k1_DIS_s006
                                 + 0.901884 w_k1_FDX_s006
                                 + 0.583965 w_k1_IBM_s006
                                 + 0.79231 w_k1_INTC_s006 + 0.9972 w_k1_KO_s006
                                 + 0.688609 w_k1_MAT_s006
                                 + 1.00178 w_k1_MSFT_s006
                                 + 1.09601 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.640656 w_k1_ADM_s007
                                 + 0.624603 w_k1_AXP_s007
                                 + 0.670849 w_k1_DIS_s007
                                 + 0.69389 w_k1_FDX_s007
                                 + 0.825392 w_k1_IBM_s007
                                 + 0.768024 w_k1_INTC_s007
                                 + 0.783228 w_k1_KO_s007
                                 + 0.497477 w_k1_MAT_s007
                                 + 0.786485 w_k1_MSFT_s007
                                 + 0.837776 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.717345 w_k1_ADM_s007
                                 + 0.671589 w_k1_AXP_s007
                                 + 1.14887 w_k1_DIS_s007
                                 + 0.841958 w_k1_FDX_s007
                                 + 1.13804 w_k1_IBM_s007
                                 + 0.875834 w_k1_INTC_s007
                                 + 1.14148 w_k1_KO_s007 + 0.81067 w_k1_MAT_s007
                                 + 1.13845 w_k1_MSFT_s007
                                 + 1.10066 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.03824 w_k1_ADM_s007
                                 + 0.883111 w_k1_AXP_s007
                                 + 1.04109 w_k1_DIS_s007
                                 + 1.01157 w_k1_FDX_s007 + 1.1158 w_k1_IBM_s007
                                 + 1.00735 w_k1_INTC_s007
                                 + 1.20424 w_k1_KO_s007
                                 + 0.863386 w_k1_MAT_s007
                                 + 1.10545 w_k1_MSFT_s007
                                 + 1.07626 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.01903 w_k1_ADM_s007
                                 + 1.32031 w_k1_AXP_s007
                                 + 1.46078 w_k1_DIS_s007
                                 + 1.26773 w_k1_FDX_s007
                                 + 1.57539 w_k1_IBM_s007
                                 + 1.24882 w_k1_INTC_s007 + 1.3794 w_k1_KO_s007
                                 + 1.13186 w_k1_MAT_s007
                                 + 1.23096 w_k1_MSFT_s007
                                 + 1.33645 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.01609 w_k1_ADM_s007
                                 + 0.771454 w_k1_AXP_s007
                                 + 0.818829 w_k1_DIS_s007
                                 + 0.926576 w_k1_FDX_s007
                                 + 0.690061 w_k1_IBM_s007
                                 + 0.759175 w_k1_INTC_s007
                                 + 0.76183 w_k1_KO_s007
                                 + 0.546205 w_k1_MAT_s007
                                 + 1.03864 w_k1_MSFT_s007
                                 + 0.975273 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.851976 w_k1_ADM_s007
                                 + 1.13647 w_k1_AXP_s007
                                 + 1.22755 w_k1_DIS_s007
                                 + 0.958536 w_k1_FDX_s007
                                 + 1.13686 w_k1_IBM_s007
                                 + 1.02689 w_k1_INTC_s007
                                 + 1.03059 w_k1_KO_s007 + 1.0631 w_k1_MAT_s007
                                 + 1.11963 w_k1_MSFT_s007
                                 + 1.18491 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.09613 w_k1_ADM_s007
                                 + 1.26091 w_k1_AXP_s007 + 1.2157 w_k1_DIS_s007
                                 + 1.24918 w_k1_FDX_s007
                                 + 1.04685 w_k1_IBM_s007
                                 + 1.12221 w_k1_INTC_s007 + 1.2534 w_k1_KO_s007
                                 + 1.15422 w_k1_MAT_s007
                                 + 1.14355 w_k1_MSFT_s007
                                 + 1.25778 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.726144 w_k1_ADM_s007
                                 + 0.948279 w_k1_AXP_s007
                                 + 0.783419 w_k1_DIS_s007
                                 + 0.746636 w_k1_FDX_s007
                                 + 1.35482 w_k1_IBM_s007
                                 + 0.845196 w_k1_INTC_s007
                                 + 1.1374 w_k1_KO_s007 + 1.05215 w_k1_MAT_s007
                                 + 0.926759 w_k1_MSFT_s007
                                 + 1.01043 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.64626 w_k1_ADM_s010
                                 + 0.734443 w_k1_AXP_s010
                                 + 0.881116 w_k1_DIS_s010
                                 + 0.86934 w_k1_FDX_s010
                                 + 0.98434 w_k1_IBM_s010
                                 + 0.781026 w_k1_INTC_s010
                                 + 0.905851 w_k1_KO_s010
                                 + 0.926109 w_k1_MAT_s010
                                 + 0.930317 w_k1_MSFT_s010
                                 + 0.812021 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.721677 w_k1_ADM_s010
                                 + 0.730831 w_k1_AXP_s010
                                 + 0.72673 w_k1_DIS_s010
                                 + 0.806751 w_k1_FDX_s010
                                 + 1.12645 w_k1_IBM_s010
                                 + 0.741987 w_k1_INTC_s010
                                 + 0.898654 w_k1_KO_s010
                                 + 1.02281 w_k1_MAT_s010
                                 + 0.83635 w_k1_MSFT_s010
                                 + 0.888604 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.901407 w_k1_ADM_s010
                                 + 0.819904 w_k1_AXP_s010
                                 + 1.03527 w_k1_DIS_s010
                                 + 1.01212 w_k1_FDX_s010
                                 + 0.863178 w_k1_IBM_s010
                                 + 0.91689 w_k1_INTC_s010
                                 + 1.00199 w_k1_KO_s010
                                 + 0.985512 w_k1_MAT_s010
                                 + 1.01957 w_k1_MSFT_s010
                                 + 0.983239 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.912308 w_k1_ADM_s010
                                 + 0.801068 w_k1_AXP_s010
                                 + 0.809636 w_k1_DIS_s010
                                 + 1.00554 w_k1_FDX_s010
                                 + 0.604658 w_k1_IBM_s010
                                 + 0.731657 w_k1_INTC_s010
                                 + 0.783252 w_k1_KO_s010
                                 + 0.753707 w_k1_MAT_s010
                                 + 0.97122 w_k1_MSFT_s010
                                 + 0.915953 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.24708 w_k1_ADM_s010
                                 + 1.32239 w_k1_AXP_s010 + 1.5611 w_k1_DIS_s010
                                 + 1.38836 w_k1_FDX_s010
                                 + 1.07057 w_k1_IBM_s010
                                 + 1.01458 w_k1_INTC_s010
                                 + 1.19501 w_k1_KO_s010 + 1.19903 w_k1_MAT_s010
                                 + 1.2921 w_k1_MSFT_s010 + 1.24402 w_k1_PG_s010
                                 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.925204 w_k1_ADM_s010
                                 + 1.13826 w_k1_AXP_s010
                                 + 1.20404 w_k1_DIS_s010
                                 + 1.08431 w_k1_FDX_s010 + 1.5175 w_k1_IBM_s010
                                 + 0.89912 w_k1_INTC_s010
                                 + 1.31423 w_k1_KO_s010 + 1.47434 w_k1_MAT_s010
                                 + 0.924413 w_k1_MSFT_s010
                                 + 1.13377 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.01188 w_k1_ADM_s010
                                 + 1.19722 w_k1_AXP_s010
                                 + 1.06389 w_k1_DIS_s010
                                 + 1.08492 w_k1_FDX_s010
                                 + 1.12385 w_k1_IBM_s010
                                 + 0.730108 w_k1_INTC_s010
                                 + 1.19197 w_k1_KO_s010 + 1.39694 w_k1_MAT_s010
                                 + 1.01487 w_k1_MSFT_s010
                                 + 1.11248 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.11412 w_k1_ADM_s010
                                 + 1.44552 w_k1_AXP_s010
                                 + 1.46278 w_k1_DIS_s010
                                 + 1.25856 w_k1_FDX_s010 + 1.9601 w_k1_IBM_s010
                                 + 1.04291 w_k1_INTC_s010 + 1.6557 w_k1_KO_s010
                                 + 2.04122 w_k1_MAT_s010
                                 + 1.16881 w_k1_MSFT_s010
                                 + 1.35641 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.05626 w_k1_ADM_s011
                                 + 1.24533 w_k1_AXP_s011
                                 + 0.833281 w_k1_DIS_s011
                                 + 1.0254 w_k1_FDX_s011
                                 + 0.944403 w_k1_IBM_s011
                                 + 0.923604 w_k1_INTC_s011
                                 + 0.984729 w_k1_KO_s011
                                 + 1.54747 w_k1_MAT_s011
                                 + 1.02586 w_k1_MSFT_s011
                                 + 0.965381 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.89101 w_k1_ADM_s011
                                 + 0.904955 w_k1_AXP_s011
                                 + 0.81226 w_k1_DIS_s011
                                 + 0.828492 w_k1_FDX_s011
                                 + 0.98146 w_k1_IBM_s011
                                 + 1.09605 w_k1_INTC_s011
                                 + 0.947989 w_k1_KO_s011
                                 + 0.809456 w_k1_MAT_s011
                                 + 0.936339 w_k1_MSFT_s011
                                 + 0.956701 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.45912 w_k1_ADM_s011
                                 + 1.70152 w_k1_AXP_s011
                                 + 1.53754 w_k1_DIS_s011
                                 + 1.42359 w_k1_FDX_s011
                                 + 1.26024 w_k1_IBM_s011
                                 + 1.41841 w_k1_INTC_s011
                                 + 1.23171 w_k1_KO_s011 + 1.22864 w_k1_MAT_s011
                                 + 1.15755 w_k1_MSFT_s011
                                 + 1.17634 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.10318 w_k1_ADM_s011
                                 + 0.978316 w_k1_AXP_s011
                                 + 0.942192 w_k1_DIS_s011
                                 + 1.14942 w_k1_FDX_s011
                                 + 0.702742 w_k1_IBM_s011
                                 + 1.14976 w_k1_INTC_s011
                                 + 0.769659 w_k1_KO_s011
                                 + 0.710492 w_k1_MAT_s011
                                 + 1.04365 w_k1_MSFT_s011
                                 + 0.858587 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.953255 w_k1_ADM_s011
                                 + 1.08636 w_k1_AXP_s011
                                 + 1.01178 w_k1_DIS_s011
                                 + 1.17566 w_k1_FDX_s011
                                 + 1.10162 w_k1_IBM_s011
                                 + 1.08211 w_k1_INTC_s011
                                 + 1.04519 w_k1_KO_s011 + 1.26791 w_k1_MAT_s011
                                 + 1.11137 w_k1_MSFT_s011 + 1.0961 w_k1_PG_s011
                                 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.06229 w_k1_ADM_s011
                                 + 0.963676 w_k1_AXP_s011
                                 + 1.17513 w_k1_DIS_s011
                                 + 1.05611 w_k1_FDX_s011
                                 + 1.20253 w_k1_IBM_s011
                                 + 1.30213 w_k1_INTC_s011
                                 + 1.33733 w_k1_KO_s011
                                 + 0.809718 w_k1_MAT_s011
                                 + 1.09089 w_k1_MSFT_s011
                                 + 0.997452 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.991989 w_k1_ADM_s011
                                 + 0.749387 w_k1_AXP_s011
                                 + 0.610242 w_k1_DIS_s011
                                 + 0.819756 w_k1_FDX_s011
                                 + 0.744818 w_k1_IBM_s011
                                 + 0.916243 w_k1_INTC_s011
                                 + 0.731205 w_k1_KO_s011
                                 + 0.693361 w_k1_MAT_s011
                                 + 0.765553 w_k1_MSFT_s011
                                 + 0.796708 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.804768 w_k1_ADM_s011
                                 + 0.647387 w_k1_AXP_s011
                                 + 0.573714 w_k1_DIS_s011
                                 + 0.653841 w_k1_FDX_s011
                                 + 0.682873 w_k1_IBM_s011
                                 + 0.828792 w_k1_INTC_s011
                                 + 0.688874 w_k1_KO_s011
                                 + 0.56151 w_k1_MAT_s011
                                 + 0.731612 w_k1_MSFT_s011
                                 + 0.736209 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.777557 w_k1_ADM_s012
                                 + 0.993949 w_k1_AXP_s012
                                 + 0.663172 w_k1_DIS_s012
                                 + 0.778003 w_k1_FDX_s012
                                 + 1.60983 w_k1_IBM_s012
                                 + 0.703971 w_k1_INTC_s012
                                 + 0.981459 w_k1_KO_s012
                                 + 1.68895 w_k1_MAT_s012
                                 + 0.629181 w_k1_MSFT_s012
                                 + 0.913924 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.05255 w_k1_ADM_s012
                                 + 0.903288 w_k1_AXP_s012
                                 + 0.744303 w_k1_DIS_s012
                                 + 0.870445 w_k1_FDX_s012
                                 + 0.967305 w_k1_IBM_s012
                                 + 0.81511 w_k1_INTC_s012
                                 + 0.892598 w_k1_KO_s012
                                 + 1.13169 w_k1_MAT_s012
                                 + 0.921952 w_k1_MSFT_s012
                                 + 0.972122 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.55866 w_k1_ADM_s012
                                 + 1.16503 w_k1_AXP_s012
                                 + 1.25262 w_k1_DIS_s012
                                 + 1.21641 w_k1_FDX_s012
                                 + 0.740116 w_k1_IBM_s012
                                 + 0.877767 w_k1_INTC_s012
                                 + 1.08909 w_k1_KO_s012
                                 + 0.941891 w_k1_MAT_s012
                                 + 1.05149 w_k1_MSFT_s012
                                 + 1.14099 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.21168 w_k1_ADM_s012
                                 + 1.2129 w_k1_AXP_s012 + 1.65986 w_k1_DIS_s012
                                 + 1.28021 w_k1_FDX_s012
                                 + 1.14748 w_k1_IBM_s012
                                 + 1.12991 w_k1_INTC_s012
                                 + 1.28603 w_k1_KO_s012 + 1.04968 w_k1_MAT_s012
                                 + 1.22523 w_k1_MSFT_s012
                                 + 1.10346 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.13997 w_k1_ADM_s012
                                 + 1.25682 w_k1_AXP_s012
                                 + 0.961843 w_k1_DIS_s012
                                 + 0.984748 w_k1_FDX_s012
                                 + 1.11341 w_k1_IBM_s012
                                 + 0.989668 w_k1_INTC_s012
                                 + 1.00147 w_k1_KO_s012 + 1.43915 w_k1_MAT_s012
                                 + 0.809413 w_k1_MSFT_s012
                                 + 0.959106 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.04321 w_k1_ADM_s012
                                 + 0.988834 w_k1_AXP_s012
                                 + 1.21576 w_k1_DIS_s012
                                 + 0.887737 w_k1_FDX_s012
                                 + 1.77335 w_k1_IBM_s012
                                 + 0.700358 w_k1_INTC_s012
                                 + 1.26238 w_k1_KO_s012 + 2.13197 w_k1_MAT_s012
                                 + 0.926388 w_k1_MSFT_s012
                                 + 1.05617 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.0756 w_k1_ADM_s012
                                 + 0.764913 w_k1_AXP_s012
                                 + 0.978957 w_k1_DIS_s012
                                 + 1.04115 w_k1_FDX_s012
                                 + 0.691261 w_k1_IBM_s012
                                 + 0.735916 w_k1_INTC_s012
                                 + 0.83305 w_k1_KO_s012 + 0.89602 w_k1_MAT_s012
                                 + 0.9822 w_k1_MSFT_s012
                                 + 0.888809 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.896487 w_k1_ADM_s012
                                 + 0.586418 w_k1_AXP_s012
                                 + 0.681707 w_k1_DIS_s012
                                 + 0.792539 w_k1_FDX_s012
                                 + 0.651246 w_k1_IBM_s012
                                 + 0.677147 w_k1_INTC_s012
                                 + 0.65523 w_k1_KO_s012
                                 + 0.567346 w_k1_MAT_s012
                                 + 0.774962 w_k1_MSFT_s012
                                 + 0.777988 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.23767 w_k1_ADM_s013
                                 + 1.29051 w_k1_AXP_s013 + 1.2543 w_k1_DIS_s013
                                 + 1.10931 w_k1_FDX_s013
                                 + 1.44985 w_k1_IBM_s013
                                 + 1.20619 w_k1_INTC_s013
                                 + 1.33843 w_k1_KO_s013 + 1.73076 w_k1_MAT_s013
                                 + 1.14759 w_k1_MSFT_s013
                                 + 1.08273 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.49772 w_k1_ADM_s013
                                 + 1.25749 w_k1_AXP_s013
                                 + 1.09971 w_k1_DIS_s013
                                 + 1.16044 w_k1_FDX_s013
                                 + 0.840479 w_k1_IBM_s013
                                 + 0.926876 w_k1_INTC_s013
                                 + 1.00471 w_k1_KO_s013
                                 + 0.809437 w_k1_MAT_s013
                                 + 0.951324 w_k1_MSFT_s013
                                 + 1.03369 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.936806 w_k1_ADM_s013
                                 + 0.977065 w_k1_AXP_s013
                                 + 0.894895 w_k1_DIS_s013
                                 + 0.92852 w_k1_FDX_s013
                                 + 1.26648 w_k1_IBM_s013
                                 + 1.00094 w_k1_INTC_s013
                                 + 1.07565 w_k1_KO_s013 + 1.22246 w_k1_MAT_s013
                                 + 0.959319 w_k1_MSFT_s013
                                 + 0.986567 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.894551 w_k1_ADM_s013
                                 + 0.817916 w_k1_AXP_s013
                                 + 1.06086 w_k1_DIS_s013
                                 + 0.944944 w_k1_FDX_s013
                                 + 0.775153 w_k1_IBM_s013
                                 + 0.887414 w_k1_INTC_s013
                                 + 1.01698 w_k1_KO_s013 + 0.68403 w_k1_MAT_s013
                                 + 1.08646 w_k1_MSFT_s013
                                 + 0.886654 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.717986 w_k1_ADM_s013
                                 + 0.795174 w_k1_AXP_s013
                                 + 0.673988 w_k1_DIS_s013
                                 + 0.799523 w_k1_FDX_s013
                                 + 0.829247 w_k1_IBM_s013
                                 + 0.732382 w_k1_INTC_s013
                                 + 0.697787 w_k1_KO_s013
                                 + 0.624603 w_k1_MAT_s013
                                 + 0.706437 w_k1_MSFT_s013
                                 + 0.758828 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.947862 w_k1_ADM_s013
                                 + 1.13757 w_k1_AXP_s013
                                 + 1.18669 w_k1_DIS_s013
                                 + 1.15044 w_k1_FDX_s013
                                 + 1.11356 w_k1_IBM_s013
                                 + 1.11004 w_k1_INTC_s013
                                 + 1.09941 w_k1_KO_s013 + 1.17012 w_k1_MAT_s013
                                 + 1.10603 w_k1_MSFT_s013
                                 + 1.14777 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.890839 w_k1_ADM_s013
                                 + 1.11699 w_k1_AXP_s013 + 1.1317 w_k1_DIS_s013
                                 + 1.03384 w_k1_FDX_s013
                                 + 0.977112 w_k1_IBM_s013
                                 + 1.13852 w_k1_INTC_s013
                                 + 1.00844 w_k1_KO_s013
                                 + 0.807659 w_k1_MAT_s013
                                 + 1.08991 w_k1_MSFT_s013
                                 + 0.942027 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.45221 w_k1_ADM_s013
                                 + 1.95176 w_k1_AXP_s013
                                 + 1.61185 w_k1_DIS_s013
                                 + 1.48839 w_k1_FDX_s013
                                 + 1.33863 w_k1_IBM_s013
                                 + 1.17097 w_k1_INTC_s013
                                 + 1.33671 w_k1_KO_s013 + 2.31777 w_k1_MAT_s013
                                 + 1.24129 w_k1_MSFT_s013
                                 + 1.23042 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.698444 w_k1_ADM_s014
                                 + 0.681408 w_k1_AXP_s014
                                 + 0.653569 w_k1_DIS_s014
                                 + 0.712684 w_k1_FDX_s014
                                 + 0.845891 w_k1_IBM_s014
                                 + 0.727413 w_k1_INTC_s014
                                 + 0.74875 w_k1_KO_s014
                                 + 0.660322 w_k1_MAT_s014
                                 + 0.872842 w_k1_MSFT_s014
                                 + 0.898446 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.987616 w_k1_ADM_s014
                                 + 0.928705 w_k1_AXP_s014
                                 + 0.920014 w_k1_DIS_s014
                                 + 0.866819 w_k1_FDX_s014
                                 + 0.889145 w_k1_IBM_s014
                                 + 0.9521 w_k1_INTC_s014
                                 + 0.944305 w_k1_KO_s014
                                 + 0.633709 w_k1_MAT_s014
                                 + 0.999343 w_k1_MSFT_s014
                                 + 0.916708 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.7152 w_k1_ADM_s014
                                 + 1.65705 w_k1_AXP_s014
                                 + 1.83904 w_k1_DIS_s014
                                 + 1.41275 w_k1_FDX_s014
                                 + 1.29663 w_k1_IBM_s014
                                 + 1.15208 w_k1_INTC_s014
                                 + 1.58302 w_k1_KO_s014 + 2.02328 w_k1_MAT_s014
                                 + 1.70558 w_k1_MSFT_s014
                                 + 1.33038 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.12109 w_k1_ADM_s014
                                 + 0.873686 w_k1_AXP_s014
                                 + 0.744407 w_k1_DIS_s014
                                 + 0.825408 w_k1_FDX_s014 + 0.667 w_k1_IBM_s014
                                 + 0.838413 w_k1_INTC_s014
                                 + 0.813374 w_k1_KO_s014
                                 + 0.529514 w_k1_MAT_s014
                                 + 0.830297 w_k1_MSFT_s014
                                 + 0.850236 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.46493 w_k1_ADM_s014
                                 + 1.15461 w_k1_AXP_s014
                                 + 1.41243 w_k1_DIS_s014
                                 + 1.17198 w_k1_FDX_s014
                                 + 1.03089 w_k1_IBM_s014
                                 + 1.06154 w_k1_INTC_s014
                                 + 1.29042 w_k1_KO_s014 + 1.46309 w_k1_MAT_s014
                                 + 1.31852 w_k1_MSFT_s014
                                 + 1.22896 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.15217 w_k1_ADM_s014
                                 + 1.35859 w_k1_AXP_s014
                                 + 1.07226 w_k1_DIS_s014
                                 + 1.15056 w_k1_FDX_s014
                                 + 1.05276 w_k1_IBM_s014
                                 + 1.00842 w_k1_INTC_s014
                                 + 1.01598 w_k1_KO_s014 + 1.35181 w_k1_MAT_s014
                                 + 1.09221 w_k1_MSFT_s014
                                 + 1.09144 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.944798 w_k1_ADM_s014
                                 + 1.05218 w_k1_AXP_s014
                                 + 1.09294 w_k1_DIS_s014
                                 + 1.04137 w_k1_FDX_s014
                                 + 0.843185 w_k1_IBM_s014
                                 + 1.07872 w_k1_INTC_s014
                                 + 0.974293 w_k1_KO_s014
                                 + 0.936917 w_k1_MAT_s014
                                 + 1.08327 w_k1_MSFT_s014
                                 + 1.01277 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.917763 w_k1_ADM_s014
                                 + 1.21194 w_k1_AXP_s014
                                 + 1.17457 w_k1_DIS_s014
                                 + 0.915732 w_k1_FDX_s014
                                 + 1.30186 w_k1_IBM_s014
                                 + 1.03204 w_k1_INTC_s014
                                 + 1.20792 w_k1_KO_s014 + 1.62552 w_k1_MAT_s014
                                 + 1.08751 w_k1_MSFT_s014
                                 + 1.06126 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.17013 w_k1_ADM_s015
                                 + 1.23244 w_k1_AXP_s015
                                 + 1.05828 w_k1_DIS_s015
                                 + 1.06715 w_k1_FDX_s015
                                 + 0.861079 w_k1_IBM_s015
                                 + 1.10725 w_k1_INTC_s015
                                 + 0.878729 w_k1_KO_s015
                                 + 0.99266 w_k1_MAT_s015
                                 + 0.96407 w_k1_MSFT_s015
                                 + 0.979236 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.806275 w_k1_ADM_s015
                                 + 0.75941 w_k1_AXP_s015
                                 + 0.986563 w_k1_DIS_s015
                                 + 0.929582 w_k1_FDX_s015
                                 + 0.800597 w_k1_IBM_s015
                                 + 1.01839 w_k1_INTC_s015
                                 + 0.862415 w_k1_KO_s015
                                 + 0.556164 w_k1_MAT_s015
                                 + 0.939901 w_k1_MSFT_s015
                                 + 0.874069 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.28366 w_k1_ADM_s015
                                 + 1.52304 w_k1_AXP_s015
                                 + 1.09694 w_k1_DIS_s015
                                 + 1.16548 w_k1_FDX_s015
                                 + 1.65566 w_k1_IBM_s015
                                 + 0.978349 w_k1_INTC_s015
                                 + 1.1927 w_k1_KO_s015 + 1.82686 w_k1_MAT_s015
                                 + 1.07778 w_k1_MSFT_s015
                                 + 1.13168 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.76928 w_k1_ADM_s015
                                 + 0.506911 w_k1_AXP_s015
                                 + 0.66678 w_k1_DIS_s015
                                 + 0.664881 w_k1_FDX_s015
                                 + 0.755926 w_k1_IBM_s015
                                 + 0.662913 w_k1_INTC_s015
                                 + 0.711032 w_k1_KO_s015
                                 + 0.571644 w_k1_MAT_s015
                                 + 0.759657 w_k1_MSFT_s015
                                 + 0.82014 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.946783 w_k1_ADM_s015
                                 + 0.677272 w_k1_AXP_s015
                                 + 0.893605 w_k1_DIS_s015
                                 + 0.765681 w_k1_FDX_s015
                                 + 0.987066 w_k1_IBM_s015
                                 + 0.775675 w_k1_INTC_s015
                                 + 0.958535 w_k1_KO_s015
                                 + 1.08935 w_k1_MAT_s015
                                 + 0.82317 w_k1_MSFT_s015
                                 + 0.955034 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.881774 w_k1_ADM_s015
                                 + 1.07491 w_k1_AXP_s015
                                 + 0.748624 w_k1_DIS_s015
                                 + 0.8685 w_k1_FDX_s015
                                 + 0.898615 w_k1_IBM_s015
                                 + 0.894095 w_k1_INTC_s015
                                 + 0.776553 w_k1_KO_s015
                                 + 0.977458 w_k1_MAT_s015
                                 + 0.787444 w_k1_MSFT_s015
                                 + 0.875502 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.03354 w_k1_ADM_s015
                                 + 1.08477 w_k1_AXP_s015
                                 + 1.05056 w_k1_DIS_s015
                                 + 0.998975 w_k1_FDX_s015
                                 + 1.14044 w_k1_IBM_s015
                                 + 0.984258 w_k1_INTC_s015
                                 + 1.14628 w_k1_KO_s015 + 1.31837 w_k1_MAT_s015
                                 + 1.04709 w_k1_MSFT_s015
                                 + 1.08943 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.09878 w_k1_ADM_s015
                                 + 1.36091 w_k1_AXP_s015
                                 + 1.43169 w_k1_DIS_s015 + 1.3043 w_k1_FDX_s015
                                 + 1.1432 w_k1_IBM_s015
                                 + 1.34512 w_k1_INTC_s015
                                 + 1.22249 w_k1_KO_s015 + 1.34218 w_k1_MAT_s015
                                 + 1.34463 w_k1_MSFT_s015
                                 + 1.15081 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.02205 w_k1_ADM_s016
                                 + 0.959434 w_k1_AXP_s016
                                 + 0.842815 w_k1_DIS_s016
                                 + 0.809892 w_k1_FDX_s016
                                 + 1.32264 w_k1_IBM_s016
                                 + 0.87656 w_k1_INTC_s016 + 1.1464 w_k1_KO_s016
                                 + 1.29525 w_k1_MAT_s016
                                 + 0.863538 w_k1_MSFT_s016
                                 + 1.08184 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.2667 w_k1_ADM_s016
                                 + 0.799146 w_k1_AXP_s016
                                 + 1.06571 w_k1_DIS_s016
                                 + 0.900685 w_k1_FDX_s016
                                 + 0.992742 w_k1_IBM_s016
                                 + 0.766839 w_k1_INTC_s016
                                 + 1.2627 w_k1_KO_s016 + 0.661469 w_k1_MAT_s016
                                 + 0.992863 w_k1_MSFT_s016
                                 + 1.01666 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.939188 w_k1_ADM_s016
                                 + 0.709927 w_k1_AXP_s016
                                 + 0.696592 w_k1_DIS_s016
                                 + 0.82134 w_k1_FDX_s016
                                 + 0.872261 w_k1_IBM_s016
                                 + 0.801527 w_k1_INTC_s016
                                 + 0.837223 w_k1_KO_s016
                                 + 0.649804 w_k1_MAT_s016
                                 + 0.812802 w_k1_MSFT_s016
                                 + 0.790142 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.59193 w_k1_ADM_s016
                                 + 1.45116 w_k1_AXP_s016 + 1.5178 w_k1_DIS_s016
                                 + 1.23968 w_k1_FDX_s016 + 1.3437 w_k1_IBM_s016
                                 + 0.871072 w_k1_INTC_s016
                                 + 1.52353 w_k1_KO_s016 + 1.48917 w_k1_MAT_s016
                                 + 1.11476 w_k1_MSFT_s016
                                 + 1.23851 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.02499 w_k1_ADM_s016
                                 + 0.964198 w_k1_AXP_s016
                                 + 0.800727 w_k1_DIS_s016
                                 + 0.981011 w_k1_FDX_s016
                                 + 0.833771 w_k1_IBM_s016
                                 + 0.869383 w_k1_INTC_s016
                                 + 1.03041 w_k1_KO_s016
                                 + 0.947309 w_k1_MAT_s016
                                 + 1.01151 w_k1_MSFT_s016
                                 + 1.05162 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.903969 w_k1_ADM_s016
                                 + 0.48682 w_k1_AXP_s016
                                 + 0.771717 w_k1_DIS_s016
                                 + 0.793341 w_k1_FDX_s016
                                 + 0.580939 w_k1_IBM_s016
                                 + 0.762017 w_k1_INTC_s016
                                 + 0.805742 w_k1_KO_s016
                                 + 0.36925 w_k1_MAT_s016
                                 + 0.859686 w_k1_MSFT_s016
                                 + 0.881511 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.948222 w_k1_ADM_s016
                                 + 0.751404 w_k1_AXP_s016
                                 + 1.05718 w_k1_DIS_s016
                                 + 0.975006 w_k1_FDX_s016
                                 + 0.578987 w_k1_IBM_s016
                                 + 0.826043 w_k1_INTC_s016
                                 + 0.876056 w_k1_KO_s016
                                 + 0.507986 w_k1_MAT_s016
                                 + 1.09535 w_k1_MSFT_s016
                                 + 0.99921 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.21962 w_k1_ADM_s016
                                 + 1.08615 w_k1_AXP_s016
                                 + 1.12948 w_k1_DIS_s016
                                 + 1.20492 w_k1_FDX_s016
                                 + 0.849264 w_k1_IBM_s016
                                 + 0.860799 w_k1_INTC_s016
                                 + 1.18879 w_k1_KO_s016
                                 + 0.879171 w_k1_MAT_s016
                                 + 1.08053 w_k1_MSFT_s016 + 1.0475 w_k1_PG_s016
                                 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.47911 w_k1_ADM_s017
                                 + 1.62073 w_k1_AXP_s017
                                 + 1.54016 w_k1_DIS_s017
                                 + 1.35744 w_k1_FDX_s017
                                 + 1.38607 w_k1_IBM_s017
                                 + 1.22495 w_k1_INTC_s017
                                 + 1.54812 w_k1_KO_s017 + 1.78802 w_k1_MAT_s017
                                 + 1.38129 w_k1_MSFT_s017
                                 + 1.30671 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.1472 w_k1_ADM_s017
                                 + 1.18942 w_k1_AXP_s017
                                 + 1.34939 w_k1_DIS_s017
                                 + 1.08513 w_k1_FDX_s017
                                 + 1.48523 w_k1_IBM_s017
                                 + 1.16335 w_k1_INTC_s017
                                 + 1.44938 w_k1_KO_s017 + 1.30166 w_k1_MAT_s017
                                 + 1.24221 w_k1_MSFT_s017
                                 + 1.31397 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.90102 w_k1_ADM_s017
                                 + 0.971498 w_k1_AXP_s017
                                 + 0.76198 w_k1_DIS_s017
                                 + 0.860339 w_k1_FDX_s017
                                 + 1.2241 w_k1_IBM_s017
                                 + 0.889535 w_k1_INTC_s017
                                 + 1.08794 w_k1_KO_s017 + 1.21122 w_k1_MAT_s017
                                 + 0.902881 w_k1_MSFT_s017
                                 + 1.07445 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.975246 w_k1_ADM_s017
                                 + 0.990992 w_k1_AXP_s017
                                 + 0.974695 w_k1_DIS_s017
                                 + 1.01304 w_k1_FDX_s017
                                 + 0.766352 w_k1_IBM_s017
                                 + 0.889131 w_k1_INTC_s017
                                 + 0.848098 w_k1_KO_s017
                                 + 0.665821 w_k1_MAT_s017
                                 + 1.02155 w_k1_MSFT_s017
                                 + 0.938725 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.872847 w_k1_ADM_s017
                                 + 0.902429 w_k1_AXP_s017
                                 + 1.23784 w_k1_DIS_s017
                                 + 1.04446 w_k1_FDX_s017
                                 + 0.972994 w_k1_IBM_s017
                                 + 1.11628 w_k1_INTC_s017
                                 + 1.04821 w_k1_KO_s017
                                 + 0.766289 w_k1_MAT_s017
                                 + 1.195 w_k1_MSFT_s017 + 1.06086 w_k1_PG_s017
                                 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.3786 w_k1_ADM_s017
                                 + 1.3671 w_k1_AXP_s017 + 1.11698 w_k1_DIS_s017
                                 + 1.22136 w_k1_FDX_s017
                                 + 1.00655 w_k1_IBM_s017
                                 + 0.999229 w_k1_INTC_s017
                                 + 1.03894 w_k1_KO_s017 + 1.05031 w_k1_MAT_s017
                                 + 0.979761 w_k1_MSFT_s017
                                 + 1.13627 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.955314 w_k1_ADM_s017
                                 + 0.70665 w_k1_AXP_s017
                                 + 0.789926 w_k1_DIS_s017
                                 + 0.798989 w_k1_FDX_s017
                                 + 1.01705 w_k1_IBM_s017
                                 + 0.776055 w_k1_INTC_s017
                                 + 0.858778 w_k1_KO_s017
                                 + 0.771895 w_k1_MAT_s017
                                 + 0.837636 w_k1_MSFT_s017
                                 + 0.902989 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.799141 w_k1_ADM_s017
                                 + 0.668521 w_k1_AXP_s017
                                 + 0.66718 w_k1_DIS_s017
                                 + 0.814894 w_k1_FDX_s017
                                 + 0.70506 w_k1_IBM_s017
                                 + 0.938007 w_k1_INTC_s017
                                 + 0.767451 w_k1_KO_s017
                                 + 0.627296 w_k1_MAT_s017
                                 + 1.02277 w_k1_MSFT_s017
                                 + 0.92585 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.936055 w_k1_ADM_s100
                                 + 1.20635 w_k1_AXP_s100 + 1.0278 w_k1_DIS_s100
                                 + 0.901189 w_k1_FDX_s100
                                 + 1.43364 w_k1_IBM_s100
                                 + 1.04084 w_k1_INTC_s100 + 1.1982 w_k1_KO_s100
                                 + 1.69658 w_k1_MAT_s100
                                 + 0.985344 w_k1_MSFT_s100
                                 + 1.05059 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.07179 w_k1_ADM_s100
                                 + 1.14222 w_k1_AXP_s100
                                 + 1.29482 w_k1_DIS_s100
                                 + 1.10076 w_k1_FDX_s100
                                 + 1.22262 w_k1_IBM_s100
                                 + 1.06374 w_k1_INTC_s100
                                 + 1.25685 w_k1_KO_s100 + 1.27014 w_k1_MAT_s100
                                 + 1.23018 w_k1_MSFT_s100
                                 + 1.07363 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.4104 w_k1_ADM_s100
                                 + 1.53909 w_k1_AXP_s100
                                 + 1.64374 w_k1_DIS_s100
                                 + 1.46479 w_k1_FDX_s100
                                 + 1.33725 w_k1_IBM_s100
                                 + 1.21751 w_k1_INTC_s100
                                 + 1.39339 w_k1_KO_s100 + 1.56955 w_k1_MAT_s100
                                 + 1.30224 w_k1_MSFT_s100
                                 + 1.31273 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.08636 w_k1_ADM_s100
                                 + 1.01287 w_k1_AXP_s100
                                 + 0.988535 w_k1_DIS_s100
                                 + 0.992598 w_k1_FDX_s100
                                 + 0.933325 w_k1_IBM_s100
                                 + 0.788258 w_k1_INTC_s100
                                 + 0.977585 w_k1_KO_s100
                                 + 1.32854 w_k1_MAT_s100
                                 + 0.953849 w_k1_MSFT_s100
                                 + 0.964494 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.05216 w_k1_ADM_s100
                                 + 1.3652 w_k1_AXP_s100 + 1.38511 w_k1_DIS_s100
                                 + 1.29607 w_k1_FDX_s100
                                 + 0.879003 w_k1_IBM_s100
                                 + 1.16143 w_k1_INTC_s100
                                 + 0.986364 w_k1_KO_s100
                                 + 1.19607 w_k1_MAT_s100
                                 + 1.29344 w_k1_MSFT_s100
                                 + 1.03883 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.972471 w_k1_ADM_s100
                                 + 0.701455 w_k1_AXP_s100
                                 + 0.734183 w_k1_DIS_s100
                                 + 0.803131 w_k1_FDX_s100
                                 + 0.752673 w_k1_IBM_s100
                                 + 0.624271 w_k1_INTC_s100
                                 + 0.813353 w_k1_KO_s100
                                 + 0.803077 w_k1_MAT_s100
                                 + 0.869075 w_k1_MSFT_s100
                                 + 0.802802 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.813695 w_k1_ADM_s100
                                 + 0.912941 w_k1_AXP_s100
                                 + 0.885314 w_k1_DIS_s100
                                 + 0.851263 w_k1_FDX_s100
                                 + 1.26375 w_k1_IBM_s100
                                 + 0.986899 w_k1_INTC_s100
                                 + 0.9868 w_k1_KO_s100 + 1.25058 w_k1_MAT_s100
                                 + 0.903999 w_k1_MSFT_s100
                                 + 0.870621 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.39555 w_k1_ADM_s100
                                 + 1.47166 w_k1_AXP_s100
                                 + 0.947008 w_k1_DIS_s100
                                 + 1.06014 w_k1_FDX_s100
                                 + 1.14995 w_k1_IBM_s100
                                 + 0.769856 w_k1_INTC_s100
                                 + 0.990532 w_k1_KO_s100
                                 + 1.46478 w_k1_MAT_s100
                                 + 0.945178 w_k1_MSFT_s100
                                 + 1.01987 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.771058 w_k1_ADM_s101
                                 + 0.677017 w_k1_AXP_s101
                                 + 0.62731 w_k1_DIS_s101
                                 + 0.755884 w_k1_FDX_s101
                                 + 0.703563 w_k1_IBM_s101
                                 + 0.757337 w_k1_INTC_s101
                                 + 0.622349 w_k1_KO_s101
                                 + 0.561043 w_k1_MAT_s101
                                 + 0.709098 w_k1_MSFT_s101
                                 + 0.672028 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.15024 w_k1_ADM_s101
                                 + 1.38669 w_k1_AXP_s101
                                 + 1.38273 w_k1_DIS_s101
                                 + 1.21453 w_k1_FDX_s101
                                 + 1.33816 w_k1_IBM_s101
                                 + 1.29907 w_k1_INTC_s101
                                 + 1.20519 w_k1_KO_s101 + 1.65268 w_k1_MAT_s101
                                 + 1.2063 w_k1_MSFT_s101 + 1.10206 w_k1_PG_s101
                                 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.850713 w_k1_ADM_s101
                                 + 1.11514 w_k1_AXP_s101
                                 + 0.795917 w_k1_DIS_s101
                                 + 0.88175 w_k1_FDX_s101
                                 + 1.13202 w_k1_IBM_s101
                                 + 0.976205 w_k1_INTC_s101
                                 + 0.926009 w_k1_KO_s101
                                 + 1.20847 w_k1_MAT_s101
                                 + 0.886815 w_k1_MSFT_s101
                                 + 0.905396 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.890931 w_k1_ADM_s101
                                 + 0.825697 w_k1_AXP_s101
                                 + 1.13511 w_k1_DIS_s101
                                 + 1.01111 w_k1_FDX_s101
                                 + 0.890104 w_k1_IBM_s101
                                 + 0.868576 w_k1_INTC_s101
                                 + 0.887551 w_k1_KO_s101
                                 + 0.797425 w_k1_MAT_s101
                                 + 1.05218 w_k1_MSFT_s101
                                 + 0.789677 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.23221 w_k1_ADM_s101
                                 + 1.10775 w_k1_AXP_s101
                                 + 0.903564 w_k1_DIS_s101
                                 + 1.12056 w_k1_FDX_s101
                                 + 0.734666 w_k1_IBM_s101
                                 + 1.1326 w_k1_INTC_s101
                                 + 0.720981 w_k1_KO_s101
                                 + 0.816971 w_k1_MAT_s101
                                 + 1.00812 w_k1_MSFT_s101
                                 + 0.818813 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.836695 w_k1_ADM_s101
                                 + 1.0012 w_k1_AXP_s101
                                 + 0.914026 w_k1_DIS_s101
                                 + 0.852103 w_k1_FDX_s101
                                 + 1.80744 w_k1_IBM_s101
                                 + 0.921003 w_k1_INTC_s101
                                 + 1.20883 w_k1_KO_s101 + 1.91979 w_k1_MAT_s101
                                 + 0.945651 w_k1_MSFT_s101
                                 + 0.975509 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.720501 w_k1_ADM_s101
                                 + 0.738105 w_k1_AXP_s101
                                 + 0.744332 w_k1_DIS_s101
                                 + 0.739095 w_k1_FDX_s101
                                 + 1.04717 w_k1_IBM_s101
                                 + 0.823837 w_k1_INTC_s101
                                 + 0.750245 w_k1_KO_s101
                                 + 0.847116 w_k1_MAT_s101
                                 + 0.856757 w_k1_MSFT_s101
                                 + 0.775896 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.21497 w_k1_ADM_s101
                                 + 1.53852 w_k1_AXP_s101
                                 + 1.38224 w_k1_DIS_s101
                                 + 1.50316 w_k1_FDX_s101
                                 + 0.858331 w_k1_IBM_s101
                                 + 1.21884 w_k1_INTC_s101
                                 + 0.974547 w_k1_KO_s101
                                 + 0.979434 w_k1_MAT_s101
                                 + 1.31795 w_k1_MSFT_s101
                                 + 0.894352 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.05936 w_k1_ADM_s102
                                 + 0.992333 w_k1_AXP_s102
                                 + 0.910676 w_k1_DIS_s102
                                 + 1.01401 w_k1_FDX_s102
                                 + 0.692874 w_k1_IBM_s102
                                 + 1.10492 w_k1_INTC_s102
                                 + 0.857537 w_k1_KO_s102
                                 + 0.556799 w_k1_MAT_s102
                                 + 1.00486 w_k1_MSFT_s102
                                 + 0.98521 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.994666 w_k1_ADM_s102
                                 + 1.07271 w_k1_AXP_s102
                                 + 1.32163 w_k1_DIS_s102
                                 + 1.03598 w_k1_FDX_s102
                                 + 1.38234 w_k1_IBM_s102
                                 + 1.08558 w_k1_INTC_s102
                                 + 1.47468 w_k1_KO_s102 + 1.58639 w_k1_MAT_s102
                                 + 1.07605 w_k1_MSFT_s102 + 1.0682 w_k1_PG_s102
                                 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.38944 w_k1_ADM_s102
                                 + 1.36353 w_k1_AXP_s102
                                 + 1.43871 w_k1_DIS_s102
                                 + 1.23912 w_k1_FDX_s102 + 1.6394 w_k1_IBM_s102
                                 + 1.17932 w_k1_INTC_s102
                                 + 1.48682 w_k1_KO_s102 + 1.54209 w_k1_MAT_s102
                                 + 1.17387 w_k1_MSFT_s102
                                 + 1.43533 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.28924 w_k1_ADM_s102
                                 + 1.62179 w_k1_AXP_s102
                                 + 1.75778 w_k1_DIS_s102
                                 + 1.49825 w_k1_FDX_s102
                                 + 1.20464 w_k1_IBM_s102
                                 + 1.45254 w_k1_INTC_s102
                                 + 1.59364 w_k1_KO_s102 + 1.54027 w_k1_MAT_s102
                                 + 1.45414 w_k1_MSFT_s102
                                 + 1.27852 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.977026 w_k1_ADM_s102
                                 + 0.916868 w_k1_AXP_s102
                                 + 0.943414 w_k1_DIS_s102
                                 + 0.822432 w_k1_FDX_s102
                                 + 1.21178 w_k1_IBM_s102
                                 + 0.991305 w_k1_INTC_s102
                                 + 1.13747 w_k1_KO_s102 + 1.18251 w_k1_MAT_s102
                                 + 0.987159 w_k1_MSFT_s102
                                 + 1.11764 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.00839 w_k1_ADM_s102
                                 + 1.19558 w_k1_AXP_s102
                                 + 1.08533 w_k1_DIS_s102
                                 + 1.13896 w_k1_FDX_s102
                                 + 0.989429 w_k1_IBM_s102
                                 + 1.3196 w_k1_INTC_s102 + 1.10354 w_k1_KO_s102
                                 + 1.18286 w_k1_MAT_s102
                                 + 1.16136 w_k1_MSFT_s102
                                 + 1.06454 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.648737 w_k1_ADM_s102
                                 + 0.609569 w_k1_AXP_s102
                                 + 0.485824 w_k1_DIS_s102
                                 + 0.624461 w_k1_FDX_s102
                                 + 0.789612 w_k1_IBM_s102
                                 + 0.816337 w_k1_INTC_s102
                                 + 0.846358 w_k1_KO_s102
                                 + 0.767161 w_k1_MAT_s102
                                 + 0.789778 w_k1_MSFT_s102
                                 + 0.843264 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.785687 w_k1_ADM_s102
                                 + 0.69408 w_k1_AXP_s102
                                 + 0.625207 w_k1_DIS_s102
                                 + 0.733049 w_k1_FDX_s102
                                 + 0.873613 w_k1_IBM_s102
                                 + 0.907437 w_k1_INTC_s102
                                 + 0.915567 w_k1_KO_s102
                                 + 0.602309 w_k1_MAT_s102
                                 + 0.742449 w_k1_MSFT_s102
                                 + 0.875887 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.931173 w_k1_ADM_s103
                                 + 1.0077 w_k1_AXP_s103
                                 + 0.900163 w_k1_DIS_s103
                                 + 0.952047 w_k1_FDX_s103
                                 + 1.23567 w_k1_IBM_s103
                                 + 1.24796 w_k1_INTC_s103
                                 + 1.13659 w_k1_KO_s103 + 1.02223 w_k1_MAT_s103
                                 + 0.929675 w_k1_MSFT_s103
                                 + 0.95619 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.0142 w_k1_ADM_s103
                                 + 1.19265 w_k1_AXP_s103
                                 + 1.75437 w_k1_DIS_s103
                                 + 1.38255 w_k1_FDX_s103
                                 + 1.12106 w_k1_IBM_s103
                                 + 1.5901 w_k1_INTC_s103 + 1.43689 w_k1_KO_s103
                                 + 1.18863 w_k1_MAT_s103
                                 + 1.64937 w_k1_MSFT_s103
                                 + 1.20124 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.823656 w_k1_ADM_s103
                                 + 0.702311 w_k1_AXP_s103
                                 + 0.746722 w_k1_DIS_s103
                                 + 0.835788 w_k1_FDX_s103
                                 + 0.935625 w_k1_IBM_s103
                                 + 1.01541 w_k1_INTC_s103
                                 + 0.904879 w_k1_KO_s103
                                 + 0.817277 w_k1_MAT_s103
                                 + 0.892537 w_k1_MSFT_s103
                                 + 0.881748 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.24588 w_k1_ADM_s103
                                 + 1.2936 w_k1_AXP_s103 + 1.06212 w_k1_DIS_s103
                                 + 1.22817 w_k1_FDX_s103
                                 + 0.896557 w_k1_IBM_s103
                                 + 1.30643 w_k1_INTC_s103
                                 + 0.979751 w_k1_KO_s103
                                 + 1.15275 w_k1_MAT_s103
                                 + 1.05193 w_k1_MSFT_s103
                                 + 1.07358 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.920453 w_k1_ADM_s103
                                 + 0.664186 w_k1_AXP_s103
                                 + 0.535347 w_k1_DIS_s103
                                 + 0.797768 w_k1_FDX_s103
                                 + 0.689892 w_k1_IBM_s103
                                 + 0.841966 w_k1_INTC_s103
                                 + 0.706646 w_k1_KO_s103
                                 + 0.774016 w_k1_MAT_s103
                                 + 0.75339 w_k1_MSFT_s103
                                 + 0.781107 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.933465 w_k1_ADM_s103
                                 + 0.913359 w_k1_AXP_s103
                                 + 1.1732 w_k1_DIS_s103 + 1.17694 w_k1_FDX_s103
                                 + 0.814189 w_k1_IBM_s103
                                 + 1.42542 w_k1_INTC_s103
                                 + 1.02757 w_k1_KO_s103
                                 + 0.569136 w_k1_MAT_s103
                                 + 1.25003 w_k1_MSFT_s103
                                 + 1.01001 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.765341 w_k1_ADM_s103
                                 + 0.867615 w_k1_AXP_s103
                                 + 0.74834 w_k1_DIS_s103
                                 + 0.981318 w_k1_FDX_s103
                                 + 0.608726 w_k1_IBM_s103
                                 + 1.21872 w_k1_INTC_s103
                                 + 0.753706 w_k1_KO_s103
                                 + 0.459808 w_k1_MAT_s103
                                 + 0.92243 w_k1_MSFT_s103
                                 + 0.92429 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.11319 w_k1_ADM_s103
                                 + 1.57729 w_k1_AXP_s103
                                 + 1.44473 w_k1_DIS_s103
                                 + 1.29486 w_k1_FDX_s103
                                 + 1.50886 w_k1_IBM_s103
                                 + 1.44567 w_k1_INTC_s103
                                 + 1.37133 w_k1_KO_s103 + 1.68823 w_k1_MAT_s103
                                 + 1.26144 w_k1_MSFT_s103
                                 + 1.37772 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.94668 w_k1_ADM_s104
                                 + 0.878014 w_k1_AXP_s104
                                 + 0.932249 w_k1_DIS_s104
                                 + 0.996132 w_k1_FDX_s104
                                 + 0.828796 w_k1_IBM_s104
                                 + 0.964904 w_k1_INTC_s104
                                 + 0.932996 w_k1_KO_s104
                                 + 0.836687 w_k1_MAT_s104
                                 + 0.931294 w_k1_MSFT_s104
                                 + 0.902709 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.740787 w_k1_ADM_s104
                                 + 0.643571 w_k1_AXP_s104
                                 + 0.612561 w_k1_DIS_s104
                                 + 0.812297 w_k1_FDX_s104
                                 + 0.550458 w_k1_IBM_s104
                                 + 0.764068 w_k1_INTC_s104
                                 + 0.648453 w_k1_KO_s104
                                 + 0.496042 w_k1_MAT_s104
                                 + 0.795659 w_k1_MSFT_s104
                                 + 0.734312 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.10402 w_k1_ADM_s104
                                 + 1.31894 w_k1_AXP_s104
                                 + 1.66746 w_k1_DIS_s104
                                 + 1.30545 w_k1_FDX_s104
                                 + 0.941783 w_k1_IBM_s104
                                 + 1.17273 w_k1_INTC_s104
                                 + 1.23124 w_k1_KO_s104 + 0.85066 w_k1_MAT_s104
                                 + 1.26495 w_k1_MSFT_s104
                                 + 1.15901 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.37105 w_k1_ADM_s104
                                 + 1.43544 w_k1_AXP_s104
                                 + 1.58433 w_k1_DIS_s104 + 1.2157 w_k1_FDX_s104
                                 + 1.44526 w_k1_IBM_s104
                                 + 1.19617 w_k1_INTC_s104
                                 + 1.68958 w_k1_KO_s104 + 1.33975 w_k1_MAT_s104
                                 + 1.17813 w_k1_MSFT_s104
                                 + 1.28176 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.02667 w_k1_ADM_s104
                                 + 1.1626 w_k1_AXP_s104
                                 + 0.935894 w_k1_DIS_s104
                                 + 0.904357 w_k1_FDX_s104
                                 + 1.1774 w_k1_IBM_s104
                                 + 0.896718 w_k1_INTC_s104
                                 + 1.08426 w_k1_KO_s104 + 1.13054 w_k1_MAT_s104
                                 + 0.929913 w_k1_MSFT_s104
                                 + 0.992443 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.820737 w_k1_ADM_s104
                                 + 1.01156 w_k1_AXP_s104
                                 + 1.26343 w_k1_DIS_s104
                                 + 1.05858 w_k1_FDX_s104 + 1.1009 w_k1_IBM_s104
                                 + 1.154 w_k1_INTC_s104 + 1.04402 w_k1_KO_s104
                                 + 0.840466 w_k1_MAT_s104
                                 + 1.16789 w_k1_MSFT_s104
                                 + 0.962997 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.615951 w_k1_ADM_s104
                                 + 0.731942 w_k1_AXP_s104
                                 + 0.816769 w_k1_DIS_s104
                                 + 0.752235 w_k1_FDX_s104
                                 + 0.876897 w_k1_IBM_s104
                                 + 0.69753 w_k1_INTC_s104
                                 + 0.844302 w_k1_KO_s104
                                 + 0.697662 w_k1_MAT_s104
                                 + 0.814526 w_k1_MSFT_s104
                                 + 0.852253 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.795892 w_k1_ADM_s104
                                 + 0.938367 w_k1_AXP_s104
                                 + 0.964939 w_k1_DIS_s104
                                 + 0.807445 w_k1_FDX_s104
                                 + 1.47499 w_k1_IBM_s104
                                 + 1.00448 w_k1_INTC_s104
                                 + 1.27764 w_k1_KO_s104 + 1.208 w_k1_MAT_s104
                                 + 0.882287 w_k1_MSFT_s104
                                 + 0.963073 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.938818 w_k1_ADM_s105
                                 + 0.890139 w_k1_AXP_s105
                                 + 1.06356 w_k1_DIS_s105
                                 + 0.972134 w_k1_FDX_s105
                                 + 1.29688 w_k1_IBM_s105
                                 + 1.1316 w_k1_INTC_s105 + 1.12118 w_k1_KO_s105
                                 + 0.977351 w_k1_MAT_s105
                                 + 1.0123 w_k1_MSFT_s105 + 1.00037 w_k1_PG_s105
                                 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.25771 w_k1_ADM_s105
                                 + 1.15719 w_k1_AXP_s105
                                 + 0.957809 w_k1_DIS_s105
                                 + 1.13111 w_k1_FDX_s105
                                 + 0.837509 w_k1_IBM_s105
                                 + 1.1345 w_k1_INTC_s105
                                 + 0.893154 w_k1_KO_s105
                                 + 0.861266 w_k1_MAT_s105
                                 + 0.985098 w_k1_MSFT_s105
                                 + 0.929033 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.886531 w_k1_ADM_s105
                                 + 0.745886 w_k1_AXP_s105
                                 + 0.767577 w_k1_DIS_s105
                                 + 0.872316 w_k1_FDX_s105
                                 + 0.659101 w_k1_IBM_s105
                                 + 0.935928 w_k1_INTC_s105
                                 + 0.784554 w_k1_KO_s105
                                 + 0.544399 w_k1_MAT_s105
                                 + 0.854266 w_k1_MSFT_s105
                                 + 0.866211 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.50077 w_k1_ADM_s105
                                 + 1.62556 w_k1_AXP_s105
                                 + 1.71777 w_k1_DIS_s105
                                 + 1.54192 w_k1_FDX_s105
                                 + 1.23511 w_k1_IBM_s105
                                 + 1.4871 w_k1_INTC_s105 + 1.27423 w_k1_KO_s105
                                 + 1.31354 w_k1_MAT_s105
                                 + 1.36204 w_k1_MSFT_s105
                                 + 1.09782 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.778458 w_k1_ADM_s105
                                 + 0.69042 w_k1_AXP_s105
                                 + 0.642486 w_k1_DIS_s105
                                 + 0.717311 w_k1_FDX_s105
                                 + 0.884559 w_k1_IBM_s105
                                 + 0.905627 w_k1_INTC_s105
                                 + 0.690917 w_k1_KO_s105
                                 + 0.847906 w_k1_MAT_s105
                                 + 0.722604 w_k1_MSFT_s105
                                 + 0.723725 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.10913 w_k1_ADM_s105
                                 + 1.32948 w_k1_AXP_s105 + 1.4379 w_k1_DIS_s105
                                 + 1.30362 w_k1_FDX_s105
                                 + 1.06904 w_k1_IBM_s105
                                 + 1.06923 w_k1_INTC_s105
                                 + 1.21919 w_k1_KO_s105 + 1.1738 w_k1_MAT_s105
                                 + 1.18269 w_k1_MSFT_s105
                                 + 1.17691 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.853189 w_k1_ADM_s105
                                 + 0.758343 w_k1_AXP_s105
                                 + 0.864566 w_k1_DIS_s105
                                 + 0.85124 w_k1_FDX_s105
                                 + 0.87186 w_k1_IBM_s105
                                 + 0.967134 w_k1_INTC_s105
                                 + 0.839829 w_k1_KO_s105
                                 + 0.785916 w_k1_MAT_s105
                                 + 0.835245 w_k1_MSFT_s105
                                 + 0.832145 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.07916 w_k1_ADM_s105
                                 + 1.1811 w_k1_AXP_s105 + 1.27121 w_k1_DIS_s105
                                 + 1.1097 w_k1_FDX_s105 + 1.24867 w_k1_IBM_s105
                                 + 1.02504 w_k1_INTC_s105
                                 + 1.28601 w_k1_KO_s105 + 1.73501 w_k1_MAT_s105
                                 + 1.05284 w_k1_MSFT_s105
                                 + 1.19643 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.853797 w_k1_ADM_s106
                                 + 0.901531 w_k1_AXP_s106
                                 + 1.02188 w_k1_DIS_s106
                                 + 0.980437 w_k1_FDX_s106
                                 + 1.10274 w_k1_IBM_s106
                                 + 0.88846 w_k1_INTC_s106
                                 + 1.03333 w_k1_KO_s106 + 1.20758 w_k1_MAT_s106
                                 + 1.2919 w_k1_MSFT_s106 + 1.02988 w_k1_PG_s106
                                 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.22403 w_k1_ADM_s106
                                 + 1.3385 w_k1_AXP_s106 + 1.42017 w_k1_DIS_s106
                                 + 1.11397 w_k1_FDX_s106
                                 + 1.39645 w_k1_IBM_s106
                                 + 1.20176 w_k1_INTC_s106 + 1.3322 w_k1_KO_s106
                                 + 1.17159 w_k1_MAT_s106
                                 + 1.55064 w_k1_MSFT_s106
                                 + 1.22785 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.945689 w_k1_ADM_s106
                                 + 0.735558 w_k1_AXP_s106
                                 + 0.674489 w_k1_DIS_s106
                                 + 0.784155 w_k1_FDX_s106
                                 + 1.0476 w_k1_IBM_s106
                                 + 0.918814 w_k1_INTC_s106
                                 + 0.903044 w_k1_KO_s106
                                 + 0.828179 w_k1_MAT_s106
                                 + 1.03795 w_k1_MSFT_s106
                                 + 0.989159 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.13331 w_k1_ADM_s106
                                 + 1.26423 w_k1_AXP_s106
                                 + 0.917046 w_k1_DIS_s106
                                 + 1.11069 w_k1_FDX_s106
                                 + 0.733522 w_k1_IBM_s106
                                 + 1.28565 w_k1_INTC_s106
                                 + 0.854788 w_k1_KO_s106
                                 + 0.974427 w_k1_MAT_s106
                                 + 1.23439 w_k1_MSFT_s106
                                 + 1.09392 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.818919 w_k1_ADM_s106
                                 + 0.692497 w_k1_AXP_s106
                                 + 0.596004 w_k1_DIS_s106
                                 + 0.752538 w_k1_FDX_s106
                                 + 0.698036 w_k1_IBM_s106
                                 + 0.902125 w_k1_INTC_s106
                                 + 0.813702 w_k1_KO_s106
                                 + 0.749805 w_k1_MAT_s106
                                 + 1.08934 w_k1_MSFT_s106
                                 + 0.882807 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.927512 w_k1_ADM_s106
                                 + 0.64754 w_k1_AXP_s106
                                 + 0.745121 w_k1_DIS_s106
                                 + 0.829047 w_k1_FDX_s106
                                 + 0.87528 w_k1_IBM_s106
                                 + 0.998054 w_k1_INTC_s106
                                 + 0.795597 w_k1_KO_s106
                                 + 0.789924 w_k1_MAT_s106
                                 + 1.15089 w_k1_MSFT_s106
                                 + 0.837306 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.858482 w_k1_ADM_s106
                                 + 0.778218 w_k1_AXP_s106
                                 + 0.887845 w_k1_DIS_s106
                                 + 0.898126 w_k1_FDX_s106
                                 + 0.620048 w_k1_IBM_s106
                                 + 1.23686 w_k1_INTC_s106
                                 + 0.774601 w_k1_KO_s106
                                 + 0.648358 w_k1_MAT_s106
                                 + 1.12403 w_k1_MSFT_s106
                                 + 0.793217 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.25203 w_k1_ADM_s106
                                 + 1.27203 w_k1_AXP_s106
                                 + 1.09428 w_k1_DIS_s106
                                 + 1.10243 w_k1_FDX_s106
                                 + 0.94288 w_k1_IBM_s106
                                 + 1.13335 w_k1_INTC_s106
                                 + 0.967149 w_k1_KO_s106
                                 + 0.994896 w_k1_MAT_s106
                                 + 1.13068 w_k1_MSFT_s106
                                 + 1.02948 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.930103 w_k1_ADM_s107
                                 + 0.939727 w_k1_AXP_s107
                                 + 1.41642 w_k1_DIS_s107
                                 + 1.08292 w_k1_FDX_s107
                                 + 1.24494 w_k1_IBM_s107
                                 + 0.958145 w_k1_INTC_s107
                                 + 1.28053 w_k1_KO_s107 + 1.34806 w_k1_MAT_s107
                                 + 1.28124 w_k1_MSFT_s107
                                 + 1.19865 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.14579 w_k1_ADM_s107
                                 + 1.33313 w_k1_AXP_s107
                                 + 1.02618 w_k1_DIS_s107
                                 + 1.07126 w_k1_FDX_s107
                                 + 1.10022 w_k1_IBM_s107
                                 + 1.0287 w_k1_INTC_s107 + 1.22456 w_k1_KO_s107
                                 + 1.28146 w_k1_MAT_s107
                                 + 1.1513 w_k1_MSFT_s107 + 1.16097 w_k1_PG_s107
                                 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.897305 w_k1_ADM_s107
                                 + 0.793245 w_k1_AXP_s107
                                 + 0.751298 w_k1_DIS_s107
                                 + 0.764685 w_k1_FDX_s107
                                 + 1.0619 w_k1_IBM_s107
                                 + 0.715751 w_k1_INTC_s107
                                 + 0.923541 w_k1_KO_s107
                                 + 0.936877 w_k1_MAT_s107
                                 + 0.823107 w_k1_MSFT_s107
                                 + 0.880183 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.932942 w_k1_ADM_s107
                                 + 1.12772 w_k1_AXP_s107
                                 + 1.16327 w_k1_DIS_s107
                                 + 1.06818 w_k1_FDX_s107
                                 + 0.966781 w_k1_IBM_s107
                                 + 0.882139 w_k1_INTC_s107
                                 + 1.14053 w_k1_KO_s107 + 1.09444 w_k1_MAT_s107
                                 + 1.13944 w_k1_MSFT_s107
                                 + 1.07932 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.58415 w_k1_ADM_s107
                                 + 1.37505 w_k1_AXP_s107
                                 + 1.42425 w_k1_DIS_s107
                                 + 1.14213 w_k1_FDX_s107
                                 + 1.58589 w_k1_IBM_s107
                                 + 1.04183 w_k1_INTC_s107
                                 + 1.46303 w_k1_KO_s107 + 1.6345 w_k1_MAT_s107
                                 + 1.19639 w_k1_MSFT_s107
                                 + 1.24754 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.1623 w_k1_ADM_s107
                                 + 1.00382 w_k1_AXP_s107
                                 + 1.13264 w_k1_DIS_s107
                                 + 1.09567 w_k1_FDX_s107
                                 + 0.779359 w_k1_IBM_s107
                                 + 0.755124 w_k1_INTC_s107
                                 + 1.03291 w_k1_KO_s107
                                 + 0.930484 w_k1_MAT_s107
                                 + 1.10856 w_k1_MSFT_s107
                                 + 1.00274 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.991517 w_k1_ADM_s107
                                 + 0.793561 w_k1_AXP_s107
                                 + 0.819542 w_k1_DIS_s107
                                 + 0.773592 w_k1_FDX_s107
                                 + 0.718401 w_k1_IBM_s107
                                 + 0.786942 w_k1_INTC_s107
                                 + 0.767864 w_k1_KO_s107
                                 + 0.61068 w_k1_MAT_s107
                                 + 0.890865 w_k1_MSFT_s107
                                 + 0.874195 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.17166 w_k1_ADM_s107
                                 + 1.48072 w_k1_AXP_s107
                                 + 1.52979 w_k1_DIS_s107
                                 + 1.34201 w_k1_FDX_s107
                                 + 0.853929 w_k1_IBM_s107
                                 + 1.0547 w_k1_INTC_s107 + 1.13772 w_k1_KO_s107
                                 + 1.28625 w_k1_MAT_s107
                                 + 1.31885 w_k1_MSFT_s107 + 1.2474 w_k1_PG_s107
                                 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.09022 w_k1_ADM_s110
                                 + 1.31999 w_k1_AXP_s110
                                 + 1.26973 w_k1_DIS_s110
                                 + 1.15427 w_k1_FDX_s110
                                 + 1.27404 w_k1_IBM_s110
                                 + 1.13321 w_k1_INTC_s110
                                 + 1.17771 w_k1_KO_s110 + 1.33099 w_k1_MAT_s110
                                 + 1.07622 w_k1_MSFT_s110
                                 + 1.01859 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.04968 w_k1_ADM_s110
                                 + 1.20367 w_k1_AXP_s110
                                 + 0.944728 w_k1_DIS_s110
                                 + 1.22118 w_k1_FDX_s110
                                 + 0.763821 w_k1_IBM_s110
                                 + 1.15032 w_k1_INTC_s110
                                 + 0.87012 w_k1_KO_s110
                                 + 0.898302 w_k1_MAT_s110
                                 + 1.05843 w_k1_MSFT_s110
                                 + 0.944093 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.93698 w_k1_ADM_s110
                                 + 1.0954 w_k1_AXP_s110
                                 + 0.823394 w_k1_DIS_s110
                                 + 0.925353 w_k1_FDX_s110
                                 + 1.13618 w_k1_IBM_s110
                                 + 0.815116 w_k1_INTC_s110
                                 + 0.89552 w_k1_KO_s110 + 1.43634 w_k1_MAT_s110
                                 + 0.859529 w_k1_MSFT_s110
                                 + 0.981491 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.952854 w_k1_ADM_s110
                                 + 0.969591 w_k1_AXP_s110
                                 + 0.84701 w_k1_DIS_s110
                                 + 0.977711 w_k1_FDX_s110
                                 + 0.94404 w_k1_IBM_s110
                                 + 1.13029 w_k1_INTC_s110
                                 + 0.88362 w_k1_KO_s110
                                 + 0.693035 w_k1_MAT_s110
                                 + 0.993633 w_k1_MSFT_s110
                                 + 0.969796 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.970537 w_k1_ADM_s110
                                 + 1.07733 w_k1_AXP_s110
                                 + 0.971677 w_k1_DIS_s110
                                 + 0.927567 w_k1_FDX_s110
                                 + 1.61825 w_k1_IBM_s110
                                 + 0.949293 w_k1_INTC_s110
                                 + 1.16576 w_k1_KO_s110 + 1.73529 w_k1_MAT_s110
                                 + 0.916246 w_k1_MSFT_s110
                                 + 1.02054 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.800817 w_k1_ADM_s110
                                 + 0.854918 w_k1_AXP_s110
                                 + 0.691966 w_k1_DIS_s110
                                 + 0.918784 w_k1_FDX_s110
                                 + 0.625046 w_k1_IBM_s110
                                 + 1.04852 w_k1_INTC_s110
                                 + 0.634303 w_k1_KO_s110
                                 + 0.652188 w_k1_MAT_s110
                                 + 0.855102 w_k1_MSFT_s110
                                 + 0.878372 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.33986 w_k1_ADM_s110
                                 + 1.53213 w_k1_AXP_s110
                                 + 1.75094 w_k1_DIS_s110
                                 + 1.33588 w_k1_FDX_s110
                                 + 1.68697 w_k1_IBM_s110
                                 + 1.30535 w_k1_INTC_s110
                                 + 1.35188 w_k1_KO_s110 + 1.70893 w_k1_MAT_s110
                                 + 1.36844 w_k1_MSFT_s110
                                 + 1.19444 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.36464 w_k1_ADM_s110
                                 + 1.28879 w_k1_AXP_s110
                                 + 1.57601 w_k1_DIS_s110
                                 + 1.47478 w_k1_FDX_s110
                                 + 0.969188 w_k1_IBM_s110
                                 + 1.30738 w_k1_INTC_s110
                                 + 1.09201 w_k1_KO_s110 + 1.06436 w_k1_MAT_s110
                                 + 1.29569 w_k1_MSFT_s110 + 1.1357 w_k1_PG_s110
                                 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.22536 w_k1_ADM_s111
                                 + 0.918889 w_k1_AXP_s111
                                 + 0.921731 w_k1_DIS_s111
                                 + 0.873442 w_k1_FDX_s111
                                 + 1.4658 w_k1_IBM_s111
                                 + 0.957284 w_k1_INTC_s111
                                 + 1.16823 w_k1_KO_s111 + 1.18848 w_k1_MAT_s111
                                 + 0.92952 w_k1_MSFT_s111 + 1.1386 w_k1_PG_s111
                                 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.29854 w_k1_ADM_s111
                                 + 1.53808 w_k1_AXP_s111
                                 + 1.01811 w_k1_DIS_s111
                                 + 1.21545 w_k1_FDX_s111
                                 + 1.07271 w_k1_IBM_s111
                                 + 1.17716 w_k1_INTC_s111
                                 + 1.11557 w_k1_KO_s111 + 1.303 w_k1_MAT_s111
                                 + 1.05503 w_k1_MSFT_s111 + 1.3104 w_k1_PG_s111
                                 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.29862 w_k1_ADM_s111
                                 + 1.2761 w_k1_AXP_s111 + 1.36387 w_k1_DIS_s111
                                 + 1.32362 w_k1_FDX_s111
                                 + 0.949009 w_k1_IBM_s111
                                 + 1.24151 w_k1_INTC_s111
                                 + 1.07478 w_k1_KO_s111 + 1.10204 w_k1_MAT_s111
                                 + 1.21713 w_k1_MSFT_s111
                                 + 1.43076 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.33904 w_k1_ADM_s111
                                 + 0.884128 w_k1_AXP_s111
                                 + 1.17911 w_k1_DIS_s111
                                 + 1.04183 w_k1_FDX_s111
                                 + 0.98825 w_k1_IBM_s111
                                 + 1.12178 w_k1_INTC_s111
                                 + 1.20538 w_k1_KO_s111
                                 + 0.771308 w_k1_MAT_s111
                                 + 1.22989 w_k1_MSFT_s111 + 1.1757 w_k1_PG_s111
                                 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.02013 w_k1_ADM_s111
                                 + 0.934243 w_k1_AXP_s111
                                 + 0.803339 w_k1_DIS_s111
                                 + 0.824802 w_k1_FDX_s111
                                 + 1.05629 w_k1_IBM_s111
                                 + 0.874947 w_k1_INTC_s111
                                 + 0.904697 w_k1_KO_s111
                                 + 1.01774 w_k1_MAT_s111
                                 + 0.801301 w_k1_MSFT_s111
                                 + 1.02483 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.889798 w_k1_ADM_s111
                                 + 0.836035 w_k1_AXP_s111
                                 + 0.787735 w_k1_DIS_s111
                                 + 0.945147 w_k1_FDX_s111
                                 + 0.728594 w_k1_IBM_s111
                                 + 1.08945 w_k1_INTC_s111
                                 + 0.874364 w_k1_KO_s111
                                 + 0.754205 w_k1_MAT_s111
                                 + 0.921312 w_k1_MSFT_s111
                                 + 0.957787 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.32838 w_k1_ADM_s111
                                 + 1.85076 w_k1_AXP_s111
                                 + 1.90517 w_k1_DIS_s111
                                 + 1.55257 w_k1_FDX_s111
                                 + 1.15864 w_k1_IBM_s111
                                 + 1.50683 w_k1_INTC_s111 + 1.4595 w_k1_KO_s111
                                 + 1.2202 w_k1_MAT_s111
                                 + 1.45445 w_k1_MSFT_s111
                                 + 1.49163 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.42108 w_k1_ADM_s111
                                 + 1.56222 w_k1_AXP_s111
                                 + 1.23496 w_k1_DIS_s111
                                 + 1.16317 w_k1_FDX_s111
                                 + 1.74272 w_k1_IBM_s111
                                 + 0.997715 w_k1_INTC_s111
                                 + 1.40509 w_k1_KO_s111 + 1.72878 w_k1_MAT_s111
                                 + 0.979743 w_k1_MSFT_s111
                                 + 1.30175 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.05756 w_k1_ADM_s112
                                 + 1.40668 w_k1_AXP_s112
                                 + 1.61109 w_k1_DIS_s112 + 1.1693 w_k1_FDX_s112
                                 + 1.53244 w_k1_IBM_s112
                                 + 1.18501 w_k1_INTC_s112
                                 + 1.73991 w_k1_KO_s112 + 2.25782 w_k1_MAT_s112
                                 + 1.30137 w_k1_MSFT_s112
                                 + 1.16566 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.88999 w_k1_ADM_s112
                                 + 0.730943 w_k1_AXP_s112
                                 + 0.684751 w_k1_DIS_s112
                                 + 0.828265 w_k1_FDX_s112
                                 + 0.774348 w_k1_IBM_s112
                                 + 0.933145 w_k1_INTC_s112
                                 + 0.939549 w_k1_KO_s112
                                 + 0.869887 w_k1_MAT_s112
                                 + 0.864887 w_k1_MSFT_s112
                                 + 0.909282 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.27147 w_k1_ADM_s112
                                 + 1.30088 w_k1_AXP_s112
                                 + 1.34796 w_k1_DIS_s112
                                 + 1.30287 w_k1_FDX_s112
                                 + 0.730552 w_k1_IBM_s112
                                 + 1.29417 w_k1_INTC_s112 + 1.0444 w_k1_KO_s112
                                 + 0.926365 w_k1_MAT_s112
                                 + 1.34558 w_k1_MSFT_s112
                                 + 1.01169 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.47765 w_k1_ADM_s112
                                 + 1.95065 w_k1_AXP_s112
                                 + 1.66167 w_k1_DIS_s112
                                 + 1.58538 w_k1_FDX_s112
                                 + 1.06138 w_k1_IBM_s112
                                 + 1.64584 w_k1_INTC_s112 + 1.3342 w_k1_KO_s112
                                 + 1.63907 w_k1_MAT_s112
                                 + 1.3897 w_k1_MSFT_s112 + 1.30447 w_k1_PG_s112
                                 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.10713 w_k1_ADM_s112
                                 + 1.01723 w_k1_AXP_s112
                                 + 0.920848 w_k1_DIS_s112
                                 + 0.94212 w_k1_FDX_s112
                                 + 1.06839 w_k1_IBM_s112
                                 + 0.96447 w_k1_INTC_s112
                                 + 1.11598 w_k1_KO_s112 + 1.26107 w_k1_MAT_s112
                                 + 1.07289 w_k1_MSFT_s112
                                 + 0.973373 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.901043 w_k1_ADM_s112
                                 + 1.26398 w_k1_AXP_s112
                                 + 0.857261 w_k1_DIS_s112
                                 + 0.891405 w_k1_FDX_s112
                                 + 1.52607 w_k1_IBM_s112
                                 + 1.16813 w_k1_INTC_s112
                                 + 1.22544 w_k1_KO_s112 + 1.53283 w_k1_MAT_s112
                                 + 0.918002 w_k1_MSFT_s112
                                 + 0.952502 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.05926 w_k1_ADM_s112
                                 + 1.33888 w_k1_AXP_s112
                                 + 1.31889 w_k1_DIS_s112
                                 + 1.14586 w_k1_FDX_s112
                                 + 1.06538 w_k1_IBM_s112
                                 + 1.24502 w_k1_INTC_s112
                                 + 1.18137 w_k1_KO_s112 + 1.37467 w_k1_MAT_s112
                                 + 1.14786 w_k1_MSFT_s112
                                 + 1.03825 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.07251 w_k1_ADM_s112
                                 + 1.07824 w_k1_AXP_s112
                                 + 1.04716 w_k1_DIS_s112
                                 + 1.10706 w_k1_FDX_s112
                                 + 0.652459 w_k1_IBM_s112
                                 + 1.21302 w_k1_INTC_s112
                                 + 1.02167 w_k1_KO_s112
                                 + 0.684518 w_k1_MAT_s112
                                 + 1.1724 w_k1_MSFT_s112
                                 + 0.877744 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.952898 w_k1_ADM_s113
                                 + 1.29046 w_k1_AXP_s113
                                 + 1.06595 w_k1_DIS_s113
                                 + 1.01077 w_k1_FDX_s113 + 1.3422 w_k1_IBM_s113
                                 + 0.948807 w_k1_INTC_s113
                                 + 1.15844 w_k1_KO_s113 + 2.17525 w_k1_MAT_s113
                                 + 1.01843 w_k1_MSFT_s113
                                 + 0.998365 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.922855 w_k1_ADM_s113
                                 + 0.974756 w_k1_AXP_s113
                                 + 0.954309 w_k1_DIS_s113
                                 + 0.972953 w_k1_FDX_s113
                                 + 0.860761 w_k1_IBM_s113
                                 + 0.891191 w_k1_INTC_s113
                                 + 0.830601 w_k1_KO_s113
                                 + 0.953713 w_k1_MAT_s113
                                 + 0.992673 w_k1_MSFT_s113
                                 + 0.989151 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.85521 w_k1_ADM_s113
                                 + 0.774528 w_k1_AXP_s113
                                 + 0.674881 w_k1_DIS_s113
                                 + 0.777746 w_k1_FDX_s113
                                 + 0.836223 w_k1_IBM_s113
                                 + 0.878861 w_k1_INTC_s113
                                 + 0.820007 w_k1_KO_s113
                                 + 0.713635 w_k1_MAT_s113
                                 + 0.865054 w_k1_MSFT_s113
                                 + 0.789992 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.00758 w_k1_ADM_s113
                                 + 0.972191 w_k1_AXP_s113
                                 + 1.33915 w_k1_DIS_s113
                                 + 0.940021 w_k1_FDX_s113
                                 + 1.0347 w_k1_IBM_s113
                                 + 1.04177 w_k1_INTC_s113
                                 + 1.05404 w_k1_KO_s113 + 1.2542 w_k1_MAT_s113
                                 + 1.1555 w_k1_MSFT_s113 + 1.02479 w_k1_PG_s113
                                 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.27184 w_k1_ADM_s113
                                 + 1.28874 w_k1_AXP_s113
                                 + 1.26346 w_k1_DIS_s113
                                 + 1.23393 w_k1_FDX_s113
                                 + 0.718874 w_k1_IBM_s113
                                 + 1.30082 w_k1_INTC_s113
                                 + 0.952817 w_k1_KO_s113
                                 + 1.15831 w_k1_MAT_s113
                                 + 1.14961 w_k1_MSFT_s113
                                 + 0.999144 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.982682 w_k1_ADM_s113
                                 + 0.812152 w_k1_AXP_s113
                                 + 0.79001 w_k1_DIS_s113
                                 + 0.914437 w_k1_FDX_s113
                                 + 0.524936 w_k1_IBM_s113
                                 + 1.00682 w_k1_INTC_s113
                                 + 0.656498 w_k1_KO_s113
                                 + 0.588795 w_k1_MAT_s113
                                 + 0.907882 w_k1_MSFT_s113
                                 + 0.774798 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.2885 w_k1_ADM_s113
                                 + 1.56973 w_k1_AXP_s113
                                 + 1.94931 w_k1_DIS_s113
                                 + 1.45426 w_k1_FDX_s113
                                 + 1.28805 w_k1_IBM_s113
                                 + 1.39903 w_k1_INTC_s113
                                 + 1.26774 w_k1_KO_s113 + 1.39114 w_k1_MAT_s113
                                 + 1.25817 w_k1_MSFT_s113
                                 + 1.08912 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.802198 w_k1_ADM_s113
                                 + 0.865661 w_k1_AXP_s113
                                 + 0.680223 w_k1_DIS_s113
                                 + 0.719955 w_k1_FDX_s113
                                 + 1.14317 w_k1_IBM_s113
                                 + 0.989123 w_k1_INTC_s113
                                 + 0.813093 w_k1_KO_s113
                                 + 1.20431 w_k1_MAT_s113
                                 + 0.895727 w_k1_MSFT_s113
                                 + 0.874726 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.927257 w_k1_ADM_s114
                                 + 1.26288 w_k1_AXP_s114 + 1.3044 w_k1_DIS_s114
                                 + 1.02146 w_k1_FDX_s114
                                 + 1.86201 w_k1_IBM_s114
                                 + 0.858512 w_k1_INTC_s114
                                 + 1.24766 w_k1_KO_s114 + 2.0628 w_k1_MAT_s114
                                 + 0.943689 w_k1_MSFT_s114
                                 + 1.11963 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.32375 w_k1_ADM_s114
                                 + 1.11349 w_k1_AXP_s114
                                 + 1.39055 w_k1_DIS_s114
                                 + 1.24642 w_k1_FDX_s114
                                 + 0.980943 w_k1_IBM_s114
                                 + 1.13675 w_k1_INTC_s114
                                 + 1.17754 w_k1_KO_s114 + 1.06002 w_k1_MAT_s114
                                 + 1.20079 w_k1_MSFT_s114
                                 + 1.06639 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.829798 w_k1_ADM_s114
                                 + 0.926345 w_k1_AXP_s114
                                 + 0.84679 w_k1_DIS_s114
                                 + 0.77199 w_k1_FDX_s114
                                 + 1.11161 w_k1_IBM_s114
                                 + 0.841368 w_k1_INTC_s114
                                 + 0.837377 w_k1_KO_s114
                                 + 1.21256 w_k1_MAT_s114
                                 + 0.783467 w_k1_MSFT_s114
                                 + 0.829528 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.22581 w_k1_ADM_s114
                                 + 1.44539 w_k1_AXP_s114
                                 + 1.73131 w_k1_DIS_s114
                                 + 1.35471 w_k1_FDX_s114
                                 + 1.38064 w_k1_IBM_s114
                                 + 1.17527 w_k1_INTC_s114
                                 + 1.28596 w_k1_KO_s114 + 1.58478 w_k1_MAT_s114
                                 + 1.18954 w_k1_MSFT_s114
                                 + 1.21064 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.738319 w_k1_ADM_s114
                                 + 1.0926 w_k1_AXP_s114 + 1.1628 w_k1_DIS_s114
                                 + 1.01485 w_k1_FDX_s114
                                 + 1.31517 w_k1_IBM_s114
                                 + 1.03533 w_k1_INTC_s114
                                 + 1.04185 w_k1_KO_s114 + 1.13167 w_k1_MAT_s114
                                 + 1.05824 w_k1_MSFT_s114
                                 + 1.01669 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.972931 w_k1_ADM_s114
                                 + 0.982798 w_k1_AXP_s114
                                 + 1.00805 w_k1_DIS_s114
                                 + 1.12138 w_k1_FDX_s114
                                 + 0.690874 w_k1_IBM_s114
                                 + 1.02812 w_k1_INTC_s114
                                 + 0.80026 w_k1_KO_s114
                                 + 0.571032 w_k1_MAT_s114
                                 + 0.917812 w_k1_MSFT_s114
                                 + 0.946422 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.04157 w_k1_ADM_s114
                                 + 0.820438 w_k1_AXP_s114
                                 + 1.02312 w_k1_DIS_s114
                                 + 0.966345 w_k1_FDX_s114
                                 + 1.09147 w_k1_IBM_s114
                                 + 0.934206 w_k1_INTC_s114
                                 + 1.16136 w_k1_KO_s114
                                 + 0.899491 w_k1_MAT_s114
                                 + 1.00046 w_k1_MSFT_s114
                                 + 0.959848 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.925594 w_k1_ADM_s114
                                 + 0.993305 w_k1_AXP_s114
                                 + 0.723195 w_k1_DIS_s114
                                 + 0.917212 w_k1_FDX_s114
                                 + 0.751934 w_k1_IBM_s114
                                 + 0.914963 w_k1_INTC_s114
                                 + 0.795074 w_k1_KO_s114
                                 + 0.68242 w_k1_MAT_s114
                                 + 0.845793 w_k1_MSFT_s114
                                 + 0.833931 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.904352 w_k1_ADM_s115
                                 + 1.35387 w_k1_AXP_s115
                                 + 1.19186 w_k1_DIS_s115 + 1.1004 w_k1_FDX_s115
                                 + 1.38605 w_k1_IBM_s115
                                 + 1.08414 w_k1_INTC_s115
                                 + 1.25104 w_k1_KO_s115 + 1.34988 w_k1_MAT_s115
                                 + 0.935233 w_k1_MSFT_s115
                                 + 1.1171 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.02687 w_k1_ADM_s115
                                 + 0.994945 w_k1_AXP_s115
                                 + 0.963579 w_k1_DIS_s115
                                 + 1.03543 w_k1_FDX_s115
                                 + 0.920887 w_k1_IBM_s115
                                 + 1.08886 w_k1_INTC_s115
                                 + 1.05895 w_k1_KO_s115
                                 + 0.753801 w_k1_MAT_s115
                                 + 1.0194 w_k1_MSFT_s115
                                 + 0.970566 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.878026 w_k1_ADM_s115
                                 + 0.947729 w_k1_AXP_s115
                                 + 0.578707 w_k1_DIS_s115
                                 + 0.749875 w_k1_FDX_s115
                                 + 1.36048 w_k1_IBM_s115
                                 + 0.913323 w_k1_INTC_s115 + 1.061 w_k1_KO_s115
                                 + 1.29799 w_k1_MAT_s115
                                 + 0.695294 w_k1_MSFT_s115
                                 + 0.953948 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.725293 w_k1_ADM_s115
                                 + 0.877637 w_k1_AXP_s115
                                 + 0.722486 w_k1_DIS_s115
                                 + 0.822285 w_k1_FDX_s115
                                 + 1.06217 w_k1_IBM_s115
                                 + 0.953821 w_k1_INTC_s115
                                 + 0.976275 w_k1_KO_s115
                                 + 1.15549 w_k1_MAT_s115
                                 + 0.821437 w_k1_MSFT_s115
                                 + 0.822422 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.763343 w_k1_ADM_s115
                                 + 0.762958 w_k1_AXP_s115
                                 + 0.568731 w_k1_DIS_s115
                                 + 0.803967 w_k1_FDX_s115
                                 + 0.787678 w_k1_IBM_s115
                                 + 0.748726 w_k1_INTC_s115
                                 + 0.744659 w_k1_KO_s115
                                 + 0.595089 w_k1_MAT_s115
                                 + 0.746728 w_k1_MSFT_s115
                                 + 0.78245 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.01077 w_k1_ADM_s115
                                 + 0.976648 w_k1_AXP_s115
                                 + 0.896726 w_k1_DIS_s115
                                 + 1.09796 w_k1_FDX_s115
                                 + 0.578746 w_k1_IBM_s115
                                 + 0.98706 w_k1_INTC_s115
                                 + 0.864866 w_k1_KO_s115
                                 + 0.564389 w_k1_MAT_s115
                                 + 0.915117 w_k1_MSFT_s115
                                 + 0.885393 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.05799 w_k1_ADM_s115
                                 + 1.45147 w_k1_AXP_s115
                                 + 0.938032 w_k1_DIS_s115
                                 + 1.17426 w_k1_FDX_s115
                                 + 0.953784 w_k1_IBM_s115
                                 + 1.10715 w_k1_INTC_s115
                                 + 1.01124 w_k1_KO_s115 + 0.95231 w_k1_MAT_s115
                                 + 1.05792 w_k1_MSFT_s115
                                 + 1.06009 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.10047 w_k1_ADM_s115
                                 + 1.40645 w_k1_AXP_s115
                                 + 1.66332 w_k1_DIS_s115
                                 + 1.49083 w_k1_FDX_s115
                                 + 0.88962 w_k1_IBM_s115
                                 + 1.24911 w_k1_INTC_s115
                                 + 1.11301 w_k1_KO_s115 + 0.92089 w_k1_MAT_s115
                                 + 1.3254 w_k1_MSFT_s115 + 1.08403 w_k1_PG_s115
                                 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.725892 w_k1_ADM_s116
                                 + 0.588043 w_k1_AXP_s116 + 0.759 w_k1_DIS_s116
                                 + 0.847566 w_k1_FDX_s116
                                 + 0.629728 w_k1_IBM_s116
                                 + 0.780927 w_k1_INTC_s116
                                 + 0.732636 w_k1_KO_s116
                                 + 0.742993 w_k1_MAT_s116
                                 + 0.822437 w_k1_MSFT_s116
                                 + 0.862947 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.782161 w_k1_ADM_s116
                                 + 1.16148 w_k1_AXP_s116 + 1.0502 w_k1_DIS_s116
                                 + 1.0928 w_k1_FDX_s116
                                 + 0.899772 w_k1_IBM_s116
                                 + 1.00855 w_k1_INTC_s116 + 1.0429 w_k1_KO_s116
                                 + 1.06922 w_k1_MAT_s116
                                 + 1.03528 w_k1_MSFT_s116 + 1.0094 w_k1_PG_s116
                                 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.947585 w_k1_ADM_s116
                                 + 1.38864 w_k1_AXP_s116
                                 + 1.20411 w_k1_DIS_s116
                                 + 1.23744 w_k1_FDX_s116
                                 + 1.28287 w_k1_IBM_s116
                                 + 1.13289 w_k1_INTC_s116
                                 + 1.21582 w_k1_KO_s116 + 1.52844 w_k1_MAT_s116
                                 + 1.12589 w_k1_MSFT_s116
                                 + 1.21195 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.771639 w_k1_ADM_s116
                                 + 1.00551 w_k1_AXP_s116
                                 + 0.718321 w_k1_DIS_s116
                                 + 0.885583 w_k1_FDX_s116
                                 + 0.835123 w_k1_IBM_s116
                                 + 0.876324 w_k1_INTC_s116
                                 + 0.763066 w_k1_KO_s116
                                 + 0.815933 w_k1_MAT_s116
                                 + 0.758792 w_k1_MSFT_s116
                                 + 0.883133 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.618504 w_k1_ADM_s116
                                 + 0.74865 w_k1_AXP_s116
                                 + 0.812637 w_k1_DIS_s116
                                 + 0.816772 w_k1_FDX_s116
                                 + 0.957013 w_k1_IBM_s116
                                 + 0.87123 w_k1_INTC_s116
                                 + 0.830346 w_k1_KO_s116
                                 + 1.00186 w_k1_MAT_s116
                                 + 0.831922 w_k1_MSFT_s116
                                 + 0.86611 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.846445 w_k1_ADM_s116
                                 + 0.929085 w_k1_AXP_s116
                                 + 0.749995 w_k1_DIS_s116
                                 + 0.987249 w_k1_FDX_s116
                                 + 0.704491 w_k1_IBM_s116
                                 + 0.910754 w_k1_INTC_s116
                                 + 0.718474 w_k1_KO_s116
                                 + 0.832986 w_k1_MAT_s116
                                 + 0.819169 w_k1_MSFT_s116
                                 + 0.86448 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.93278 w_k1_ADM_s116
                                 + 0.866978 w_k1_AXP_s116
                                 + 0.954294 w_k1_DIS_s116
                                 + 0.879982 w_k1_FDX_s116
                                 + 1.18988 w_k1_IBM_s116
                                 + 0.985459 w_k1_INTC_s116
                                 + 1.11908 w_k1_KO_s116 + 1.06231 w_k1_MAT_s116
                                 + 0.899521 w_k1_MSFT_s116
                                 + 1.0883 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.03147 w_k1_ADM_s116
                                 + 0.780305 w_k1_AXP_s116
                                 + 1.00116 w_k1_DIS_s116
                                 + 0.950267 w_k1_FDX_s116
                                 + 0.787169 w_k1_IBM_s116
                                 + 1.05919 w_k1_INTC_s116
                                 + 0.952576 w_k1_KO_s116
                                 + 0.866653 w_k1_MAT_s116
                                 + 0.978715 w_k1_MSFT_s116
                                 + 1.13276 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.0463 w_k1_ADM_s117
                                 + 1.08923 w_k1_AXP_s117
                                 + 1.18155 w_k1_DIS_s117
                                 + 1.06605 w_k1_FDX_s117
                                 + 0.920372 w_k1_IBM_s117
                                 + 1.05086 w_k1_INTC_s117
                                 + 1.10128 w_k1_KO_s117 + 1.02596 w_k1_MAT_s117
                                 + 1.10778 w_k1_MSFT_s117
                                 + 0.977697 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.994998 w_k1_ADM_s117
                                 + 0.914566 w_k1_AXP_s117
                                 + 0.861558 w_k1_DIS_s117
                                 + 0.839165 w_k1_FDX_s117
                                 + 0.999934 w_k1_IBM_s117
                                 + 0.971444 w_k1_INTC_s117
                                 + 0.955213 w_k1_KO_s117
                                 + 1.03056 w_k1_MAT_s117
                                 + 0.894752 w_k1_MSFT_s117
                                 + 0.881932 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.833853 w_k1_ADM_s117
                                 + 0.802739 w_k1_AXP_s117
                                 + 0.795815 w_k1_DIS_s117
                                 + 0.936291 w_k1_FDX_s117
                                 + 0.675131 w_k1_IBM_s117
                                 + 0.939146 w_k1_INTC_s117
                                 + 0.7058 w_k1_KO_s117 + 0.713984 w_k1_MAT_s117
                                 + 0.908164 w_k1_MSFT_s117
                                 + 0.840067 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.03373 w_k1_ADM_s117
                                 + 1.07825 w_k1_AXP_s117
                                 + 1.36935 w_k1_DIS_s117
                                 + 1.03123 w_k1_FDX_s117
                                 + 1.23266 w_k1_IBM_s117
                                 + 1.27213 w_k1_INTC_s117 + 1.3288 w_k1_KO_s117
                                 + 1.13115 w_k1_MAT_s117
                                 + 1.20268 w_k1_MSFT_s117
                                 + 1.01183 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.15323 w_k1_ADM_s117
                                 + 1.62906 w_k1_AXP_s117
                                 + 1.70037 w_k1_DIS_s117
                                 + 1.45222 w_k1_FDX_s117
                                 + 1.26311 w_k1_IBM_s117
                                 + 1.30178 w_k1_INTC_s117
                                 + 1.40437 w_k1_KO_s117 + 1.65374 w_k1_MAT_s117
                                 + 1.40782 w_k1_MSFT_s117
                                 + 1.09025 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.938272 w_k1_ADM_s117
                                 + 0.978455 w_k1_AXP_s117
                                 + 1.18761 w_k1_DIS_s117
                                 + 1.01296 w_k1_FDX_s117
                                 + 1.48586 w_k1_IBM_s117
                                 + 1.0096 w_k1_INTC_s117 + 1.21054 w_k1_KO_s117
                                 + 1.47716 w_k1_MAT_s117
                                 + 0.962414 w_k1_MSFT_s117
                                 + 0.97094 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.28701 w_k1_ADM_s117
                                 + 1.25848 w_k1_AXP_s117
                                 + 1.45991 w_k1_DIS_s117
                                 + 1.27659 w_k1_FDX_s117
                                 + 0.967195 w_k1_IBM_s117
                                 + 1.25663 w_k1_INTC_s117
                                 + 1.18255 w_k1_KO_s117 + 1.28621 w_k1_MAT_s117
                                 + 1.31033 w_k1_MSFT_s117
                                 + 1.18389 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.754479 w_k1_ADM_s117
                                 + 0.631433 w_k1_AXP_s117
                                 + 0.694595 w_k1_DIS_s117
                                 + 0.684958 w_k1_FDX_s117
                                 + 0.807158 w_k1_IBM_s117
                                 + 0.735977 w_k1_INTC_s117
                                 + 0.728005 w_k1_KO_s117
                                 + 0.825693 w_k1_MAT_s117
                                 + 0.717563 w_k1_MSFT_s117
                                 + 0.728537 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.00894972388412087 w_k1_ADM_s000
                                 + 0.00337074940922 w_k1_ADM_s001
                                 + 0.00640011435503314 w_k1_ADM_s002
                                 + 0.00498970476547213 w_k1_ADM_s003
                                 + 0.00450703054694175 w_k1_ADM_s004
                                 + 0.00327339740107489 w_k1_ADM_s005
                                 + 0.00433135177685099 w_k1_ADM_s006
                                 + 0.0105141303828944 w_k1_ADM_s007
                                 + 0.0398019497093625 w_k1_ADM_s010
                                 + 0.0410435133459823 w_k1_ADM_s011
                                 + 0.0189898479745109 w_k1_ADM_s012
                                 + 0.0159974995190803 w_k1_ADM_s013
                                 + 0.0155786035674357 w_k1_ADM_s014
                                 + 0.012560454944304 w_k1_ADM_s015
                                 + 0.0135233166036906 w_k1_ADM_s016
                                 + 0.0338780383456432 w_k1_ADM_s017
                                 + 0.0473522383444728 w_k1_ADM_s100
                                 + 0.0340273842078316 w_k1_ADM_s101
                                 + 0.0402996788296817 w_k1_ADM_s102
                                 + 0.0305949810634012 w_k1_ADM_s103
                                 + 0.0262085229132289 w_k1_ADM_s104
                                 + 0.0608312121790277 w_k1_ADM_s105
                                 + 0.0147830176476386 w_k1_ADM_s106
                                 + 0.0227139037187887 w_k1_ADM_s107
                                 + 0.0887795701419149 w_k1_ADM_s110
                                 + 0.0354176023945474 w_k1_ADM_s111
                                 + 0.0395142356395451 w_k1_ADM_s112
                                 + 0.0748495395616871 w_k1_ADM_s113
                                 + 0.0919923711150936 w_k1_ADM_s114
                                 + 0.0750111043917382 w_k1_ADM_s115
                                 + 0.0307251683973687 w_k1_ADM_s116
                                 + 0.0459326997959777 w_k1_ADM_s117
                                 + 0.0121002503523821 w_k1_AXP_s000
                                 + 0.00356478140140333 w_k1_AXP_s001
                                 + 0.00619417273675927 w_k1_AXP_s002
                                 + 0.00588718245922662 w_k1_AXP_s003
                                 + 0.00617381562591253 w_k1_AXP_s004
                                 + 0.00424212964728864 w_k1_AXP_s005
                                 + 0.00389889154182828 w_k1_AXP_s006
                                 + 0.0103601554405174 w_k1_AXP_s007
                                 + 0.0413767838388211 w_k1_AXP_s010
                                 + 0.040836712692763 w_k1_AXP_s011
                                 + 0.0169830471189376 w_k1_AXP_s012
                                 + 0.0171459180011796 w_k1_AXP_s013
                                 + 0.0155540378980679 w_k1_AXP_s014
                                 + 0.0131681164202912 w_k1_AXP_s015
                                 + 0.0113610887313992 w_k1_AXP_s016
                                 + 0.0339427753053858 w_k1_AXP_s017
                                 + 0.0465943887047046 w_k1_AXP_s100
                                 + 0.0360719286255446 w_k1_AXP_s101
                                 + 0.0412679838588086 w_k1_AXP_s102
                                 + 0.0325918318349836 w_k1_AXP_s103
                                 + 0.0286462352471841 w_k1_AXP_s104
                                 + 0.0594438969092069 w_k1_AXP_s105
                                 + 0.014501983520959 w_k1_AXP_s106
                                 + 0.0219027731381608 w_k1_AXP_s107
                                 + 0.0974637739974496 w_k1_AXP_s110
                                 + 0.0342016455260136 w_k1_AXP_s111
                                 + 0.0420235545417439 w_k1_AXP_s112
                                 + 0.0793150302022273 w_k1_AXP_s113
                                 + 0.0987955314761413 w_k1_AXP_s114
                                 + 0.0909502443812959 w_k1_AXP_s115
                                 + 0.0377734394679178 w_k1_AXP_s116
                                 + 0.0461680562767809 w_k1_AXP_s117
                                 + 0.0119999870514186 w_k1_DIS_s000
                                 + 0.00327897889702234 w_k1_DIS_s001
                                 + 0.00600318753676885 w_k1_DIS_s002
                                 + 0.00583423896953976 w_k1_DIS_s003
                                 + 0.00538596662525721 w_k1_DIS_s004
                                 + 0.0042167358021813 w_k1_DIS_s005
                                 + 0.00473014549097068 w_k1_DIS_s006
                                 + 0.0112255718876848 w_k1_DIS_s007
                                 + 0.0440361368451863 w_k1_DIS_s010
                                 + 0.03932208671542 w_k1_DIS_s011
                                 + 0.0181691160428284 w_k1_DIS_s012
                                 + 0.0166057170121609 w_k1_DIS_s013
                                 + 0.0153416641390602 w_k1_DIS_s014
                                 + 0.0134291327919121 w_k1_DIS_s015
                                 + 0.011954678696016 w_k1_DIS_s016
                                 + 0.0345491480716474 w_k1_DIS_s017
                                 + 0.0452244304673368 w_k1_DIS_s100
                                 + 0.0357782719005854 w_k1_DIS_s101
                                 + 0.0416735260039009 w_k1_DIS_s102
                                 + 0.0322523300414456 w_k1_DIS_s103
                                 + 0.0317538972778888 w_k1_DIS_s104
                                 + 0.0605807570405884 w_k1_DIS_s105
                                 + 0.0146401514233137 w_k1_DIS_s106
                                 + 0.0224236815527572 w_k1_DIS_s107
                                 + 0.0855141869727686 w_k1_DIS_s110
                                 + 0.0319733279332769 w_k1_DIS_s111
                                 + 0.0382857562461493 w_k1_DIS_s112
                                 + 0.0806316205241278 w_k1_DIS_s113
                                 + 0.100218242653706 w_k1_DIS_s114
                                 + 0.0798241618317731 w_k1_DIS_s115
                                 + 0.0355981243780516 w_k1_DIS_s116
                                 + 0.0501088574613745 w_k1_DIS_s117
                                 + 0.011545143187661 w_k1_FDX_s000
                                 + 0.00349288281024084 w_k1_FDX_s001
                                 + 0.00595312830242341 w_k1_FDX_s002
                                 + 0.00601570411456166 w_k1_FDX_s003
                                 + 0.0054145785501807 w_k1_FDX_s004
                                 + 0.00390740086171434 w_k1_FDX_s005
                                 + 0.0044838010243083 w_k1_FDX_s006
                                 + 0.0109495468537103 w_k1_FDX_s007
                                 + 0.044857001809428 w_k1_FDX_s010
                                 + 0.0400491791787657 w_k1_FDX_s011
                                 + 0.017061523933281 w_k1_FDX_s012
                                 + 0.0162787652707112 w_k1_FDX_s013
                                 + 0.0141352972903798 w_k1_FDX_s014
                                 + 0.0127817505513492 w_k1_FDX_s015
                                 + 0.0117389166046569 w_k1_FDX_s016
                                 + 0.0336122783790211 w_k1_FDX_s017
                                 + 0.0444156333771129 w_k1_FDX_s100
                                 + 0.0358056560721053 w_k1_FDX_s101
                                 + 0.0395922020559496 w_k1_FDX_s102
                                 + 0.0335552564804836 w_k1_FDX_s103
                                 + 0.0284961095255862 w_k1_FDX_s104
                                 + 0.0605539470704201 w_k1_FDX_s105
                                 + 0.0142576096117865 w_k1_FDX_s106
                                 + 0.0204015996594103 w_k1_FDX_s107
                                 + 0.0946187683341803 w_k1_FDX_s110
                                 + 0.0322626565679128 w_k1_FDX_s111
                                 + 0.0384869660320201 w_k1_FDX_s112
                                 + 0.0741136641050223 w_k1_FDX_s113
                                 + 0.0965312006429716 w_k1_FDX_s114
                                 + 0.0845097321495649 w_k1_FDX_s115
                                 + 0.0368455114395743 w_k1_FDX_s116
                                 + 0.0463481755350484 w_k1_FDX_s117
                                 + 0.0126012524773162 w_k1_IBM_s000
                                 + 0.0033962600519226 w_k1_IBM_s001
                                 + 0.00570079749387421 w_k1_IBM_s002
                                 + 0.00599828653804355 w_k1_IBM_s003
                                 + 0.00565200920140414 w_k1_IBM_s004
                                 + 0.00368616333347467 w_k1_IBM_s005
                                 + 0.00436090552425353 w_k1_IBM_s006
                                 + 0.0113200493070634 w_k1_IBM_s007
                                 + 0.0441511756630959 w_k1_IBM_s010
                                 + 0.0400756767464106 w_k1_IBM_s011
                                 + 0.017147847543979 w_k1_IBM_s012
                                 + 0.0165097664121909 w_k1_IBM_s013
                                 + 0.0144109092722172 w_k1_IBM_s014
                                 + 0.0130210050937172 w_k1_IBM_s015
                                 + 0.0112461125827398 w_k1_IBM_s016
                                 + 0.0351218497218947 w_k1_IBM_s017
                                 + 0.0451668834750233 w_k1_IBM_s100
                                 + 0.0370110630031446 w_k1_IBM_s101
                                 + 0.0438962945101032 w_k1_IBM_s102
                                 + 0.0335214729582064 w_k1_IBM_s103
                                 + 0.0294529122946732 w_k1_IBM_s104
                                 + 0.0566168748594583 w_k1_IBM_s105
                                 + 0.0152182892331836 w_k1_IBM_s106
                                 + 0.0211000472689848 w_k1_IBM_s107
                                 + 0.0886430628011889 w_k1_IBM_s110
                                 + 0.0327979506586851 w_k1_IBM_s111
                                 + 0.0383430160934936 w_k1_IBM_s112
                                 + 0.0723586293978733 w_k1_IBM_s113
                                 + 0.0983748229078951 w_k1_IBM_s114
                                 + 0.0816128660862883 w_k1_IBM_s115
                                 + 0.0366265658439958 w_k1_IBM_s116
                                 + 0.0464871692866447 w_k1_IBM_s117
                                 + 0.0117093573651385 w_k1_INTC_s000
                                 + 0.00299490288588423 w_k1_INTC_s001
                                 + 0.00628242118257945 w_k1_INTC_s002
                                 + 0.00547661074784089 w_k1_INTC_s003
                                 + 0.00639522598329853 w_k1_INTC_s004
                                 + 0.00357927417127974 w_k1_INTC_s005
                                 + 0.00446300955294767 w_k1_INTC_s006
                                 + 0.0104565161631122 w_k1_INTC_s007
                                 + 0.0357627980056762 w_k1_INTC_s010
                                 + 0.0449437659622499 w_k1_INTC_s011
                                 + 0.0144762990079587 w_k1_INTC_s012
                                 + 0.0157669448263316 w_k1_INTC_s013
                                 + 0.0141269647105918 w_k1_INTC_s014
                                 + 0.0129648677037516 w_k1_INTC_s015
                                 + 0.00971330281582771 w_k1_INTC_s016
                                 + 0.0336894576257757 w_k1_INTC_s017
                                 + 0.0384860108851357 w_k1_INTC_s100
                                 + 0.0353192820078001 w_k1_INTC_s101
                                 + 0.0439681801987809 w_k1_INTC_s102
                                 + 0.0404592650667236 w_k1_INTC_s103
                                 + 0.0283418241358194 w_k1_INTC_s104
                                 + 0.0622703406862902 w_k1_INTC_s105
                                 + 0.0153213563861199 w_k1_INTC_s106
                                 + 0.0178618373776002 w_k1_INTC_s107
                                 + 0.0961067090998778 w_k1_INTC_s110
                                 + 0.033262785297096 w_k1_INTC_s111
                                 + 0.0420001189990617 w_k1_INTC_s112
                                 + 0.0773295101176746 w_k1_INTC_s113
                                 + 0.0928160213840929 w_k1_INTC_s114
                                 + 0.0826647647758774 w_k1_INTC_s115
                                 + 0.0358299747840202 w_k1_INTC_s116
                                 + 0.0482983962339516 w_k1_INTC_s117
                                 + 0.0123656931131266 w_k1_KO_s000
                                 + 0.00330843527746578 w_k1_KO_s001
                                 + 0.00583839553636632 w_k1_KO_s002
                                 + 0.00552441761081654 w_k1_KO_s003
                                 + 0.00568754393831642 w_k1_KO_s004
                                 + 0.00351810966702973 w_k1_KO_s005
                                 + 0.00478798584360508 w_k1_KO_s006
                                 + 0.0116156818648282 w_k1_KO_s007
                                 + 0.0447782759481881 w_k1_KO_s010
                                 + 0.0410629457612412 w_k1_KO_s011
                                 + 0.0170137202681134 w_k1_KO_s012
                                 + 0.016276250056987 w_k1_KO_s013
                                 + 0.0151518883194478 w_k1_KO_s014
                                 + 0.0128813204019578 w_k1_KO_s015
                                 + 0.0130800334153482 w_k1_KO_s016
                                 + 0.0356506802543504 w_k1_KO_s017
                                 + 0.0445904890289846 w_k1_KO_s100
                                 + 0.0322010386967601 w_k1_KO_s101
                                 + 0.0465957031892994 w_k1_KO_s102
                                 + 0.0341554163388799 w_k1_KO_s103
                                 + 0.0308031973811313 w_k1_KO_s104
                                 + 0.0565532295959835 w_k1_KO_s105
                                 + 0.014827611907437 w_k1_KO_s106
                                 + 0.0222484679693963 w_k1_KO_s107
                                 + 0.0815832395212874 w_k1_KO_s110
                                 + 0.0329602662173611 w_k1_KO_s111
                                 + 0.0423386837666804 w_k1_KO_s112
                                 + 0.0710175235530832 w_k1_KO_s113
                                 + 0.0920588844332131 w_k1_KO_s114
                                 + 0.0834896823301181 w_k1_KO_s115
                                 + 0.036621258428672 w_k1_KO_s116
                                 + 0.0477044745538675 w_k1_KO_s117
                                 + 0.012168893703591 w_k1_MAT_s000
                                 + 0.00408150718309998 w_k1_MAT_s001
                                 + 0.0055157969110334 w_k1_MAT_s002
                                 + 0.00560583048759513 w_k1_MAT_s003
                                 + 0.00518254411929565 w_k1_MAT_s004
                                 + 0.00356831155441759 w_k1_MAT_s005
                                 + 0.00436336193906553 w_k1_MAT_s006
                                 + 0.00922616568836381 w_k1_MAT_s007
                                 + 0.0479125456111741 w_k1_MAT_s010
                                 + 0.0359555689062945 w_k1_MAT_s011
                                 + 0.0193672857320722 w_k1_MAT_s012
                                 + 0.0166912681416068 w_k1_MAT_s013
                                 + 0.0157036821493181 w_k1_MAT_s014
                                 + 0.0138848629986199 w_k1_MAT_s015
                                 + 0.0107452436267326 w_k1_MAT_s016
                                 + 0.0331583137944609 w_k1_MAT_s017
                                 + 0.0540160819911373 w_k1_MAT_s100
                                 + 0.0374985836548489 w_k1_MAT_s101
                                 + 0.0438066087294322 w_k1_MAT_s102
                                 + 0.0309169683847599 w_k1_MAT_s103
                                 + 0.0258721112760718 w_k1_MAT_s104
                                 + 0.0555352470270258 w_k1_MAT_s105
                                 + 0.0148590299921523 w_k1_MAT_s106
                                 + 0.022573065234284 w_k1_MAT_s107
                                 + 0.0924184408449729 w_k1_MAT_s110
                                 + 0.0324915228041086 w_k1_MAT_s111
                                 + 0.0454439835280015 w_k1_MAT_s112
                                 + 0.0911043489346088 w_k1_MAT_s113
                                 + 0.0964708830719429 w_k1_MAT_s114
                                 + 0.0777467812627456 w_k1_MAT_s115
                                 + 0.0404164723300656 w_k1_MAT_s116
                                 + 0.0497797848578647 w_k1_MAT_s117
                                 + 0.0110596845653906 w_k1_MSFT_s000
                                 + 0.00348929975843082 w_k1_MSFT_s001
                                 + 0.00605036645356991 w_k1_MSFT_s002
                                 + 0.00574533539988934 w_k1_MSFT_s003
                                 + 0.00522497147409967 w_k1_MSFT_s004
                                 + 0.00373244373080402 w_k1_MSFT_s005
                                 + 0.00503203069999381 w_k1_MSFT_s006
                                 + 0.0118269121802228 w_k1_MSFT_s007
                                 + 0.0431158283407619 w_k1_MSFT_s010
                                 + 0.039296863691155 w_k1_MSFT_s011
                                 + 0.0159503602742316 w_k1_MSFT_s012
                                 + 0.0158274755419746 w_k1_MSFT_s013
                                 + 0.0157557615299863 w_k1_MSFT_s014
                                 + 0.013000628729494 w_k1_MSFT_s015
                                 + 0.0114881545670629 w_k1_MSFT_s016
                                 + 0.0360493432560092 w_k1_MSFT_s017
                                 + 0.0446569163006027 w_k1_MSFT_s100
                                 + 0.0358011148848634 w_k1_MSFT_s101
                                 + 0.0412454740190582 w_k1_MSFT_s102
                                 + 0.0336172754539536 w_k1_MSFT_s103
                                 + 0.0289252763052072 w_k1_MSFT_s104
                                 + 0.0567805174147905 w_k1_MSFT_s105
                                 + 0.0185004279947495 w_k1_MSFT_s106
                                 + 0.0217393814382696 w_k1_MSFT_s107
                                 + 0.0888749012366032 w_k1_MSFT_s110
                                 + 0.0311819984091769 w_k1_MSFT_s111
                                 + 0.0402415850071345 w_k1_MSFT_s112
                                 + 0.0770247709960121 w_k1_MSFT_s113
                                 + 0.0909007604850579 w_k1_MSFT_s114
                                 + 0.0760806420675064 w_k1_MSFT_s115
                                 + 0.0347401276970809 w_k1_MSFT_s116
                                 + 0.0475851213898031 w_k1_MSFT_s117
                                 + 0.0117796670920949 w_k1_PG_s000
                                 + 0.00346137022676214 w_k1_PG_s001
                                 + 0.00621397291633113 w_k1_PG_s002
                                 + 0.00630588400681176 w_k1_PG_s003
                                 + 0.00541237709994996 w_k1_PG_s004
                                 + 0.00348925379308173 w_k1_PG_s005
                                 + 0.00484330038794328 w_k1_PG_s006
                                 + 0.0120313042830556 w_k1_PG_s007
                                 + 0.0439181172066053 w_k1_PG_s010
                                 + 0.0380747870172742 w_k1_PG_s011
                                 + 0.0165444613369203 w_k1_PG_s012
                                 + 0.0155173222893761 w_k1_PG_s013
                                 + 0.0149812986562967 w_k1_PG_s014
                                 + 0.0127678757312906 w_k1_PG_s015
                                 + 0.0118701913681487 w_k1_PG_s016
                                 + 0.0359954414701306 w_k1_PG_s017
                                 + 0.0428093904665638 w_k1_PG_s100
                                 + 0.0306906118033123 w_k1_PG_s101
                                 + 0.0430714260837406 w_k1_PG_s102
                                 + 0.0323419877689433 w_k1_PG_s103
                                 + 0.02808318259849 w_k1_PG_s104
                                 + 0.0549682738662356 w_k1_PG_s105
                                 + 0.0151269602411561 w_k1_PG_s106
                                 + 0.0214351251234704 w_k1_PG_s107
                                 + 0.0874231143399944 w_k1_PG_s110
                                 + 0.035697958321079 w_k1_PG_s111
                                 + 0.0371033985434404 w_k1_PG_s112
                                 + 0.0708221683684796 w_k1_PG_s113
                                 + 0.0905587844618418 w_k1_PG_s114
                                 + 0.0785151600483426 w_k1_PG_s115
                                 + 0.0375610073484108 w_k1_PG_s116
                                 + 0.0437843841251024 w_k1_PG_s117 >= 1.2
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
