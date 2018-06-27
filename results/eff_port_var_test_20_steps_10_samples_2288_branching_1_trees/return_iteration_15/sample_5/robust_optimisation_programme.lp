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
 w_k1_ADM_s0_return_constraint:  1.12145 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.910087 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.02671 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.03395 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.11109 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.861456 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.70809 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.04239 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.00734 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.807916 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.17467 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.788345 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.13325 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.933036 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.860891 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.27019 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.867864 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.28055 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.53576 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.939327 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.958303 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.17112 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.16274 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.921843 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.925929 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.09093 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.720456 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.00655 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.953168 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.37517 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.972374 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.815526 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.8772 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.846552 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.903789 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.801977 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.28525 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.16766 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.25895 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.92704 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.89904 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.11121 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.12153 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.739648 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.737144 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.11623 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.917269 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.620804 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.33175 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.70035 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.6191 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.781387 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.993616 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.32177 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.61846 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.24247 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.38208 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.786373 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.00844 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.07259 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.2262 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.09544 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.721736 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.942742 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.713691 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.941126 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.15366 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.50822 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.07168 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.628536 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.642951 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.645779 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.922815 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.843786 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.12898 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.39839 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.27011 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.831711 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.93859 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.20851 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.1713 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.753959 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.752063 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.993858 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.02697 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.521675 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.34172 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.574926 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.85969 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.823384 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.839141 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.23721 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.535856 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.61972 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.67037 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.984262 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.941784 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.843844 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.21628 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.15072 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.715263 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.07225 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.768798 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.1916 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.925214 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.87925 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.920254 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.620554 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.634567 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.740154 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.982088 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.866972 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.28987 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.42826 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.15217 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.913142 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.997328 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.08365 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.12578 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.783161 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.800816 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.997515 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.00492 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.700649 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.311 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.631113 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.47661 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.979639 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.866096 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.37987 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.694877 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.40192 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.32393 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.935912 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.791175 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.969293 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.32182 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.881885 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.813985 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.957848 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.719617 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.12752 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.924215 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.33075 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.05023 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.710719 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.824113 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.949411 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.920057 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.81862 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.31724 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.25591 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.21056 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.900708 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.802902 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.35797 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.11394 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.840886 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.0138 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.41458 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.934668 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.631976 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.965621 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.07427 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.25364 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.689061 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.09285 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.68068 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.641631 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.890252 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.35231 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.819576 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.59472 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.825463 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.683488 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.87772 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.651972 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.03803 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.20385 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.836594 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.23377 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.37627 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.881683 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.876615 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.706742 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.483581 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.17069 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.48895 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.891192 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.62163 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.13601 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.94291 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.987774 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.09671 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.13787 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.833867 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.775809 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.874765 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.92497 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.67063 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.17226 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.784732 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.08025 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.904639 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.01291 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.12687 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.656447 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.36368 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.21996 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.881973 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.908391 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.911103 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.07584 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.923398 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.786741 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.00561 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.818573 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.07168 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.881522 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.0337 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.990847 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.755852 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.888299 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.915805 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.01267 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.863203 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.10897 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.20096 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.22107 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.896779 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.88684 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.22249 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.13368 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.891784 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.819419 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.05747 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.0008 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.642828 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.05858 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.825782 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.3503 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.773022 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.05925 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.935436 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.634544 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.20595 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.23029 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.969569 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.05923 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.744297 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.984811 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.28736 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.782372 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.15812 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.901435 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.00207 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.978668 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.31627 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.854883 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.731936 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.662454 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.657305 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.94983 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.12002 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.95217 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.32034 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.40159 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.844121 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.704344 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.28238 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.19993 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.861272 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.997386 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.5892 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.979208 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.628569 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.16567 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.919437 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.73446 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.616886 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.1137 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.837303 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.605623 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.742976 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.37602 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.669709 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.53283 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.996008 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.714648 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.65154 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.679736 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.0344 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.10091 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.845585 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.46294 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.56919 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.940286 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.580733 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.642 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.434045 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.04479 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.35858 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.906081 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.46992 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.14376 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.91811 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.975261 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.13464 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.1314 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.810177 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.81383 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.971429 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.00731 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.737663 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.16886 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.782413 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.39811 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.933285 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.895569 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.17433 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.676934 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.31793 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.24116 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.888958 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.822015 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.775623 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.19519 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.892501 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.87703 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.03732 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.736551 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.06574 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.953654 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.29678 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.949763 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.732924 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.881705 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.937366 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.05291 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.910143 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.23724 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.20804 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.10454 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.933714 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.972781 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.12355 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.11709 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.842321 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.918485 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.03879 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.00186 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.677018 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.18359 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.857271 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.23691 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.901882 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.995942 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.11639 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.741797 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.15734 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.17121 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.949821 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.00167 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.895483 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.09106 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.989554 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.825712 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.995478 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.881201 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.04723 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.02309 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.24203 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.953222 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.839856 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.870998 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.866715 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.01393 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.996585 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.17353 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.22197 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.495998820002528 z_k1_s0000
                                 - 0.151999871871441 z_k1_s0001
                                 - 0.27599905987274 z_k1_s0002
                                 - 0.137999736736103 z_k1_s0003
                                 - 0.221000670838292 z_k1_s0004
                                 - 0.377000049549087 z_k1_s0005
                                 - 0.212000746451079 z_k1_s0006
                                 - 0.195999823897399 z_k1_s0007
                                 - 0.76600003912814 z_k1_s0010
                                 - 1.13899891807145 z_k1_s0011
                                 - 0.272999573011058 z_k1_s0012
                                 - 0.640998933841845 z_k1_s0013
                                 - 0.618997476674029 z_k1_s0014
                                 - 0.356999158345807 z_k1_s0015
                                 - 0.537999285723364 z_k1_s0016
                                 - 0.92799642102947 z_k1_s0017
                                 - 0.697000987522841 z_k1_s0020
                                 - 0.777000192347431 z_k1_s0021
                                 - 0.877996549942274 z_k1_s0022
                                 - 0.60899658522339 z_k1_s0023
                                 - 0.411999598741318 z_k1_s0024
                                 - 0.874999218257553 z_k1_s0025
                                 - 0.701998299328177 z_k1_s0026
                                 - 0.326999388216711 z_k1_s0027
                                 - 0.139999924193628 z_k1_s0030
                                 - 0.349000348846507 z_k1_s0031
                                 - 0.227999353258645 z_k1_s0032
                                 - 0.22500050584636 z_k1_s0033
                                 - 0.199999627419696 z_k1_s0034
                                 - 0.170999792443744 z_k1_s0035
                                 - 0.180999773814729 z_k1_s0036
                                 - 0.355998892807442 z_k1_s0037
                                 - 0.526999317811264 z_k1_s0040
                                 - 0.186999731329827 z_k1_s0041
                                 - 0.197999844890288 z_k1_s0042
                                 - 0.297999577732019 z_k1_s0043
                                 - 0.358999832613097 z_k1_s0044
                                 - 0.301999369117867 z_k1_s0045
                                 - 0.494998237726963 z_k1_s0046
                                 - 0.382998580928185 z_k1_s0047
                                 - 0.190000053639279 z_k1_s0050
                                 - 0.219999896066596 z_k1_s0051
                                 - 0.374999464340921 z_k1_s0052
                                 - 0.450998568036975 z_k1_s0053
                                 - 0.248999954066256 z_k1_s0054
                                 - 0.162999851294822 z_k1_s0055
                                 - 0.407000596422908 z_k1_s0056
                                 - 0.813000261218587 z_k1_s0057
                                 - 0.153999799069579 z_k1_s0060
                                 - 0.394999312873438 z_k1_s0061
                                 - 0.397000310912322 z_k1_s0062
                                 - 0.191999901826858 z_k1_s0063
                                 - 0.247999556943335 z_k1_s0064
                                 - 0.291000314799627 z_k1_s0065
                                 - 0.207999836145819 z_k1_s0066
                                 - 0.415000093271294 z_k1_s0067
                                 - 0.216999566443567 z_k1_s0070
                                 - 0.133999622817034 z_k1_s0071
                                 - 0.513998900079411 z_k1_s0072
                                 - 0.35399885489611 z_k1_s0073
                                 - 0.5419983333379 z_k1_s0074
                                 - 0.401000001503423 z_k1_s0075
                                 - 0.116999729753533 z_k1_s0076
                                 - 0.27499872084963 z_k1_s0077
                                 - 0.257999559763214 z_k1_s0100
                                 - 0.419999313593566 z_k1_s0101
                                 - 0.243999741666559 z_k1_s0102
                                 - 0.39699936460652 z_k1_s0103
                                 - 0.327999517645383 z_k1_s0104
                                 - 0.5979978062643 z_k1_s0105
                                 - 1.16199791300577 z_k1_s0106
                                 - 0.929999068692144 z_k1_s0107
                                 - 0.155000128014788 z_k1_s0110
                                 - 0.214000368743156 z_k1_s0111
                                 - 0.104000373894031 z_k1_s0112
                                 - 0.205999882135545 z_k1_s0113
                                 - 0.0700000418131169 z_k1_s0114
                                 - 0.102000500242333 z_k1_s0115
                                 - 0.0700000418131169 z_k1_s0116
                                 - 0.0710000778390479 z_k1_s0117
                                 - 0.184999783264471 z_k1_s0120
                                 - 0.414999980849607 z_k1_s0121
                                 - 0.236000591660918 z_k1_s0122
                                 - 0.236998793758567 z_k1_s0123
                                 - 0.278000300694152 z_k1_s0124
                                 - 0.236000591660918 z_k1_s0125
                                 - 0.262000605157507 z_k1_s0126
                                 - 0.183999751468476 z_k1_s0127
                                 - 0.0860000699178698 z_k1_s0130
                                 - 0.268000223051043 z_k1_s0131
                                 - 0.0840001561385022 z_k1_s0132
                                 - 0.0860000699178698 z_k1_s0133
                                 - 0.113999862730357 z_k1_s0134
                                 - 0.100000132974336 z_k1_s0135
                                 - 0.229000737801811 z_k1_s0136
                                 - 0.144000235923104 z_k1_s0137
                                 - 0.170999854317771 z_k1_s0140
                                 - 0.198999853607213 z_k1_s0141
                                 - 0.359000226460891 z_k1_s0142
                                 - 0.364999284023612 z_k1_s0143
                                 - 0.832998329862132 z_k1_s0144
                                 - 0.729998568047223 z_k1_s0145
                                 - 0.424000460382637 z_k1_s0146
                                 - 0.216999994493179 z_k1_s0147
                                 - 0.702000498756939 z_k1_s0150
                                 - 0.531999622649482 z_k1_s0151
                                 - 0.63200106906696 z_k1_s0152
                                 - 0.484001144292249 z_k1_s0153
                                 - 0.403001238935896 z_k1_s0154
                                 - 0.29400037460186 z_k1_s0155
                                 - 0.678001259578323 z_k1_s0156
                                 - 0.232000427504377 z_k1_s0157
                                 - 0.193000686752287 z_k1_s0160
                                 - 0.249999418922234 z_k1_s0161
                                 - 0.0730000239093473 z_k1_s0162
                                 - 0.0760000535604245 z_k1_s0163
                                 - 0.182999715915115 z_k1_s0164
                                 - 0.186000268766341 z_k1_s0165
                                 - 0.157000592539435 z_k1_s0166
                                 - 0.190000133901061 z_k1_s0167
                                 - 0.365999288864753 z_k1_s0170
                                 - 0.297998130959061 z_k1_s0171
                                 - 0.326998725889147 z_k1_s0172
                                 - 0.468999188932969 z_k1_s0173
                                 - 0.164999472751081 z_k1_s0174
                                 - 0.22899926824241 z_k1_s0175
                                 - 0.202999351324057 z_k1_s0176
                                 - 0.235000395066518 z_k1_s0177
                                 - 0.279999878876104 z_k1_s1000
                                 - 0.255999957458154 z_k1_s1001
                                 - 0.308999883381129 z_k1_s1002
                                 - 0.31999986157269 z_k1_s1003
                                 - 0.461001318028184 z_k1_s1004
                                 - 0.865001075063704 z_k1_s1005
                                 - 0.40499888705292 z_k1_s1006
                                 - 0.51399953895113 z_k1_s1007
                                 - 0.262000176778871 z_k1_s1010
                                 - 0.316000383936494 z_k1_s1011
                                 - 0.825000754128293 z_k1_s1012
                                 - 0.205000106468174 z_k1_s1013
                                 - 0.768999080467294 z_k1_s1014
                                 - 0.391000300766209 z_k1_s1015
                                 - 0.302000132421386 z_k1_s1016
                                 - 0.268000236885305 z_k1_s1017
                                 - 1.48700066015428 z_k1_s1020
                                 - 0.816000078961564 z_k1_s1021
                                 - 1.5229967629893 z_k1_s1022
                                 - 1.21500155163342 z_k1_s1023
                                 - 1.16199862101836 z_k1_s1024
                                 - 1.03200147656265 z_k1_s1025
                                 - 1.18699948387471 z_k1_s1026
                                 - 0.755999843705651 z_k1_s1027
                                 - 1.36200121786646 z_k1_s1030
                                 - 0.310000710598891 z_k1_s1031
                                 - 0.619004232082946 z_k1_s1032
                                 - 0.385001235144674 z_k1_s1033
                                 - 1.35800517378873 z_k1_s1034
                                 - 0.776003824024318 z_k1_s1035
                                 - 0.92300115974074 z_k1_s1036
                                 - 0.340000798956898 z_k1_s1037
                                 - 0.754001650735102 z_k1_s1040
                                 - 0.315000330314807 z_k1_s1041
                                 - 0.282000117867415 z_k1_s1042
                                 - 0.45100016044574 z_k1_s1043
                                 - 0.316000154525205 z_k1_s1044
                                 - 0.305000189209482 z_k1_s1045
                                 - 0.231000221130844 z_k1_s1046
                                 - 0.510999108076339 z_k1_s1047
                                 - 0.646999542579548 z_k1_s1050
                                 - 0.537002714799693 z_k1_s1051
                                 - 0.477001509456159 z_k1_s1052
                                 - 0.251000529739845 z_k1_s1053
                                 - 0.808002983918671 z_k1_s1054
                                 - 0.938001041482261 z_k1_s1055
                                 - 0.771001205620294 z_k1_s1056
                                 - 0.538999856968859 z_k1_s1057
                                 - 0.44099950049767 z_k1_s1060
                                 - 0.365001008294817 z_k1_s1061
                                 - 0.396999194905859 z_k1_s1062
                                 - 0.23400017480106 z_k1_s1063
                                 - 0.462999653293575 z_k1_s1064
                                 - 0.281000170437575 z_k1_s1065
                                 - 0.280000926856387 z_k1_s1066
                                 - 0.337999039589715 z_k1_s1067
                                 - 0.658001438108478 z_k1_s1070
                                 - 0.775002185115831 z_k1_s1071
                                 - 0.595999742018523 z_k1_s1072
                                 - 1.0859994980136 z_k1_s1073
                                 - 0.441000255795787 z_k1_s1074
                                 - 0.319999966776436 z_k1_s1075
                                 - 0.638001351047411 z_k1_s1076
                                 - 0.240000093932381 z_k1_s1077
                                 - 0.788997577423961 z_k1_s1100
                                 - 0.363999173165769 z_k1_s1101
                                 - 0.423997840569484 z_k1_s1102
                                 - 0.546998756438672 z_k1_s1103
                                 - 0.25899933171429 z_k1_s1104
                                 - 0.222999450873633 z_k1_s1105
                                 - 0.932996344788335 z_k1_s1106
                                 - 0.240999391293961 z_k1_s1107
                                 - 0.219000281194299 z_k1_s1110
                                 - 0.109999978545099 z_k1_s1111
                                 - 0.0549999892725494 z_k1_s1112
                                 - 0.169999874296981 z_k1_s1113
                                 - 0.0609999788477376 z_k1_s1114
                                 - 0.188000148422513 z_k1_s1115
                                 - 0.138999741525194 z_k1_s1116
                                 - 0.0760000036857027 z_k1_s1117
                                 - 0.218999856448704 z_k1_s1120
                                 - 0.33499912135773 z_k1_s1121
                                 - 0.49900032569076 z_k1_s1122
                                 - 0.167999914985302 z_k1_s1123
                                 - 0.52900096508087 z_k1_s1124
                                 - 0.228000048265737 z_k1_s1125
                                 - 0.122000080086919 z_k1_s1126
                                 - 0.190999946984472 z_k1_s1127
                                 - 0.179000255684273 z_k1_s1130
                                 - 0.102999549929964 z_k1_s1131
                                 - 0.0789999992181809 z_k1_s1132
                                 - 0.105000340239407 z_k1_s1133
                                 - 0.172999776756235 z_k1_s1134
                                 - 0.080000015972844 z_k1_s1135
                                 - 0.0820000494821703 z_k1_s1136
                                 - 0.145000064425785 z_k1_s1137
                                 - 0.173000320127045 z_k1_s1140
                                 - 0.413000284413393 z_k1_s1141
                                 - 0.109000197890444 z_k1_s1142
                                 - 0.182000423097975 z_k1_s1143
                                 - 0.499000194755859 z_k1_s1144
                                 - 0.275001194330381 z_k1_s1145
                                 - 0.144000329932455 z_k1_s1146
                                 - 0.401001318320976 z_k1_s1147
                                 - 0.459999055450766 z_k1_s1150
                                 - 0.147999875788302 z_k1_s1151
                                 - 0.310999732006507 z_k1_s1152
                                 - 0.154999797775669 z_k1_s1153
                                 - 0.202999459631675 z_k1_s1154
                                 - 0.118999879183861 z_k1_s1155
                                 - 0.178999973103278 z_k1_s1156
                                 - 0.146999911532935 z_k1_s1157
                                 - 0.5399985223025 z_k1_s1160
                                 - 0.392999636430866 z_k1_s1161
                                 - 0.33900045860024 z_k1_s1162
                                 - 0.825997914514468 z_k1_s1163
                                 - 0.191999886729548 z_k1_s1164
                                 - 0.553999058480392 z_k1_s1165
                                 - 0.233000011822753 z_k1_s1166
                                 - 0.294999929583362 z_k1_s1167
                                 - 0.194000421607574 z_k1_s1170
                                 - 0.360000415817597 z_k1_s1171
                                 - 0.360000415817597 z_k1_s1172
                                 - 0.390000678386091 z_k1_s1173
                                 - 0.342999628861105 z_k1_s1174
                                 - 0.647000830886202 z_k1_s1175
                                 - 0.277001239213886 z_k1_s1176
                                 - 0.164000159039079 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.08778 w_k1_ADM_s000
                                 + 0.727093 w_k1_AXP_s000
                                 + 0.850865 w_k1_DIS_s000
                                 + 0.902642 w_k1_FDX_s000
                                 + 0.806483 w_k1_IBM_s000
                                 + 1.27575 w_k1_INTC_s000
                                 + 1.05053 w_k1_KO_s000
                                 + 0.570418 w_k1_MAT_s000
                                 + 1.00289 w_k1_MSFT_s000
                                 + 0.918662 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.26638 w_k1_ADM_s000
                                 + 0.914299 w_k1_AXP_s000
                                 + 0.818455 w_k1_DIS_s000
                                 + 1.02985 w_k1_FDX_s000
                                 + 0.654935 w_k1_IBM_s000
                                 + 1.03744 w_k1_INTC_s000
                                 + 0.764207 w_k1_KO_s000
                                 + 0.466012 w_k1_MAT_s000
                                 + 1.00209 w_k1_MSFT_s000
                                 + 0.868303 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.49638 w_k1_ADM_s000
                                 + 1.06431 w_k1_AXP_s000
                                 + 1.16258 w_k1_DIS_s000
                                 + 1.05527 w_k1_FDX_s000
                                 + 1.13051 w_k1_IBM_s000
                                 + 1.07232 w_k1_INTC_s000
                                 + 1.06789 w_k1_KO_s000 + 1.01068 w_k1_MAT_s000
                                 + 1.08885 w_k1_MSFT_s000
                                 + 1.05481 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.79395 w_k1_ADM_s000
                                 + 1.34782 w_k1_AXP_s000
                                 + 1.12376 w_k1_DIS_s000 + 1.2287 w_k1_FDX_s000
                                 + 0.839901 w_k1_IBM_s000
                                 + 0.935973 w_k1_INTC_s000
                                 + 1.1473 w_k1_KO_s000 + 0.821088 w_k1_MAT_s000
                                 + 1.07384 w_k1_MSFT_s000
                                 + 1.04097 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.19766 w_k1_ADM_s000
                                 + 1.13968 w_k1_AXP_s000 + 1.3298 w_k1_DIS_s000
                                 + 1.04552 w_k1_FDX_s000
                                 + 1.41734 w_k1_IBM_s000
                                 + 1.07964 w_k1_INTC_s000
                                 + 1.37421 w_k1_KO_s000 + 1.07133 w_k1_MAT_s000
                                 + 1.00604 w_k1_MSFT_s000
                                 + 1.12562 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.15897 w_k1_ADM_s000
                                 + 1.03478 w_k1_AXP_s000
                                 + 0.931939 w_k1_DIS_s000
                                 + 0.779322 w_k1_FDX_s000
                                 + 1.09209 w_k1_IBM_s000
                                 + 0.903087 w_k1_INTC_s000
                                 + 1.08754 w_k1_KO_s000
                                 + 0.819633 w_k1_MAT_s000
                                 + 0.894137 w_k1_MSFT_s000
                                 + 0.870635 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.32356 w_k1_ADM_s000
                                 + 1.67738 w_k1_AXP_s000
                                 + 1.45611 w_k1_DIS_s000
                                 + 1.27578 w_k1_FDX_s000 + 1.3828 w_k1_IBM_s000
                                 + 1.17258 w_k1_INTC_s000
                                 + 1.43451 w_k1_KO_s000 + 1.4991 w_k1_MAT_s000
                                 + 1.13782 w_k1_MSFT_s000
                                 + 1.11636 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.24328 w_k1_ADM_s000
                                 + 1.20817 w_k1_AXP_s000
                                 + 0.810442 w_k1_DIS_s000
                                 + 0.909311 w_k1_FDX_s000
                                 + 0.875092 w_k1_IBM_s000
                                 + 0.899585 w_k1_INTC_s000
                                 + 0.891558 w_k1_KO_s000
                                 + 0.94058 w_k1_MAT_s000
                                 + 0.938013 w_k1_MSFT_s000
                                 + 0.886584 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.713244 w_k1_ADM_s001
                                 + 0.828535 w_k1_AXP_s001
                                 + 0.813193 w_k1_DIS_s001
                                 + 0.784172 w_k1_FDX_s001
                                 + 1.02323 w_k1_IBM_s001
                                 + 0.832459 w_k1_INTC_s001
                                 + 0.982277 w_k1_KO_s001
                                 + 0.989893 w_k1_MAT_s001
                                 + 0.796798 w_k1_MSFT_s001
                                 + 0.852215 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.1123 w_k1_ADM_s001
                                 + 0.894862 w_k1_AXP_s001
                                 + 0.90319 w_k1_DIS_s001
                                 + 0.989995 w_k1_FDX_s001
                                 + 0.733005 w_k1_IBM_s001
                                 + 1.01653 w_k1_INTC_s001
                                 + 0.846393 w_k1_KO_s001
                                 + 0.605295 w_k1_MAT_s001
                                 + 0.940874 w_k1_MSFT_s001
                                 + 0.888589 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.29099 w_k1_ADM_s001
                                 + 1.87941 w_k1_AXP_s001
                                 + 1.79874 w_k1_DIS_s001
                                 + 1.37578 w_k1_FDX_s001
                                 + 1.51342 w_k1_IBM_s001
                                 + 1.5146 w_k1_INTC_s001 + 1.76275 w_k1_KO_s001
                                 + 1.54654 w_k1_MAT_s001
                                 + 1.45043 w_k1_MSFT_s001
                                 + 1.32656 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.822585 w_k1_ADM_s001
                                 + 0.775046 w_k1_AXP_s001
                                 + 0.665845 w_k1_DIS_s001
                                 + 0.758278 w_k1_FDX_s001
                                 + 0.741714 w_k1_IBM_s001
                                 + 0.793346 w_k1_INTC_s001
                                 + 0.834219 w_k1_KO_s001
                                 + 0.771278 w_k1_MAT_s001
                                 + 0.814245 w_k1_MSFT_s001
                                 + 0.879776 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.00413 w_k1_ADM_s001
                                 + 1.18473 w_k1_AXP_s001
                                 + 1.25774 w_k1_DIS_s001
                                 + 0.916781 w_k1_FDX_s001
                                 + 1.56955 w_k1_IBM_s001
                                 + 1.04451 w_k1_INTC_s001
                                 + 1.34915 w_k1_KO_s001 + 1.14731 w_k1_MAT_s001
                                 + 0.958312 w_k1_MSFT_s001
                                 + 1.02595 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.36878 w_k1_ADM_s001
                                 + 1.49268 w_k1_AXP_s001
                                 + 1.45142 w_k1_DIS_s001
                                 + 1.37828 w_k1_FDX_s001
                                 + 0.952655 w_k1_IBM_s001
                                 + 1.45472 w_k1_INTC_s001
                                 + 1.20969 w_k1_KO_s001
                                 + 0.939234 w_k1_MAT_s001
                                 + 1.34341 w_k1_MSFT_s001
                                 + 1.11448 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.90643 w_k1_ADM_s001
                                 + 1.1113 w_k1_AXP_s001 + 1.33185 w_k1_DIS_s001
                                 + 1.01646 w_k1_FDX_s001
                                 + 1.15882 w_k1_IBM_s001
                                 + 1.33333 w_k1_INTC_s001
                                 + 1.28248 w_k1_KO_s001
                                 + 0.904598 w_k1_MAT_s001
                                 + 1.29651 w_k1_MSFT_s001
                                 + 1.08627 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.10116 w_k1_ADM_s001
                                 + 1.25459 w_k1_AXP_s001
                                 + 1.09277 w_k1_DIS_s001
                                 + 1.13343 w_k1_FDX_s001
                                 + 0.966147 w_k1_IBM_s001
                                 + 1.14575 w_k1_INTC_s001
                                 + 1.06677 w_k1_KO_s001 + 1.1112 w_k1_MAT_s001
                                 + 1.07337 w_k1_MSFT_s001
                                 + 1.05548 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.914005 w_k1_ADM_s002
                                 + 1.32275 w_k1_AXP_s002
                                 + 1.19127 w_k1_DIS_s002 + 1.0371 w_k1_FDX_s002
                                 + 1.38977 w_k1_IBM_s002
                                 + 1.13791 w_k1_INTC_s002
                                 + 1.14742 w_k1_KO_s002 + 1.31018 w_k1_MAT_s002
                                 + 0.986632 w_k1_MSFT_s002
                                 + 1.03858 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.691506 w_k1_ADM_s002
                                 + 0.852329 w_k1_AXP_s002
                                 + 0.708936 w_k1_DIS_s002
                                 + 0.822817 w_k1_FDX_s002
                                 + 0.751208 w_k1_IBM_s002
                                 + 0.81578 w_k1_INTC_s002
                                 + 0.746061 w_k1_KO_s002
                                 + 0.823138 w_k1_MAT_s002
                                 + 0.825661 w_k1_MSFT_s002
                                 + 0.82927 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.924966 w_k1_ADM_s002
                                 + 0.793728 w_k1_AXP_s002
                                 + 0.914394 w_k1_DIS_s002
                                 + 0.851834 w_k1_FDX_s002
                                 + 1.13291 w_k1_IBM_s002
                                 + 0.933119 w_k1_INTC_s002
                                 + 0.928842 w_k1_KO_s002
                                 + 0.874176 w_k1_MAT_s002
                                 + 1.01805 w_k1_MSFT_s002
                                 + 0.935379 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.968798 w_k1_ADM_s002
                                 + 1.28811 w_k1_AXP_s002
                                 + 0.819835 w_k1_DIS_s002
                                 + 1.00536 w_k1_FDX_s002
                                 + 0.912346 w_k1_IBM_s002
                                 + 1.08573 w_k1_INTC_s002
                                 + 0.866572 w_k1_KO_s002
                                 + 0.960733 w_k1_MAT_s002
                                 + 0.920382 w_k1_MSFT_s002
                                 + 0.881741 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.10685 w_k1_ADM_s002
                                 + 1.3375 w_k1_AXP_s002 + 1.81699 w_k1_DIS_s002
                                 + 1.39526 w_k1_FDX_s002
                                 + 1.34367 w_k1_IBM_s002
                                 + 1.3687 w_k1_INTC_s002 + 1.49594 w_k1_KO_s002
                                 + 1.59953 w_k1_MAT_s002
                                 + 1.48287 w_k1_MSFT_s002
                                 + 1.23916 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.99037 w_k1_ADM_s002
                                 + 0.961156 w_k1_AXP_s002
                                 + 1.07103 w_k1_DIS_s002
                                 + 1.07131 w_k1_FDX_s002
                                 + 0.859079 w_k1_IBM_s002
                                 + 1.18581 w_k1_INTC_s002
                                 + 0.956136 w_k1_KO_s002
                                 + 0.691903 w_k1_MAT_s002
                                 + 1.10172 w_k1_MSFT_s002
                                 + 0.979951 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.989561 w_k1_ADM_s002
                                 + 0.926077 w_k1_AXP_s002
                                 + 1.39413 w_k1_DIS_s002
                                 + 1.14506 w_k1_FDX_s002
                                 + 0.930697 w_k1_IBM_s002
                                 + 1.21854 w_k1_INTC_s002
                                 + 1.12131 w_k1_KO_s002
                                 + 0.935487 w_k1_MAT_s002
                                 + 1.21788 w_k1_MSFT_s002
                                 + 1.04107 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.33027 w_k1_ADM_s002
                                 + 1.63475 w_k1_AXP_s002
                                 + 1.39371 w_k1_DIS_s002
                                 + 1.35287 w_k1_FDX_s002
                                 + 1.08569 w_k1_IBM_s002
                                 + 1.51969 w_k1_INTC_s002
                                 + 1.13697 w_k1_KO_s002
                                 + 0.980949 w_k1_MAT_s002
                                 + 1.31418 w_k1_MSFT_s002
                                 + 1.07693 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.938994 w_k1_ADM_s003
                                 + 1.03935 w_k1_AXP_s003
                                 + 1.41084 w_k1_DIS_s003
                                 + 1.09149 w_k1_FDX_s003
                                 + 1.36253 w_k1_IBM_s003
                                 + 1.28822 w_k1_INTC_s003 + 1.282 w_k1_KO_s003
                                 + 1.0673 w_k1_MAT_s003 + 1.1143 w_k1_MSFT_s003
                                 + 1.2612 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.838535 w_k1_ADM_s003
                                 + 0.673456 w_k1_AXP_s003
                                 + 0.720299 w_k1_DIS_s003
                                 + 0.829102 w_k1_FDX_s003
                                 + 0.880386 w_k1_IBM_s003
                                 + 1.00455 w_k1_INTC_s003
                                 + 0.809688 w_k1_KO_s003
                                 + 0.86177 w_k1_MAT_s003
                                 + 0.833857 w_k1_MSFT_s003
                                 + 1.01729 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.634737 w_k1_ADM_s003
                                 + 0.775086 w_k1_AXP_s003
                                 + 0.618414 w_k1_DIS_s003
                                 + 0.707586 w_k1_FDX_s003
                                 + 1.12453 w_k1_IBM_s003
                                 + 0.768764 w_k1_INTC_s003
                                 + 0.824994 w_k1_KO_s003
                                 + 0.905435 w_k1_MAT_s003
                                 + 0.657961 w_k1_MSFT_s003
                                 + 0.989238 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.782314 w_k1_ADM_s003
                                 + 0.974823 w_k1_AXP_s003
                                 + 0.925198 w_k1_DIS_s003
                                 + 1.03145 w_k1_FDX_s003
                                 + 0.963117 w_k1_IBM_s003
                                 + 1.25381 w_k1_INTC_s003
                                 + 1.02596 w_k1_KO_s003 + 1.00465 w_k1_MAT_s003
                                 + 1.01641 w_k1_MSFT_s003 + 1.0821 w_k1_PG_s003
                                 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.02065 w_k1_ADM_s003
                                 + 1.1536 w_k1_AXP_s003 + 1.0062 w_k1_DIS_s003
                                 + 1.04278 w_k1_FDX_s003
                                 + 0.972621 w_k1_IBM_s003
                                 + 1.01698 w_k1_INTC_s003
                                 + 0.943564 w_k1_KO_s003
                                 + 0.98101 w_k1_MAT_s003
                                 + 0.946014 w_k1_MSFT_s003
                                 + 1.13095 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.976435 w_k1_ADM_s003
                                 + 1.14911 w_k1_AXP_s003
                                 + 0.970485 w_k1_DIS_s003
                                 + 0.891679 w_k1_FDX_s003
                                 + 1.05417 w_k1_IBM_s003
                                 + 0.965276 w_k1_INTC_s003
                                 + 1.03511 w_k1_KO_s003 + 1.13088 w_k1_MAT_s003
                                 + 0.891021 w_k1_MSFT_s003
                                 + 1.18682 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.908395 w_k1_ADM_s003
                                 + 0.714871 w_k1_AXP_s003
                                 + 1.10218 w_k1_DIS_s003
                                 + 0.999382 w_k1_FDX_s003
                                 + 0.691348 w_k1_IBM_s003
                                 + 1.18567 w_k1_INTC_s003
                                 + 0.968519 w_k1_KO_s003
                                 + 0.55851 w_k1_MAT_s003
                                 + 1.13003 w_k1_MSFT_s003
                                 + 1.05182 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.19315 w_k1_ADM_s003
                                 + 1.41395 w_k1_AXP_s003
                                 + 1.57976 w_k1_DIS_s003
                                 + 1.37034 w_k1_FDX_s003
                                 + 1.24597 w_k1_IBM_s003
                                 + 1.20937 w_k1_INTC_s003
                                 + 1.24288 w_k1_KO_s003 + 1.4054 w_k1_MAT_s003
                                 + 1.15037 w_k1_MSFT_s003 + 1.4495 w_k1_PG_s003
                                 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.06246 w_k1_ADM_s004
                                 + 0.806582 w_k1_AXP_s004
                                 + 0.958788 w_k1_DIS_s004
                                 + 0.984931 w_k1_FDX_s004
                                 + 0.63036 w_k1_IBM_s004
                                 + 0.80856 w_k1_INTC_s004
                                 + 0.856453 w_k1_KO_s004
                                 + 0.649898 w_k1_MAT_s004
                                 + 0.939556 w_k1_MSFT_s004
                                 + 0.882182 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.07582 w_k1_ADM_s004
                                 + 1.27452 w_k1_AXP_s004
                                 + 1.14638 w_k1_DIS_s004
                                 + 0.979073 w_k1_FDX_s004
                                 + 1.56188 w_k1_IBM_s004
                                 + 0.90025 w_k1_INTC_s004
                                 + 1.32536 w_k1_KO_s004 + 1.67408 w_k1_MAT_s004
                                 + 0.997502 w_k1_MSFT_s004
                                 + 1.11218 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.37402 w_k1_ADM_s004
                                 + 1.4287 w_k1_AXP_s004 + 1.62836 w_k1_DIS_s004
                                 + 1.21675 w_k1_FDX_s004
                                 + 1.42375 w_k1_IBM_s004
                                 + 1.0519 w_k1_INTC_s004 + 1.57765 w_k1_KO_s004
                                 + 1.4988 w_k1_MAT_s004
                                 + 1.39707 w_k1_MSFT_s004
                                 + 1.08589 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.887887 w_k1_ADM_s004
                                 + 0.760194 w_k1_AXP_s004
                                 + 0.688433 w_k1_DIS_s004
                                 + 0.743815 w_k1_FDX_s004
                                 + 0.824282 w_k1_IBM_s004
                                 + 0.779443 w_k1_INTC_s004
                                 + 0.753078 w_k1_KO_s004
                                 + 0.84009 w_k1_MAT_s004
                                 + 0.780281 w_k1_MSFT_s004
                                 + 0.805695 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.02572 w_k1_ADM_s004
                                 + 0.949473 w_k1_AXP_s004
                                 + 1.25568 w_k1_DIS_s004
                                 + 1.07703 w_k1_FDX_s004
                                 + 0.975903 w_k1_IBM_s004
                                 + 1.13028 w_k1_INTC_s004
                                 + 1.12046 w_k1_KO_s004
                                 + 0.726771 w_k1_MAT_s004
                                 + 1.10145 w_k1_MSFT_s004
                                 + 0.958942 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.17886 w_k1_ADM_s004
                                 + 1.22995 w_k1_AXP_s004
                                 + 1.49312 w_k1_DIS_s004
                                 + 1.39738 w_k1_FDX_s004
                                 + 0.831548 w_k1_IBM_s004
                                 + 1.33352 w_k1_INTC_s004
                                 + 0.955578 w_k1_KO_s004
                                 + 0.741882 w_k1_MAT_s004
                                 + 1.44812 w_k1_MSFT_s004
                                 + 0.974821 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.962792 w_k1_ADM_s004
                                 + 0.956747 w_k1_AXP_s004
                                 + 0.850498 w_k1_DIS_s004
                                 + 0.874612 w_k1_FDX_s004
                                 + 0.970897 w_k1_IBM_s004
                                 + 1.04702 w_k1_INTC_s004
                                 + 0.954092 w_k1_KO_s004
                                 + 1.06527 w_k1_MAT_s004
                                 + 0.984621 w_k1_MSFT_s004
                                 + 0.989551 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.19728 w_k1_ADM_s004
                                 + 1.26138 w_k1_AXP_s004
                                 + 1.00463 w_k1_DIS_s004
                                 + 1.13533 w_k1_FDX_s004
                                 + 0.986035 w_k1_IBM_s004
                                 + 1.0202 w_k1_INTC_s004
                                 + 0.993346 w_k1_KO_s004
                                 + 1.32571 w_k1_MAT_s004
                                 + 1.10453 w_k1_MSFT_s004
                                 + 0.977192 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.08563 w_k1_ADM_s005
                                 + 1.00289 w_k1_AXP_s005
                                 + 1.31346 w_k1_DIS_s005
                                 + 1.30315 w_k1_FDX_s005
                                 + 0.723071 w_k1_IBM_s005
                                 + 1.0309 w_k1_INTC_s005
                                 + 0.995162 w_k1_KO_s005
                                 + 0.673713 w_k1_MAT_s005
                                 + 1.00663 w_k1_MSFT_s005
                                 + 1.05051 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.15697 w_k1_ADM_s005
                                 + 1.47086 w_k1_AXP_s005
                                 + 1.61466 w_k1_DIS_s005
                                 + 1.41061 w_k1_FDX_s005
                                 + 1.05067 w_k1_IBM_s005
                                 + 1.14422 w_k1_INTC_s005
                                 + 1.11496 w_k1_KO_s005 + 1.24478 w_k1_MAT_s005
                                 + 1.09159 w_k1_MSFT_s005
                                 + 1.07362 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.957183 w_k1_ADM_s005
                                 + 1.39444 w_k1_AXP_s005
                                 + 1.38611 w_k1_DIS_s005 + 1.1964 w_k1_FDX_s005
                                 + 1.59951 w_k1_IBM_s005
                                 + 1.06709 w_k1_INTC_s005
                                 + 1.43496 w_k1_KO_s005 + 2.34144 w_k1_MAT_s005
                                 + 1.13808 w_k1_MSFT_s005
                                 + 1.10039 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.755956 w_k1_ADM_s005
                                 + 0.912811 w_k1_AXP_s005
                                 + 0.77408 w_k1_DIS_s005
                                 + 0.991809 w_k1_FDX_s005
                                 + 0.783098 w_k1_IBM_s005
                                 + 1.03742 w_k1_INTC_s005
                                 + 0.766349 w_k1_KO_s005
                                 + 0.87799 w_k1_MAT_s005
                                 + 0.855955 w_k1_MSFT_s005
                                 + 0.851571 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.778602 w_k1_ADM_s005
                                 + 0.716237 w_k1_AXP_s005
                                 + 0.780013 w_k1_DIS_s005
                                 + 0.866888 w_k1_FDX_s005
                                 + 0.653728 w_k1_IBM_s005
                                 + 0.770134 w_k1_INTC_s005
                                 + 0.705472 w_k1_KO_s005
                                 + 0.799346 w_k1_MAT_s005
                                 + 0.741242 w_k1_MSFT_s005
                                 + 0.838227 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.600959 w_k1_ADM_s005
                                 + 0.491433 w_k1_AXP_s005
                                 + 0.55808 w_k1_DIS_s005
                                 + 0.706683 w_k1_FDX_s005
                                 + 0.674207 w_k1_IBM_s005
                                 + 0.712782 w_k1_INTC_s005
                                 + 0.655469 w_k1_KO_s005
                                 + 0.485951 w_k1_MAT_s005
                                 + 0.639202 w_k1_MSFT_s005
                                 + 0.78415 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.819778 w_k1_ADM_s005
                                 + 0.798275 w_k1_AXP_s005
                                 + 0.801961 w_k1_DIS_s005
                                 + 0.803856 w_k1_FDX_s005
                                 + 0.986352 w_k1_IBM_s005
                                 + 0.841482 w_k1_INTC_s005
                                 + 0.924063 w_k1_KO_s005
                                 + 0.904057 w_k1_MAT_s005
                                 + 0.821905 w_k1_MSFT_s005
                                 + 0.887288 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.85933 w_k1_ADM_s005
                                 + 1.0595 w_k1_AXP_s005 + 1.14439 w_k1_DIS_s005
                                 + 1.06262 w_k1_FDX_s005
                                 + 1.13869 w_k1_IBM_s005
                                 + 0.971621 w_k1_INTC_s005
                                 + 1.06252 w_k1_KO_s005 + 1.34048 w_k1_MAT_s005
                                 + 0.921398 w_k1_MSFT_s005
                                 + 0.939815 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.09943 w_k1_ADM_s006
                                 + 1.48336 w_k1_AXP_s006
                                 + 1.13329 w_k1_DIS_s006
                                 + 1.24711 w_k1_FDX_s006
                                 + 0.949323 w_k1_IBM_s006
                                 + 1.50466 w_k1_INTC_s006
                                 + 1.01913 w_k1_KO_s006 + 1.12531 w_k1_MAT_s006
                                 + 1.06312 w_k1_MSFT_s006
                                 + 1.02034 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.26003 w_k1_ADM_s006
                                 + 1.30093 w_k1_AXP_s006
                                 + 1.79226 w_k1_DIS_s006
                                 + 1.37873 w_k1_FDX_s006
                                 + 1.29647 w_k1_IBM_s006
                                 + 1.63552 w_k1_INTC_s006
                                 + 1.56281 w_k1_KO_s006 + 1.0136 w_k1_MAT_s006
                                 + 1.37957 w_k1_MSFT_s006
                                 + 1.31455 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.30344 w_k1_ADM_s006
                                 + 0.77689 w_k1_AXP_s006
                                 + 1.07592 w_k1_DIS_s006 + 1.1102 w_k1_FDX_s006
                                 + 0.780597 w_k1_IBM_s006
                                 + 1.29694 w_k1_INTC_s006
                                 + 1.05566 w_k1_KO_s006
                                 + 0.579109 w_k1_MAT_s006
                                 + 1.17346 w_k1_MSFT_s006
                                 + 1.00169 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.02674 w_k1_ADM_s006
                                 + 0.736838 w_k1_AXP_s006
                                 + 1.09139 w_k1_DIS_s006
                                 + 0.918519 w_k1_FDX_s006
                                 + 1.1589 w_k1_IBM_s006
                                 + 1.32856 w_k1_INTC_s006
                                 + 1.16397 w_k1_KO_s006
                                 + 0.850499 w_k1_MAT_s006
                                 + 1.09734 w_k1_MSFT_s006
                                 + 1.02187 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.24041 w_k1_ADM_s006
                                 + 1.0284 w_k1_AXP_s006
                                 + 0.705786 w_k1_DIS_s006
                                 + 0.793645 w_k1_FDX_s006
                                 + 1.07112 w_k1_IBM_s006
                                 + 1.21157 w_k1_INTC_s006
                                 + 0.980964 w_k1_KO_s006 + 1.1525 w_k1_MAT_s006
                                 + 0.827745 w_k1_MSFT_s006
                                 + 1.03074 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.03092 w_k1_ADM_s006
                                 + 0.965298 w_k1_AXP_s006
                                 + 0.902706 w_k1_DIS_s006
                                 + 0.821333 w_k1_FDX_s006
                                 + 1.5267 w_k1_IBM_s006
                                 + 1.19606 w_k1_INTC_s006
                                 + 1.09948 w_k1_KO_s006 + 1.27137 w_k1_MAT_s006
                                 + 0.886903 w_k1_MSFT_s006
                                 + 0.934327 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.02226 w_k1_ADM_s006
                                 + 1.1199 w_k1_AXP_s006 + 1.07151 w_k1_DIS_s006
                                 + 1.10971 w_k1_FDX_s006
                                 + 1.32625 w_k1_IBM_s006
                                 + 1.69403 w_k1_INTC_s006
                                 + 1.04331 w_k1_KO_s006 + 1.11731 w_k1_MAT_s006
                                 + 1.04155 w_k1_MSFT_s006
                                 + 0.993835 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.04498 w_k1_ADM_s006
                                 + 0.73522 w_k1_AXP_s006
                                 + 0.774501 w_k1_DIS_s006
                                 + 0.845246 w_k1_FDX_s006
                                 + 0.737286 w_k1_IBM_s006
                                 + 1.29298 w_k1_INTC_s006
                                 + 0.720515 w_k1_KO_s006
                                 + 0.505402 w_k1_MAT_s006
                                 + 0.862876 w_k1_MSFT_s006
                                 + 0.841329 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.926079 w_k1_ADM_s007
                                 + 1.06531 w_k1_AXP_s007
                                 + 0.973667 w_k1_DIS_s007
                                 + 1.05504 w_k1_FDX_s007
                                 + 0.859578 w_k1_IBM_s007
                                 + 1.30388 w_k1_INTC_s007
                                 + 0.982937 w_k1_KO_s007
                                 + 0.90548 w_k1_MAT_s007
                                 + 1.00527 w_k1_MSFT_s007
                                 + 0.995879 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.46038 w_k1_ADM_s007
                                 + 1.35826 w_k1_AXP_s007 + 1.2183 w_k1_DIS_s007
                                 + 1.23515 w_k1_FDX_s007
                                 + 0.782339 w_k1_IBM_s007
                                 + 0.928527 w_k1_INTC_s007
                                 + 0.920563 w_k1_KO_s007
                                 + 1.03705 w_k1_MAT_s007
                                 + 1.13405 w_k1_MSFT_s007
                                 + 1.10522 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.788736 w_k1_ADM_s007
                                 + 0.72304 w_k1_AXP_s007
                                 + 0.732471 w_k1_DIS_s007
                                 + 0.74706 w_k1_FDX_s007
                                 + 0.948241 w_k1_IBM_s007
                                 + 0.944884 w_k1_INTC_s007
                                 + 0.888249 w_k1_KO_s007
                                 + 0.689352 w_k1_MAT_s007
                                 + 0.812971 w_k1_MSFT_s007
                                 + 0.83738 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.20267 w_k1_ADM_s007
                                 + 1.4536 w_k1_AXP_s007 + 1.57573 w_k1_DIS_s007
                                 + 1.25964 w_k1_FDX_s007
                                 + 1.41615 w_k1_IBM_s007
                                 + 1.39119 w_k1_INTC_s007
                                 + 1.38517 w_k1_KO_s007 + 1.5081 w_k1_MAT_s007
                                 + 1.34402 w_k1_MSFT_s007
                                 + 1.27533 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.955536 w_k1_ADM_s007
                                 + 0.835074 w_k1_AXP_s007
                                 + 0.814449 w_k1_DIS_s007
                                 + 0.880185 w_k1_FDX_s007
                                 + 0.718226 w_k1_IBM_s007
                                 + 0.853479 w_k1_INTC_s007
                                 + 0.666427 w_k1_KO_s007
                                 + 0.715326 w_k1_MAT_s007
                                 + 0.926746 w_k1_MSFT_s007
                                 + 0.82577 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.910752 w_k1_ADM_s007
                                 + 1.42473 w_k1_AXP_s007
                                 + 1.00954 w_k1_DIS_s007
                                 + 0.995745 w_k1_FDX_s007
                                 + 1.34788 w_k1_IBM_s007
                                 + 1.31952 w_k1_INTC_s007
                                 + 1.09194 w_k1_KO_s007 + 1.73063 w_k1_MAT_s007
                                 + 1.0495 w_k1_MSFT_s007 + 1.10764 w_k1_PG_s007
                                 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.963449 w_k1_ADM_s007
                                 + 1.1439 w_k1_AXP_s007 + 1.36385 w_k1_DIS_s007
                                 + 1.22777 w_k1_FDX_s007
                                 + 0.835121 w_k1_IBM_s007
                                 + 1.45828 w_k1_INTC_s007
                                 + 1.07971 w_k1_KO_s007
                                 + 0.723556 w_k1_MAT_s007
                                 + 1.42718 w_k1_MSFT_s007
                                 + 1.11814 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.982698 w_k1_ADM_s007
                                 + 0.984285 w_k1_AXP_s007
                                 + 1.19243 w_k1_DIS_s007
                                 + 0.954516 w_k1_FDX_s007
                                 + 0.894983 w_k1_IBM_s007
                                 + 0.960368 w_k1_INTC_s007
                                 + 1.06096 w_k1_KO_s007
                                 + 0.717804 w_k1_MAT_s007
                                 + 1.06949 w_k1_MSFT_s007
                                 + 0.899647 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.23059 w_k1_ADM_s010
                                 + 1.48378 w_k1_AXP_s010
                                 + 1.33177 w_k1_DIS_s010
                                 + 1.09745 w_k1_FDX_s010 + 1.5362 w_k1_IBM_s010
                                 + 0.931649 w_k1_INTC_s010
                                 + 1.13944 w_k1_KO_s010 + 1.49446 w_k1_MAT_s010
                                 + 0.957307 w_k1_MSFT_s010
                                 + 1.13233 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.762567 w_k1_ADM_s010
                                 + 0.787443 w_k1_AXP_s010
                                 + 0.797591 w_k1_DIS_s010
                                 + 0.762105 w_k1_FDX_s010
                                 + 1.23813 w_k1_IBM_s010
                                 + 0.660482 w_k1_INTC_s010
                                 + 0.827473 w_k1_KO_s010
                                 + 1.15274 w_k1_MAT_s010
                                 + 0.67273 w_k1_MSFT_s010
                                 + 0.864797 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.35244 w_k1_ADM_s010
                                 + 1.35515 w_k1_AXP_s010
                                 + 1.76569 w_k1_DIS_s010
                                 + 1.41349 w_k1_FDX_s010
                                 + 1.17334 w_k1_IBM_s010
                                 + 1.13818 w_k1_INTC_s010
                                 + 1.19158 w_k1_KO_s010 + 1.15138 w_k1_MAT_s010
                                 + 1.39849 w_k1_MSFT_s010
                                 + 1.18228 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.885515 w_k1_ADM_s010
                                 + 0.867638 w_k1_AXP_s010
                                 + 1.13107 w_k1_DIS_s010
                                 + 0.953682 w_k1_FDX_s010
                                 + 1.31956 w_k1_IBM_s010
                                 + 0.938232 w_k1_INTC_s010
                                 + 1.00088 w_k1_KO_s010 + 1.1663 w_k1_MAT_s010
                                 + 0.945171 w_k1_MSFT_s010
                                 + 0.974109 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.33129 w_k1_ADM_s010
                                 + 1.30114 w_k1_AXP_s010
                                 + 1.68434 w_k1_DIS_s010
                                 + 1.23888 w_k1_FDX_s010 + 1.4137 w_k1_IBM_s010
                                 + 1.02514 w_k1_INTC_s010
                                 + 1.27081 w_k1_KO_s010 + 1.3632 w_k1_MAT_s010
                                 + 1.27074 w_k1_MSFT_s010
                                 + 1.15626 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.24512 w_k1_ADM_s010
                                 + 1.25919 w_k1_AXP_s010
                                 + 1.26794 w_k1_DIS_s010
                                 + 1.19683 w_k1_FDX_s010
                                 + 0.979294 w_k1_IBM_s010
                                 + 1.08634 w_k1_INTC_s010
                                 + 0.98849 w_k1_KO_s010 + 1.05869 w_k1_MAT_s010
                                 + 1.06796 w_k1_MSFT_s010
                                 + 1.01723 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.78934 w_k1_ADM_s010
                                 + 0.696475 w_k1_AXP_s010
                                 + 0.695572 w_k1_DIS_s010
                                 + 0.756659 w_k1_FDX_s010
                                 + 0.84844 w_k1_IBM_s010
                                 + 0.72995 w_k1_INTC_s010
                                 + 0.728449 w_k1_KO_s010
                                 + 0.649183 w_k1_MAT_s010
                                 + 0.729969 w_k1_MSFT_s010
                                 + 0.783774 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.18376 w_k1_ADM_s010
                                 + 0.845575 w_k1_AXP_s010
                                 + 1.0096 w_k1_DIS_s010 + 1.06522 w_k1_FDX_s010
                                 + 0.847767 w_k1_IBM_s010
                                 + 0.810246 w_k1_INTC_s010
                                 + 0.870621 w_k1_KO_s010
                                 + 0.634242 w_k1_MAT_s010
                                 + 0.93207 w_k1_MSFT_s010
                                 + 0.898367 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.866507 w_k1_ADM_s011
                                 + 0.927396 w_k1_AXP_s011
                                 + 0.949894 w_k1_DIS_s011
                                 + 0.818524 w_k1_FDX_s011
                                 + 1.1662 w_k1_IBM_s011
                                 + 1.12097 w_k1_INTC_s011
                                 + 1.01226 w_k1_KO_s011
                                 + 0.718909 w_k1_MAT_s011
                                 + 0.996837 w_k1_MSFT_s011
                                 + 0.974463 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.10916 w_k1_ADM_s011
                                 + 1.06686 w_k1_AXP_s011
                                 + 1.07212 w_k1_DIS_s011
                                 + 1.04908 w_k1_FDX_s011
                                 + 1.03507 w_k1_IBM_s011
                                 + 1.16261 w_k1_INTC_s011
                                 + 0.941602 w_k1_KO_s011
                                 + 0.693377 w_k1_MAT_s011
                                 + 0.956001 w_k1_MSFT_s011
                                 + 0.934793 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.33948 w_k1_ADM_s011
                                 + 1.35741 w_k1_AXP_s011
                                 + 1.38837 w_k1_DIS_s011
                                 + 1.12727 w_k1_FDX_s011
                                 + 1.51791 w_k1_IBM_s011
                                 + 1.11228 w_k1_INTC_s011
                                 + 1.14323 w_k1_KO_s011 + 1.18144 w_k1_MAT_s011
                                 + 1.05543 w_k1_MSFT_s011
                                 + 1.00096 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.999573 w_k1_ADM_s011
                                 + 0.801237 w_k1_AXP_s011
                                 + 0.807608 w_k1_DIS_s011
                                 + 0.926335 w_k1_FDX_s011
                                 + 0.890891 w_k1_IBM_s011
                                 + 0.977789 w_k1_INTC_s011
                                 + 0.963661 w_k1_KO_s011
                                 + 0.772916 w_k1_MAT_s011
                                 + 0.930121 w_k1_MSFT_s011
                                 + 0.861459 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.809511 w_k1_ADM_s011
                                 + 0.812962 w_k1_AXP_s011
                                 + 0.924978 w_k1_DIS_s011
                                 + 0.895858 w_k1_FDX_s011
                                 + 0.820491 w_k1_IBM_s011
                                 + 0.907211 w_k1_INTC_s011
                                 + 0.937083 w_k1_KO_s011
                                 + 0.845669 w_k1_MAT_s011
                                 + 0.932683 w_k1_MSFT_s011
                                 + 0.84605 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.768126 w_k1_ADM_s011
                                 + 0.680518 w_k1_AXP_s011
                                 + 0.676748 w_k1_DIS_s011
                                 + 0.715349 w_k1_FDX_s011
                                 + 0.756414 w_k1_IBM_s011
                                 + 0.780118 w_k1_INTC_s011
                                 + 0.632741 w_k1_KO_s011
                                 + 0.626066 w_k1_MAT_s011
                                 + 0.767126 w_k1_MSFT_s011
                                 + 0.703492 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.86169 w_k1_ADM_s011
                                 + 1.68977 w_k1_AXP_s011
                                 + 1.86375 w_k1_DIS_s011
                                 + 1.52513 w_k1_FDX_s011 + 1.3171 w_k1_IBM_s011
                                 + 1.36962 w_k1_INTC_s011
                                 + 1.45163 w_k1_KO_s011 + 1.64967 w_k1_MAT_s011
                                 + 1.63432 w_k1_MSFT_s011
                                 + 1.16526 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.29907 w_k1_ADM_s011
                                 + 1.09391 w_k1_AXP_s011
                                 + 1.36672 w_k1_DIS_s011
                                 + 1.23474 w_k1_FDX_s011
                                 + 0.849299 w_k1_IBM_s011
                                 + 1.33498 w_k1_INTC_s011
                                 + 1.25726 w_k1_KO_s011
                                 + 0.642685 w_k1_MAT_s011
                                 + 1.37297 w_k1_MSFT_s011
                                 + 1.04259 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.71002 w_k1_ADM_s012
                                 + 0.827627 w_k1_AXP_s012 + 1.041 w_k1_DIS_s012
                                 + 0.978244 w_k1_FDX_s012
                                 + 0.655227 w_k1_IBM_s012
                                 + 1.01931 w_k1_INTC_s012
                                 + 0.843733 w_k1_KO_s012
                                 + 0.745145 w_k1_MAT_s012
                                 + 1.08985 w_k1_MSFT_s012
                                 + 0.913639 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.805593 w_k1_ADM_s012
                                 + 0.742909 w_k1_AXP_s012
                                 + 0.843727 w_k1_DIS_s012
                                 + 0.79354 w_k1_FDX_s012 + 1.1541 w_k1_IBM_s012
                                 + 1.02574 w_k1_INTC_s012
                                 + 1.03865 w_k1_KO_s012 + 1.13509 w_k1_MAT_s012
                                 + 1.02283 w_k1_MSFT_s012
                                 + 1.01999 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.885439 w_k1_ADM_s012
                                 + 1.65945 w_k1_AXP_s012
                                 + 1.53725 w_k1_DIS_s012 + 1.2338 w_k1_FDX_s012
                                 + 1.36855 w_k1_IBM_s012
                                 + 1.46888 w_k1_INTC_s012
                                 + 1.35529 w_k1_KO_s012 + 1.65552 w_k1_MAT_s012
                                 + 1.25702 w_k1_MSFT_s012
                                 + 1.34493 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.935033 w_k1_ADM_s012
                                 + 1.26826 w_k1_AXP_s012
                                 + 1.07883 w_k1_DIS_s012
                                 + 1.10438 w_k1_FDX_s012
                                 + 1.10116 w_k1_IBM_s012
                                 + 1.37258 w_k1_INTC_s012
                                 + 1.05792 w_k1_KO_s012 + 1.1766 w_k1_MAT_s012
                                 + 1.01404 w_k1_MSFT_s012
                                 + 1.08352 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.05923 w_k1_ADM_s012
                                 + 1.02368 w_k1_AXP_s012
                                 + 0.993294 w_k1_DIS_s012
                                 + 1.05647 w_k1_FDX_s012
                                 + 0.778788 w_k1_IBM_s012
                                 + 1.04979 w_k1_INTC_s012
                                 + 0.957598 w_k1_KO_s012
                                 + 0.651019 w_k1_MAT_s012
                                 + 0.951873 w_k1_MSFT_s012
                                 + 0.958753 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.641926 w_k1_ADM_s012
                                 + 0.765893 w_k1_AXP_s012
                                 + 0.545573 w_k1_DIS_s012
                                 + 0.64706 w_k1_FDX_s012
                                 + 0.874183 w_k1_IBM_s012
                                 + 0.805973 w_k1_INTC_s012
                                 + 0.745053 w_k1_KO_s012
                                 + 0.963442 w_k1_MAT_s012
                                 + 0.672265 w_k1_MSFT_s012
                                 + 0.719633 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.05889 w_k1_ADM_s012
                                 + 1.03373 w_k1_AXP_s012
                                 + 1.55403 w_k1_DIS_s012
                                 + 1.33825 w_k1_FDX_s012 + 0.9107 w_k1_IBM_s012
                                 + 1.26991 w_k1_INTC_s012
                                 + 1.13228 w_k1_KO_s012
                                 + 0.774375 w_k1_MAT_s012
                                 + 1.30784 w_k1_MSFT_s012
                                 + 1.16627 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.761881 w_k1_ADM_s012
                                 + 0.852952 w_k1_AXP_s012
                                 + 1.02126 w_k1_DIS_s012
                                 + 0.804256 w_k1_FDX_s012
                                 + 1.61695 w_k1_IBM_s012
                                 + 1.19299 w_k1_INTC_s012 + 1.2696 w_k1_KO_s012
                                 + 1.04546 w_k1_MAT_s012
                                 + 0.912989 w_k1_MSFT_s012
                                 + 0.962428 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.01761 w_k1_ADM_s013
                                 + 0.960756 w_k1_AXP_s013 + 1.299 w_k1_DIS_s013
                                 + 0.987174 w_k1_FDX_s013
                                 + 1.20669 w_k1_IBM_s013
                                 + 0.844525 w_k1_INTC_s013
                                 + 1.30111 w_k1_KO_s013 + 1.20391 w_k1_MAT_s013
                                 + 0.930105 w_k1_MSFT_s013
                                 + 1.06195 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.0874 w_k1_ADM_s013
                                 + 0.716335 w_k1_AXP_s013
                                 + 0.719287 w_k1_DIS_s013
                                 + 0.753298 w_k1_FDX_s013
                                 + 0.734366 w_k1_IBM_s013
                                 + 0.73768 w_k1_INTC_s013
                                 + 0.701395 w_k1_KO_s013
                                 + 0.728479 w_k1_MAT_s013
                                 + 0.828431 w_k1_MSFT_s013
                                 + 0.828198 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.16324 w_k1_ADM_s013
                                 + 1.32893 w_k1_AXP_s013
                                 + 1.13059 w_k1_DIS_s013
                                 + 1.15707 w_k1_FDX_s013
                                 + 1.07999 w_k1_IBM_s013
                                 + 1.17226 w_k1_INTC_s013
                                 + 0.889997 w_k1_KO_s013
                                 + 1.30824 w_k1_MAT_s013
                                 + 0.999453 w_k1_MSFT_s013
                                 + 1.00771 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.25495 w_k1_ADM_s013
                                 + 1.00119 w_k1_AXP_s013
                                 + 1.02032 w_k1_DIS_s013
                                 + 1.00365 w_k1_FDX_s013
                                 + 0.901399 w_k1_IBM_s013
                                 + 0.835939 w_k1_INTC_s013
                                 + 0.779014 w_k1_KO_s013 + 0.7202 w_k1_MAT_s013
                                 + 0.971322 w_k1_MSFT_s013
                                 + 0.9125 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.883423 w_k1_ADM_s013
                                 + 1.29652 w_k1_AXP_s013 + 1.2536 w_k1_DIS_s013
                                 + 1.21271 w_k1_FDX_s013
                                 + 0.855845 w_k1_IBM_s013
                                 + 1.09488 w_k1_INTC_s013
                                 + 0.914283 w_k1_KO_s013
                                 + 1.39355 w_k1_MAT_s013
                                 + 1.08517 w_k1_MSFT_s013
                                 + 1.09355 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.08863 w_k1_ADM_s013
                                 + 1.06381 w_k1_AXP_s013
                                 + 0.938276 w_k1_DIS_s013
                                 + 0.908864 w_k1_FDX_s013
                                 + 1.34707 w_k1_IBM_s013
                                 + 0.773819 w_k1_INTC_s013
                                 + 1.20144 w_k1_KO_s013 + 1.50994 w_k1_MAT_s013
                                 + 1.00593 w_k1_MSFT_s013 + 1.0285 w_k1_PG_s013
                                 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.01135 w_k1_ADM_s013
                                 + 0.915549 w_k1_AXP_s013
                                 + 0.767798 w_k1_DIS_s013
                                 + 0.818263 w_k1_FDX_s013
                                 + 1.00022 w_k1_IBM_s013
                                 + 0.971844 w_k1_INTC_s013
                                 + 0.772539 w_k1_KO_s013
                                 + 1.19602 w_k1_MAT_s013
                                 + 0.922228 w_k1_MSFT_s013
                                 + 0.875252 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.941969 w_k1_ADM_s013
                                 + 1.25828 w_k1_AXP_s013
                                 + 1.70524 w_k1_DIS_s013
                                 + 1.31756 w_k1_FDX_s013
                                 + 1.26349 w_k1_IBM_s013
                                 + 1.20865 w_k1_INTC_s013
                                 + 1.23206 w_k1_KO_s013 + 1.08126 w_k1_MAT_s013
                                 + 1.28556 w_k1_MSFT_s013
                                 + 1.15598 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.974433 w_k1_ADM_s014
                                 + 1.11185 w_k1_AXP_s014
                                 + 1.05959 w_k1_DIS_s014
                                 + 1.18244 w_k1_FDX_s014
                                 + 0.619449 w_k1_IBM_s014
                                 + 0.828408 w_k1_INTC_s014
                                 + 0.774337 w_k1_KO_s014
                                 + 0.842341 w_k1_MAT_s014
                                 + 1.04957 w_k1_MSFT_s014 + 1.1349 w_k1_PG_s014
                                 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.24259 w_k1_ADM_s014
                                 + 1.49717 w_k1_AXP_s014
                                 + 1.62831 w_k1_DIS_s014 + 1.3599 w_k1_FDX_s014
                                 + 1.72211 w_k1_IBM_s014
                                 + 1.46997 w_k1_INTC_s014
                                 + 1.80754 w_k1_KO_s014 + 1.54301 w_k1_MAT_s014
                                 + 1.57846 w_k1_MSFT_s014
                                 + 1.35279 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.26126 w_k1_ADM_s014
                                 + 1.5524 w_k1_AXP_s014 + 1.41038 w_k1_DIS_s014
                                 + 1.36427 w_k1_FDX_s014
                                 + 0.792374 w_k1_IBM_s014
                                 + 1.27268 w_k1_INTC_s014
                                 + 1.25821 w_k1_KO_s014
                                 + 0.725192 w_k1_MAT_s014
                                 + 1.18066 w_k1_MSFT_s014
                                 + 1.35651 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.11409 w_k1_ADM_s014
                                 + 1.4284 w_k1_AXP_s014 + 1.35593 w_k1_DIS_s014
                                 + 1.17762 w_k1_FDX_s014 + 1.2776 w_k1_IBM_s014
                                 + 1.10866 w_k1_INTC_s014
                                 + 1.33344 w_k1_KO_s014 + 1.54466 w_k1_MAT_s014
                                 + 1.15582 w_k1_MSFT_s014
                                 + 1.38222 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.617548 w_k1_ADM_s014
                                 + 1.01285 w_k1_AXP_s014
                                 + 0.780998 w_k1_DIS_s014
                                 + 0.838352 w_k1_FDX_s014
                                 + 1.13256 w_k1_IBM_s014
                                 + 0.827354 w_k1_INTC_s014
                                 + 1.00436 w_k1_KO_s014 + 1.38632 w_k1_MAT_s014
                                 + 0.914097 w_k1_MSFT_s014
                                 + 0.898494 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.957095 w_k1_ADM_s014
                                 + 1.31419 w_k1_AXP_s014
                                 + 1.02161 w_k1_DIS_s014
                                 + 1.07777 w_k1_FDX_s014
                                 + 0.984648 w_k1_IBM_s014
                                 + 0.92587 w_k1_INTC_s014
                                 + 1.11727 w_k1_KO_s014 + 1.22011 w_k1_MAT_s014
                                 + 0.952112 w_k1_MSFT_s014
                                 + 1.10274 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.725138 w_k1_ADM_s014
                                 + 0.835865 w_k1_AXP_s014
                                 + 0.544662 w_k1_DIS_s014
                                 + 0.763898 w_k1_FDX_s014
                                 + 0.714263 w_k1_IBM_s014
                                 + 0.95271 w_k1_INTC_s014
                                 + 0.781834 w_k1_KO_s014
                                 + 0.583103 w_k1_MAT_s014
                                 + 0.80405 w_k1_MSFT_s014
                                 + 0.745271 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.721478 w_k1_ADM_s014
                                 + 1.23859 w_k1_AXP_s014
                                 + 0.799641 w_k1_DIS_s014
                                 + 1.0087 w_k1_FDX_s014
                                 + 0.671373 w_k1_IBM_s014
                                 + 0.870089 w_k1_INTC_s014
                                 + 0.843365 w_k1_KO_s014
                                 + 1.08737 w_k1_MAT_s014
                                 + 0.958406 w_k1_MSFT_s014
                                 + 0.96962 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.825562 w_k1_ADM_s015
                                 + 0.996575 w_k1_AXP_s015
                                 + 0.905205 w_k1_DIS_s015
                                 + 0.912142 w_k1_FDX_s015
                                 + 1.34572 w_k1_IBM_s015
                                 + 1.08968 w_k1_INTC_s015
                                 + 1.11449 w_k1_KO_s015 + 1.15315 w_k1_MAT_s015
                                 + 0.996942 w_k1_MSFT_s015
                                 + 0.937999 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.07615 w_k1_ADM_s015
                                 + 1.09551 w_k1_AXP_s015
                                 + 0.80512 w_k1_DIS_s015
                                 + 0.960864 w_k1_FDX_s015
                                 + 0.833104 w_k1_IBM_s015
                                 + 0.999343 w_k1_INTC_s015
                                 + 0.765063 w_k1_KO_s015
                                 + 1.05804 w_k1_MAT_s015
                                 + 0.892231 w_k1_MSFT_s015
                                 + 1.0076 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.63306 w_k1_ADM_s015
                                 + 1.34313 w_k1_AXP_s015 + 1.4752 w_k1_DIS_s015
                                 + 1.32699 w_k1_FDX_s015
                                 + 1.19443 w_k1_IBM_s015
                                 + 1.24435 w_k1_INTC_s015 + 1.1745 w_k1_KO_s015
                                 + 1.08441 w_k1_MAT_s015
                                 + 1.41074 w_k1_MSFT_s015
                                 + 1.24205 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.817791 w_k1_ADM_s015
                                 + 0.740099 w_k1_AXP_s015
                                 + 0.677655 w_k1_DIS_s015
                                 + 0.693615 w_k1_FDX_s015
                                 + 0.935808 w_k1_IBM_s015
                                 + 0.811349 w_k1_INTC_s015
                                 + 0.82755 w_k1_KO_s015
                                 + 0.931422 w_k1_MAT_s015
                                 + 0.832681 w_k1_MSFT_s015
                                 + 0.873543 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.883707 w_k1_ADM_s015
                                 + 0.707805 w_k1_AXP_s015
                                 + 0.665206 w_k1_DIS_s015
                                 + 0.827725 w_k1_FDX_s015
                                 + 0.687649 w_k1_IBM_s015
                                 + 0.824309 w_k1_INTC_s015
                                 + 0.659103 w_k1_KO_s015
                                 + 0.837122 w_k1_MAT_s015
                                 + 0.886347 w_k1_MSFT_s015
                                 + 0.790802 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.01958 w_k1_ADM_s015
                                 + 1.27253 w_k1_AXP_s015
                                 + 1.14944 w_k1_DIS_s015
                                 + 1.32209 w_k1_FDX_s015
                                 + 0.833514 w_k1_IBM_s015
                                 + 1.3341 w_k1_INTC_s015
                                 + 0.940895 w_k1_KO_s015
                                 + 1.02345 w_k1_MAT_s015
                                 + 1.30747 w_k1_MSFT_s015
                                 + 1.10438 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.02269 w_k1_ADM_s015
                                 + 1.02423 w_k1_AXP_s015
                                 + 1.04786 w_k1_DIS_s015
                                 + 1.05694 w_k1_FDX_s015
                                 + 0.908279 w_k1_IBM_s015
                                 + 1.27185 w_k1_INTC_s015
                                 + 0.919175 w_k1_KO_s015
                                 + 0.983949 w_k1_MAT_s015
                                 + 1.14487 w_k1_MSFT_s015
                                 + 0.973364 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.937632 w_k1_ADM_s015
                                 + 1.4185 w_k1_AXP_s015 + 1.47472 w_k1_DIS_s015
                                 + 1.10536 w_k1_FDX_s015
                                 + 1.56432 w_k1_IBM_s015
                                 + 1.15423 w_k1_INTC_s015
                                 + 1.17077 w_k1_KO_s015 + 1.33479 w_k1_MAT_s015
                                 + 1.21851 w_k1_MSFT_s015
                                 + 1.09104 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.782649 w_k1_ADM_s016
                                 + 0.575904 w_k1_AXP_s016
                                 + 0.593884 w_k1_DIS_s016
                                 + 0.693766 w_k1_FDX_s016
                                 + 0.637162 w_k1_IBM_s016
                                 + 0.763019 w_k1_INTC_s016
                                 + 0.729462 w_k1_KO_s016
                                 + 0.631962 w_k1_MAT_s016
                                 + 0.803412 w_k1_MSFT_s016
                                 + 0.754827 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.00953 w_k1_ADM_s016
                                 + 0.836408 w_k1_AXP_s016
                                 + 0.820051 w_k1_DIS_s016
                                 + 0.939628 w_k1_FDX_s016
                                 + 0.780038 w_k1_IBM_s016
                                 + 1.03144 w_k1_INTC_s016
                                 + 0.962501 w_k1_KO_s016
                                 + 0.729264 w_k1_MAT_s016
                                 + 1.10892 w_k1_MSFT_s016
                                 + 0.89954 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.756701 w_k1_ADM_s016
                                 + 1.04902 w_k1_AXP_s016
                                 + 0.833683 w_k1_DIS_s016
                                 + 1.03674 w_k1_FDX_s016
                                 + 0.863904 w_k1_IBM_s016
                                 + 0.795486 w_k1_INTC_s016
                                 + 0.865979 w_k1_KO_s016
                                 + 0.985008 w_k1_MAT_s016
                                 + 1.01646 w_k1_MSFT_s016
                                 + 0.880582 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.750282 w_k1_ADM_s016
                                 + 0.628089 w_k1_AXP_s016
                                 + 0.60521 w_k1_DIS_s016
                                 + 0.655116 w_k1_FDX_s016
                                 + 0.998159 w_k1_IBM_s016
                                 + 0.722081 w_k1_INTC_s016
                                 + 0.84879 w_k1_KO_s016
                                 + 0.549479 w_k1_MAT_s016
                                 + 0.656538 w_k1_MSFT_s016
                                 + 0.798771 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.08466 w_k1_ADM_s016
                                 + 1.0131 w_k1_AXP_s016
                                 + 0.973056 w_k1_DIS_s016
                                 + 1.05704 w_k1_FDX_s016
                                 + 1.30968 w_k1_IBM_s016
                                 + 0.951142 w_k1_INTC_s016
                                 + 1.29312 w_k1_KO_s016 + 1.12471 w_k1_MAT_s016
                                 + 1.06565 w_k1_MSFT_s016
                                 + 1.03327 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.33845 w_k1_ADM_s016
                                 + 1.59102 w_k1_AXP_s016
                                 + 1.75678 w_k1_DIS_s016
                                 + 1.47662 w_k1_FDX_s016
                                 + 1.44844 w_k1_IBM_s016
                                 + 1.20991 w_k1_INTC_s016
                                 + 1.79646 w_k1_KO_s016 + 1.91318 w_k1_MAT_s016
                                 + 1.29415 w_k1_MSFT_s016
                                 + 1.37176 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.895173 w_k1_ADM_s016
                                 + 0.966868 w_k1_AXP_s016
                                 + 0.858889 w_k1_DIS_s016
                                 + 1.02398 w_k1_FDX_s016
                                 + 1.01871 w_k1_IBM_s016
                                 + 0.970292 w_k1_INTC_s016
                                 + 0.974692 w_k1_KO_s016
                                 + 1.00917 w_k1_MAT_s016
                                 + 0.854073 w_k1_MSFT_s016
                                 + 0.922674 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.939254 w_k1_ADM_s016
                                 + 1.13682 w_k1_AXP_s016
                                 + 1.28455 w_k1_DIS_s016
                                 + 1.19088 w_k1_FDX_s016
                                 + 1.06334 w_k1_IBM_s016
                                 + 0.845291 w_k1_INTC_s016
                                 + 1.18711 w_k1_KO_s016 + 1.04327 w_k1_MAT_s016
                                 + 0.935568 w_k1_MSFT_s016
                                 + 0.990001 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.12153 w_k1_ADM_s017
                                 + 1.31564 w_k1_AXP_s017
                                 + 1.22582 w_k1_DIS_s017
                                 + 1.15231 w_k1_FDX_s017
                                 + 1.35053 w_k1_IBM_s017
                                 + 1.03024 w_k1_INTC_s017
                                 + 1.47933 w_k1_KO_s017 + 1.4579 w_k1_MAT_s017
                                 + 1.18753 w_k1_MSFT_s017
                                 + 1.14214 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.967551 w_k1_ADM_s017
                                 + 0.946734 w_k1_AXP_s017
                                 + 0.818843 w_k1_DIS_s017
                                 + 0.994923 w_k1_FDX_s017
                                 + 0.711527 w_k1_IBM_s017
                                 + 1.04372 w_k1_INTC_s017
                                 + 0.90299 w_k1_KO_s017 + 0.4506 w_k1_MAT_s017
                                 + 1.05291 w_k1_MSFT_s017
                                 + 0.863731 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.781981 w_k1_ADM_s017
                                 + 0.868787 w_k1_AXP_s017
                                 + 0.7541 w_k1_DIS_s017
                                 + 0.803903 w_k1_FDX_s017
                                 + 0.895063 w_k1_IBM_s017
                                 + 0.863242 w_k1_INTC_s017
                                 + 1.02941 w_k1_KO_s017
                                 + 0.816203 w_k1_MAT_s017
                                 + 1.02871 w_k1_MSFT_s017
                                 + 0.847308 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.09184 w_k1_ADM_s017
                                 + 1.09324 w_k1_AXP_s017
                                 + 0.947477 w_k1_DIS_s017
                                 + 0.940402 w_k1_FDX_s017
                                 + 1.03242 w_k1_IBM_s017
                                 + 0.787942 w_k1_INTC_s017
                                 + 1.01318 w_k1_KO_s017 + 1.17227 w_k1_MAT_s017
                                 + 0.983341 w_k1_MSFT_s017
                                 + 0.969462 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.686467 w_k1_ADM_s017
                                 + 0.761556 w_k1_AXP_s017
                                 + 0.560937 w_k1_DIS_s017
                                 + 0.711881 w_k1_FDX_s017
                                 + 0.736321 w_k1_IBM_s017
                                 + 0.688795 w_k1_INTC_s017
                                 + 0.7195 w_k1_KO_s017 + 0.544571 w_k1_MAT_s017
                                 + 0.77807 w_k1_MSFT_s017
                                 + 0.797242 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.976327 w_k1_ADM_s017
                                 + 1.40743 w_k1_AXP_s017
                                 + 1.24703 w_k1_DIS_s017
                                 + 1.30791 w_k1_FDX_s017
                                 + 0.986786 w_k1_IBM_s017
                                 + 1.10715 w_k1_INTC_s017
                                 + 1.09721 w_k1_KO_s017 + 1.0906 w_k1_MAT_s017
                                 + 1.17229 w_k1_MSFT_s017
                                 + 1.14778 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.837888 w_k1_ADM_s017
                                 + 0.792636 w_k1_AXP_s017
                                 + 0.700275 w_k1_DIS_s017
                                 + 0.690188 w_k1_FDX_s017
                                 + 1.28323 w_k1_IBM_s017
                                 + 0.621171 w_k1_INTC_s017
                                 + 1.10131 w_k1_KO_s017 + 1.09378 w_k1_MAT_s017
                                 + 0.748977 w_k1_MSFT_s017
                                 + 0.938954 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.22406 w_k1_ADM_s017
                                 + 1.6638 w_k1_AXP_s017 + 1.82756 w_k1_DIS_s017
                                 + 1.45506 w_k1_FDX_s017
                                 + 1.31891 w_k1_IBM_s017
                                 + 1.25353 w_k1_INTC_s017
                                 + 1.78335 w_k1_KO_s017 + 1.19709 w_k1_MAT_s017
                                 + 1.81354 w_k1_MSFT_s017
                                 + 1.29445 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.55577 w_k1_ADM_s100
                                 + 1.46522 w_k1_AXP_s100
                                 + 1.88479 w_k1_DIS_s100 + 1.3952 w_k1_FDX_s100
                                 + 1.36228 w_k1_IBM_s100
                                 + 1.63577 w_k1_INTC_s100
                                 + 1.38464 w_k1_KO_s100 + 1.20218 w_k1_MAT_s100
                                 + 1.41386 w_k1_MSFT_s100
                                 + 1.28162 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.813445 w_k1_ADM_s100
                                 + 0.953538 w_k1_AXP_s100
                                 + 0.665801 w_k1_DIS_s100
                                 + 0.771973 w_k1_FDX_s100
                                 + 1.11922 w_k1_IBM_s100
                                 + 0.831125 w_k1_INTC_s100
                                 + 0.734337 w_k1_KO_s100
                                 + 1.77443 w_k1_MAT_s100
                                 + 0.690149 w_k1_MSFT_s100
                                 + 0.812634 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.42641 w_k1_ADM_s100
                                 + 1.21369 w_k1_AXP_s100
                                 + 1.35041 w_k1_DIS_s100
                                 + 1.27331 w_k1_FDX_s100
                                 + 0.766367 w_k1_IBM_s100
                                 + 1.18331 w_k1_INTC_s100
                                 + 0.908079 w_k1_KO_s100
                                 + 0.661669 w_k1_MAT_s100
                                 + 1.0799 w_k1_MSFT_s100 + 1.0724 w_k1_PG_s100
                                 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.943417 w_k1_ADM_s100
                                 + 0.699633 w_k1_AXP_s100
                                 + 0.783727 w_k1_DIS_s100
                                 + 0.876278 w_k1_FDX_s100
                                 + 0.636717 w_k1_IBM_s100
                                 + 0.891284 w_k1_INTC_s100
                                 + 0.690355 w_k1_KO_s100
                                 + 0.594044 w_k1_MAT_s100
                                 + 0.832324 w_k1_MSFT_s100
                                 + 0.777408 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.24274 w_k1_ADM_s100
                                 + 1.20923 w_k1_AXP_s100
                                 + 1.29956 w_k1_DIS_s100
                                 + 1.17528 w_k1_FDX_s100
                                 + 1.15875 w_k1_IBM_s100
                                 + 1.12569 w_k1_INTC_s100
                                 + 1.01862 w_k1_KO_s100
                                 + 0.949962 w_k1_MAT_s100
                                 + 1.15452 w_k1_MSFT_s100
                                 + 1.02113 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.14117 w_k1_ADM_s100
                                 + 1.03212 w_k1_AXP_s100
                                 + 0.951404 w_k1_DIS_s100
                                 + 1.0068 w_k1_FDX_s100
                                 + 0.979004 w_k1_IBM_s100
                                 + 0.999098 w_k1_INTC_s100
                                 + 0.840352 w_k1_KO_s100
                                 + 1.26079 w_k1_MAT_s100
                                 + 0.922444 w_k1_MSFT_s100
                                 + 0.905777 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.770542 w_k1_ADM_s100
                                 + 0.693314 w_k1_AXP_s100
                                 + 0.610781 w_k1_DIS_s100
                                 + 0.751368 w_k1_FDX_s100
                                 + 0.747889 w_k1_IBM_s100
                                 + 0.947311 w_k1_INTC_s100
                                 + 0.615668 w_k1_KO_s100
                                 + 0.837847 w_k1_MAT_s100
                                 + 0.798411 w_k1_MSFT_s100
                                 + 0.80612 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.11415 w_k1_ADM_s100
                                 + 1.11256 w_k1_AXP_s100
                                 + 1.12007 w_k1_DIS_s100
                                 + 0.962751 w_k1_FDX_s100
                                 + 1.29642 w_k1_IBM_s100
                                 + 1.06316 w_k1_INTC_s100
                                 + 1.04572 w_k1_KO_s100 + 1.25375 w_k1_MAT_s100
                                 + 1.01501 w_k1_MSFT_s100
                                 + 0.980607 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.788687 w_k1_ADM_s101
                                 + 0.654979 w_k1_AXP_s101
                                 + 0.531056 w_k1_DIS_s101
                                 + 0.707909 w_k1_FDX_s101
                                 + 0.687002 w_k1_IBM_s101
                                 + 0.774757 w_k1_INTC_s101
                                 + 0.702275 w_k1_KO_s101
                                 + 0.792392 w_k1_MAT_s101
                                 + 0.73484 w_k1_MSFT_s101
                                 + 0.714104 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.844767 w_k1_ADM_s101
                                 + 0.773583 w_k1_AXP_s101
                                 + 0.90057 w_k1_DIS_s101
                                 + 0.957672 w_k1_FDX_s101
                                 + 0.637546 w_k1_IBM_s101
                                 + 0.871134 w_k1_INTC_s101
                                 + 0.738513 w_k1_KO_s101
                                 + 0.517065 w_k1_MAT_s101
                                 + 1.08548 w_k1_MSFT_s101
                                 + 0.909055 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.867558 w_k1_ADM_s101
                                 + 0.796867 w_k1_AXP_s101
                                 + 0.836723 w_k1_DIS_s101
                                 + 0.898634 w_k1_FDX_s101
                                 + 1.05451 w_k1_IBM_s101
                                 + 1.03625 w_k1_INTC_s101
                                 + 1.10138 w_k1_KO_s101 + 1.10533 w_k1_MAT_s101
                                 + 0.9869 w_k1_MSFT_s101 + 1.08647 w_k1_PG_s101
                                 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.35583 w_k1_ADM_s101
                                 + 1.47607 w_k1_AXP_s101
                                 + 1.75286 w_k1_DIS_s101
                                 + 1.33135 w_k1_FDX_s101
                                 + 1.73288 w_k1_IBM_s101
                                 + 1.17207 w_k1_INTC_s101
                                 + 1.55501 w_k1_KO_s101 + 1.81071 w_k1_MAT_s101
                                 + 1.25945 w_k1_MSFT_s101
                                 + 1.36118 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.950453 w_k1_ADM_s101
                                 + 1.18277 w_k1_AXP_s101
                                 + 1.08094 w_k1_DIS_s101
                                 + 1.20033 w_k1_FDX_s101
                                 + 0.862595 w_k1_IBM_s101
                                 + 1.05041 w_k1_INTC_s101
                                 + 0.919664 w_k1_KO_s101
                                 + 1.22616 w_k1_MAT_s101
                                 + 1.05369 w_k1_MSFT_s101
                                 + 1.12973 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.12315 w_k1_ADM_s101
                                 + 1.04371 w_k1_AXP_s101
                                 + 1.30593 w_k1_DIS_s101
                                 + 1.04193 w_k1_FDX_s101 + 1.3458 w_k1_IBM_s101
                                 + 0.987244 w_k1_INTC_s101
                                 + 1.32072 w_k1_KO_s101 + 1.27753 w_k1_MAT_s101
                                 + 0.994436 w_k1_MSFT_s101
                                 + 1.12803 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.4214 w_k1_ADM_s101
                                 + 1.84699 w_k1_AXP_s101
                                 + 1.65128 w_k1_DIS_s101
                                 + 1.44745 w_k1_FDX_s101
                                 + 1.26521 w_k1_IBM_s101
                                 + 1.27007 w_k1_INTC_s101
                                 + 1.35576 w_k1_KO_s101 + 1.66508 w_k1_MAT_s101
                                 + 1.37734 w_k1_MSFT_s101
                                 + 1.16076 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.971857 w_k1_ADM_s101
                                 + 0.894018 w_k1_AXP_s101
                                 + 0.734795 w_k1_DIS_s101
                                 + 0.958936 w_k1_FDX_s101
                                 + 0.786594 w_k1_IBM_s101
                                 + 1.08939 w_k1_INTC_s101
                                 + 0.873974 w_k1_KO_s101
                                 + 0.803423 w_k1_MAT_s101
                                 + 0.946321 w_k1_MSFT_s101
                                 + 0.954095 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.853707 w_k1_ADM_s102
                                 + 0.723559 w_k1_AXP_s102
                                 + 0.684998 w_k1_DIS_s102
                                 + 0.854801 w_k1_FDX_s102
                                 + 0.724652 w_k1_IBM_s102
                                 + 0.984336 w_k1_INTC_s102
                                 + 0.763913 w_k1_KO_s102
                                 + 0.663664 w_k1_MAT_s102
                                 + 0.834046 w_k1_MSFT_s102
                                 + 0.88059 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.2739 w_k1_ADM_s102
                                 + 1.33598 w_k1_AXP_s102
                                 + 1.38648 w_k1_DIS_s102
                                 + 1.41851 w_k1_FDX_s102
                                 + 1.02445 w_k1_IBM_s102
                                 + 1.3434 w_k1_INTC_s102 + 1.2067 w_k1_KO_s102
                                 + 0.861474 w_k1_MAT_s102
                                 + 1.15071 w_k1_MSFT_s102
                                 + 1.13759 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.983668 w_k1_ADM_s102
                                 + 1.17065 w_k1_AXP_s102
                                 + 0.964426 w_k1_DIS_s102
                                 + 1.06006 w_k1_FDX_s102
                                 + 1.29721 w_k1_IBM_s102
                                 + 1.18622 w_k1_INTC_s102 + 0.9055 w_k1_KO_s102
                                 + 1.16128 w_k1_MAT_s102
                                 + 0.922602 w_k1_MSFT_s102
                                 + 1.03782 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.05455 w_k1_ADM_s102
                                 + 0.933439 w_k1_AXP_s102
                                 + 1.17441 w_k1_DIS_s102
                                 + 1.11513 w_k1_FDX_s102
                                 + 0.791009 w_k1_IBM_s102
                                 + 1.02829 w_k1_INTC_s102
                                 + 0.848671 w_k1_KO_s102
                                 + 0.731212 w_k1_MAT_s102
                                 + 1.04595 w_k1_MSFT_s102
                                 + 1.10727 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.828084 w_k1_ADM_s102
                                 + 0.865946 w_k1_AXP_s102
                                 + 0.866784 w_k1_DIS_s102
                                 + 0.836503 w_k1_FDX_s102
                                 + 1.20902 w_k1_IBM_s102
                                 + 1.0618 w_k1_INTC_s102 + 0.94388 w_k1_KO_s102
                                 + 1.01977 w_k1_MAT_s102
                                 + 0.844605 w_k1_MSFT_s102
                                 + 0.949211 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.8463 w_k1_ADM_s102
                                 + 1.11267 w_k1_AXP_s102
                                 + 1.13699 w_k1_DIS_s102
                                 + 0.917965 w_k1_FDX_s102
                                 + 1.7809 w_k1_IBM_s102
                                 + 1.21419 w_k1_INTC_s102
                                 + 1.15492 w_k1_KO_s102 + 1.75239 w_k1_MAT_s102
                                 + 0.977652 w_k1_MSFT_s102
                                 + 1.10825 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.06588 w_k1_ADM_s102
                                 + 1.35903 w_k1_AXP_s102
                                 + 1.26014 w_k1_DIS_s102
                                 + 1.17024 w_k1_FDX_s102
                                 + 1.06228 w_k1_IBM_s102
                                 + 1.09521 w_k1_INTC_s102
                                 + 1.02022 w_k1_KO_s102
                                 + 0.933523 w_k1_MAT_s102
                                 + 0.976289 w_k1_MSFT_s102
                                 + 0.988213 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.762516 w_k1_ADM_s102
                                 + 0.656025 w_k1_AXP_s102
                                 + 0.580758 w_k1_DIS_s102
                                 + 0.660805 w_k1_FDX_s102
                                 + 0.940215 w_k1_IBM_s102
                                 + 0.803507 w_k1_INTC_s102
                                 + 0.701347 w_k1_KO_s102
                                 + 0.721657 w_k1_MAT_s102
                                 + 0.660112 w_k1_MSFT_s102
                                 + 0.809169 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.867522 w_k1_ADM_s103
                                 + 0.836337 w_k1_AXP_s103
                                 + 0.826561 w_k1_DIS_s103
                                 + 0.891512 w_k1_FDX_s103
                                 + 0.993414 w_k1_IBM_s103
                                 + 0.816552 w_k1_INTC_s103
                                 + 0.799684 w_k1_KO_s103
                                 + 1.08374 w_k1_MAT_s103
                                 + 0.837474 w_k1_MSFT_s103
                                 + 0.91934 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.03874 w_k1_ADM_s103
                                 + 0.825312 w_k1_AXP_s103
                                 + 0.603618 w_k1_DIS_s103
                                 + 0.866304 w_k1_FDX_s103
                                 + 0.565798 w_k1_IBM_s103
                                 + 0.79583 w_k1_INTC_s103
                                 + 0.572842 w_k1_KO_s103
                                 + 0.515852 w_k1_MAT_s103
                                 + 0.799593 w_k1_MSFT_s103
                                 + 0.804895 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.1912 w_k1_ADM_s103
                                 + 1.44201 w_k1_AXP_s103
                                 + 1.69141 w_k1_DIS_s103
                                 + 1.31769 w_k1_FDX_s103
                                 + 1.48212 w_k1_IBM_s103
                                 + 1.28386 w_k1_INTC_s103 + 1.2293 w_k1_KO_s103
                                 + 1.37833 w_k1_MAT_s103
                                 + 1.16603 w_k1_MSFT_s103
                                 + 1.06753 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.10676 w_k1_ADM_s103
                                 + 0.899567 w_k1_AXP_s103
                                 + 1.28374 w_k1_DIS_s103
                                 + 1.07425 w_k1_FDX_s103
                                 + 0.993741 w_k1_IBM_s103
                                 + 1.10869 w_k1_INTC_s103
                                 + 0.961262 w_k1_KO_s103
                                 + 0.889692 w_k1_MAT_s103
                                 + 1.194 w_k1_MSFT_s103 + 1.01376 w_k1_PG_s103
                                 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.01708 w_k1_ADM_s103
                                 + 1.05729 w_k1_AXP_s103
                                 + 0.887788 w_k1_DIS_s103
                                 + 1.08188 w_k1_FDX_s103
                                 + 0.885463 w_k1_IBM_s103
                                 + 0.873101 w_k1_INTC_s103
                                 + 0.79934 w_k1_KO_s103
                                 + 0.866026 w_k1_MAT_s103
                                 + 0.890283 w_k1_MSFT_s103
                                 + 0.953282 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.926403 w_k1_ADM_s103
                                 + 1.10041 w_k1_AXP_s103
                                 + 1.04697 w_k1_DIS_s103
                                 + 1.03106 w_k1_FDX_s103
                                 + 1.41913 w_k1_IBM_s103
                                 + 1.16466 w_k1_INTC_s103
                                 + 1.08908 w_k1_KO_s103 + 1.22342 w_k1_MAT_s103
                                 + 1.01821 w_k1_MSFT_s103
                                 + 1.08103 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.15741 w_k1_ADM_s103
                                 + 1.27497 w_k1_AXP_s103
                                 + 1.27031 w_k1_DIS_s103 + 1.3155 w_k1_FDX_s103
                                 + 1.05782 w_k1_IBM_s103
                                 + 0.990066 w_k1_INTC_s103 + 1.025 w_k1_KO_s103
                                 + 1.22126 w_k1_MAT_s103
                                 + 1.11629 w_k1_MSFT_s103 + 1.1336 w_k1_PG_s103
                                 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.62191 w_k1_ADM_s103
                                 + 0.673429 w_k1_AXP_s103
                                 + 0.604034 w_k1_DIS_s103
                                 + 0.68738 w_k1_FDX_s103
                                 + 0.718737 w_k1_IBM_s103
                                 + 0.697082 w_k1_INTC_s103
                                 + 0.569012 w_k1_KO_s103
                                 + 0.608235 w_k1_MAT_s103
                                 + 0.67035 w_k1_MSFT_s103
                                 + 0.733222 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.09262 w_k1_ADM_s104
                                 + 1.09742 w_k1_AXP_s104
                                 + 1.15938 w_k1_DIS_s104
                                 + 1.01309 w_k1_FDX_s104
                                 + 1.20341 w_k1_IBM_s104
                                 + 1.12112 w_k1_INTC_s104
                                 + 1.08173 w_k1_KO_s104
                                 + 0.961553 w_k1_MAT_s104
                                 + 1.13272 w_k1_MSFT_s104
                                 + 1.03528 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.08241 w_k1_ADM_s104
                                 + 1.32131 w_k1_AXP_s104
                                 + 1.37176 w_k1_DIS_s104
                                 + 1.11405 w_k1_FDX_s104
                                 + 1.49622 w_k1_IBM_s104
                                 + 1.1357 w_k1_INTC_s104 + 1.34938 w_k1_KO_s104
                                 + 1.28629 w_k1_MAT_s104
                                 + 1.27441 w_k1_MSFT_s104
                                 + 1.10322 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.777771 w_k1_ADM_s104
                                 + 0.690153 w_k1_AXP_s104
                                 + 0.761844 w_k1_DIS_s104
                                 + 0.810685 w_k1_FDX_s104
                                 + 0.604662 w_k1_IBM_s104
                                 + 0.794952 w_k1_INTC_s104
                                 + 0.750236 w_k1_KO_s104
                                 + 0.48593 w_k1_MAT_s104
                                 + 0.888454 w_k1_MSFT_s104
                                 + 0.802774 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.01716 w_k1_ADM_s104
                                 + 0.926725 w_k1_AXP_s104
                                 + 0.93127 w_k1_DIS_s104
                                 + 0.965425 w_k1_FDX_s104
                                 + 0.816656 w_k1_IBM_s104
                                 + 0.941077 w_k1_INTC_s104
                                 + 0.906588 w_k1_KO_s104
                                 + 0.787505 w_k1_MAT_s104
                                 + 1.00953 w_k1_MSFT_s104
                                 + 0.979186 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.2443 w_k1_ADM_s104
                                 + 0.927108 w_k1_AXP_s104
                                 + 1.03844 w_k1_DIS_s104
                                 + 1.08179 w_k1_FDX_s104
                                 + 0.744781 w_k1_IBM_s104
                                 + 1.10006 w_k1_INTC_s104
                                 + 1.07581 w_k1_KO_s104
                                 + 0.544817 w_k1_MAT_s104
                                 + 1.1941 w_k1_MSFT_s104 + 1.04488 w_k1_PG_s104
                                 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.42985 w_k1_ADM_s104
                                 + 1.31198 w_k1_AXP_s104
                                 + 1.67903 w_k1_DIS_s104
                                 + 1.38576 w_k1_FDX_s104
                                 + 0.997216 w_k1_IBM_s104
                                 + 1.03909 w_k1_INTC_s104
                                 + 1.22211 w_k1_KO_s104 + 1.163 w_k1_MAT_s104
                                 + 1.30456 w_k1_MSFT_s104
                                 + 1.05192 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.26344 w_k1_ADM_s104
                                 + 1.05091 w_k1_AXP_s104
                                 + 0.829087 w_k1_DIS_s104
                                 + 0.927124 w_k1_FDX_s104
                                 + 0.948321 w_k1_IBM_s104
                                 + 0.912535 w_k1_INTC_s104
                                 + 0.900747 w_k1_KO_s104
                                 + 0.898381 w_k1_MAT_s104
                                 + 0.982482 w_k1_MSFT_s104
                                 + 0.954494 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.853775 w_k1_ADM_s104
                                 + 0.790688 w_k1_AXP_s104
                                 + 0.696034 w_k1_DIS_s104
                                 + 0.864966 w_k1_FDX_s104
                                 + 0.986328 w_k1_IBM_s104
                                 + 0.916063 w_k1_INTC_s104
                                 + 0.811762 w_k1_KO_s104
                                 + 0.672881 w_k1_MAT_s104
                                 + 0.840544 w_k1_MSFT_s104
                                 + 0.851013 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.10426 w_k1_ADM_s105
                                 + 1.01683 w_k1_AXP_s105
                                 + 1.27052 w_k1_DIS_s105
                                 + 1.13452 w_k1_FDX_s105
                                 + 0.91631 w_k1_IBM_s105
                                 + 1.06573 w_k1_INTC_s105
                                 + 1.07181 w_k1_KO_s105 + 1.10179 w_k1_MAT_s105
                                 + 1.28167 w_k1_MSFT_s105
                                 + 1.02446 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.07825 w_k1_ADM_s105
                                 + 1.42208 w_k1_AXP_s105
                                 + 1.73086 w_k1_DIS_s105
                                 + 1.21686 w_k1_FDX_s105
                                 + 1.60053 w_k1_IBM_s105
                                 + 1.27619 w_k1_INTC_s105
                                 + 1.37024 w_k1_KO_s105 + 1.37978 w_k1_MAT_s105
                                 + 1.2814 w_k1_MSFT_s105 + 1.15679 w_k1_PG_s105
                                 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.39728 w_k1_ADM_s105
                                 + 1.30412 w_k1_AXP_s105
                                 + 1.13452 w_k1_DIS_s105
                                 + 1.20378 w_k1_FDX_s105
                                 + 0.753635 w_k1_IBM_s105
                                 + 1.1024 w_k1_INTC_s105 + 1.02233 w_k1_KO_s105
                                 + 0.829097 w_k1_MAT_s105
                                 + 1.13365 w_k1_MSFT_s105
                                 + 1.13482 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.895014 w_k1_ADM_s105
                                 + 0.621622 w_k1_AXP_s105
                                 + 0.628664 w_k1_DIS_s105
                                 + 0.692313 w_k1_FDX_s105
                                 + 0.782303 w_k1_IBM_s105
                                 + 0.609398 w_k1_INTC_s105
                                 + 0.814654 w_k1_KO_s105
                                 + 0.713281 w_k1_MAT_s105
                                 + 0.796017 w_k1_MSFT_s105
                                 + 0.795858 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.33795 w_k1_ADM_s105
                                 + 1.30198 w_k1_AXP_s105
                                 + 1.22568 w_k1_DIS_s105
                                 + 1.06972 w_k1_FDX_s105
                                 + 1.19068 w_k1_IBM_s105
                                 + 0.974728 w_k1_INTC_s105
                                 + 1.10392 w_k1_KO_s105 + 1.06167 w_k1_MAT_s105
                                 + 1.22473 w_k1_MSFT_s105
                                 + 1.10353 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.02604 w_k1_ADM_s105
                                 + 0.785698 w_k1_AXP_s105
                                 + 0.917085 w_k1_DIS_s105
                                 + 0.895199 w_k1_FDX_s105
                                 + 0.705279 w_k1_IBM_s105
                                 + 0.926015 w_k1_INTC_s105
                                 + 0.884981 w_k1_KO_s105
                                 + 0.676674 w_k1_MAT_s105
                                 + 1.01903 w_k1_MSFT_s105
                                 + 0.901257 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.829254 w_k1_ADM_s105
                                 + 1.12649 w_k1_AXP_s105
                                 + 0.879306 w_k1_DIS_s105
                                 + 0.919184 w_k1_FDX_s105
                                 + 1.06278 w_k1_IBM_s105
                                 + 0.83722 w_k1_INTC_s105
                                 + 0.92429 w_k1_KO_s105 + 1.21184 w_k1_MAT_s105
                                 + 0.85961 w_k1_MSFT_s105
                                 + 0.976951 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.979758 w_k1_ADM_s105
                                 + 0.884589 w_k1_AXP_s105
                                 + 0.642848 w_k1_DIS_s105
                                 + 0.848513 w_k1_FDX_s105
                                 + 0.753502 w_k1_IBM_s105
                                 + 0.777376 w_k1_INTC_s105
                                 + 0.762227 w_k1_KO_s105
                                 + 0.706036 w_k1_MAT_s105
                                 + 0.789332 w_k1_MSFT_s105
                                 + 0.793163 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.13893 w_k1_ADM_s106
                                 + 1.01255 w_k1_AXP_s106
                                 + 1.01852 w_k1_DIS_s106
                                 + 1.05533 w_k1_FDX_s106
                                 + 1.02772 w_k1_IBM_s106
                                 + 0.991351 w_k1_INTC_s106
                                 + 1.08447 w_k1_KO_s106
                                 + 0.974566 w_k1_MAT_s106
                                 + 0.987802 w_k1_MSFT_s106
                                 + 1.01752 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.808231 w_k1_ADM_s106
                                 + 0.717283 w_k1_AXP_s106
                                 + 0.695681 w_k1_DIS_s106
                                 + 0.759075 w_k1_FDX_s106
                                 + 0.78664 w_k1_IBM_s106
                                 + 0.880593 w_k1_INTC_s106
                                 + 0.892856 w_k1_KO_s106
                                 + 0.736103 w_k1_MAT_s106
                                 + 0.773634 w_k1_MSFT_s106
                                 + 0.798946 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.20247 w_k1_ADM_s106
                                 + 1.11504 w_k1_AXP_s106
                                 + 1.38029 w_k1_DIS_s106
                                 + 1.18218 w_k1_FDX_s106
                                 + 0.954309 w_k1_IBM_s106
                                 + 1.43798 w_k1_INTC_s106
                                 + 1.15838 w_k1_KO_s106
                                 + 0.801693 w_k1_MAT_s106
                                 + 1.21275 w_k1_MSFT_s106
                                 + 1.11332 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.5515 w_k1_ADM_s106
                                 + 1.62859 w_k1_AXP_s106
                                 + 1.60258 w_k1_DIS_s106
                                 + 1.47845 w_k1_FDX_s106
                                 + 1.20945 w_k1_IBM_s106
                                 + 1.22987 w_k1_INTC_s106
                                 + 1.41501 w_k1_KO_s106 + 1.54415 w_k1_MAT_s106
                                 + 1.18893 w_k1_MSFT_s106
                                 + 1.26097 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.974696 w_k1_ADM_s106
                                 + 1.08946 w_k1_AXP_s106
                                 + 1.32287 w_k1_DIS_s106
                                 + 1.05642 w_k1_FDX_s106 + 1.5008 w_k1_IBM_s106
                                 + 1.1761 w_k1_INTC_s106 + 1.23532 w_k1_KO_s106
                                 + 1.5277 w_k1_MAT_s106
                                 + 1.00007 w_k1_MSFT_s106
                                 + 0.981029 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.92122 w_k1_ADM_s106
                                 + 0.872387 w_k1_AXP_s106
                                 + 0.823648 w_k1_DIS_s106
                                 + 0.909128 w_k1_FDX_s106
                                 + 0.818386 w_k1_IBM_s106
                                 + 0.955818 w_k1_INTC_s106
                                 + 0.832112 w_k1_KO_s106
                                 + 0.623868 w_k1_MAT_s106
                                 + 0.847695 w_k1_MSFT_s106
                                 + 0.932679 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.744604 w_k1_ADM_s106
                                 + 0.820216 w_k1_AXP_s106
                                 + 0.813681 w_k1_DIS_s106
                                 + 0.808435 w_k1_FDX_s106
                                 + 1.15703 w_k1_IBM_s106
                                 + 0.902081 w_k1_INTC_s106
                                 + 0.982658 w_k1_KO_s106
                                 + 0.973425 w_k1_MAT_s106
                                 + 0.872655 w_k1_MSFT_s106
                                 + 0.932508 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.917093 w_k1_ADM_s106
                                 + 1.17147 w_k1_AXP_s106
                                 + 1.04806 w_k1_DIS_s106
                                 + 1.01416 w_k1_FDX_s106
                                 + 0.826343 w_k1_IBM_s106
                                 + 0.90442 w_k1_INTC_s106
                                 + 0.858951 w_k1_KO_s106
                                 + 0.83523 w_k1_MAT_s106
                                 + 0.915105 w_k1_MSFT_s106
                                 + 0.864429 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.740197 w_k1_ADM_s107
                                 + 0.730404 w_k1_AXP_s107
                                 + 0.61737 w_k1_DIS_s107
                                 + 0.870288 w_k1_FDX_s107
                                 + 0.618659 w_k1_IBM_s107
                                 + 1.08512 w_k1_INTC_s107
                                 + 0.766967 w_k1_KO_s107
                                 + 0.553577 w_k1_MAT_s107
                                 + 0.892379 w_k1_MSFT_s107
                                 + 0.865094 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.956435 w_k1_ADM_s107
                                 + 1.05575 w_k1_AXP_s107
                                 + 1.07259 w_k1_DIS_s107
                                 + 1.13169 w_k1_FDX_s107
                                 + 0.836429 w_k1_IBM_s107
                                 + 1.10691 w_k1_INTC_s107
                                 + 1.01674 w_k1_KO_s107
                                 + 0.861312 w_k1_MAT_s107
                                 + 1.05467 w_k1_MSFT_s107
                                 + 0.948977 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.842157 w_k1_ADM_s107
                                 + 0.755883 w_k1_AXP_s107
                                 + 0.684622 w_k1_DIS_s107
                                 + 0.781022 w_k1_FDX_s107
                                 + 0.955815 w_k1_IBM_s107
                                 + 0.953611 w_k1_INTC_s107
                                 + 0.857129 w_k1_KO_s107
                                 + 0.87572 w_k1_MAT_s107
                                 + 0.825949 w_k1_MSFT_s107
                                 + 0.889423 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.921577 w_k1_ADM_s107
                                 + 0.919046 w_k1_AXP_s107
                                 + 0.87296 w_k1_DIS_s107
                                 + 0.901064 w_k1_FDX_s107
                                 + 1.1275 w_k1_IBM_s107
                                 + 1.00703 w_k1_INTC_s107 + 1.1202 w_k1_KO_s107
                                 + 1.18231 w_k1_MAT_s107
                                 + 1.01937 w_k1_MSFT_s107
                                 + 1.03763 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.35824 w_k1_ADM_s107
                                 + 1.45653 w_k1_AXP_s107
                                 + 1.42561 w_k1_DIS_s107
                                 + 1.46902 w_k1_FDX_s107
                                 + 0.824187 w_k1_IBM_s107
                                 + 1.58614 w_k1_INTC_s107
                                 + 1.19238 w_k1_KO_s107 + 0.87599 w_k1_MAT_s107
                                 + 1.4236 w_k1_MSFT_s107 + 1.13164 w_k1_PG_s107
                                 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.2311 w_k1_ADM_s107
                                 + 1.31297 w_k1_AXP_s107
                                 + 1.12892 w_k1_DIS_s107
                                 + 1.31131 w_k1_FDX_s107
                                 + 1.26434 w_k1_IBM_s107
                                 + 1.36267 w_k1_INTC_s107
                                 + 1.24636 w_k1_KO_s107 + 1.23642 w_k1_MAT_s107
                                 + 1.17355 w_k1_MSFT_s107
                                 + 1.10407 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.925986 w_k1_ADM_s107
                                 + 1.23295 w_k1_AXP_s107 + 1.4587 w_k1_DIS_s107
                                 + 1.16468 w_k1_FDX_s107
                                 + 1.60261 w_k1_IBM_s107
                                 + 1.16635 w_k1_INTC_s107
                                 + 1.42467 w_k1_KO_s107 + 1.33951 w_k1_MAT_s107
                                 + 1.24406 w_k1_MSFT_s107
                                 + 1.10759 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.01128 w_k1_ADM_s107
                                 + 0.940923 w_k1_AXP_s107
                                 + 1.15498 w_k1_DIS_s107
                                 + 1.16397 w_k1_FDX_s107
                                 + 0.596418 w_k1_IBM_s107
                                 + 1.09373 w_k1_INTC_s107
                                 + 0.999148 w_k1_KO_s107
                                 + 0.613068 w_k1_MAT_s107
                                 + 1.20946 w_k1_MSFT_s107
                                 + 0.942634 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.01643 w_k1_ADM_s110
                                 + 0.826558 w_k1_AXP_s110
                                 + 1.03689 w_k1_DIS_s110
                                 + 0.904232 w_k1_FDX_s110
                                 + 1.00537 w_k1_IBM_s110
                                 + 0.986566 w_k1_INTC_s110
                                 + 1.03698 w_k1_KO_s110
                                 + 0.791551 w_k1_MAT_s110
                                 + 1.15105 w_k1_MSFT_s110
                                 + 1.09319 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.18431 w_k1_ADM_s110
                                 + 1.16914 w_k1_AXP_s110
                                 + 1.41043 w_k1_DIS_s110
                                 + 1.25178 w_k1_FDX_s110
                                 + 1.22089 w_k1_IBM_s110
                                 + 1.22806 w_k1_INTC_s110 + 1.2062 w_k1_KO_s110
                                 + 1.30768 w_k1_MAT_s110
                                 + 1.21175 w_k1_MSFT_s110
                                 + 1.12946 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.00818 w_k1_ADM_s110
                                 + 1.17391 w_k1_AXP_s110
                                 + 0.864823 w_k1_DIS_s110
                                 + 1.06553 w_k1_FDX_s110
                                 + 1.05623 w_k1_IBM_s110
                                 + 1.10587 w_k1_INTC_s110
                                 + 0.952675 w_k1_KO_s110
                                 + 1.16124 w_k1_MAT_s110
                                 + 1.03982 w_k1_MSFT_s110
                                 + 1.08632 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.03887 w_k1_ADM_s110
                                 + 1.25426 w_k1_AXP_s110
                                 + 1.37425 w_k1_DIS_s110
                                 + 1.13167 w_k1_FDX_s110
                                 + 1.61714 w_k1_IBM_s110
                                 + 1.07717 w_k1_INTC_s110
                                 + 1.34656 w_k1_KO_s110 + 1.50654 w_k1_MAT_s110
                                 + 1.02843 w_k1_MSFT_s110
                                 + 1.09308 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.08516 w_k1_ADM_s110
                                 + 1.25932 w_k1_AXP_s110 + 1.487 w_k1_DIS_s110
                                 + 1.34215 w_k1_FDX_s110
                                 + 0.830559 w_k1_IBM_s110
                                 + 1.25633 w_k1_INTC_s110
                                 + 1.12382 w_k1_KO_s110
                                 + 0.991886 w_k1_MAT_s110
                                 + 1.38401 w_k1_MSFT_s110
                                 + 1.34195 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.03722 w_k1_ADM_s110
                                 + 0.855995 w_k1_AXP_s110
                                 + 0.844205 w_k1_DIS_s110
                                 + 0.980437 w_k1_FDX_s110
                                 + 0.572484 w_k1_IBM_s110
                                 + 1.14189 w_k1_INTC_s110
                                 + 0.726906 w_k1_KO_s110
                                 + 0.524334 w_k1_MAT_s110
                                 + 1.02484 w_k1_MSFT_s110
                                 + 0.973051 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.827135 w_k1_ADM_s110
                                 + 0.712149 w_k1_AXP_s110
                                 + 0.766583 w_k1_DIS_s110
                                 + 0.862193 w_k1_FDX_s110
                                 + 0.808167 w_k1_IBM_s110
                                 + 0.974981 w_k1_INTC_s110
                                 + 0.885812 w_k1_KO_s110
                                 + 0.842951 w_k1_MAT_s110
                                 + 1.05763 w_k1_MSFT_s110
                                 + 1.00636 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.01702 w_k1_ADM_s110
                                 + 0.93437 w_k1_AXP_s110
                                 + 1.32305 w_k1_DIS_s110
                                 + 1.13427 w_k1_FDX_s110
                                 + 0.909495 w_k1_IBM_s110
                                 + 1.23716 w_k1_INTC_s110
                                 + 1.13235 w_k1_KO_s110
                                 + 0.758565 w_k1_MAT_s110
                                 + 1.10318 w_k1_MSFT_s110
                                 + 1.12005 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.979749 w_k1_ADM_s111
                                 + 1.20558 w_k1_AXP_s111
                                 + 1.39697 w_k1_DIS_s111
                                 + 1.21254 w_k1_FDX_s111
                                 + 1.00001 w_k1_IBM_s111
                                 + 1.14315 w_k1_INTC_s111
                                 + 1.03568 w_k1_KO_s111 + 1.16461 w_k1_MAT_s111
                                 + 1.27226 w_k1_MSFT_s111 + 1.1648 w_k1_PG_s111
                                 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.73524 w_k1_ADM_s111
                                 + 0.600422 w_k1_AXP_s111
                                 + 0.618198 w_k1_DIS_s111
                                 + 0.750433 w_k1_FDX_s111
                                 + 0.726389 w_k1_IBM_s111
                                 + 0.89168 w_k1_INTC_s111
                                 + 0.750891 w_k1_KO_s111
                                 + 0.822627 w_k1_MAT_s111
                                 + 0.942801 w_k1_MSFT_s111
                                 + 0.786557 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.14956 w_k1_ADM_s111
                                 + 0.684775 w_k1_AXP_s111
                                 + 0.972628 w_k1_DIS_s111
                                 + 1.11452 w_k1_FDX_s111
                                 + 0.75482 w_k1_IBM_s111
                                 + 1.03099 w_k1_INTC_s111
                                 + 0.929549 w_k1_KO_s111
                                 + 0.697465 w_k1_MAT_s111
                                 + 1.2742 w_k1_MSFT_s111 + 1.00246 w_k1_PG_s111
                                 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.719976 w_k1_ADM_s111
                                 + 0.883177 w_k1_AXP_s111
                                 + 0.708819 w_k1_DIS_s111
                                 + 0.824147 w_k1_FDX_s111
                                 + 0.955537 w_k1_IBM_s111
                                 + 0.995832 w_k1_INTC_s111
                                 + 0.828072 w_k1_KO_s111
                                 + 1.23584 w_k1_MAT_s111
                                 + 0.858593 w_k1_MSFT_s111
                                 + 0.864639 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.1818 w_k1_ADM_s111
                                 + 1.20518 w_k1_AXP_s111
                                 + 1.36488 w_k1_DIS_s111
                                 + 1.13468 w_k1_FDX_s111
                                 + 1.48884 w_k1_IBM_s111
                                 + 1.2443 w_k1_INTC_s111 + 1.60728 w_k1_KO_s111
                                 + 1.73307 w_k1_MAT_s111
                                 + 1.29401 w_k1_MSFT_s111
                                 + 1.28483 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.753967 w_k1_ADM_s111
                                 + 1.06826 w_k1_AXP_s111
                                 + 1.04341 w_k1_DIS_s111
                                 + 0.895702 w_k1_FDX_s111
                                 + 1.18469 w_k1_IBM_s111
                                 + 1.05167 w_k1_INTC_s111
                                 + 1.02184 w_k1_KO_s111 + 1.09741 w_k1_MAT_s111
                                 + 0.879303 w_k1_MSFT_s111
                                 + 0.935272 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.85571 w_k1_ADM_s111
                                 + 0.929074 w_k1_AXP_s111
                                 + 0.808881 w_k1_DIS_s111
                                 + 0.977268 w_k1_FDX_s111
                                 + 0.760814 w_k1_IBM_s111
                                 + 1.05099 w_k1_INTC_s111
                                 + 0.796232 w_k1_KO_s111
                                 + 0.995655 w_k1_MAT_s111
                                 + 1.0812 w_k1_MSFT_s111 + 0.96165 w_k1_PG_s111
                                 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.12385 w_k1_ADM_s111
                                 + 1.64233 w_k1_AXP_s111
                                 + 1.37159 w_k1_DIS_s111
                                 + 1.07613 w_k1_FDX_s111
                                 + 1.65182 w_k1_IBM_s111
                                 + 1.03862 w_k1_INTC_s111
                                 + 1.30021 w_k1_KO_s111 + 2.82413 w_k1_MAT_s111
                                 + 1.20557 w_k1_MSFT_s111
                                 + 1.17628 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.694927 w_k1_ADM_s112
                                 + 0.722992 w_k1_AXP_s112
                                 + 0.69883 w_k1_DIS_s112
                                 + 0.726622 w_k1_FDX_s112
                                 + 0.760905 w_k1_IBM_s112
                                 + 0.70956 w_k1_INTC_s112
                                 + 0.824271 w_k1_KO_s112
                                 + 0.689786 w_k1_MAT_s112
                                 + 0.776028 w_k1_MSFT_s112
                                 + 0.765833 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.986448 w_k1_ADM_s112
                                 + 1.15684 w_k1_AXP_s112
                                 + 1.14107 w_k1_DIS_s112
                                 + 1.03427 w_k1_FDX_s112
                                 + 0.854434 w_k1_IBM_s112
                                 + 0.889571 w_k1_INTC_s112
                                 + 0.993518 w_k1_KO_s112
                                 + 0.846177 w_k1_MAT_s112
                                 + 1.01449 w_k1_MSFT_s112
                                 + 0.965239 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.26818 w_k1_ADM_s112
                                 + 1.40824 w_k1_AXP_s112
                                 + 1.66367 w_k1_DIS_s112
                                 + 1.28902 w_k1_FDX_s112 + 1.2592 w_k1_IBM_s112
                                 + 1.04723 w_k1_INTC_s112
                                 + 1.33085 w_k1_KO_s112 + 1.17225 w_k1_MAT_s112
                                 + 1.21027 w_k1_MSFT_s112
                                 + 1.17965 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.82349 w_k1_ADM_s112
                                 + 0.881615 w_k1_AXP_s112
                                 + 0.724081 w_k1_DIS_s112
                                 + 0.877583 w_k1_FDX_s112
                                 + 0.861031 w_k1_IBM_s112
                                 + 0.884983 w_k1_INTC_s112
                                 + 0.907433 w_k1_KO_s112
                                 + 0.958313 w_k1_MAT_s112
                                 + 0.796316 w_k1_MSFT_s112
                                 + 0.914447 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.740746 w_k1_ADM_s112
                                 + 1.25094 w_k1_AXP_s112
                                 + 1.03419 w_k1_DIS_s112
                                 + 0.95614 w_k1_FDX_s112
                                 + 1.34585 w_k1_IBM_s112
                                 + 0.939981 w_k1_INTC_s112
                                 + 1.13482 w_k1_KO_s112 + 1.44702 w_k1_MAT_s112
                                 + 0.900042 w_k1_MSFT_s112
                                 + 1.08773 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.0927 w_k1_ADM_s112
                                 + 1.17178 w_k1_AXP_s112
                                 + 1.13063 w_k1_DIS_s112
                                 + 1.14159 w_k1_FDX_s112
                                 + 0.850139 w_k1_IBM_s112
                                 + 1.00248 w_k1_INTC_s112
                                 + 1.08659 w_k1_KO_s112 + 1.02352 w_k1_MAT_s112
                                 + 1.09715 w_k1_MSFT_s112
                                 + 1.03866 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.864561 w_k1_ADM_s112
                                 + 1.04755 w_k1_AXP_s112
                                 + 0.930705 w_k1_DIS_s112
                                 + 0.807361 w_k1_FDX_s112
                                 + 1.04364 w_k1_IBM_s112
                                 + 0.811292 w_k1_INTC_s112
                                 + 0.936286 w_k1_KO_s112
                                 + 0.957734 w_k1_MAT_s112
                                 + 0.762249 w_k1_MSFT_s112
                                 + 0.852002 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.771299 w_k1_ADM_s112
                                 + 0.794175 w_k1_AXP_s112
                                 + 0.907967 w_k1_DIS_s112
                                 + 0.85829 w_k1_FDX_s112
                                 + 0.812335 w_k1_IBM_s112
                                 + 0.868807 w_k1_INTC_s112
                                 + 0.884341 w_k1_KO_s112
                                 + 0.657495 w_k1_MAT_s112
                                 + 0.951175 w_k1_MSFT_s112
                                 + 0.991753 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.36596 w_k1_ADM_s113
                                 + 1.2956 w_k1_AXP_s113 + 1.20735 w_k1_DIS_s113
                                 + 1.38128 w_k1_FDX_s113
                                 + 0.848207 w_k1_IBM_s113
                                 + 1.30145 w_k1_INTC_s113
                                 + 0.972306 w_k1_KO_s113
                                 + 0.677902 w_k1_MAT_s113
                                 + 1.26211 w_k1_MSFT_s113
                                 + 1.10047 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.926539 w_k1_ADM_s113
                                 + 1.32972 w_k1_AXP_s113
                                 + 1.18671 w_k1_DIS_s113 + 1.1529 w_k1_FDX_s113
                                 + 0.968637 w_k1_IBM_s113
                                 + 1.03533 w_k1_INTC_s113
                                 + 0.982337 w_k1_KO_s113
                                 + 1.75581 w_k1_MAT_s113
                                 + 1.19925 w_k1_MSFT_s113
                                 + 1.04627 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.28405 w_k1_ADM_s113
                                 + 1.56141 w_k1_AXP_s113
                                 + 0.858274 w_k1_DIS_s113
                                 + 1.07724 w_k1_FDX_s113
                                 + 0.939639 w_k1_IBM_s113
                                 + 1.16963 w_k1_INTC_s113
                                 + 0.751157 w_k1_KO_s113
                                 + 0.848523 w_k1_MAT_s113
                                 + 0.906598 w_k1_MSFT_s113
                                 + 0.963051 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.902914 w_k1_ADM_s113
                                 + 0.952092 w_k1_AXP_s113
                                 + 0.787241 w_k1_DIS_s113
                                 + 0.822921 w_k1_FDX_s113
                                 + 1.16569 w_k1_IBM_s113
                                 + 0.948838 w_k1_INTC_s113
                                 + 1.0005 w_k1_KO_s113 + 1.31357 w_k1_MAT_s113
                                 + 0.953417 w_k1_MSFT_s113
                                 + 1.06904 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.949336 w_k1_ADM_s113
                                 + 0.824252 w_k1_AXP_s113
                                 + 0.880831 w_k1_DIS_s113
                                 + 0.940187 w_k1_FDX_s113
                                 + 0.856658 w_k1_IBM_s113
                                 + 0.919869 w_k1_INTC_s113
                                 + 0.810598 w_k1_KO_s113
                                 + 1.05896 w_k1_MAT_s113
                                 + 0.882068 w_k1_MSFT_s113
                                 + 0.936708 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.1463 w_k1_ADM_s113
                                 + 0.954657 w_k1_AXP_s113
                                 + 0.926194 w_k1_DIS_s113
                                 + 0.983401 w_k1_FDX_s113
                                 + 1.0666 w_k1_IBM_s113
                                 + 0.94882 w_k1_INTC_s113
                                 + 0.898171 w_k1_KO_s113
                                 + 1.11146 w_k1_MAT_s113
                                 + 0.87255 w_k1_MSFT_s113
                                 + 0.88362 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.12533 w_k1_ADM_s113
                                 + 1.90771 w_k1_AXP_s113 + 1.4197 w_k1_DIS_s113
                                 + 1.17591 w_k1_FDX_s113
                                 + 1.62912 w_k1_IBM_s113
                                 + 1.39519 w_k1_INTC_s113
                                 + 1.21106 w_k1_KO_s113 + 1.6217 w_k1_MAT_s113
                                 + 1.11533 w_k1_MSFT_s113
                                 + 1.14226 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.916279 w_k1_ADM_s113
                                 + 0.74847 w_k1_AXP_s113
                                 + 0.628103 w_k1_DIS_s113
                                 + 0.716064 w_k1_FDX_s113
                                 + 0.809689 w_k1_IBM_s113
                                 + 0.762872 w_k1_INTC_s113
                                 + 0.717533 w_k1_KO_s113 + 0.8378 w_k1_MAT_s113
                                 + 0.751629 w_k1_MSFT_s113
                                 + 0.708162 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.11043 w_k1_ADM_s114
                                 + 1.24562 w_k1_AXP_s114
                                 + 1.17586 w_k1_DIS_s114 + 1.1576 w_k1_FDX_s114
                                 + 1.18236 w_k1_IBM_s114
                                 + 1.05998 w_k1_INTC_s114
                                 + 1.13905 w_k1_KO_s114 + 1.56231 w_k1_MAT_s114
                                 + 1.04037 w_k1_MSFT_s114
                                 + 1.13062 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.11325 w_k1_ADM_s114
                                 + 1.06767 w_k1_AXP_s114
                                 + 0.994855 w_k1_DIS_s114
                                 + 1.0871 w_k1_FDX_s114 + 1.16266 w_k1_IBM_s114
                                 + 0.975176 w_k1_INTC_s114
                                 + 1.07346 w_k1_KO_s114 + 1.27795 w_k1_MAT_s114
                                 + 1.01754 w_k1_MSFT_s114
                                 + 1.06495 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.32199 w_k1_ADM_s114
                                 + 1.3168 w_k1_AXP_s114 + 1.72256 w_k1_DIS_s114
                                 + 1.5082 w_k1_FDX_s114 + 1.2785 w_k1_IBM_s114
                                 + 1.02447 w_k1_INTC_s114
                                 + 1.39058 w_k1_KO_s114 + 1.74631 w_k1_MAT_s114
                                 + 1.23794 w_k1_MSFT_s114
                                 + 1.28598 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.982972 w_k1_ADM_s114
                                 + 1.14117 w_k1_AXP_s114
                                 + 0.74595 w_k1_DIS_s114
                                 + 0.923923 w_k1_FDX_s114
                                 + 1.12265 w_k1_IBM_s114
                                 + 1.01658 w_k1_INTC_s114
                                 + 0.985166 w_k1_KO_s114
                                 + 1.15478 w_k1_MAT_s114
                                 + 0.871323 w_k1_MSFT_s114
                                 + 0.990858 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.826103 w_k1_ADM_s114
                                 + 0.710365 w_k1_AXP_s114
                                 + 0.740172 w_k1_DIS_s114
                                 + 0.812746 w_k1_FDX_s114
                                 + 0.933912 w_k1_IBM_s114
                                 + 0.825441 w_k1_INTC_s114
                                 + 0.958164 w_k1_KO_s114
                                 + 0.670811 w_k1_MAT_s114
                                 + 0.77915 w_k1_MSFT_s114
                                 + 0.824522 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.87674 w_k1_ADM_s114
                                 + 0.767873 w_k1_AXP_s114
                                 + 0.744367 w_k1_DIS_s114
                                 + 0.853065 w_k1_FDX_s114
                                 + 0.653102 w_k1_IBM_s114
                                 + 0.82768 w_k1_INTC_s114
                                 + 0.737896 w_k1_KO_s114
                                 + 0.59581 w_k1_MAT_s114
                                 + 0.795613 w_k1_MSFT_s114
                                 + 0.803363 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.158 w_k1_ADM_s114
                                 + 1.3036 w_k1_AXP_s114 + 1.38492 w_k1_DIS_s114
                                 + 1.05912 w_k1_FDX_s114 + 1.6451 w_k1_IBM_s114
                                 + 1.14109 w_k1_INTC_s114
                                 + 1.33642 w_k1_KO_s114 + 1.73986 w_k1_MAT_s114
                                 + 0.999742 w_k1_MSFT_s114
                                 + 1.19021 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.965992 w_k1_ADM_s114
                                 + 1.0961 w_k1_AXP_s114
                                 + 0.892446 w_k1_DIS_s114
                                 + 1.08786 w_k1_FDX_s114
                                 + 0.880357 w_k1_IBM_s114
                                 + 0.977341 w_k1_INTC_s114
                                 + 0.991393 w_k1_KO_s114
                                 + 0.908009 w_k1_MAT_s114
                                 + 0.939303 w_k1_MSFT_s114
                                 + 1.05826 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.863474 w_k1_ADM_s115
                                 + 0.931983 w_k1_AXP_s115
                                 + 0.963498 w_k1_DIS_s115
                                 + 0.900795 w_k1_FDX_s115
                                 + 1.18659 w_k1_IBM_s115
                                 + 0.93629 w_k1_INTC_s115
                                 + 0.924616 w_k1_KO_s115
                                 + 0.975404 w_k1_MAT_s115
                                 + 0.946093 w_k1_MSFT_s115
                                 + 0.933369 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.00546 w_k1_ADM_s115
                                 + 0.987453 w_k1_AXP_s115
                                 + 0.678605 w_k1_DIS_s115 + 0.956 w_k1_FDX_s115
                                 + 0.92146 w_k1_IBM_s115
                                 + 0.844924 w_k1_INTC_s115
                                 + 0.822938 w_k1_KO_s115
                                 + 0.968783 w_k1_MAT_s115
                                 + 0.851096 w_k1_MSFT_s115
                                 + 0.974232 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.912067 w_k1_ADM_s115
                                 + 0.868132 w_k1_AXP_s115
                                 + 0.882298 w_k1_DIS_s115
                                 + 0.931566 w_k1_FDX_s115
                                 + 0.690501 w_k1_IBM_s115
                                 + 0.88014 w_k1_INTC_s115
                                 + 0.734516 w_k1_KO_s115
                                 + 0.712282 w_k1_MAT_s115
                                 + 0.815447 w_k1_MSFT_s115
                                 + 0.87496 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.6185 w_k1_ADM_s115
                                 + 0.657482 w_k1_AXP_s115
                                 + 0.580372 w_k1_DIS_s115
                                 + 0.640128 w_k1_FDX_s115
                                 + 0.825586 w_k1_IBM_s115
                                 + 0.770811 w_k1_INTC_s115
                                 + 0.653399 w_k1_KO_s115
                                 + 0.614135 w_k1_MAT_s115
                                 + 0.567347 w_k1_MSFT_s115
                                 + 0.717407 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.17024 w_k1_ADM_s115
                                 + 1.45803 w_k1_AXP_s115
                                 + 1.24541 w_k1_DIS_s115
                                 + 1.13316 w_k1_FDX_s115
                                 + 1.11471 w_k1_IBM_s115
                                 + 1.3071 w_k1_INTC_s115 + 1.15559 w_k1_KO_s115
                                 + 1.03306 w_k1_MAT_s115
                                 + 1.17765 w_k1_MSFT_s115
                                 + 1.08377 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.14825 w_k1_ADM_s115
                                 + 1.32578 w_k1_AXP_s115
                                 + 1.05422 w_k1_DIS_s115
                                 + 1.13435 w_k1_FDX_s115
                                 + 0.862449 w_k1_IBM_s115
                                 + 0.927549 w_k1_INTC_s115
                                 + 0.803015 w_k1_KO_s115
                                 + 0.833622 w_k1_MAT_s115
                                 + 0.999855 w_k1_MSFT_s115
                                 + 0.945508 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.03498 w_k1_ADM_s115
                                 + 1.43851 w_k1_AXP_s115
                                 + 1.57301 w_k1_DIS_s115
                                 + 1.21657 w_k1_FDX_s115
                                 + 1.65016 w_k1_IBM_s115
                                 + 1.2926 w_k1_INTC_s115 + 1.33038 w_k1_KO_s115
                                 + 1.25548 w_k1_MAT_s115
                                 + 1.20601 w_k1_MSFT_s115
                                 + 1.11264 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.48596 w_k1_ADM_s115
                                 + 1.36027 w_k1_AXP_s115
                                 + 1.51356 w_k1_DIS_s115 + 1.2929 w_k1_FDX_s115
                                 + 1.02361 w_k1_IBM_s115
                                 + 1.13075 w_k1_INTC_s115
                                 + 1.09438 w_k1_KO_s115
                                 + 0.906948 w_k1_MAT_s115
                                 + 1.18809 w_k1_MSFT_s115
                                 + 1.24009 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.19301 w_k1_ADM_s116
                                 + 1.19795 w_k1_AXP_s116
                                 + 1.27906 w_k1_DIS_s116
                                 + 1.22399 w_k1_FDX_s116
                                 + 0.711338 w_k1_IBM_s116
                                 + 1.06445 w_k1_INTC_s116
                                 + 1.12169 w_k1_KO_s116 + 1.04006 w_k1_MAT_s116
                                 + 1.22136 w_k1_MSFT_s116 + 1.0435 w_k1_PG_s116
                                 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.835291 w_k1_ADM_s116
                                 + 1.24545 w_k1_AXP_s116
                                 + 1.27642 w_k1_DIS_s116
                                 + 0.975494 w_k1_FDX_s116
                                 + 1.58332 w_k1_IBM_s116
                                 + 1.28109 w_k1_INTC_s116
                                 + 1.50731 w_k1_KO_s116 + 2.00672 w_k1_MAT_s116
                                 + 1.1487 w_k1_MSFT_s116 + 1.01901 w_k1_PG_s116
                                 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.787954 w_k1_ADM_s116
                                 + 0.859493 w_k1_AXP_s116 + 1.144 w_k1_DIS_s116
                                 + 0.997401 w_k1_FDX_s116
                                 + 0.863309 w_k1_IBM_s116
                                 + 1.22846 w_k1_INTC_s116 + 1.0136 w_k1_KO_s116
                                 + 0.96431 w_k1_MAT_s116
                                 + 1.18743 w_k1_MSFT_s116
                                 + 0.974688 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.889251 w_k1_ADM_s116
                                 + 1.28287 w_k1_AXP_s116
                                 + 1.19285 w_k1_DIS_s116
                                 + 1.07844 w_k1_FDX_s116
                                 + 1.21965 w_k1_IBM_s116
                                 + 1.09565 w_k1_INTC_s116
                                 + 1.09363 w_k1_KO_s116 + 1.40704 w_k1_MAT_s116
                                 + 1.10979 w_k1_MSFT_s116
                                 + 1.07442 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.721402 w_k1_ADM_s116
                                 + 0.769874 w_k1_AXP_s116
                                 + 0.730474 w_k1_DIS_s116
                                 + 0.841608 w_k1_FDX_s116
                                 + 0.525725 w_k1_IBM_s116
                                 + 0.802429 w_k1_INTC_s116
                                 + 0.721418 w_k1_KO_s116
                                 + 0.594942 w_k1_MAT_s116
                                 + 0.814972 w_k1_MSFT_s116
                                 + 0.747842 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.883312 w_k1_ADM_s116
                                 + 0.895802 w_k1_AXP_s116
                                 + 0.79457 w_k1_DIS_s116
                                 + 0.85019 w_k1_FDX_s116
                                 + 0.910398 w_k1_IBM_s116
                                 + 0.797444 w_k1_INTC_s116
                                 + 0.788232 w_k1_KO_s116
                                 + 1.09481 w_k1_MAT_s116
                                 + 0.879375 w_k1_MSFT_s116
                                 + 0.82779 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.755542 w_k1_ADM_s116
                                 + 0.645943 w_k1_AXP_s116
                                 + 0.706477 w_k1_DIS_s116
                                 + 0.702365 w_k1_FDX_s116
                                 + 0.984099 w_k1_IBM_s116
                                 + 0.943037 w_k1_INTC_s116
                                 + 0.939707 w_k1_KO_s116
                                 + 0.756453 w_k1_MAT_s116
                                 + 0.729439 w_k1_MSFT_s116
                                 + 0.848579 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.894605 w_k1_ADM_s116
                                 + 1.14069 w_k1_AXP_s116
                                 + 0.821714 w_k1_DIS_s116
                                 + 0.97006 w_k1_FDX_s116
                                 + 0.802531 w_k1_IBM_s116
                                 + 0.897539 w_k1_INTC_s116
                                 + 0.892767 w_k1_KO_s116
                                 + 1.12787 w_k1_MAT_s116
                                 + 0.899433 w_k1_MSFT_s116
                                 + 0.945462 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.11449 w_k1_ADM_s117
                                 + 1.85568 w_k1_AXP_s117 + 1.2693 w_k1_DIS_s117
                                 + 1.26366 w_k1_FDX_s117
                                 + 1.12295 w_k1_IBM_s117
                                 + 1.15735 w_k1_INTC_s117
                                 + 0.97537 w_k1_KO_s117 + 1.95927 w_k1_MAT_s117
                                 + 1.09865 w_k1_MSFT_s117
                                 + 1.06136 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.718569 w_k1_ADM_s117
                                 + 0.603198 w_k1_AXP_s117
                                 + 0.866824 w_k1_DIS_s117
                                 + 0.791297 w_k1_FDX_s117
                                 + 0.960893 w_k1_IBM_s117
                                 + 0.982682 w_k1_INTC_s117
                                 + 0.961557 w_k1_KO_s117 + 0.6795 w_k1_MAT_s117
                                 + 0.869598 w_k1_MSFT_s117
                                 + 0.89625 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.01492 w_k1_ADM_s117
                                 + 1.08322 w_k1_AXP_s117
                                 + 1.36173 w_k1_DIS_s117
                                 + 1.11377 w_k1_FDX_s117
                                 + 1.20768 w_k1_IBM_s117
                                 + 1.05917 w_k1_INTC_s117
                                 + 1.20933 w_k1_KO_s117 + 1.05554 w_k1_MAT_s117
                                 + 1.27569 w_k1_MSFT_s117
                                 + 1.04201 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.849165 w_k1_ADM_s117
                                 + 0.889895 w_k1_AXP_s117
                                 + 0.908027 w_k1_DIS_s117
                                 + 0.881961 w_k1_FDX_s117
                                 + 1.7292 w_k1_IBM_s117
                                 + 0.991305 w_k1_INTC_s117
                                 + 1.21694 w_k1_KO_s117 + 1.96787 w_k1_MAT_s117
                                 + 0.960926 w_k1_MSFT_s117
                                 + 1.09932 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.923562 w_k1_ADM_s117
                                 + 1.01813 w_k1_AXP_s117
                                 + 1.05404 w_k1_DIS_s117
                                 + 1.09318 w_k1_FDX_s117
                                 + 0.666831 w_k1_IBM_s117
                                 + 1.37425 w_k1_INTC_s117
                                 + 0.858314 w_k1_KO_s117
                                 + 0.650053 w_k1_MAT_s117
                                 + 1.10589 w_k1_MSFT_s117
                                 + 1.03899 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.770637 w_k1_ADM_s117
                                 + 0.865276 w_k1_AXP_s117
                                 + 0.657083 w_k1_DIS_s117
                                 + 0.7739 w_k1_FDX_s117 + 0.81725 w_k1_IBM_s117
                                 + 0.914808 w_k1_INTC_s117
                                 + 0.757034 w_k1_KO_s117
                                 + 0.957688 w_k1_MAT_s117
                                 + 0.840426 w_k1_MSFT_s117
                                 + 0.892354 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.873334 w_k1_ADM_s117
                                 + 0.783803 w_k1_AXP_s117
                                 + 1.02476 w_k1_DIS_s117
                                 + 1.00374 w_k1_FDX_s117
                                 + 0.901682 w_k1_IBM_s117
                                 + 0.962587 w_k1_INTC_s117
                                 + 0.900408 w_k1_KO_s117
                                 + 0.828008 w_k1_MAT_s117
                                 + 1.03652 w_k1_MSFT_s117 + 1.0116 w_k1_PG_s117
                                 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.0519 w_k1_ADM_s117
                                 + 1.16954 w_k1_AXP_s117
                                 + 1.66579 w_k1_DIS_s117
                                 + 1.39636 w_k1_FDX_s117
                                 + 0.881142 w_k1_IBM_s117
                                 + 1.51861 w_k1_INTC_s117
                                 + 1.33595 w_k1_KO_s117 + 1.06169 w_k1_MAT_s117
                                 + 1.4928 w_k1_MSFT_s117 + 1.15181 w_k1_PG_s117
                                 >= 1
 k1_return_constraint:           0.0261848507498305 w_k1_ADM_s000
                                 + 0.0531270773098694 w_k1_ADM_s001
                                 + 0.0502873666009247 w_k1_ADM_s002
                                 + 0.0170512793810665 w_k1_ADM_s003
                                 + 0.0295712461296843 w_k1_ADM_s004
                                 + 0.0248479496303731 w_k1_ADM_s005
                                 + 0.0263553893267463 w_k1_ADM_s006
                                 + 0.0249387678835938 w_k1_ADM_s007
                                 + 0.045186618343761 w_k1_ADM_s010
                                 + 0.0107445517498907 w_k1_ADM_s011
                                 + 0.017598173564407 w_k1_ADM_s012
                                 + 0.0116139300296561 w_k1_ADM_s013
                                 + 0.029504518070629 w_k1_ADM_s014
                                 + 0.0414677182610424 w_k1_ADM_s015
                                 + 0.0128213997993541 w_k1_ADM_s016
                                 + 0.0226117660285581 w_k1_ADM_s017
                                 + 0.0383127042355935 w_k1_ADM_s100
                                 + 0.0332703449363821 w_k1_ADM_s101
                                 + 0.0876564474169644 w_k1_ADM_s102
                                 + 0.0604687080883882 w_k1_ADM_s103
                                 + 0.0340030432373541 w_k1_ADM_s104
                                 + 0.0539557198594514 w_k1_ADM_s105
                                 + 0.028663168345718 w_k1_ADM_s106
                                 + 0.0455747118009828 w_k1_ADM_s107
                                 + 0.0375794574721829 w_k1_ADM_s110
                                 + 0.00899255063760962 w_k1_ADM_s111
                                 + 0.0214760219050214 w_k1_ADM_s112
                                 + 0.0101726307954217 w_k1_ADM_s113
                                 + 0.0218232235615945 w_k1_ADM_s114
                                 + 0.0170342222662878 w_k1_ADM_s115
                                 + 0.0304194176081029 w_k1_ADM_s116
                                 + 0.0240125378683708 w_k1_ADM_s117
                                 + 0.0221374704059872 w_k1_AXP_s000
                                 + 0.0569215049973677 w_k1_AXP_s001
                                 + 0.056422898310481 w_k1_AXP_s002
                                 + 0.018365733404521 w_k1_AXP_s003
                                 + 0.0284182253180144 w_k1_AXP_s004
                                 + 0.0289344757175928 w_k1_AXP_s005
                                 + 0.0226620214456187 w_k1_AXP_s006
                                 + 0.027278311911782 w_k1_AXP_s007
                                 + 0.0416410069993655 w_k1_AXP_s010
                                 + 0.0100055302342771 w_k1_AXP_s011
                                 + 0.0204674069894833 w_k1_AXP_s012
                                 + 0.0111737401728912 w_k1_AXP_s013
                                 + 0.0399298007224542 w_k1_AXP_s014
                                 + 0.0417236783757288 w_k1_AXP_s015
                                 + 0.0129368662514245 w_k1_AXP_s016
                                 + 0.0256032421201616 w_k1_AXP_s017
                                 + 0.0355616790040176 w_k1_AXP_s100
                                 + 0.0349109574856533 w_k1_AXP_s101
                                 + 0.0934674633694081 w_k1_AXP_s102
                                 + 0.0632937686267897 w_k1_AXP_s103
                                 + 0.0319616765368949 w_k1_AXP_s104
                                 + 0.0533395178524181 w_k1_AXP_s105
                                 + 0.0290728086252408 w_k1_AXP_s106
                                 + 0.0482233084009356 w_k1_AXP_s107
                                 + 0.0366819668674214 w_k1_AXP_s110
                                 + 0.0104617848772879 w_k1_AXP_s111
                                 + 0.026051019220721 w_k1_AXP_s112
                                 + 0.0107612316537376 w_k1_AXP_s113
                                 + 0.0220056019637852 w_k1_AXP_s114
                                 + 0.0185795240035123 w_k1_AXP_s115
                                 + 0.0361846647443612 w_k1_AXP_s116
                                 + 0.0263214518868379 w_k1_AXP_s117
                                 + 0.0213515329974683 w_k1_DIS_s000
                                 + 0.0559681409211082 w_k1_DIS_s001
                                 + 0.0580344003754417 w_k1_DIS_s002
                                 + 0.0192714960566133 w_k1_DIS_s003
                                 + 0.0295469822803092 w_k1_DIS_s004
                                 + 0.0301566011412514 w_k1_DIS_s005
                                 + 0.0250116948528535 w_k1_DIS_s006
                                 + 0.0264257530953748 w_k1_DIS_s007
                                 + 0.0461631403954541 w_k1_DIS_s010
                                 + 0.0104870285637996 w_k1_DIS_s011
                                 + 0.0216116474951145 w_k1_DIS_s012
                                 + 0.0114531971860193 w_k1_DIS_s013
                                 + 0.0330726630392362 w_k1_DIS_s014
                                 + 0.0398269237322872 w_k1_DIS_s015
                                 + 0.0131022844234696 w_k1_DIS_s016
                                 + 0.0233337457502988 w_k1_DIS_s017
                                 + 0.0361140050791266 w_k1_DIS_s100
                                 + 0.0351082372580832 w_k1_DIS_s101
                                 + 0.0916110357447796 w_k1_DIS_s102
                                 + 0.0625007359805692 w_k1_DIS_s103
                                 + 0.0332856600583786 w_k1_DIS_s104
                                 + 0.0532548587593201 w_k1_DIS_s105
                                 + 0.0305207738389859 w_k1_DIS_s106
                                 + 0.0479135521516575 w_k1_DIS_s107
                                 + 0.0405735781668628 w_k1_DIS_s110
                                 + 0.0104402624099727 w_k1_DIS_s111
                                 + 0.0257895747142276 w_k1_DIS_s112
                                 + 0.00922780319646981 w_k1_DIS_s113
                                 + 0.0206917089649607 w_k1_DIS_s114
                                 + 0.0179032515326663 w_k1_DIS_s115
                                 + 0.0355288792148239 w_k1_DIS_s116
                                 + 0.0274637443839984 w_k1_DIS_s117
                                 + 0.0203861640153595 w_k1_FDX_s000
                                 + 0.0524812542177358 w_k1_FDX_s001
                                 + 0.0548081661908717 w_k1_FDX_s002
                                 + 0.0186533130074896 w_k1_FDX_s003
                                 + 0.0284113986594172 w_k1_FDX_s004
                                 + 0.0297601092630159 w_k1_FDX_s005
                                 + 0.023711858319888 w_k1_FDX_s006
                                 + 0.0250684397933379 w_k1_FDX_s007
                                 + 0.043186710803344 w_k1_FDX_s010
                                 + 0.00989538087109696 w_k1_FDX_s011
                                 + 0.0200821900510902 w_k1_FDX_s012
                                 + 0.0107653617182734 w_k1_FDX_s013
                                 + 0.0342031803511712 w_k1_FDX_s014
                                 + 0.040211927762946 w_k1_FDX_s015
                                 + 0.0134939722710981 w_k1_FDX_s016
                                 + 0.0232117068073491 w_k1_FDX_s017
                                 + 0.0347398785169222 w_k1_FDX_s100
                                 + 0.0352697039872614 w_k1_FDX_s101
                                 + 0.0920594587279575 w_k1_FDX_s102
                                 + 0.0642926146380777 w_k1_FDX_s103
                                 + 0.0319948331867694 w_k1_FDX_s104
                                 + 0.0500554175549188 w_k1_FDX_s105
                                 + 0.0287148008187545 w_k1_FDX_s106
                                 + 0.0498363461570632 w_k1_FDX_s107
                                 + 0.0388392240691724 w_k1_FDX_s110
                                 + 0.0100473113761358 w_k1_FDX_s111
                                 + 0.0232477793316204 w_k1_FDX_s112
                                 + 0.00979084839570164 w_k1_FDX_s113
                                 + 0.022106874848484 w_k1_FDX_s114
                                 + 0.0171762978083294 w_k1_FDX_s115
                                 + 0.033556419511865 w_k1_FDX_s116
                                 + 0.0265765563646465 w_k1_FDX_s117
                                 + 0.0211711405150193 w_k1_IBM_s000
                                 + 0.0533895683215798 w_k1_IBM_s001
                                 + 0.0541631912410494 w_k1_IBM_s002
                                 + 0.0191458750365985 w_k1_IBM_s003
                                 + 0.0261152669223735 w_k1_IBM_s004
                                 + 0.0292139729186954 w_k1_IBM_s005
                                 + 0.0248244725583128 w_k1_IBM_s006
                                 + 0.0255367678209086 w_k1_IBM_s007
                                 + 0.045501284238567 w_k1_IBM_s010
                                 + 0.010307395203772 w_k1_IBM_s011
                                 + 0.021430538849911 w_k1_IBM_s012
                                 + 0.0111209274951703 w_k1_IBM_s013
                                 + 0.0331015885399947 w_k1_IBM_s014
                                 + 0.0409663434656957 w_k1_IBM_s015
                                 + 0.0132796101366675 w_k1_IBM_s016
                                 + 0.0240112041629027 w_k1_IBM_s017
                                 + 0.0345879345755275 w_k1_IBM_s100
                                 + 0.0338911548841728 w_k1_IBM_s101
                                 + 0.100647322837218 w_k1_IBM_s102
                                 + 0.0635291280779266 w_k1_IBM_s103
                                 + 0.0318008673177758 w_k1_IBM_s104
                                 + 0.0485735216693822 w_k1_IBM_s105
                                 + 0.0293032891384011 w_k1_IBM_s106
                                 + 0.0478310589886351 w_k1_IBM_s107
                                 + 0.0388603269501971 w_k1_IBM_s110
                                 + 0.0104769388718053 w_k1_IBM_s111
                                 + 0.0241413485734101 w_k1_IBM_s112
                                 + 0.00932750309368069 w_k1_IBM_s113
                                 + 0.0226395245744474 w_k1_IBM_s114
                                 + 0.0179968356325646 w_k1_IBM_s115
                                 + 0.0337779659897917 w_k1_IBM_s116
                                 + 0.0282468780230796 w_k1_IBM_s117
                                 + 0.0221955592603271 w_k1_INTC_s000
                                 + 0.0566397464722764 w_k1_INTC_s001
                                 + 0.0586130411752712 w_k1_INTC_s002
                                 + 0.0200192258137988 w_k1_INTC_s003
                                 + 0.0275250753488524 w_k1_INTC_s004
                                 + 0.0275599094254562 w_k1_INTC_s005
                                 + 0.0318518302941362 w_k1_INTC_s006
                                 + 0.0281194272108802 w_k1_INTC_s007
                                 + 0.0375556291670454 w_k1_INTC_s010
                                 + 0.0107338579219653 w_k1_INTC_s011
                                 + 0.023204896553804 w_k1_INTC_s012
                                 + 0.0103948549985068 w_k1_INTC_s013
                                 + 0.0325356103778289 w_k1_INTC_s014
                                 + 0.0433025158055213 w_k1_INTC_s015
                                 + 0.0123011540645236 w_k1_INTC_s016
                                 + 0.0212778161662205 w_k1_INTC_s017
                                 + 0.0363492584344667 w_k1_INTC_s100
                                 + 0.034427417669626 w_k1_INTC_s101
                                 + 0.100102268333886 w_k1_INTC_s102
                                 + 0.0582071126884839 w_k1_INTC_s103
                                 + 0.0319511958082545 w_k1_INTC_s104
                                 + 0.0477433935107812 w_k1_INTC_s105
                                 + 0.0298866484661967 w_k1_INTC_s106
                                 + 0.0537602485270349 w_k1_INTC_s107
                                 + 0.0407134614053244 w_k1_INTC_s110
                                 + 0.0107306955802273 w_k1_INTC_s111
                                 + 0.0211538034582037 w_k1_INTC_s112
                                 + 0.00991692373692919 w_k1_INTC_s113
                                 + 0.0207855084174175 w_k1_INTC_s114
                                 + 0.0172225434375086 w_k1_INTC_s115
                                 + 0.034800735259569 w_k1_INTC_s116
                                 + 0.0292514242890627 w_k1_INTC_s117
                                 + 0.022828511080105 w_k1_KO_s000
                                 + 0.0567933924214399 w_k1_KO_s001
                                 + 0.0533459301362749 w_k1_KO_s002
                                 + 0.0186448258212432 w_k1_KO_s003
                                 + 0.0277953636453582 w_k1_KO_s004
                                 + 0.0284053052527111 w_k1_KO_s005
                                 + 0.0249608393007055 w_k1_KO_s006
                                 + 0.0250071828647158 w_k1_KO_s007
                                 + 0.0399367990206431 w_k1_KO_s010
                                 + 0.0099682947092633 w_k1_KO_s011
                                 + 0.021300147684427 w_k1_KO_s012
                                 + 0.01020325940757 w_k1_KO_s013
                                 + 0.0359725790880684 w_k1_KO_s014
                                 + 0.0376926992207049 w_k1_KO_s015
                                 + 0.0145849684680918 w_k1_KO_s016
                                 + 0.026349499948329 w_k1_KO_s017
                                 + 0.030605340812517 w_k1_KO_s100
                                 + 0.0351207073796041 w_k1_KO_s101
                                 + 0.085607018208371 w_k1_KO_s102
                                 + 0.0546795979240143 w_k1_KO_s103
                                 + 0.0319670164688421 w_k1_KO_s104
                                 + 0.0496696742039642 w_k1_KO_s105
                                 + 0.0296637869267432 w_k1_KO_s106
                                 + 0.0509343845347418 w_k1_KO_s107
                                 + 0.0395025258544951 w_k1_KO_s110
                                 + 0.0100095165070691 w_k1_KO_s111
                                 + 0.0246108563676692 w_k1_KO_s112
                                 + 0.00855053812844062 w_k1_KO_s113
                                 + 0.0224231054875353 w_k1_KO_s114
                                 + 0.016059827813966 w_k1_KO_s115
                                 + 0.0350254009295081 w_k1_KO_s116
                                 + 0.0269806299419332 w_k1_KO_s117
                                 + 0.0179394724053559 w_k1_MAT_s000
                                 + 0.049276320055543 w_k1_MAT_s001
                                 + 0.0514728207471509 w_k1_MAT_s002
                                 + 0.0187365820973222 w_k1_MAT_s003
                                 + 0.0272266733038366 w_k1_MAT_s004
                                 + 0.0341187734167866 w_k1_MAT_s005
                                 + 0.0206480172864305 w_k1_MAT_s006
                                 + 0.0258738408977952 w_k1_MAT_s007
                                 + 0.0403809381278862 w_k1_MAT_s010
                                 + 0.00826068291891655 w_k1_MAT_s011
                                 + 0.0208207793058366 w_k1_MAT_s012
                                 + 0.012100483602954 w_k1_MAT_s013
                                 + 0.0380391886066501 w_k1_MAT_s014
                                 + 0.0412359274753167 w_k1_MAT_s015
                                 + 0.0133628604957674 w_k1_MAT_s016
                                 + 0.0232751312456554 w_k1_MAT_s017
                                 + 0.0369768636449161 w_k1_MAT_s100
                                 + 0.0381469587049684 w_k1_MAT_s101
                                 + 0.0899398535330495 w_k1_MAT_s102
                                 + 0.0629116547784159 w_k1_MAT_s103
                                 + 0.0270063679951691 w_k1_MAT_s104
                                 + 0.0483575450989865 w_k1_MAT_s105
                                 + 0.0281556272662041 w_k1_MAT_s106
                                 + 0.0462140184507192 w_k1_MAT_s107
                                 + 0.0376007292938467 w_k1_MAT_s110
                                 + 0.0125905223628064 w_k1_MAT_s111
                                 + 0.0242174394155054 w_k1_MAT_s112
                                 + 0.0103372807992442 w_k1_MAT_s113
                                 + 0.0231182892908887 w_k1_MAT_s114
                                 + 0.0157573869531752 w_k1_MAT_s115
                                 + 0.0406911236574384 w_k1_MAT_s116
                                 + 0.0301825440287788 w_k1_MAT_s117
                                 + 0.0208295530216938 w_k1_MSFT_s000
                                 + 0.0536629263634065 w_k1_MSFT_s001
                                 + 0.0564320784542191 w_k1_MSFT_s002
                                 + 0.0178135822035336 w_k1_MSFT_s003
                                 + 0.0293399180331987 w_k1_MSFT_s004
                                 + 0.0261659500501204 w_k1_MSFT_s005
                                 + 0.0242330792083955 w_k1_MSFT_s006
                                 + 0.0264798551244484 w_k1_MSFT_s007
                                 + 0.0401648093870534 w_k1_MSFT_s010
                                 + 0.0101588289007345 w_k1_MSFT_s011
                                 + 0.0209700070327395 w_k1_MSFT_s012
                                 + 0.0109011095629599 w_k1_MSFT_s013
                                 + 0.0334469364370009 w_k1_MSFT_s014
                                 + 0.0426964183343345 w_k1_MSFT_s015
                                 + 0.0130396085827801 w_k1_MSFT_s016
                                 + 0.0252103249188453 w_k1_MSFT_s017
                                 + 0.0334781181405202 w_k1_MSFT_s100
                                 + 0.0347660346574927 w_k1_MSFT_s101
                                 + 0.0850342439672977 w_k1_MSFT_s102
                                 + 0.0582738425960351 w_k1_MSFT_s103
                                 + 0.0339305227038139 w_k1_MSFT_s104
                                 + 0.0529155550961974 w_k1_MSFT_s105
                                 + 0.0273255130278805 w_k1_MSFT_s106
                                 + 0.0509118531867737 w_k1_MSFT_s107
                                 + 0.0419231008018805 w_k1_MSFT_s110
                                 + 0.0108452859804544 w_k1_MSFT_s111
                                 + 0.0224845157688877 w_k1_MSFT_s112
                                 + 0.00944015523270588 w_k1_MSFT_s113
                                 + 0.0202196095820306 w_k1_MSFT_s114
                                 + 0.0165127627767872 w_k1_MSFT_s115
                                 + 0.0350913423870009 w_k1_MSFT_s116
                                 + 0.0281521786886959 w_k1_MSFT_s117
                                 + 0.0200984904144784 w_k1_PG_s000
                                 + 0.0518780745430684 w_k1_PG_s001
                                 + 0.0517744563041015 w_k1_PG_s002
                                 + 0.0213615627038288 w_k1_PG_s003
                                 + 0.0263073398964835 w_k1_PG_s004
                                 + 0.0269422761049751 w_k1_PG_s005
                                 + 0.023536303279154 w_k1_PG_s006
                                 + 0.0251603559035176 w_k1_PG_s007
                                 + 0.0406432971289032 w_k1_PG_s010
                                 + 0.00919220894331603 w_k1_PG_s011
                                 + 0.0208553345085805 w_k1_PG_s012
                                 + 0.0107081731725988 w_k1_PG_s013
                                 + 0.0353461485827597 w_k1_PG_s014
                                 + 0.0395873740894659 w_k1_PG_s015
                                 + 0.0127275228494257 w_k1_PG_s016
                                 + 0.0229634678364768 w_k1_PG_s017
                                 + 0.0323177924488712 w_k1_PG_s100
                                 + 0.0356580647768816 w_k1_PG_s101
                                 + 0.091950801777396 w_k1_PG_s102
                                 + 0.0598182014406617 w_k1_PG_s103
                                 + 0.0310248614863303 w_k1_PG_s104
                                 + 0.0494287683427481 w_k1_PG_s105
                                 + 0.0274697457233997 w_k1_PG_s106
                                 + 0.0474688594013094 w_k1_PG_s107
                                 + 0.0410557306426936 w_k1_PG_s110
                                 + 0.0102100885462953 w_k1_PG_s111
                                 + 0.0233893811895858 w_k1_PG_s112
                                 + 0.00922169143599801 w_k1_PG_s113
                                 + 0.0218404763541784 w_k1_PG_s114
                                 + 0.0167082105131418 w_k1_PG_s115
                                 + 0.0326065797621423 w_k1_PG_s116
                                 + 0.0273525267981002 w_k1_PG_s117 >= 1.18
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
