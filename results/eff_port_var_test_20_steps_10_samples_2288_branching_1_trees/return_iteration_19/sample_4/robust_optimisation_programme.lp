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
 w_k1_ADM_s0_return_constraint:  1.18346 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.939208 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.941841 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.10028 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.915072 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.00588 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.850929 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.658854 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.4411 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.04535 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.899903 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.898275 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.07605 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.01727 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.996858 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.943109 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.58086 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.02779 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.20406 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.05438 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.46574 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.27159 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.10645 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.08526 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.23219 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.767358 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.963782 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.55166 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.20242 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.682794 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.697005 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.681019 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.56757 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.21739 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.956921 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.925376 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.11188 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.03992 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.28611 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.905105 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.868577 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.36705 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.913051 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.11222 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.861074 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.627617 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.51651 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.22278 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.880894 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.890035 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.1827 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.06557 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.828431 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.732514 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.55609 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.03154 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.43816 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.845074 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.22849 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.02785 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.924342 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.25249 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.11732 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.653953 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.797024 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.50012 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.15693 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.580316 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.700307 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.594328 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.66798 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.38051 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.868356 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.889611 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.55075 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.03644 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.32195 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.873228 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.914525 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.43469 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.891332 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.09842 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.716625 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.529883 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.62353 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.14163 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.816957 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.723445 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.21748 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.04241 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.76261 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.654244 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.4103 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.00495 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.38238 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.03704 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.62095 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.17104 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.878586 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.29302 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.01278 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.536385 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.777935 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.79349 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.24034 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.493778 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.726017 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.455279 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 2.23493 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.58457 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.803133 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.918628 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.1649 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.1943 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.16304 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.915317 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.931175 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.29231 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.988131 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.0165 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.960323 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.669717 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.41261 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.00313 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.873751 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.762124 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.17801 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.13728 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.991529 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.793271 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.44203 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.906857 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.11798 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.975567 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.21518 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.25708 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.00266 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.10886 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.970308 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.724087 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.824699 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.53162 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.23736 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.568484 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.782542 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.614489 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.71675 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.25684 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.836084 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.02212 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.18831 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.12787 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.21322 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.926905 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.9787 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.1149 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.731953 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.20959 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.511731 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.736126 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.32082 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.81071 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.92487 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.26174 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.09566 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.729222 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.554457 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.774416 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.05546 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.42346 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.97503 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.900535 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.36592 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.627082 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.721674 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.26404 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.36725 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.596345 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.985961 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.28269 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.802832 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.878922 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.910455 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.810582 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.16941 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.72711 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.43695 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.690008 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.11744 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.06126 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.12421 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.877409 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.928829 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.20005 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.950604 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.22026 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.908061 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.695468 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.31306 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.16348 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.99285 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.886055 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.19241 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.07856 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.884112 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.921861 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.45127 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.8083 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.02413 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.873747 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.2277 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.0083 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.16587 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.26588 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.05447 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.772603 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.975922 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.40202 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.30608 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.714968 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.731827 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.584729 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.31671 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.11592 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.792139 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.02138 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.918145 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.965162 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.18254 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.910707 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.973568 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.21005 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.789572 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.0976 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.616853 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.688727 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.40322 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.4257 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.89629 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.868077 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.957989 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.852086 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.733989 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.839333 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.26604 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.19123 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.43043 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.967337 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.36492 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.875059 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.895367 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.11505 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.19265 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.64228 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.978282 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.43376 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.26273 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.692387 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.919652 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.679084 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.52315 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.57702 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.12533 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.832893 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.16325 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.10006 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.33027 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.892496 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.891979 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.01783 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.807822 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.26217 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.515959 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.739954 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.70356 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.97452 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.824222 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.02443 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.13382 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.82422 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.665793 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.794818 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.63015 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.68177 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 2.35396 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.957711 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.3522 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.875964 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.655538 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.22372 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.08917 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.553752 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.902905 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.03696 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.852327 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.694836 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.83694 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.707 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.75789 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.73232 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.32964 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.677576 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.69239 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.05114 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.13107 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.91069 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.970521 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.2407 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.968872 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.0856 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.85544 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.698517 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.25115 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.11392 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.903397 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.779437 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.13845 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.05297 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.913224 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.864724 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.34404 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.931159 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.10345 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.945791 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.28346 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.09043 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.03974 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.14694 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.94582 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.75207 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.90961 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.46068 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.24156 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.605208 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.758867 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.588769 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.55723 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.32568 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.853575 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.01873 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.03159 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.03943 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.13736 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.967728 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.972092 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.08638 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.879752 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.02659 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.848893 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.733142 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.28833 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.15506 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.945529 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.868162 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.02047 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.927554 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.866959 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.861683 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.2811 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.13083 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.32656 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.06418 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.20994 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.04529 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.93737 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.12114 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.04641 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.72928 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.942384 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.27767 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.09103 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.763476 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.861972 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.699073 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.44165 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.08954 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.926906 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.95254 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.09858 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.05056 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.218999645980965 z_k1_s0000
                                 - 0.343998232409751 z_k1_s0001
                                 - 0.569998490909988 z_k1_s0002
                                 - 0.654999800079136 z_k1_s0003
                                 - 0.546000916553359 z_k1_s0004
                                 - 0.176999930861096 z_k1_s0005
                                 - 0.6019985567153 z_k1_s0006
                                 - 0.1869999101753 z_k1_s0007
                                 - 0.260000622194079 z_k1_s0010
                                 - 0.20299990619046 z_k1_s0011
                                 - 0.0989998666126597 z_k1_s0012
                                 - 0.408999078960681 z_k1_s0013
                                 - 0.282000127441712 z_k1_s0014
                                 - 0.389998840292808 z_k1_s0015
                                 - 0.259000169001389 z_k1_s0016
                                 - 0.190999909673784 z_k1_s0017
                                 - 0.33600013938268 z_k1_s0020
                                 - 0.204999821135452 z_k1_s0021
                                 - 0.22399972345538 z_k1_s0022
                                 - 0.295000452255394 z_k1_s0023
                                 - 0.301999993326192 z_k1_s0024
                                 - 0.551999437272103 z_k1_s0025
                                 - 0.539000685426057 z_k1_s0026
                                 - 0.317999736058886 z_k1_s0027
                                 - 0.370997098781377 z_k1_s0030
                                 - 0.197999064512399 z_k1_s0031
                                 - 0.408999896234433 z_k1_s0032
                                 - 0.311998798006324 z_k1_s0033
                                 - 0.4359995208553 z_k1_s0034
                                 - 0.305998845790719 z_k1_s0035
                                 - 0.85899791589839 z_k1_s0036
                                 - 0.315998873049631 z_k1_s0037
                                 - 0.567000389061939 z_k1_s0040
                                 - 0.575999914519907 z_k1_s0041
                                 - 1.61599815414699 z_k1_s0042
                                 - 0.760000460797288 z_k1_s0043
                                 - 0.536000035172729 z_k1_s0044
                                 - 0.869997719951236 z_k1_s0045
                                 - 0.798001516287492 z_k1_s0046
                                 - 1.15999925426907 z_k1_s0047
                                 - 0.163999902617627 z_k1_s0050
                                 - 0.241000774456294 z_k1_s0051
                                 - 0.287999969474842 z_k1_s0052
                                 - 0.196000059225926 z_k1_s0053
                                 - 0.212999977424102 z_k1_s0054
                                 - 0.710999924641017 z_k1_s0055
                                 - 0.277999170534828 z_k1_s0056
                                 - 0.30899996724905 z_k1_s0057
                                 - 0.595001506668605 z_k1_s0060
                                 - 0.333002238492986 z_k1_s0061
                                 - 0.404002015800041 z_k1_s0062
                                 - 0.264000966340477 z_k1_s0063
                                 - 0.252000675683295 z_k1_s0064
                                 - 0.66600128397566 z_k1_s0065
                                 - 0.406000840255618 z_k1_s0066
                                 - 0.273000785207075 z_k1_s0067
                                 - 0.371000151882365 z_k1_s0070
                                 - 0.511999038416615 z_k1_s0071
                                 - 0.491998355907812 z_k1_s0072
                                 - 0.483000901630206 z_k1_s0073
                                 - 0.767000059650213 z_k1_s0074
                                 - 0.712001474502568 z_k1_s0075
                                 - 0.658999885304342 z_k1_s0076
                                 - 0.393000024841632 z_k1_s0077
                                 - 0.231999673358032 z_k1_s0100
                                 - 0.0770000674637393 z_k1_s0101
                                 - 0.246000589897316 z_k1_s0102
                                 - 0.0740000305625387 z_k1_s0103
                                 - 0.0650000213589841 z_k1_s0104
                                 - 0.140000030388574 z_k1_s0105
                                 - 0.0500000398530756 z_k1_s0106
                                 - 0.131000021185019 z_k1_s0107
                                 - 0.406999984261903 z_k1_s0110
                                 - 0.137000140150838 z_k1_s0111
                                 - 0.266999188093051 z_k1_s0112
                                 - 0.156999953203479 z_k1_s0113
                                 - 0.552999461466329 z_k1_s0114
                                 - 0.263999847575264 z_k1_s0115
                                 - 0.468999509965109 z_k1_s0116
                                 - 0.377001317083123 z_k1_s0117
                                 - 0.154999864245812 z_k1_s0120
                                 - 0.0430000347166185 z_k1_s0121
                                 - 0.0979998820005607 z_k1_s0122
                                 - 0.0820000488107516 z_k1_s0123
                                 - 0.21300037102196 z_k1_s0124
                                 - 0.100000100061914 z_k1_s0125
                                 - 0.0490000241003245 z_k1_s0126
                                 - 0.0910003652864869 z_k1_s0127
                                 - 0.249999877834464 z_k1_s0130
                                 - 0.200998734126348 z_k1_s0131
                                 - 0.185999000295518 z_k1_s0132
                                 - 0.177000022794479 z_k1_s0133
                                 - 0.0999996635346326 z_k1_s0134
                                 - 0.168999014354757 z_k1_s0135
                                 - 0.110999612143485 z_k1_s0136
                                 - 0.243999121504673 z_k1_s0137
                                 - 0.092999745748782 z_k1_s0140
                                 - 0.147000545539271 z_k1_s0141
                                 - 0.107999819579337 z_k1_s0142
                                 - 0.0400000188813148 z_k1_s0143
                                 - 0.16499992213528 z_k1_s0144
                                 - 0.127000447098531 z_k1_s0145
                                 - 0.0840000574507775 z_k1_s0146
                                 - 0.126000086176166 z_k1_s0147
                                 - 0.233000569996602 z_k1_s0150
                                 - 0.834002087582806 z_k1_s0151
                                 - 0.292000646108872 z_k1_s0152
                                 - 0.433000492606831 z_k1_s0153
                                 - 0.247000736511756 z_k1_s0154
                                 - 0.563000880244597 z_k1_s0155
                                 - 0.316000792534496 z_k1_s0156
                                 - 0.326000123643311 z_k1_s0157
                                 - 0.170000160044358 z_k1_s0160
                                 - 0.0860001502813645 z_k1_s0161
                                 - 0.181000540094826 z_k1_s0162
                                 - 0.16499980874843 z_k1_s0163
                                 - 0.179000203176168 z_k1_s0164
                                 - 0.0520001182724929 z_k1_s0165
                                 - 0.0930001510949473 z_k1_s0166
                                 - 0.0560001047088055 z_k1_s0167
                                 - 0.0540000845225699 z_k1_s0170
                                 - 0.0730000668952598 z_k1_s0171
                                 - 0.119999795827385 z_k1_s0172
                                 - 0.0880000854737446 z_k1_s0173
                                 - 0.1240005534227 z_k1_s0174
                                 - 0.239000284257175 z_k1_s0175
                                 - 0.100999725254572 z_k1_s0176
                                 - 0.0830001086809574 z_k1_s0177
                                 - 0.286000981455299 z_k1_s1000
                                 - 0.307000719201771 z_k1_s1001
                                 - 0.569002318033429 z_k1_s1002
                                 - 0.802002902506543 z_k1_s1003
                                 - 0.349000935896811 z_k1_s1004
                                 - 0.904003428765925 z_k1_s1005
                                 - 0.270000651837443 z_k1_s1006
                                 - 0.2190007593088 z_k1_s1007
                                 - 0.271002028590819 z_k1_s1010
                                 - 0.464001929372094 z_k1_s1011
                                 - 0.246000786265503 z_k1_s1012
                                 - 0.159000524446046 z_k1_s1013
                                 - 0.586000679136872 z_k1_s1014
                                 - 0.339002380266254 z_k1_s1015
                                 - 0.41600246495658 z_k1_s1016
                                 - 0.184000700768044 z_k1_s1017
                                 - 0.211000695623542 z_k1_s1020
                                 - 0.101000083974719 z_k1_s1021
                                 - 0.317999722280673 z_k1_s1022
                                 - 0.325999386124136 z_k1_s1023
                                 - 0.302000394593747 z_k1_s1024
                                 - 0.229000945771566 z_k1_s1025
                                 - 0.197000076097204 z_k1_s1026
                                 - 0.329000769815783 z_k1_s1027
                                 - 0.22200003208504 z_k1_s1030
                                 - 0.157999891574122 z_k1_s1031
                                 - 0.129999866525572 z_k1_s1032
                                 - 0.245000219174812 z_k1_s1033
                                 - 0.128999928073798 z_k1_s1034
                                 - 0.132999973280707 z_k1_s1035
                                 - 0.141999856446393 z_k1_s1036
                                 - 0.298000016816872 z_k1_s1037
                                 - 0.566000740889419 z_k1_s1040
                                 - 0.46200097466584 z_k1_s1041
                                 - 0.410000496753045 z_k1_s1042
                                 - 1.21100295133992 z_k1_s1043
                                 - 0.834999495662785 z_k1_s1044
                                 - 0.434000717328181 z_k1_s1045
                                 - 0.834999495662785 z_k1_s1046
                                 - 1.19500280428983 z_k1_s1047
                                 - 0.269999708490167 z_k1_s1050
                                 - 0.867997812848282 z_k1_s1051
                                 - 0.557998593973896 z_k1_s1052
                                 - 0.479000020904556 z_k1_s1053
                                 - 0.190999866622223 z_k1_s1054
                                 - 0.201999716117901 z_k1_s1055
                                 - 0.3039997046763 z_k1_s1056
                                 - 0.299999817077808 z_k1_s1057
                                 - 0.495000759549247 z_k1_s1060
                                 - 0.392000411900918 z_k1_s1061
                                 - 0.548001267476315 z_k1_s1062
                                 - 0.254000958550626 z_k1_s1063
                                 - 0.228000729054402 z_k1_s1064
                                 - 0.276999998439096 z_k1_s1065
                                 - 0.229000777989022 z_k1_s1066
                                 - 0.184000661538914 z_k1_s1067
                                 - 0.289000973265089 z_k1_s1070
                                 - 0.151000170529307 z_k1_s1071
                                 - 0.126000015540935 z_k1_s1072
                                 - 0.139000043334857 z_k1_s1073
                                 - 0.398000038613425 z_k1_s1074
                                 - 0.137000077135815 z_k1_s1075
                                 - 0.140000108934428 z_k1_s1076
                                 - 0.26999956187315 z_k1_s1077
                                 - 0.607999878863613 z_k1_s1100
                                 - 1.20599846877263 z_k1_s1101
                                 - 0.620999714905154 z_k1_s1102
                                 - 0.577999691198691 z_k1_s1103
                                 - 0.975999482807477 z_k1_s1104
                                 - 0.793002262530258 z_k1_s1105
                                 - 0.934001427595914 z_k1_s1106
                                 - 0.415999658927814 z_k1_s1107
                                 - 0.28699999405376 z_k1_s1110
                                 - 0.358000121875433 z_k1_s1111
                                 - 0.41399930186146 z_k1_s1112
                                 - 0.191999881095188 z_k1_s1113
                                 - 0.213000088964994 z_k1_s1114
                                 - 0.333000014649486 z_k1_s1115
                                 - 0.683999870901633 z_k1_s1116
                                 - 0.463999516313353 z_k1_s1117
                                 - 0.11000003786857 z_k1_s1120
                                 - 0.118999958903252 z_k1_s1121
                                 - 0.266999250362611 z_k1_s1122
                                 - 0.480999378743295 z_k1_s1123
                                 - 0.111000088094659 z_k1_s1124
                                 - 0.125999956485796 z_k1_s1125
                                 - 0.337999276414142 z_k1_s1126
                                 - 0.218000329285041 z_k1_s1127
                                 - 0.425999217266154 z_k1_s1130
                                 - 0.435999872836158 z_k1_s1131
                                 - 0.375000658861802 z_k1_s1132
                                 - 0.654001404255125 z_k1_s1133
                                 - 0.410999828912038 z_k1_s1134
                                 - 0.442998736734272 z_k1_s1135
                                 - 0.188000202709312 z_k1_s1136
                                 - 0.257000260139848 z_k1_s1137
                                 - 0.237999937122353 z_k1_s1140
                                 - 1.06200198537882 z_k1_s1141
                                 - 0.538001199546036 z_k1_s1142
                                 - 0.393001476374951 z_k1_s1143
                                 - 0.251999996035459 z_k1_s1144
                                 - 0.545001406402661 z_k1_s1145
                                 - 0.237999937122353 z_k1_s1146
                                 - 0.282000122277827 z_k1_s1147
                                 - 0.953002020378833 z_k1_s1150
                                 - 1.53999748713493 z_k1_s1151
                                 - 1.32800250523388 z_k1_s1152
                                 - 0.63400036932795 z_k1_s1153
                                 - 0.717000420342494 z_k1_s1154
                                 - 1.17000489075008 z_k1_s1155
                                 - 1.38600280542495 z_k1_s1156
                                 - 0.717000420342494 z_k1_s1157
                                 - 0.315999080502767 z_k1_s1160
                                 - 0.255999819343686 z_k1_s1161
                                 - 0.334999589469378 z_k1_s1162
                                 - 0.208999748036117 z_k1_s1163
                                 - 0.169999949682876 z_k1_s1164
                                 - 0.760998244096588 z_k1_s1165
                                 - 0.520998413461882 z_k1_s1166
                                 - 0.217999915809817 z_k1_s1167
                                 - 1.1990036952422 z_k1_s1170
                                 - 1.45899732498035 z_k1_s1171
                                 - 0.534000806681087 z_k1_s1172
                                 - 0.851998968311774 z_k1_s1173
                                 - 0.917997952553481 z_k1_s1174
                                 - 1.06899906336682 z_k1_s1175
                                 - 1.05199707465123 z_k1_s1176
                                 - 0.734000476421809 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.796008 w_k1_ADM_s000
                                 + 0.518293 w_k1_AXP_s000
                                 + 0.590024 w_k1_DIS_s000
                                 + 0.651949 w_k1_FDX_s000
                                 + 0.741575 w_k1_IBM_s000
                                 + 0.841945 w_k1_INTC_s000
                                 + 0.895714 w_k1_KO_s000
                                 + 0.644535 w_k1_MAT_s000
                                 + 0.798594 w_k1_MSFT_s000
                                 + 0.70907 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.45034 w_k1_ADM_s000
                                 + 1.26698 w_k1_AXP_s000
                                 + 1.58616 w_k1_DIS_s000 + 1.232 w_k1_FDX_s000
                                 + 1.41813 w_k1_IBM_s000
                                 + 1.21267 w_k1_INTC_s000
                                 + 1.68518 w_k1_KO_s000 + 1.18765 w_k1_MAT_s000
                                 + 1.29051 w_k1_MSFT_s000
                                 + 1.16751 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.23325 w_k1_ADM_s000
                                 + 0.931874 w_k1_AXP_s000
                                 + 1.08273 w_k1_DIS_s000
                                 + 1.11668 w_k1_FDX_s000
                                 + 0.76349 w_k1_IBM_s000
                                 + 1.14608 w_k1_INTC_s000
                                 + 1.02829 w_k1_KO_s000
                                 + 0.852338 w_k1_MAT_s000
                                 + 1.09131 w_k1_MSFT_s000
                                 + 1.02792 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.950271 w_k1_ADM_s000
                                 + 0.846231 w_k1_AXP_s000
                                 + 0.683771 w_k1_DIS_s000
                                 + 0.843322 w_k1_FDX_s000
                                 + 0.800785 w_k1_IBM_s000
                                 + 0.950778 w_k1_INTC_s000
                                 + 0.852419 w_k1_KO_s000
                                 + 0.823626 w_k1_MAT_s000
                                 + 0.885451 w_k1_MSFT_s000
                                 + 0.801366 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.921598 w_k1_ADM_s000
                                 + 0.754183 w_k1_AXP_s000
                                 + 0.826409 w_k1_DIS_s000
                                 + 0.84062 w_k1_FDX_s000 + 1.091 w_k1_IBM_s000
                                 + 0.896459 w_k1_INTC_s000
                                 + 1.00417 w_k1_KO_s000 + 1.0031 w_k1_MAT_s000
                                 + 0.916518 w_k1_MSFT_s000
                                 + 0.930726 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.53764 w_k1_ADM_s000
                                 + 0.976203 w_k1_AXP_s000 + 1.629 w_k1_DIS_s000
                                 + 1.41186 w_k1_FDX_s000
                                 + 0.761845 w_k1_IBM_s000
                                 + 1.19198 w_k1_INTC_s000
                                 + 1.12562 w_k1_KO_s000
                                 + 0.744304 w_k1_MAT_s000
                                 + 1.52306 w_k1_MSFT_s000
                                 + 1.18003 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.842454 w_k1_ADM_s000
                                 + 0.917181 w_k1_AXP_s000
                                 + 1.20098 w_k1_DIS_s000
                                 + 0.992162 w_k1_FDX_s000
                                 + 1.29445 w_k1_IBM_s000
                                 + 1.11352 w_k1_INTC_s000
                                 + 1.34143 w_k1_KO_s000 + 1.00737 w_k1_MAT_s000
                                 + 1.05687 w_k1_MSFT_s000
                                 + 1.05327 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.44861 w_k1_ADM_s000
                                 + 1.33253 w_k1_AXP_s000 + 1.9661 w_k1_DIS_s000
                                 + 1.39424 w_k1_FDX_s000
                                 + 0.948181 w_k1_IBM_s000
                                 + 1.41047 w_k1_INTC_s000
                                 + 1.32046 w_k1_KO_s000
                                 + 0.723089 w_k1_MAT_s000
                                 + 1.46798 w_k1_MSFT_s000 + 1.1737 w_k1_PG_s000
                                 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.10535 w_k1_ADM_s001
                                 + 1.05149 w_k1_AXP_s001
                                 + 1.01806 w_k1_DIS_s001
                                 + 0.880548 w_k1_FDX_s001
                                 + 1.37581 w_k1_IBM_s001
                                 + 1.06539 w_k1_INTC_s001
                                 + 1.12887 w_k1_KO_s001 + 1.46967 w_k1_MAT_s001
                                 + 0.999442 w_k1_MSFT_s001
                                 + 0.991715 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.08012 w_k1_ADM_s001
                                 + 0.760068 w_k1_AXP_s001
                                 + 1.07143 w_k1_DIS_s001
                                 + 0.990698 w_k1_FDX_s001
                                 + 1.04851 w_k1_IBM_s001
                                 + 1.13317 w_k1_INTC_s001
                                 + 1.05327 w_k1_KO_s001
                                 + 0.640287 w_k1_MAT_s001
                                 + 1.06381 w_k1_MSFT_s001
                                 + 0.915274 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.987296 w_k1_ADM_s001
                                 + 1.05606 w_k1_AXP_s001
                                 + 0.94809 w_k1_DIS_s001
                                 + 0.856236 w_k1_FDX_s001
                                 + 1.91247 w_k1_IBM_s001
                                 + 1.06537 w_k1_INTC_s001
                                 + 1.31349 w_k1_KO_s001 + 1.91557 w_k1_MAT_s001
                                 + 0.851216 w_k1_MSFT_s001
                                 + 1.17384 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.92554 w_k1_ADM_s001
                                 + 0.831807 w_k1_AXP_s001
                                 + 0.776939 w_k1_DIS_s001
                                 + 0.932863 w_k1_FDX_s001
                                 + 0.682116 w_k1_IBM_s001
                                 + 1.19094 w_k1_INTC_s001
                                 + 0.70053 w_k1_KO_s001
                                 + 0.428051 w_k1_MAT_s001
                                 + 0.928023 w_k1_MSFT_s001
                                 + 0.810238 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.811596 w_k1_ADM_s001
                                 + 0.777457 w_k1_AXP_s001
                                 + 0.771402 w_k1_DIS_s001
                                 + 0.85483 w_k1_FDX_s001 + 1.12 w_k1_IBM_s001
                                 + 1.0173 w_k1_INTC_s001 + 0.90988 w_k1_KO_s001
                                 + 0.890812 w_k1_MAT_s001
                                 + 0.856002 w_k1_MSFT_s001
                                 + 0.893416 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.0262 w_k1_ADM_s001
                                 + 1.31008 w_k1_AXP_s001
                                 + 1.11136 w_k1_DIS_s001
                                 + 1.15062 w_k1_FDX_s001
                                 + 0.948022 w_k1_IBM_s001
                                 + 1.17238 w_k1_INTC_s001
                                 + 0.961725 w_k1_KO_s001
                                 + 1.08243 w_k1_MAT_s001
                                 + 1.09766 w_k1_MSFT_s001
                                 + 1.05377 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.08066 w_k1_ADM_s001
                                 + 1.19948 w_k1_AXP_s001
                                 + 1.41118 w_k1_DIS_s001
                                 + 1.27278 w_k1_FDX_s001
                                 + 1.45297 w_k1_IBM_s001
                                 + 1.37295 w_k1_INTC_s001
                                 + 1.28416 w_k1_KO_s001 + 1.18448 w_k1_MAT_s001
                                 + 1.20956 w_k1_MSFT_s001
                                 + 1.17551 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.885922 w_k1_ADM_s001
                                 + 0.639423 w_k1_AXP_s001
                                 + 0.588924 w_k1_DIS_s001
                                 + 0.729195 w_k1_FDX_s001
                                 + 0.713666 w_k1_IBM_s001
                                 + 0.655556 w_k1_INTC_s001
                                 + 0.623636 w_k1_KO_s001
                                 + 0.53363 w_k1_MAT_s001
                                 + 0.63909 w_k1_MSFT_s001
                                 + 0.684728 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.00393 w_k1_ADM_s002
                                 + 1.18702 w_k1_AXP_s002
                                 + 0.801155 w_k1_DIS_s002
                                 + 0.925624 w_k1_FDX_s002
                                 + 1.24065 w_k1_IBM_s002
                                 + 0.916652 w_k1_INTC_s002
                                 + 0.921275 w_k1_KO_s002
                                 + 1.02829 w_k1_MAT_s002
                                 + 0.84895 w_k1_MSFT_s002
                                 + 0.929003 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.746212 w_k1_ADM_s002
                                 + 0.887346 w_k1_AXP_s002
                                 + 0.631908 w_k1_DIS_s002
                                 + 0.698972 w_k1_FDX_s002
                                 + 0.958821 w_k1_IBM_s002
                                 + 0.672079 w_k1_INTC_s002
                                 + 0.709727 w_k1_KO_s002
                                 + 0.934102 w_k1_MAT_s002
                                 + 0.833984 w_k1_MSFT_s002
                                 + 0.808429 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.852363 w_k1_ADM_s002
                                 + 0.96892 w_k1_AXP_s002
                                 + 1.11996 w_k1_DIS_s002
                                 + 0.925813 w_k1_FDX_s002
                                 + 1.53215 w_k1_IBM_s002
                                 + 1.03451 w_k1_INTC_s002
                                 + 1.32949 w_k1_KO_s002 + 1.14104 w_k1_MAT_s002
                                 + 1.01745 w_k1_MSFT_s002 + 1.1337 w_k1_PG_s002
                                 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.688734 w_k1_ADM_s002
                                 + 0.673794 w_k1_AXP_s002
                                 + 0.602236 w_k1_DIS_s002
                                 + 0.756598 w_k1_FDX_s002
                                 + 0.661018 w_k1_IBM_s002
                                 + 0.807172 w_k1_INTC_s002
                                 + 0.655382 w_k1_KO_s002
                                 + 0.479535 w_k1_MAT_s002
                                 + 0.80739 w_k1_MSFT_s002
                                 + 0.71735 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.846308 w_k1_ADM_s002
                                 + 0.826233 w_k1_AXP_s002
                                 + 0.855908 w_k1_DIS_s002
                                 + 0.847729 w_k1_FDX_s002
                                 + 0.96867 w_k1_IBM_s002
                                 + 0.935072 w_k1_INTC_s002
                                 + 0.896146 w_k1_KO_s002
                                 + 0.649686 w_k1_MAT_s002
                                 + 0.992009 w_k1_MSFT_s002
                                 + 0.876426 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.930907 w_k1_ADM_s002
                                 + 0.923893 w_k1_AXP_s002
                                 + 0.997093 w_k1_DIS_s002
                                 + 0.92108 w_k1_FDX_s002
                                 + 0.90005 w_k1_IBM_s002
                                 + 0.954845 w_k1_INTC_s002
                                 + 0.979246 w_k1_KO_s002
                                 + 0.66388 w_k1_MAT_s002
                                 + 0.999643 w_k1_MSFT_s002
                                 + 0.954254 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.06029 w_k1_ADM_s002
                                 + 1.19049 w_k1_AXP_s002
                                 + 1.05652 w_k1_DIS_s002
                                 + 1.17709 w_k1_FDX_s002
                                 + 0.760497 w_k1_IBM_s002
                                 + 0.993454 w_k1_INTC_s002
                                 + 0.890177 w_k1_KO_s002
                                 + 0.839387 w_k1_MAT_s002
                                 + 1.04916 w_k1_MSFT_s002 + 1.0028 w_k1_PG_s002
                                 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.85258 w_k1_ADM_s002
                                 + 1.64976 w_k1_AXP_s002
                                 + 1.64425 w_k1_DIS_s002
                                 + 1.23517 w_k1_FDX_s002
                                 + 1.39777 w_k1_IBM_s002
                                 + 1.11596 w_k1_INTC_s002
                                 + 1.28561 w_k1_KO_s002 + 1.49338 w_k1_MAT_s002
                                 + 1.14263 w_k1_MSFT_s002
                                 + 1.01087 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.697464 w_k1_ADM_s003
                                 + 0.848832 w_k1_AXP_s003
                                 + 0.763536 w_k1_DIS_s003
                                 + 0.865157 w_k1_FDX_s003
                                 + 1.11216 w_k1_IBM_s003
                                 + 0.822432 w_k1_INTC_s003
                                 + 0.908754 w_k1_KO_s003
                                 + 0.941743 w_k1_MAT_s003
                                 + 0.731486 w_k1_MSFT_s003
                                 + 0.935126 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.09128 w_k1_ADM_s003
                                 + 0.998979 w_k1_AXP_s003
                                 + 0.94116 w_k1_DIS_s003
                                 + 0.769461 w_k1_FDX_s003
                                 + 1.47833 w_k1_IBM_s003
                                 + 0.803633 w_k1_INTC_s003
                                 + 1.01831 w_k1_KO_s003 + 1.21444 w_k1_MAT_s003
                                 + 0.792974 w_k1_MSFT_s003
                                 + 1.05086 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.15357 w_k1_ADM_s003
                                 + 1.31469 w_k1_AXP_s003
                                 + 1.70555 w_k1_DIS_s003
                                 + 1.34939 w_k1_FDX_s003
                                 + 1.04304 w_k1_IBM_s003
                                 + 1.13027 w_k1_INTC_s003
                                 + 1.07514 w_k1_KO_s003 + 1.06194 w_k1_MAT_s003
                                 + 1.38955 w_k1_MSFT_s003
                                 + 1.14286 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.951441 w_k1_ADM_s003
                                 + 1.12408 w_k1_AXP_s003
                                 + 1.31478 w_k1_DIS_s003
                                 + 1.21019 w_k1_FDX_s003
                                 + 0.802191 w_k1_IBM_s003
                                 + 1.08411 w_k1_INTC_s003
                                 + 1.02002 w_k1_KO_s003 + 1.00229 w_k1_MAT_s003
                                 + 1.23227 w_k1_MSFT_s003
                                 + 1.00386 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.934798 w_k1_ADM_s003
                                 + 1.39903 w_k1_AXP_s003
                                 + 1.17404 w_k1_DIS_s003
                                 + 1.14976 w_k1_FDX_s003
                                 + 1.20225 w_k1_IBM_s003
                                 + 0.945904 w_k1_INTC_s003
                                 + 1.15655 w_k1_KO_s003 + 1.33753 w_k1_MAT_s003
                                 + 1.08064 w_k1_MSFT_s003
                                 + 1.02782 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.12343 w_k1_ADM_s003
                                 + 1.44137 w_k1_AXP_s003
                                 + 1.60199 w_k1_DIS_s003
                                 + 1.23876 w_k1_FDX_s003
                                 + 1.75827 w_k1_IBM_s003
                                 + 1.23656 w_k1_INTC_s003
                                 + 1.40051 w_k1_KO_s003 + 2.04138 w_k1_MAT_s003
                                 + 1.11843 w_k1_MSFT_s003
                                 + 1.20479 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.728308 w_k1_ADM_s003
                                 + 0.670586 w_k1_AXP_s003
                                 + 0.743696 w_k1_DIS_s003
                                 + 0.835142 w_k1_FDX_s003
                                 + 0.855732 w_k1_IBM_s003
                                 + 0.964333 w_k1_INTC_s003
                                 + 0.781186 w_k1_KO_s003
                                 + 0.656688 w_k1_MAT_s003
                                 + 0.833961 w_k1_MSFT_s003
                                 + 0.793123 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.855887 w_k1_ADM_s003
                                 + 0.892694 w_k1_AXP_s003
                                 + 0.861588 w_k1_DIS_s003
                                 + 1.01048 w_k1_FDX_s003
                                 + 0.646253 w_k1_IBM_s003
                                 + 0.919339 w_k1_INTC_s003
                                 + 0.710074 w_k1_KO_s003
                                 + 0.687315 w_k1_MAT_s003
                                 + 0.848663 w_k1_MSFT_s003
                                 + 0.836094 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.876571 w_k1_ADM_s004
                                 + 1.17584 w_k1_AXP_s004
                                 + 0.884489 w_k1_DIS_s004
                                 + 0.930002 w_k1_FDX_s004
                                 + 1.34438 w_k1_IBM_s004
                                 + 0.840163 w_k1_INTC_s004
                                 + 0.992639 w_k1_KO_s004
                                 + 1.64876 w_k1_MAT_s004
                                 + 0.74388 w_k1_MSFT_s004
                                 + 1.06075 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.1914 w_k1_ADM_s004
                                 + 1.12075 w_k1_AXP_s004
                                 + 0.84131 w_k1_DIS_s004
                                 + 1.12033 w_k1_FDX_s004
                                 + 0.737563 w_k1_IBM_s004
                                 + 0.919803 w_k1_INTC_s004
                                 + 0.835456 w_k1_KO_s004
                                 + 1.03991 w_k1_MAT_s004
                                 + 0.970615 w_k1_MSFT_s004
                                 + 0.913101 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.20985 w_k1_ADM_s004
                                 + 1.17888 w_k1_AXP_s004
                                 + 1.27996 w_k1_DIS_s004
                                 + 1.07119 w_k1_FDX_s004
                                 + 1.37293 w_k1_IBM_s004
                                 + 1.05563 w_k1_INTC_s004
                                 + 1.27894 w_k1_KO_s004 + 1.27627 w_k1_MAT_s004
                                 + 1.0593 w_k1_MSFT_s004 + 1.14992 w_k1_PG_s004
                                 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.03567 w_k1_ADM_s004
                                 + 0.825148 w_k1_AXP_s004
                                 + 1.01114 w_k1_DIS_s004
                                 + 0.99343 w_k1_FDX_s004
                                 + 0.853834 w_k1_IBM_s004
                                 + 0.906931 w_k1_INTC_s004
                                 + 0.913286 w_k1_KO_s004
                                 + 0.943574 w_k1_MAT_s004
                                 + 1.09643 w_k1_MSFT_s004
                                 + 0.972779 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.857939 w_k1_ADM_s004
                                 + 1.00937 w_k1_AXP_s004
                                 + 0.787153 w_k1_DIS_s004
                                 + 0.977493 w_k1_FDX_s004
                                 + 0.850292 w_k1_IBM_s004
                                 + 0.980252 w_k1_INTC_s004
                                 + 0.850555 w_k1_KO_s004
                                 + 1.02642 w_k1_MAT_s004
                                 + 0.999132 w_k1_MSFT_s004
                                 + 0.934364 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.967934 w_k1_ADM_s004
                                 + 1.01923 w_k1_AXP_s004
                                 + 0.825635 w_k1_DIS_s004
                                 + 0.900765 w_k1_FDX_s004
                                 + 1.05602 w_k1_IBM_s004
                                 + 0.863551 w_k1_INTC_s004
                                 + 0.881615 w_k1_KO_s004
                                 + 0.963646 w_k1_MAT_s004
                                 + 0.841957 w_k1_MSFT_s004
                                 + 0.812482 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.847804 w_k1_ADM_s004
                                 + 0.678652 w_k1_AXP_s004
                                 + 0.702912 w_k1_DIS_s004
                                 + 0.789726 w_k1_FDX_s004
                                 + 0.628737 w_k1_IBM_s004
                                 + 0.852773 w_k1_INTC_s004
                                 + 0.721964 w_k1_KO_s004
                                 + 0.503578 w_k1_MAT_s004
                                 + 0.852534 w_k1_MSFT_s004
                                 + 0.797942 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.41433 w_k1_ADM_s004
                                 + 1.56827 w_k1_AXP_s004
                                 + 1.46064 w_k1_DIS_s004
                                 + 1.33195 w_k1_FDX_s004
                                 + 1.09255 w_k1_IBM_s004
                                 + 1.00296 w_k1_INTC_s004
                                 + 1.22627 w_k1_KO_s004 + 1.13101 w_k1_MAT_s004
                                 + 1.10845 w_k1_MSFT_s004
                                 + 1.21616 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.801245 w_k1_ADM_s005
                                 + 1.3967 w_k1_AXP_s005 + 1.44998 w_k1_DIS_s005
                                 + 1.14046 w_k1_FDX_s005
                                 + 1.57504 w_k1_IBM_s005
                                 + 1.2909 w_k1_INTC_s005 + 1.54826 w_k1_KO_s005
                                 + 2.70636 w_k1_MAT_s005
                                 + 1.30009 w_k1_MSFT_s005
                                 + 1.15463 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.08106 w_k1_ADM_s005
                                 + 0.867569 w_k1_AXP_s005
                                 + 1.17353 w_k1_DIS_s005
                                 + 1.08771 w_k1_FDX_s005
                                 + 0.986274 w_k1_IBM_s005
                                 + 0.940843 w_k1_INTC_s005
                                 + 1.32918 w_k1_KO_s005 + 1.17568 w_k1_MAT_s005
                                 + 1.27801 w_k1_MSFT_s005
                                 + 1.15152 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.26061 w_k1_ADM_s005
                                 + 1.27265 w_k1_AXP_s005
                                 + 1.38074 w_k1_DIS_s005
                                 + 1.26903 w_k1_FDX_s005 + 1.1273 w_k1_IBM_s005
                                 + 0.901112 w_k1_INTC_s005
                                 + 1.22525 w_k1_KO_s005 + 1.3914 w_k1_MAT_s005
                                 + 1.13391 w_k1_MSFT_s005
                                 + 1.28107 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.974436 w_k1_ADM_s005
                                 + 0.989598 w_k1_AXP_s005
                                 + 1.03114 w_k1_DIS_s005
                                 + 1.04724 w_k1_FDX_s005
                                 + 0.816206 w_k1_IBM_s005
                                 + 1.12847 w_k1_INTC_s005
                                 + 0.992364 w_k1_KO_s005
                                 + 0.897124 w_k1_MAT_s005
                                 + 1.15967 w_k1_MSFT_s005
                                 + 0.971213 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.963018 w_k1_ADM_s005
                                 + 1.20291 w_k1_AXP_s005
                                 + 1.14662 w_k1_DIS_s005
                                 + 1.27653 w_k1_FDX_s005
                                 + 0.812556 w_k1_IBM_s005
                                 + 0.912025 w_k1_INTC_s005
                                 + 1.12121 w_k1_KO_s005 + 1.38135 w_k1_MAT_s005
                                 + 1.16251 w_k1_MSFT_s005
                                 + 1.11818 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.697151 w_k1_ADM_s005
                                 + 0.76129 w_k1_AXP_s005
                                 + 0.82196 w_k1_DIS_s005
                                 + 0.851936 w_k1_FDX_s005
                                 + 0.768556 w_k1_IBM_s005
                                 + 0.816883 w_k1_INTC_s005
                                 + 0.864031 w_k1_KO_s005
                                 + 0.94548 w_k1_MAT_s005
                                 + 0.823369 w_k1_MSFT_s005
                                 + 0.920175 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.843194 w_k1_ADM_s005
                                 + 0.888702 w_k1_AXP_s005
                                 + 1.20416 w_k1_DIS_s005
                                 + 1.10711 w_k1_FDX_s005
                                 + 1.20941 w_k1_IBM_s005
                                 + 1.09957 w_k1_INTC_s005
                                 + 1.21089 w_k1_KO_s005 + 1.20224 w_k1_MAT_s005
                                 + 1.14282 w_k1_MSFT_s005
                                 + 1.05334 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.797479 w_k1_ADM_s005
                                 + 0.973092 w_k1_AXP_s005
                                 + 1.02353 w_k1_DIS_s005
                                 + 0.915276 w_k1_FDX_s005
                                 + 1.30807 w_k1_IBM_s005
                                 + 0.983333 w_k1_INTC_s005
                                 + 1.20709 w_k1_KO_s005 + 1.43803 w_k1_MAT_s005
                                 + 0.954297 w_k1_MSFT_s005
                                 + 1.0286 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.01194 w_k1_ADM_s006
                                 + 0.994632 w_k1_AXP_s006
                                 + 0.883013 w_k1_DIS_s006
                                 + 0.967342 w_k1_FDX_s006
                                 + 0.943823 w_k1_IBM_s006
                                 + 0.945301 w_k1_INTC_s006
                                 + 1.12313 w_k1_KO_s006
                                 + 0.936978 w_k1_MAT_s006
                                 + 0.993473 w_k1_MSFT_s006
                                 + 1.06492 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.885971 w_k1_ADM_s006
                                 + 0.879139 w_k1_AXP_s006
                                 + 1.20174 w_k1_DIS_s006
                                 + 1.13898 w_k1_FDX_s006
                                 + 0.678946 w_k1_IBM_s006
                                 + 0.951283 w_k1_INTC_s006
                                 + 1.04141 w_k1_KO_s006
                                 + 0.782155 w_k1_MAT_s006
                                 + 1.19405 w_k1_MSFT_s006
                                 + 1.08926 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.01629 w_k1_ADM_s006
                                 + 1.11359 w_k1_AXP_s006
                                 + 1.30839 w_k1_DIS_s006
                                 + 1.06444 w_k1_FDX_s006
                                 + 1.49278 w_k1_IBM_s006
                                 + 1.00941 w_k1_INTC_s006
                                 + 1.56134 w_k1_KO_s006 + 1.47697 w_k1_MAT_s006
                                 + 1.1389 w_k1_MSFT_s006 + 1.10667 w_k1_PG_s006
                                 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.630245 w_k1_ADM_s006
                                 + 0.752194 w_k1_AXP_s006
                                 + 0.599625 w_k1_DIS_s006
                                 + 0.710153 w_k1_FDX_s006
                                 + 0.84341 w_k1_IBM_s006
                                 + 0.812399 w_k1_INTC_s006
                                 + 0.914224 w_k1_KO_s006
                                 + 0.82436 w_k1_MAT_s006
                                 + 0.804038 w_k1_MSFT_s006
                                 + 0.856788 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.822901 w_k1_ADM_s006
                                 + 0.982668 w_k1_AXP_s006
                                 + 0.959914 w_k1_DIS_s006
                                 + 0.99997 w_k1_FDX_s006
                                 + 0.771093 w_k1_IBM_s006
                                 + 1.11774 w_k1_INTC_s006
                                 + 0.943471 w_k1_KO_s006
                                 + 0.570055 w_k1_MAT_s006
                                 + 0.982687 w_k1_MSFT_s006
                                 + 0.954394 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.05409 w_k1_ADM_s006
                                 + 1.18108 w_k1_AXP_s006
                                 + 1.22829 w_k1_DIS_s006
                                 + 1.21523 w_k1_FDX_s006
                                 + 1.11617 w_k1_IBM_s006
                                 + 0.891863 w_k1_INTC_s006
                                 + 1.33841 w_k1_KO_s006 + 1.42214 w_k1_MAT_s006
                                 + 1.0729 w_k1_MSFT_s006 + 1.1514 w_k1_PG_s006
                                 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.912995 w_k1_ADM_s006
                                 + 0.72148 w_k1_AXP_s006
                                 + 0.813239 w_k1_DIS_s006
                                 + 0.820966 w_k1_FDX_s006
                                 + 0.818943 w_k1_IBM_s006
                                 + 0.904996 w_k1_INTC_s006
                                 + 1.18391 w_k1_KO_s006
                                 + 0.810124 w_k1_MAT_s006
                                 + 1.00118 w_k1_MSFT_s006
                                 + 0.928126 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.775018 w_k1_ADM_s006
                                 + 0.933453 w_k1_AXP_s006
                                 + 1.32799 w_k1_DIS_s006
                                 + 1.03493 w_k1_FDX_s006
                                 + 1.00609 w_k1_IBM_s006
                                 + 0.964327 w_k1_INTC_s006
                                 + 1.31746 w_k1_KO_s006 + 1.09074 w_k1_MAT_s006
                                 + 1.13196 w_k1_MSFT_s006
                                 + 1.05315 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.61383 w_k1_ADM_s007
                                 + 1.84899 w_k1_AXP_s007
                                 + 1.41905 w_k1_DIS_s007
                                 + 1.26573 w_k1_FDX_s007 + 1.346 w_k1_IBM_s007
                                 + 1.05824 w_k1_INTC_s007
                                 + 1.32254 w_k1_KO_s007 + 1.48265 w_k1_MAT_s007
                                 + 1.10171 w_k1_MSFT_s007
                                 + 1.29742 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.09052 w_k1_ADM_s007
                                 + 0.989107 w_k1_AXP_s007
                                 + 1.27954 w_k1_DIS_s007 + 1.1453 w_k1_FDX_s007
                                 + 1.03869 w_k1_IBM_s007
                                 + 1.00027 w_k1_INTC_s007 + 1.13 w_k1_KO_s007
                                 + 1.02558 w_k1_MAT_s007
                                 + 1.17771 w_k1_MSFT_s007
                                 + 1.09451 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.13215 w_k1_ADM_s007
                                 + 1.19341 w_k1_AXP_s007
                                 + 0.96429 w_k1_DIS_s007
                                 + 1.06376 w_k1_FDX_s007
                                 + 0.811731 w_k1_IBM_s007
                                 + 0.92919 w_k1_INTC_s007
                                 + 0.920191 w_k1_KO_s007
                                 + 0.857948 w_k1_MAT_s007
                                 + 0.949177 w_k1_MSFT_s007
                                 + 0.932079 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.805097 w_k1_ADM_s007
                                 + 0.731229 w_k1_AXP_s007
                                 + 0.706274 w_k1_DIS_s007
                                 + 0.786711 w_k1_FDX_s007
                                 + 0.686073 w_k1_IBM_s007
                                 + 0.875719 w_k1_INTC_s007
                                 + 0.796792 w_k1_KO_s007
                                 + 0.595726 w_k1_MAT_s007
                                 + 0.790825 w_k1_MSFT_s007
                                 + 0.837535 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.1603 w_k1_ADM_s007
                                 + 1.27172 w_k1_AXP_s007
                                 + 1.20574 w_k1_DIS_s007
                                 + 1.00234 w_k1_FDX_s007
                                 + 1.48506 w_k1_IBM_s007
                                 + 1.11199 w_k1_INTC_s007
                                 + 1.21125 w_k1_KO_s007 + 1.19699 w_k1_MAT_s007
                                 + 0.973341 w_k1_MSFT_s007
                                 + 1.17639 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.885983 w_k1_ADM_s007
                                 + 1.05191 w_k1_AXP_s007
                                 + 0.79897 w_k1_DIS_s007
                                 + 0.964124 w_k1_FDX_s007
                                 + 1.02432 w_k1_IBM_s007
                                 + 0.936343 w_k1_INTC_s007
                                 + 0.883823 w_k1_KO_s007
                                 + 1.16893 w_k1_MAT_s007
                                 + 0.969272 w_k1_MSFT_s007
                                 + 1.03271 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.91992 w_k1_ADM_s007
                                 + 0.755987 w_k1_AXP_s007
                                 + 0.808111 w_k1_DIS_s007
                                 + 0.799309 w_k1_FDX_s007
                                 + 1.08531 w_k1_IBM_s007
                                 + 0.875188 w_k1_INTC_s007
                                 + 1.11435 w_k1_KO_s007 + 1.24951 w_k1_MAT_s007
                                 + 0.862515 w_k1_MSFT_s007
                                 + 0.991732 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.39717 w_k1_ADM_s007
                                 + 1.35255 w_k1_AXP_s007
                                 + 1.41739 w_k1_DIS_s007
                                 + 1.48762 w_k1_FDX_s007
                                 + 0.957632 w_k1_IBM_s007
                                 + 1.42651 w_k1_INTC_s007
                                 + 1.27673 w_k1_KO_s007
                                 + 0.841698 w_k1_MAT_s007
                                 + 1.34999 w_k1_MSFT_s007
                                 + 1.21029 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.12876 w_k1_ADM_s010
                                 + 1.00392 w_k1_AXP_s010
                                 + 1.00223 w_k1_DIS_s010
                                 + 1.00453 w_k1_FDX_s010
                                 + 0.860678 w_k1_IBM_s010
                                 + 0.92795 w_k1_INTC_s010
                                 + 0.917911 w_k1_KO_s010
                                 + 1.07623 w_k1_MAT_s010
                                 + 1.12588 w_k1_MSFT_s010
                                 + 1.10283 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.972541 w_k1_ADM_s010
                                 + 1.23476 w_k1_AXP_s010 + 1.0339 w_k1_DIS_s010
                                 + 1.06138 w_k1_FDX_s010
                                 + 1.12628 w_k1_IBM_s010
                                 + 1.08237 w_k1_INTC_s010
                                 + 1.18715 w_k1_KO_s010 + 1.76964 w_k1_MAT_s010
                                 + 1.15077 w_k1_MSFT_s010
                                 + 1.18335 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.25001 w_k1_ADM_s010
                                 + 1.27417 w_k1_AXP_s010
                                 + 1.40944 w_k1_DIS_s010
                                 + 1.26281 w_k1_FDX_s010
                                 + 1.24558 w_k1_IBM_s010
                                 + 1.10612 w_k1_INTC_s010
                                 + 1.21934 w_k1_KO_s010 + 1.45762 w_k1_MAT_s010
                                 + 1.22701 w_k1_MSFT_s010
                                 + 1.25889 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.23339 w_k1_ADM_s010
                                 + 0.814667 w_k1_AXP_s010
                                 + 0.787692 w_k1_DIS_s010
                                 + 1.07262 w_k1_FDX_s010
                                 + 0.624283 w_k1_IBM_s010
                                 + 0.92897 w_k1_INTC_s010
                                 + 0.702826 w_k1_KO_s010
                                 + 0.650686 w_k1_MAT_s010
                                 + 0.983908 w_k1_MSFT_s010
                                 + 0.938101 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.06004 w_k1_ADM_s010
                                 + 0.868604 w_k1_AXP_s010
                                 + 1.05599 w_k1_DIS_s010
                                 + 0.917127 w_k1_FDX_s010
                                 + 1.31305 w_k1_IBM_s010
                                 + 1.10132 w_k1_INTC_s010 + 1.1515 w_k1_KO_s010
                                 + 1.0567 w_k1_MAT_s010
                                 + 0.98888 w_k1_MSFT_s010
                                 + 1.05789 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.860305 w_k1_ADM_s010
                                 + 0.6271 w_k1_AXP_s010 + 1.02435 w_k1_DIS_s010
                                 + 0.813412 w_k1_FDX_s010
                                 + 1.07759 w_k1_IBM_s010
                                 + 1.00502 w_k1_INTC_s010
                                 + 1.01641 w_k1_KO_s010
                                 + 0.728266 w_k1_MAT_s010
                                 + 1.0798 w_k1_MSFT_s010
                                 + 0.943032 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.761787 w_k1_ADM_s010
                                 + 0.699302 w_k1_AXP_s010
                                 + 1.48907 w_k1_DIS_s010
                                 + 1.03738 w_k1_FDX_s010
                                 + 0.768845 w_k1_IBM_s010
                                 + 1.03195 w_k1_INTC_s010
                                 + 1.11962 w_k1_KO_s010 + 0.74312 w_k1_MAT_s010
                                 + 1.26266 w_k1_MSFT_s010
                                 + 0.971936 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.827539 w_k1_ADM_s010
                                 + 0.611295 w_k1_AXP_s010
                                 + 0.624425 w_k1_DIS_s010
                                 + 0.733351 w_k1_FDX_s010
                                 + 0.688667 w_k1_IBM_s010
                                 + 0.954875 w_k1_INTC_s010
                                 + 0.75897 w_k1_KO_s010
                                 + 0.660226 w_k1_MAT_s010
                                 + 0.799045 w_k1_MSFT_s010
                                 + 0.782332 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.14729 w_k1_ADM_s011
                                 + 1.16499 w_k1_AXP_s011
                                 + 1.68179 w_k1_DIS_s011
                                 + 1.25041 w_k1_FDX_s011
                                 + 1.18868 w_k1_IBM_s011
                                 + 1.15399 w_k1_INTC_s011
                                 + 1.26821 w_k1_KO_s011 + 1.39927 w_k1_MAT_s011
                                 + 1.31443 w_k1_MSFT_s011
                                 + 1.21135 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.73392 w_k1_ADM_s011
                                 + 0.556543 w_k1_AXP_s011
                                 + 0.699775 w_k1_DIS_s011
                                 + 0.78351 w_k1_FDX_s011
                                 + 0.512586 w_k1_IBM_s011
                                 + 0.617249 w_k1_INTC_s011
                                 + 0.607366 w_k1_KO_s011
                                 + 0.533013 w_k1_MAT_s011
                                 + 0.652144 w_k1_MSFT_s011
                                 + 0.759136 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.07843 w_k1_ADM_s011
                                 + 0.944817 w_k1_AXP_s011
                                 + 0.942767 w_k1_DIS_s011
                                 + 0.875908 w_k1_FDX_s011
                                 + 1.50917 w_k1_IBM_s011
                                 + 0.779141 w_k1_INTC_s011
                                 + 1.19829 w_k1_KO_s011 + 1.8104 w_k1_MAT_s011
                                 + 0.859965 w_k1_MSFT_s011
                                 + 1.11534 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.904083 w_k1_ADM_s011
                                 + 1.30585 w_k1_AXP_s011
                                 + 1.45418 w_k1_DIS_s011
                                 + 1.18708 w_k1_FDX_s011
                                 + 1.89864 w_k1_IBM_s011
                                 + 1.27235 w_k1_INTC_s011
                                 + 1.43004 w_k1_KO_s011 + 2.51001 w_k1_MAT_s011
                                 + 1.14938 w_k1_MSFT_s011
                                 + 1.18116 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.899362 w_k1_ADM_s011
                                 + 1.20522 w_k1_AXP_s011
                                 + 1.07441 w_k1_DIS_s011
                                 + 1.04358 w_k1_FDX_s011
                                 + 1.01867 w_k1_IBM_s011
                                 + 1.01457 w_k1_INTC_s011
                                 + 0.979481 w_k1_KO_s011
                                 + 1.22796 w_k1_MAT_s011
                                 + 0.940104 w_k1_MSFT_s011
                                 + 1.01429 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.890249 w_k1_ADM_s011
                                 + 0.856613 w_k1_AXP_s011
                                 + 0.712645 w_k1_DIS_s011
                                 + 0.804573 w_k1_FDX_s011
                                 + 1.08981 w_k1_IBM_s011
                                 + 0.631795 w_k1_INTC_s011
                                 + 0.748234 w_k1_KO_s011
                                 + 1.11936 w_k1_MAT_s011
                                 + 0.702167 w_k1_MSFT_s011
                                 + 0.903435 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.907938 w_k1_ADM_s011
                                 + 0.656047 w_k1_AXP_s011
                                 + 0.931887 w_k1_DIS_s011
                                 + 0.908288 w_k1_FDX_s011
                                 + 0.861801 w_k1_IBM_s011
                                 + 0.826869 w_k1_INTC_s011
                                 + 0.863647 w_k1_KO_s011
                                 + 0.686971 w_k1_MAT_s011
                                 + 0.84988 w_k1_MSFT_s011
                                 + 0.86377 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.708255 w_k1_ADM_s011
                                 + 0.94156 w_k1_AXP_s011
                                 + 0.83365 w_k1_DIS_s011
                                 + 0.935638 w_k1_FDX_s011
                                 + 0.67259 w_k1_IBM_s011
                                 + 0.765587 w_k1_INTC_s011
                                 + 0.665846 w_k1_KO_s011
                                 + 0.895526 w_k1_MAT_s011
                                 + 0.942201 w_k1_MSFT_s011
                                 + 0.870403 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.900475 w_k1_ADM_s012
                                 + 1.00154 w_k1_AXP_s012
                                 + 0.935817 w_k1_DIS_s012
                                 + 0.930525 w_k1_FDX_s012
                                 + 1.05691 w_k1_IBM_s012
                                 + 0.761498 w_k1_INTC_s012
                                 + 0.927572 w_k1_KO_s012
                                 + 0.913834 w_k1_MAT_s012
                                 + 0.800767 w_k1_MSFT_s012
                                 + 0.962601 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.944842 w_k1_ADM_s012
                                 + 0.829506 w_k1_AXP_s012
                                 + 1.45761 w_k1_DIS_s012
                                 + 1.13265 w_k1_FDX_s012
                                 + 1.29655 w_k1_IBM_s012
                                 + 0.897278 w_k1_INTC_s012
                                 + 1.34737 w_k1_KO_s012
                                 + 0.738924 w_k1_MAT_s012
                                 + 1.19631 w_k1_MSFT_s012
                                 + 1.23348 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.750054 w_k1_ADM_s012
                                 + 0.828173 w_k1_AXP_s012
                                 + 1.00708 w_k1_DIS_s012
                                 + 1.03246 w_k1_FDX_s012
                                 + 0.740938 w_k1_IBM_s012
                                 + 0.897564 w_k1_INTC_s012
                                 + 0.8041 w_k1_KO_s012 + 0.551471 w_k1_MAT_s012
                                 + 0.968123 w_k1_MSFT_s012
                                 + 0.933272 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.2399 w_k1_ADM_s012
                                 + 1.26621 w_k1_AXP_s012
                                 + 1.81673 w_k1_DIS_s012
                                 + 1.38553 w_k1_FDX_s012
                                 + 1.29049 w_k1_IBM_s012
                                 + 1.14405 w_k1_INTC_s012
                                 + 1.44782 w_k1_KO_s012 + 1.43592 w_k1_MAT_s012
                                 + 1.44655 w_k1_MSFT_s012
                                 + 1.35364 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.887174 w_k1_ADM_s012
                                 + 0.725004 w_k1_AXP_s012
                                 + 0.765955 w_k1_DIS_s012
                                 + 0.80162 w_k1_FDX_s012
                                 + 0.928436 w_k1_IBM_s012
                                 + 0.676389 w_k1_INTC_s012 + 0.801 w_k1_KO_s012
                                 + 0.663869 w_k1_MAT_s012
                                 + 0.808754 w_k1_MSFT_s012
                                 + 0.807063 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.930517 w_k1_ADM_s012
                                 + 1.19692 w_k1_AXP_s012
                                 + 1.31449 w_k1_DIS_s012 + 1.1404 w_k1_FDX_s012
                                 + 1.02433 w_k1_IBM_s012
                                 + 0.805017 w_k1_INTC_s012
                                 + 0.960945 w_k1_KO_s012
                                 + 1.22667 w_k1_MAT_s012
                                 + 0.987888 w_k1_MSFT_s012
                                 + 1.15798 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.897013 w_k1_ADM_s012
                                 + 1.48085 w_k1_AXP_s012 + 1.3103 w_k1_DIS_s012
                                 + 1.26342 w_k1_FDX_s012
                                 + 1.40032 w_k1_IBM_s012
                                 + 0.992048 w_k1_INTC_s012
                                 + 1.11091 w_k1_KO_s012 + 1.31179 w_k1_MAT_s012
                                 + 1.07265 w_k1_MSFT_s012
                                 + 0.955421 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.884755 w_k1_ADM_s012
                                 + 1.08699 w_k1_AXP_s012
                                 + 0.836816 w_k1_DIS_s012
                                 + 0.849487 w_k1_FDX_s012
                                 + 1.44134 w_k1_IBM_s012
                                 + 0.778515 w_k1_INTC_s012
                                 + 1.00674 w_k1_KO_s012 + 1.22067 w_k1_MAT_s012
                                 + 0.805318 w_k1_MSFT_s012
                                 + 0.966506 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.62012 w_k1_ADM_s013
                                 + 1.51552 w_k1_AXP_s013
                                 + 1.19544 w_k1_DIS_s013
                                 + 1.37625 w_k1_FDX_s013 + 1.0522 w_k1_IBM_s013
                                 + 1.44675 w_k1_INTC_s013
                                 + 1.06453 w_k1_KO_s013 + 1.09944 w_k1_MAT_s013
                                 + 1.16021 w_k1_MSFT_s013
                                 + 1.09882 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.965498 w_k1_ADM_s013
                                 + 0.938674 w_k1_AXP_s013
                                 + 0.943673 w_k1_DIS_s013
                                 + 1.10855 w_k1_FDX_s013
                                 + 0.692317 w_k1_IBM_s013
                                 + 1.4088 w_k1_INTC_s013
                                 + 0.862559 w_k1_KO_s013
                                 + 0.674504 w_k1_MAT_s013
                                 + 1.07539 w_k1_MSFT_s013
                                 + 0.926745 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.78667 w_k1_ADM_s013
                                 + 0.776007 w_k1_AXP_s013
                                 + 1.12086 w_k1_DIS_s013 + 1.0179 w_k1_FDX_s013
                                 + 1.08741 w_k1_IBM_s013
                                 + 1.36218 w_k1_INTC_s013
                                 + 0.961734 w_k1_KO_s013
                                 + 0.952044 w_k1_MAT_s013
                                 + 1.0166 w_k1_MSFT_s013
                                 + 0.860051 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.916683 w_k1_ADM_s013
                                 + 1.22324 w_k1_AXP_s013
                                 + 1.04733 w_k1_DIS_s013
                                 + 1.08991 w_k1_FDX_s013
                                 + 0.874498 w_k1_IBM_s013
                                 + 1.40442 w_k1_INTC_s013
                                 + 0.992001 w_k1_KO_s013
                                 + 1.05508 w_k1_MAT_s013
                                 + 1.1068 w_k1_MSFT_s013 + 1.00013 w_k1_PG_s013
                                 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.824279 w_k1_ADM_s013
                                 + 0.812528 w_k1_AXP_s013
                                 + 0.702902 w_k1_DIS_s013
                                 + 0.768887 w_k1_FDX_s013
                                 + 1.08966 w_k1_IBM_s013
                                 + 0.861529 w_k1_INTC_s013
                                 + 0.886741 w_k1_KO_s013
                                 + 0.846808 w_k1_MAT_s013
                                 + 0.781152 w_k1_MSFT_s013
                                 + 0.85829 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.900994 w_k1_ADM_s013
                                 + 0.685783 w_k1_AXP_s013
                                 + 0.704926 w_k1_DIS_s013
                                 + 0.813386 w_k1_FDX_s013
                                 + 0.890161 w_k1_IBM_s013
                                 + 1.05341 w_k1_INTC_s013
                                 + 0.737036 w_k1_KO_s013
                                 + 0.687269 w_k1_MAT_s013
                                 + 0.793143 w_k1_MSFT_s013
                                 + 0.72571 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.944557 w_k1_ADM_s013
                                 + 0.774319 w_k1_AXP_s013
                                 + 0.710451 w_k1_DIS_s013
                                 + 0.878783 w_k1_FDX_s013
                                 + 0.686715 w_k1_IBM_s013
                                 + 1.24089 w_k1_INTC_s013
                                 + 0.701167 w_k1_KO_s013
                                 + 0.54016 w_k1_MAT_s013
                                 + 0.843013 w_k1_MSFT_s013
                                 + 0.78557 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.18622 w_k1_ADM_s013
                                 + 1.24346 w_k1_AXP_s013
                                 + 1.21102 w_k1_DIS_s013
                                 + 1.18157 w_k1_FDX_s013
                                 + 1.45457 w_k1_IBM_s013
                                 + 1.2288 w_k1_INTC_s013 + 1.18525 w_k1_KO_s013
                                 + 1.42065 w_k1_MAT_s013
                                 + 1.10454 w_k1_MSFT_s013
                                 + 1.06002 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.859446 w_k1_ADM_s014
                                 + 0.909132 w_k1_AXP_s014
                                 + 0.93557 w_k1_DIS_s014
                                 + 0.86038 w_k1_FDX_s014
                                 + 0.991508 w_k1_IBM_s014
                                 + 1.28495 w_k1_INTC_s014
                                 + 0.888028 w_k1_KO_s014
                                 + 1.03365 w_k1_MAT_s014
                                 + 0.936591 w_k1_MSFT_s014
                                 + 0.889732 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.881268 w_k1_ADM_s014
                                 + 0.800192 w_k1_AXP_s014
                                 + 1.10196 w_k1_DIS_s014
                                 + 0.986612 w_k1_FDX_s014
                                 + 1.01124 w_k1_IBM_s014
                                 + 1.19928 w_k1_INTC_s014
                                 + 1.08659 w_k1_KO_s014
                                 + 0.855503 w_k1_MAT_s014
                                 + 1.14133 w_k1_MSFT_s014
                                 + 1.03797 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.857832 w_k1_ADM_s014
                                 + 0.588199 w_k1_AXP_s014
                                 + 0.752836 w_k1_DIS_s014
                                 + 0.761114 w_k1_FDX_s014
                                 + 0.765807 w_k1_IBM_s014
                                 + 1.08225 w_k1_INTC_s014
                                 + 0.975835 w_k1_KO_s014
                                 + 0.55985 w_k1_MAT_s014
                                 + 0.980119 w_k1_MSFT_s014
                                 + 0.747885 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.731661 w_k1_ADM_s014
                                 + 0.700513 w_k1_AXP_s014
                                 + 0.6134 w_k1_DIS_s014
                                 + 0.774791 w_k1_FDX_s014
                                 + 0.613478 w_k1_IBM_s014
                                 + 0.943442 w_k1_INTC_s014
                                 + 0.771441 w_k1_KO_s014
                                 + 0.574581 w_k1_MAT_s014
                                 + 0.729364 w_k1_MSFT_s014
                                 + 0.714059 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.11775 w_k1_ADM_s014
                                 + 0.990001 w_k1_AXP_s014
                                 + 1.14553 w_k1_DIS_s014
                                 + 1.14108 w_k1_FDX_s014
                                 + 0.802475 w_k1_IBM_s014
                                 + 1.04757 w_k1_INTC_s014
                                 + 1.09852 w_k1_KO_s014
                                 + 0.993248 w_k1_MAT_s014
                                 + 1.17026 w_k1_MSFT_s014
                                 + 1.09579 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.065 w_k1_ADM_s014
                                 + 1.11753 w_k1_AXP_s014
                                 + 1.15572 w_k1_DIS_s014 + 1.0725 w_k1_FDX_s014
                                 + 1.25148 w_k1_IBM_s014
                                 + 1.31005 w_k1_INTC_s014
                                 + 1.43654 w_k1_KO_s014 + 1.6214 w_k1_MAT_s014
                                 + 1.11228 w_k1_MSFT_s014 + 1.1661 w_k1_PG_s014
                                 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.09261 w_k1_ADM_s014
                                 + 0.599039 w_k1_AXP_s014
                                 + 0.731199 w_k1_DIS_s014
                                 + 0.735491 w_k1_FDX_s014
                                 + 1.2172 w_k1_IBM_s014
                                 + 0.738925 w_k1_INTC_s014
                                 + 1.01853 w_k1_KO_s014 + 1.0092 w_k1_MAT_s014
                                 + 0.758547 w_k1_MSFT_s014
                                 + 0.974327 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.56536 w_k1_ADM_s014
                                 + 1.2872 w_k1_AXP_s014 + 1.65658 w_k1_DIS_s014
                                 + 1.41866 w_k1_FDX_s014
                                 + 1.50677 w_k1_IBM_s014
                                 + 1.40196 w_k1_INTC_s014
                                 + 1.58415 w_k1_KO_s014 + 1.87123 w_k1_MAT_s014
                                 + 1.25449 w_k1_MSFT_s014
                                 + 1.27721 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.03169 w_k1_ADM_s015
                                 + 1.38967 w_k1_AXP_s015
                                 + 1.24954 w_k1_DIS_s015
                                 + 1.12508 w_k1_FDX_s015
                                 + 1.62748 w_k1_IBM_s015
                                 + 1.2422 w_k1_INTC_s015 + 1.31423 w_k1_KO_s015
                                 + 1.37932 w_k1_MAT_s015
                                 + 1.18518 w_k1_MSFT_s015
                                 + 1.15344 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.960082 w_k1_ADM_s015
                                 + 1.05666 w_k1_AXP_s015
                                 + 0.863638 w_k1_DIS_s015
                                 + 0.98811 w_k1_FDX_s015
                                 + 0.948772 w_k1_IBM_s015
                                 + 1.31602 w_k1_INTC_s015
                                 + 1.11821 w_k1_KO_s015 + 1.22427 w_k1_MAT_s015
                                 + 1.13513 w_k1_MSFT_s015
                                 + 0.923362 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.25173 w_k1_ADM_s015
                                 + 1.42127 w_k1_AXP_s015
                                 + 1.34936 w_k1_DIS_s015
                                 + 1.24635 w_k1_FDX_s015
                                 + 1.17014 w_k1_IBM_s015
                                 + 1.43306 w_k1_INTC_s015
                                 + 1.28565 w_k1_KO_s015 + 1.06909 w_k1_MAT_s015
                                 + 1.26602 w_k1_MSFT_s015 + 1.0659 w_k1_PG_s015
                                 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.910743 w_k1_ADM_s015
                                 + 0.785557 w_k1_AXP_s015
                                 + 0.668633 w_k1_DIS_s015
                                 + 0.806267 w_k1_FDX_s015
                                 + 0.74291 w_k1_IBM_s015
                                 + 1.09702 w_k1_INTC_s015
                                 + 0.686048 w_k1_KO_s015
                                 + 0.705379 w_k1_MAT_s015
                                 + 0.754199 w_k1_MSFT_s015
                                 + 0.78661 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.12375 w_k1_ADM_s015
                                 + 1.59138 w_k1_AXP_s015
                                 + 1.69299 w_k1_DIS_s015
                                 + 1.43718 w_k1_FDX_s015
                                 + 1.35506 w_k1_IBM_s015
                                 + 1.32844 w_k1_INTC_s015
                                 + 1.31099 w_k1_KO_s015 + 1.85281 w_k1_MAT_s015
                                 + 1.26989 w_k1_MSFT_s015
                                 + 1.14943 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.0954 w_k1_ADM_s015
                                 + 1.10371 w_k1_AXP_s015
                                 + 1.12711 w_k1_DIS_s015
                                 + 1.18143 w_k1_FDX_s015
                                 + 0.79938 w_k1_IBM_s015
                                 + 1.3663 w_k1_INTC_s015 + 1.1224 w_k1_KO_s015
                                 + 0.844757 w_k1_MAT_s015
                                 + 1.29518 w_k1_MSFT_s015
                                 + 1.07161 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.930689 w_k1_ADM_s015
                                 + 0.83059 w_k1_AXP_s015
                                 + 0.800737 w_k1_DIS_s015
                                 + 0.823575 w_k1_FDX_s015
                                 + 1.42638 w_k1_IBM_s015
                                 + 1.15082 w_k1_INTC_s015
                                 + 1.13741 w_k1_KO_s015 + 1.08914 w_k1_MAT_s015
                                 + 0.979217 w_k1_MSFT_s015
                                 + 0.879177 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.908057 w_k1_ADM_s015
                                 + 0.873674 w_k1_AXP_s015
                                 + 1.08557 w_k1_DIS_s015
                                 + 0.916028 w_k1_FDX_s015
                                 + 1.10325 w_k1_IBM_s015
                                 + 1.26844 w_k1_INTC_s015
                                 + 1.12335 w_k1_KO_s015
                                 + 0.954352 w_k1_MAT_s015
                                 + 1.16591 w_k1_MSFT_s015
                                 + 0.959139 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.18027 w_k1_ADM_s016
                                 + 0.897889 w_k1_AXP_s016
                                 + 1.18124 w_k1_DIS_s016
                                 + 1.15362 w_k1_FDX_s016
                                 + 0.837186 w_k1_IBM_s016
                                 + 1.07464 w_k1_INTC_s016 + 1.0165 w_k1_KO_s016
                                 + 0.874267 w_k1_MAT_s016
                                 + 1.19906 w_k1_MSFT_s016 + 1.0951 w_k1_PG_s016
                                 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.863475 w_k1_ADM_s016
                                 + 1.33926 w_k1_AXP_s016
                                 + 0.831449 w_k1_DIS_s016
                                 + 1.04799 w_k1_FDX_s016
                                 + 1.18554 w_k1_IBM_s016
                                 + 1.07511 w_k1_INTC_s016
                                 + 1.00637 w_k1_KO_s016 + 1.67433 w_k1_MAT_s016
                                 + 0.999232 w_k1_MSFT_s016
                                 + 1.00452 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.802446 w_k1_ADM_s016
                                 + 0.941756 w_k1_AXP_s016
                                 + 0.754993 w_k1_DIS_s016
                                 + 0.949973 w_k1_FDX_s016
                                 + 0.803953 w_k1_IBM_s016
                                 + 0.83905 w_k1_INTC_s016
                                 + 0.860246 w_k1_KO_s016
                                 + 0.930328 w_k1_MAT_s016
                                 + 0.804667 w_k1_MSFT_s016
                                 + 0.871464 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.24452 w_k1_ADM_s016
                                 + 1.4169 w_k1_AXP_s016 + 1.65265 w_k1_DIS_s016
                                 + 1.47434 w_k1_FDX_s016
                                 + 1.24662 w_k1_IBM_s016
                                 + 1.14204 w_k1_INTC_s016
                                 + 1.31996 w_k1_KO_s016 + 1.14079 w_k1_MAT_s016
                                 + 1.28536 w_k1_MSFT_s016
                                 + 1.26917 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.853664 w_k1_ADM_s016
                                 + 0.729862 w_k1_AXP_s016
                                 + 0.687331 w_k1_DIS_s016
                                 + 0.775502 w_k1_FDX_s016
                                 + 0.941002 w_k1_IBM_s016
                                 + 0.800703 w_k1_INTC_s016
                                 + 0.908484 w_k1_KO_s016
                                 + 1.06981 w_k1_MAT_s016
                                 + 0.834877 w_k1_MSFT_s016
                                 + 0.850649 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.941412 w_k1_ADM_s016
                                 + 0.814747 w_k1_AXP_s016
                                 + 0.990076 w_k1_DIS_s016
                                 + 0.992996 w_k1_FDX_s016
                                 + 1.04712 w_k1_IBM_s016
                                 + 1.04325 w_k1_INTC_s016
                                 + 1.03363 w_k1_KO_s016
                                 + 0.931636 w_k1_MAT_s016
                                 + 1.12531 w_k1_MSFT_s016
                                 + 1.03674 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.839082 w_k1_ADM_s016
                                 + 1.04613 w_k1_AXP_s016
                                 + 1.14087 w_k1_DIS_s016
                                 + 1.01109 w_k1_FDX_s016
                                 + 1.44303 w_k1_IBM_s016
                                 + 0.921516 w_k1_INTC_s016
                                 + 1.21323 w_k1_KO_s016 + 1.84604 w_k1_MAT_s016
                                 + 0.986494 w_k1_MSFT_s016
                                 + 1.13347 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.99844 w_k1_ADM_s016
                                 + 1.19185 w_k1_AXP_s016
                                 + 0.949782 w_k1_DIS_s016
                                 + 1.10907 w_k1_FDX_s016
                                 + 0.915201 w_k1_IBM_s016
                                 + 1.15282 w_k1_INTC_s016
                                 + 1.02664 w_k1_KO_s016 + 1.37694 w_k1_MAT_s016
                                 + 0.997284 w_k1_MSFT_s016
                                 + 1.07729 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.26923 w_k1_ADM_s017
                                 + 1.55848 w_k1_AXP_s017 + 1.129 w_k1_DIS_s017
                                 + 1.23536 w_k1_FDX_s017
                                 + 0.895681 w_k1_IBM_s017
                                 + 1.02611 w_k1_INTC_s017
                                 + 1.00713 w_k1_KO_s017
                                 + 0.978881 w_k1_MAT_s017
                                 + 1.19058 w_k1_MSFT_s017
                                 + 1.00134 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.773201 w_k1_ADM_s017
                                 + 0.744313 w_k1_AXP_s017
                                 + 0.791677 w_k1_DIS_s017
                                 + 0.910453 w_k1_FDX_s017
                                 + 0.931695 w_k1_IBM_s017
                                 + 0.898919 w_k1_INTC_s017
                                 + 0.802729 w_k1_KO_s017
                                 + 0.598255 w_k1_MAT_s017
                                 + 0.938402 w_k1_MSFT_s017
                                 + 0.909127 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.04889 w_k1_ADM_s017
                                 + 0.921714 w_k1_AXP_s017
                                 + 0.795965 w_k1_DIS_s017
                                 + 0.892253 w_k1_FDX_s017
                                 + 0.898754 w_k1_IBM_s017
                                 + 1.10298 w_k1_INTC_s017
                                 + 0.949291 w_k1_KO_s017
                                 + 0.685221 w_k1_MAT_s017
                                 + 0.972578 w_k1_MSFT_s017
                                 + 1.08703 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.846416 w_k1_ADM_s017
                                 + 1.15798 w_k1_AXP_s017
                                 + 0.878081 w_k1_DIS_s017
                                 + 0.960128 w_k1_FDX_s017
                                 + 1.24648 w_k1_IBM_s017
                                 + 0.997082 w_k1_INTC_s017
                                 + 1.12359 w_k1_KO_s017 + 1.45415 w_k1_MAT_s017
                                 + 1.02948 w_k1_MSFT_s017
                                 + 1.06934 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.863351 w_k1_ADM_s017
                                 + 0.810688 w_k1_AXP_s017
                                 + 0.708091 w_k1_DIS_s017
                                 + 0.888305 w_k1_FDX_s017
                                 + 0.617778 w_k1_IBM_s017
                                 + 1.08195 w_k1_INTC_s017
                                 + 0.684075 w_k1_KO_s017
                                 + 0.625881 w_k1_MAT_s017
                                 + 1.00908 w_k1_MSFT_s017
                                 + 0.893723 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.15167 w_k1_ADM_s017
                                 + 1.2651 w_k1_AXP_s017 + 1.19015 w_k1_DIS_s017
                                 + 1.05929 w_k1_FDX_s017
                                 + 1.21252 w_k1_IBM_s017
                                 + 1.33163 w_k1_INTC_s017
                                 + 1.21663 w_k1_KO_s017 + 1.0186 w_k1_MAT_s017
                                 + 1.20736 w_k1_MSFT_s017
                                 + 1.11709 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.19573 w_k1_ADM_s017
                                 + 1.52757 w_k1_AXP_s017
                                 + 1.78098 w_k1_DIS_s017
                                 + 1.32156 w_k1_FDX_s017
                                 + 1.23029 w_k1_IBM_s017
                                 + 1.36062 w_k1_INTC_s017
                                 + 1.38663 w_k1_KO_s017 + 1.11928 w_k1_MAT_s017
                                 + 1.28492 w_k1_MSFT_s017
                                 + 1.23431 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.908181 w_k1_ADM_s017
                                 + 0.879217 w_k1_AXP_s017
                                 + 0.998985 w_k1_DIS_s017
                                 + 0.894777 w_k1_FDX_s017
                                 + 1.26963 w_k1_IBM_s017
                                 + 1.08135 w_k1_INTC_s017
                                 + 0.998609 w_k1_KO_s017
                                 + 1.08992 w_k1_MAT_s017
                                 + 1.05064 w_k1_MSFT_s017
                                 + 0.958531 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.74745 w_k1_ADM_s100
                                 + 1.02856 w_k1_AXP_s100
                                 + 1.33121 w_k1_DIS_s100
                                 + 1.04066 w_k1_FDX_s100
                                 + 1.79198 w_k1_IBM_s100
                                 + 1.0755 w_k1_INTC_s100 + 1.24477 w_k1_KO_s100
                                 + 1.39969 w_k1_MAT_s100
                                 + 1.07689 w_k1_MSFT_s100
                                 + 1.07815 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.986828 w_k1_ADM_s100
                                 + 1.36654 w_k1_AXP_s100
                                 + 1.05948 w_k1_DIS_s100
                                 + 1.15929 w_k1_FDX_s100
                                 + 1.14479 w_k1_IBM_s100
                                 + 1.03023 w_k1_INTC_s100
                                 + 1.08949 w_k1_KO_s100 + 1.51818 w_k1_MAT_s100
                                 + 0.972583 w_k1_MSFT_s100
                                 + 1.04686 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.77174 w_k1_ADM_s100
                                 + 0.583493 w_k1_AXP_s100
                                 + 0.861883 w_k1_DIS_s100
                                 + 0.843708 w_k1_FDX_s100
                                 + 0.702078 w_k1_IBM_s100
                                 + 0.690508 w_k1_INTC_s100
                                 + 0.792698 w_k1_KO_s100
                                 + 0.650643 w_k1_MAT_s100
                                 + 0.863876 w_k1_MSFT_s100
                                 + 0.873302 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.05855 w_k1_ADM_s100
                                 + 0.974256 w_k1_AXP_s100
                                 + 1.11453 w_k1_DIS_s100
                                 + 1.02636 w_k1_FDX_s100
                                 + 1.03731 w_k1_IBM_s100
                                 + 0.97814 w_k1_INTC_s100 + 1.191 w_k1_KO_s100
                                 + 1.27612 w_k1_MAT_s100
                                 + 1.07127 w_k1_MSFT_s100
                                 + 1.10122 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.1416 w_k1_ADM_s100
                                 + 1.4786 w_k1_AXP_s100 + 1.66719 w_k1_DIS_s100
                                 + 1.23077 w_k1_FDX_s100
                                 + 1.47858 w_k1_IBM_s100
                                 + 1.26937 w_k1_INTC_s100
                                 + 1.54433 w_k1_KO_s100 + 1.59027 w_k1_MAT_s100
                                 + 1.29364 w_k1_MSFT_s100
                                 + 1.27601 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.833874 w_k1_ADM_s100
                                 + 1.01466 w_k1_AXP_s100
                                 + 0.687393 w_k1_DIS_s100
                                 + 0.83665 w_k1_FDX_s100
                                 + 0.951039 w_k1_IBM_s100
                                 + 0.853157 w_k1_INTC_s100
                                 + 0.741999 w_k1_KO_s100
                                 + 1.02092 w_k1_MAT_s100
                                 + 0.804535 w_k1_MSFT_s100
                                 + 0.897544 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.992258 w_k1_ADM_s100
                                 + 0.938891 w_k1_AXP_s100
                                 + 1.55223 w_k1_DIS_s100
                                 + 1.19007 w_k1_FDX_s100
                                 + 1.01852 w_k1_IBM_s100
                                 + 0.991833 w_k1_INTC_s100
                                 + 1.24411 w_k1_KO_s100
                                 + 0.939211 w_k1_MAT_s100
                                 + 1.15042 w_k1_MSFT_s100
                                 + 1.16323 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.48027 w_k1_ADM_s100
                                 + 1.45539 w_k1_AXP_s100
                                 + 1.50754 w_k1_DIS_s100
                                 + 1.41112 w_k1_FDX_s100
                                 + 1.09742 w_k1_IBM_s100
                                 + 0.962255 w_k1_INTC_s100
                                 + 1.18492 w_k1_KO_s100 + 1.11206 w_k1_MAT_s100
                                 + 1.17438 w_k1_MSFT_s100
                                 + 1.18464 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.28125 w_k1_ADM_s101
                                 + 1.07946 w_k1_AXP_s101 + 1.3413 w_k1_DIS_s101
                                 + 1.13412 w_k1_FDX_s101
                                 + 1.42104 w_k1_IBM_s101
                                 + 1.12344 w_k1_INTC_s101
                                 + 1.51034 w_k1_KO_s101 + 1.44367 w_k1_MAT_s101
                                 + 1.06835 w_k1_MSFT_s101
                                 + 1.20941 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.978498 w_k1_ADM_s101
                                 + 1.02861 w_k1_AXP_s101
                                 + 0.92592 w_k1_DIS_s101
                                 + 0.950822 w_k1_FDX_s101
                                 + 1.18197 w_k1_IBM_s101
                                 + 0.82127 w_k1_INTC_s101
                                 + 1.25188 w_k1_KO_s101 + 1.51913 w_k1_MAT_s101
                                 + 0.947213 w_k1_MSFT_s101
                                 + 1.03966 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.57415 w_k1_ADM_s101
                                 + 1.71696 w_k1_AXP_s101
                                 + 1.43197 w_k1_DIS_s101
                                 + 1.40677 w_k1_FDX_s101
                                 + 1.20943 w_k1_IBM_s101
                                 + 1.40168 w_k1_INTC_s101
                                 + 1.47358 w_k1_KO_s101
                                 + 0.934995 w_k1_MAT_s101
                                 + 1.05707 w_k1_MSFT_s101
                                 + 1.18506 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.19582 w_k1_ADM_s101
                                 + 1.27114 w_k1_AXP_s101
                                 + 1.11038 w_k1_DIS_s101
                                 + 1.25418 w_k1_FDX_s101
                                 + 1.01914 w_k1_IBM_s101
                                 + 1.06092 w_k1_INTC_s101
                                 + 1.08208 w_k1_KO_s101 + 1.12925 w_k1_MAT_s101
                                 + 1.10343 w_k1_MSFT_s101
                                 + 1.08085 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.800088 w_k1_ADM_s101
                                 + 0.85251 w_k1_AXP_s101
                                 + 0.735233 w_k1_DIS_s101
                                 + 0.874939 w_k1_FDX_s101
                                 + 0.732551 w_k1_IBM_s101
                                 + 0.815682 w_k1_INTC_s101
                                 + 1.00588 w_k1_KO_s101
                                 + 0.746478 w_k1_MAT_s101
                                 + 0.831912 w_k1_MSFT_s101
                                 + 1.01707 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.16873 w_k1_ADM_s101
                                 + 1.14066 w_k1_AXP_s101
                                 + 0.949915 w_k1_DIS_s101
                                 + 1.09636 w_k1_FDX_s101
                                 + 0.873895 w_k1_IBM_s101
                                 + 0.871358 w_k1_INTC_s101
                                 + 1.19481 w_k1_KO_s101
                                 + 0.891875 w_k1_MAT_s101
                                 + 1.02642 w_k1_MSFT_s101
                                 + 1.13085 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.01347 w_k1_ADM_s101
                                 + 0.697642 w_k1_AXP_s101
                                 + 0.90065 w_k1_DIS_s101
                                 + 0.907443 w_k1_FDX_s101
                                 + 0.879792 w_k1_IBM_s101
                                 + 0.918804 w_k1_INTC_s101
                                 + 1.14377 w_k1_KO_s101
                                 + 0.642495 w_k1_MAT_s101
                                 + 1.05813 w_k1_MSFT_s101
                                 + 0.940481 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.23202 w_k1_ADM_s101
                                 + 0.865073 w_k1_AXP_s101
                                 + 1.34274 w_k1_DIS_s101
                                 + 1.08582 w_k1_FDX_s101
                                 + 1.26021 w_k1_IBM_s101
                                 + 1.07073 w_k1_INTC_s101
                                 + 1.61588 w_k1_KO_s101 + 1.06667 w_k1_MAT_s101
                                 + 1.24315 w_k1_MSFT_s101
                                 + 1.13614 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.104 w_k1_ADM_s102
                                 + 1.24785 w_k1_AXP_s102
                                 + 1.20658 w_k1_DIS_s102
                                 + 1.27928 w_k1_FDX_s102
                                 + 0.816576 w_k1_IBM_s102
                                 + 1.10812 w_k1_INTC_s102 + 1.0005 w_k1_KO_s102
                                 + 0.840287 w_k1_MAT_s102
                                 + 1.03338 w_k1_MSFT_s102
                                 + 1.17706 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.711223 w_k1_ADM_s102
                                 + 0.751145 w_k1_AXP_s102
                                 + 0.555747 w_k1_DIS_s102
                                 + 0.712301 w_k1_FDX_s102
                                 + 0.624173 w_k1_IBM_s102
                                 + 0.892387 w_k1_INTC_s102
                                 + 0.741184 w_k1_KO_s102
                                 + 1.06178 w_k1_MAT_s102
                                 + 0.75861 w_k1_MSFT_s102 + 0.8831 w_k1_PG_s102
                                 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.05603 w_k1_ADM_s102
                                 + 1.51318 w_k1_AXP_s102
                                 + 1.47594 w_k1_DIS_s102
                                 + 1.27729 w_k1_FDX_s102
                                 + 1.61808 w_k1_IBM_s102
                                 + 1.0766 w_k1_INTC_s102 + 1.34307 w_k1_KO_s102
                                 + 2.08573 w_k1_MAT_s102
                                 + 1.26442 w_k1_MSFT_s102
                                 + 1.23021 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.99816 w_k1_ADM_s102
                                 + 0.834394 w_k1_AXP_s102
                                 + 0.782615 w_k1_DIS_s102
                                 + 0.960232 w_k1_FDX_s102
                                 + 0.763792 w_k1_IBM_s102
                                 + 0.9434 w_k1_INTC_s102
                                 + 0.868531 w_k1_KO_s102
                                 + 0.769651 w_k1_MAT_s102
                                 + 0.986101 w_k1_MSFT_s102
                                 + 0.979396 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.846002 w_k1_ADM_s102
                                 + 0.752201 w_k1_AXP_s102
                                 + 0.960092 w_k1_DIS_s102
                                 + 0.846118 w_k1_FDX_s102
                                 + 1.32331 w_k1_IBM_s102
                                 + 0.971582 w_k1_INTC_s102
                                 + 1.17032 w_k1_KO_s102 + 1.24578 w_k1_MAT_s102
                                 + 0.988922 w_k1_MSFT_s102
                                 + 1.04509 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.17462 w_k1_ADM_s102
                                 + 1.36519 w_k1_AXP_s102
                                 + 1.15356 w_k1_DIS_s102
                                 + 1.12668 w_k1_FDX_s102
                                 + 1.21036 w_k1_IBM_s102
                                 + 1.1911 w_k1_INTC_s102 + 1.22027 w_k1_KO_s102
                                 + 2.06004 w_k1_MAT_s102
                                 + 1.20064 w_k1_MSFT_s102
                                 + 1.15271 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.698358 w_k1_ADM_s102
                                 + 0.73454 w_k1_AXP_s102
                                 + 0.830097 w_k1_DIS_s102
                                 + 0.794224 w_k1_FDX_s102
                                 + 1.03063 w_k1_IBM_s102
                                 + 0.855192 w_k1_INTC_s102
                                 + 0.983915 w_k1_KO_s102
                                 + 1.24426 w_k1_MAT_s102
                                 + 0.854229 w_k1_MSFT_s102
                                 + 0.928626 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.975274 w_k1_ADM_s102
                                 + 1.03504 w_k1_AXP_s102
                                 + 0.986953 w_k1_DIS_s102
                                 + 1.0875 w_k1_FDX_s102
                                 + 0.890727 w_k1_IBM_s102
                                 + 1.05235 w_k1_INTC_s102
                                 + 1.03937 w_k1_KO_s102
                                 + 0.860518 w_k1_MAT_s102
                                 + 1.01212 w_k1_MSFT_s102
                                 + 0.943527 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.23724 w_k1_ADM_s103
                                 + 1.16203 w_k1_AXP_s103
                                 + 1.19468 w_k1_DIS_s103
                                 + 1.16612 w_k1_FDX_s103
                                 + 1.14104 w_k1_IBM_s103
                                 + 1.06824 w_k1_INTC_s103
                                 + 1.11543 w_k1_KO_s103 + 1.15564 w_k1_MAT_s103
                                 + 1.09141 w_k1_MSFT_s103
                                 + 1.30314 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.844097 w_k1_ADM_s103
                                 + 0.545662 w_k1_AXP_s103
                                 + 0.647468 w_k1_DIS_s103
                                 + 0.742897 w_k1_FDX_s103
                                 + 0.686902 w_k1_IBM_s103
                                 + 0.745467 w_k1_INTC_s103
                                 + 0.76836 w_k1_KO_s103
                                 + 0.534773 w_k1_MAT_s103
                                 + 0.714328 w_k1_MSFT_s103
                                 + 0.850049 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.885264 w_k1_ADM_s103
                                 + 0.821846 w_k1_AXP_s103
                                 + 0.782808 w_k1_DIS_s103
                                 + 0.826881 w_k1_FDX_s103
                                 + 0.772169 w_k1_IBM_s103
                                 + 0.896228 w_k1_INTC_s103
                                 + 0.935402 w_k1_KO_s103
                                 + 0.701489 w_k1_MAT_s103
                                 + 0.772219 w_k1_MSFT_s103
                                 + 1.10631 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.01956 w_k1_ADM_s103
                                 + 0.936721 w_k1_AXP_s103
                                 + 1.30893 w_k1_DIS_s103
                                 + 1.11679 w_k1_FDX_s103
                                 + 0.86963 w_k1_IBM_s103
                                 + 1.22309 w_k1_INTC_s103
                                 + 1.07192 w_k1_KO_s103
                                 + 0.752141 w_k1_MAT_s103
                                 + 1.05172 w_k1_MSFT_s103 + 1.1316 w_k1_PG_s103
                                 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.4696 w_k1_ADM_s103
                                 + 1.65956 w_k1_AXP_s103
                                 + 1.56231 w_k1_DIS_s103
                                 + 1.27945 w_k1_FDX_s103
                                 + 1.58341 w_k1_IBM_s103
                                 + 1.00979 w_k1_INTC_s103
                                 + 1.43723 w_k1_KO_s103 + 2.21524 w_k1_MAT_s103
                                 + 1.04724 w_k1_MSFT_s103
                                 + 1.16504 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.96344 w_k1_ADM_s103
                                 + 1.07411 w_k1_AXP_s103
                                 + 0.914417 w_k1_DIS_s103
                                 + 1.04891 w_k1_FDX_s103
                                 + 0.736848 w_k1_IBM_s103
                                 + 1.10455 w_k1_INTC_s103
                                 + 0.793726 w_k1_KO_s103
                                 + 0.956301 w_k1_MAT_s103
                                 + 0.878893 w_k1_MSFT_s103
                                 + 1.01809 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.85754 w_k1_ADM_s103
                                 + 0.719799 w_k1_AXP_s103
                                 + 0.846741 w_k1_DIS_s103
                                 + 0.855339 w_k1_FDX_s103
                                 + 1.00638 w_k1_IBM_s103
                                 + 0.941194 w_k1_INTC_s103
                                 + 1.06537 w_k1_KO_s103 + 1.01131 w_k1_MAT_s103
                                 + 0.955995 w_k1_MSFT_s103
                                 + 1.04541 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.06473 w_k1_ADM_s103
                                 + 0.979355 w_k1_AXP_s103
                                 + 1.02931 w_k1_DIS_s103
                                 + 0.960096 w_k1_FDX_s103
                                 + 1.35856 w_k1_IBM_s103
                                 + 1.16848 w_k1_INTC_s103
                                 + 1.19015 w_k1_KO_s103 + 1.09111 w_k1_MAT_s103
                                 + 1.07243 w_k1_MSFT_s103
                                 + 1.12454 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.30045 w_k1_ADM_s104
                                 + 1.46002 w_k1_AXP_s104
                                 + 1.30697 w_k1_DIS_s104
                                 + 1.31588 w_k1_FDX_s104 + 1.2494 w_k1_IBM_s104
                                 + 1.41826 w_k1_INTC_s104
                                 + 1.29353 w_k1_KO_s104 + 1.38888 w_k1_MAT_s104
                                 + 1.17182 w_k1_MSFT_s104
                                 + 1.20864 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.882972 w_k1_ADM_s104
                                 + 0.944394 w_k1_AXP_s104
                                 + 0.825034 w_k1_DIS_s104
                                 + 0.799524 w_k1_FDX_s104
                                 + 0.932957 w_k1_IBM_s104
                                 + 0.834767 w_k1_INTC_s104
                                 + 0.995004 w_k1_KO_s104
                                 + 1.23853 w_k1_MAT_s104
                                 + 0.83089 w_k1_MSFT_s104
                                 + 0.976928 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.20042 w_k1_ADM_s104
                                 + 1.37942 w_k1_AXP_s104
                                 + 1.78143 w_k1_DIS_s104
                                 + 1.34004 w_k1_FDX_s104 + 1.3275 w_k1_IBM_s104
                                 + 1.35052 w_k1_INTC_s104
                                 + 1.46541 w_k1_KO_s104 + 1.44074 w_k1_MAT_s104
                                 + 1.33108 w_k1_MSFT_s104
                                 + 1.16537 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.0366 w_k1_ADM_s104
                                 + 0.932088 w_k1_AXP_s104
                                 + 1.21813 w_k1_DIS_s104
                                 + 1.08286 w_k1_FDX_s104
                                 + 1.16176 w_k1_IBM_s104
                                 + 1.06441 w_k1_INTC_s104 + 1.1172 w_k1_KO_s104
                                 + 1.08978 w_k1_MAT_s104
                                 + 1.06334 w_k1_MSFT_s104 + 1.0744 w_k1_PG_s104
                                 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.953366 w_k1_ADM_s104
                                 + 0.854372 w_k1_AXP_s104
                                 + 0.970364 w_k1_DIS_s104
                                 + 1.0054 w_k1_FDX_s104
                                 + 0.798894 w_k1_IBM_s104
                                 + 1.04964 w_k1_INTC_s104
                                 + 0.941629 w_k1_KO_s104
                                 + 0.571889 w_k1_MAT_s104
                                 + 0.954429 w_k1_MSFT_s104
                                 + 0.902419 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.905433 w_k1_ADM_s104
                                 + 1.17019 w_k1_AXP_s104
                                 + 0.731093 w_k1_DIS_s104
                                 + 0.96401 w_k1_FDX_s104
                                 + 0.858154 w_k1_IBM_s104
                                 + 0.902357 w_k1_INTC_s104
                                 + 0.79504 w_k1_KO_s104
                                 + 0.996484 w_k1_MAT_s104
                                 + 0.852128 w_k1_MSFT_s104
                                 + 0.889167 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.729769 w_k1_ADM_s104
                                 + 0.620047 w_k1_AXP_s104
                                 + 0.712986 w_k1_DIS_s104
                                 + 0.771029 w_k1_FDX_s104
                                 + 0.756073 w_k1_IBM_s104
                                 + 0.796971 w_k1_INTC_s104
                                 + 0.809887 w_k1_KO_s104
                                 + 0.667355 w_k1_MAT_s104
                                 + 0.757107 w_k1_MSFT_s104
                                 + 0.845916 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.78288 w_k1_ADM_s104
                                 + 0.877873 w_k1_AXP_s104
                                 + 1.01436 w_k1_DIS_s104
                                 + 0.856571 w_k1_FDX_s104
                                 + 1.42915 w_k1_IBM_s104
                                 + 0.862949 w_k1_INTC_s104
                                 + 1.02652 w_k1_KO_s104 + 1.16602 w_k1_MAT_s104
                                 + 0.845915 w_k1_MSFT_s104
                                 + 1.02015 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.702027 w_k1_ADM_s105
                                 + 0.612435 w_k1_AXP_s105
                                 + 0.654189 w_k1_DIS_s105
                                 + 0.736934 w_k1_FDX_s105
                                 + 0.744605 w_k1_IBM_s105
                                 + 1.0088 w_k1_INTC_s105
                                 + 0.866514 w_k1_KO_s105
                                 + 0.783083 w_k1_MAT_s105
                                 + 0.816507 w_k1_MSFT_s105
                                 + 0.833734 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.752134 w_k1_ADM_s105
                                 + 0.95447 w_k1_AXP_s105
                                 + 0.943472 w_k1_DIS_s105
                                 + 0.96557 w_k1_FDX_s105
                                 + 0.869513 w_k1_IBM_s105
                                 + 0.932602 w_k1_INTC_s105
                                 + 1.08266 w_k1_KO_s105 + 1.07472 w_k1_MAT_s105
                                 + 0.920682 w_k1_MSFT_s105
                                 + 1.0702 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.809281 w_k1_ADM_s105
                                 + 1.13536 w_k1_AXP_s105
                                 + 0.868166 w_k1_DIS_s105
                                 + 0.864202 w_k1_FDX_s105
                                 + 1.07531 w_k1_IBM_s105
                                 + 0.770392 w_k1_INTC_s105
                                 + 1.02031 w_k1_KO_s105 + 1.3256 w_k1_MAT_s105
                                 + 0.754433 w_k1_MSFT_s105
                                 + 0.883244 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.23143 w_k1_ADM_s105
                                 + 1.06852 w_k1_AXP_s105
                                 + 1.40766 w_k1_DIS_s105
                                 + 1.12785 w_k1_FDX_s105
                                 + 1.10394 w_k1_IBM_s105
                                 + 1.36127 w_k1_INTC_s105
                                 + 1.14278 w_k1_KO_s105 + 1.26554 w_k1_MAT_s105
                                 + 1.28624 w_k1_MSFT_s105
                                 + 1.15958 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.794198 w_k1_ADM_s105
                                 + 0.74379 w_k1_AXP_s105
                                 + 0.809647 w_k1_DIS_s105
                                 + 0.707462 w_k1_FDX_s105
                                 + 1.15605 w_k1_IBM_s105
                                 + 0.865988 w_k1_INTC_s105
                                 + 0.953189 w_k1_KO_s105 + 1.1344 w_k1_MAT_s105
                                 + 0.838621 w_k1_MSFT_s105
                                 + 0.983085 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.3831 w_k1_ADM_s105
                                 + 1.87168 w_k1_AXP_s105
                                 + 1.94951 w_k1_DIS_s105
                                 + 1.38599 w_k1_FDX_s105
                                 + 1.46012 w_k1_IBM_s105
                                 + 1.46908 w_k1_INTC_s105
                                 + 1.65609 w_k1_KO_s105 + 1.57587 w_k1_MAT_s105
                                 + 1.42428 w_k1_MSFT_s105
                                 + 1.36461 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.19953 w_k1_ADM_s105
                                 + 1.79526 w_k1_AXP_s105
                                 + 1.13951 w_k1_DIS_s105 + 1.1436 w_k1_FDX_s105
                                 + 1.17337 w_k1_IBM_s105
                                 + 1.12076 w_k1_INTC_s105
                                 + 1.26605 w_k1_KO_s105 + 2.13661 w_k1_MAT_s105
                                 + 0.974149 w_k1_MSFT_s105
                                 + 1.28518 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.04659 w_k1_ADM_s105
                                 + 1.14209 w_k1_AXP_s105
                                 + 1.51315 w_k1_DIS_s105
                                 + 1.20197 w_k1_FDX_s105
                                 + 0.931074 w_k1_IBM_s105
                                 + 1.01078 w_k1_INTC_s105
                                 + 1.12169 w_k1_KO_s105 + 1.5578 w_k1_MAT_s105
                                 + 1.1913 w_k1_MSFT_s105 + 1.14829 w_k1_PG_s105
                                 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.919037 w_k1_ADM_s106
                                 + 0.697169 w_k1_AXP_s106
                                 + 0.878259 w_k1_DIS_s106
                                 + 0.913622 w_k1_FDX_s106
                                 + 0.874677 w_k1_IBM_s106
                                 + 0.941013 w_k1_INTC_s106
                                 + 0.775589 w_k1_KO_s106
                                 + 0.544183 w_k1_MAT_s106
                                 + 0.97161 w_k1_MSFT_s106
                                 + 0.949859 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.3048 w_k1_ADM_s106
                                 + 1.46538 w_k1_AXP_s106
                                 + 1.56336 w_k1_DIS_s106
                                 + 1.34426 w_k1_FDX_s106
                                 + 1.22482 w_k1_IBM_s106
                                 + 1.20636 w_k1_INTC_s106
                                 + 1.23766 w_k1_KO_s106 + 1.15604 w_k1_MAT_s106
                                 + 1.18776 w_k1_MSFT_s106
                                 + 1.18818 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.992967 w_k1_ADM_s106
                                 + 1.05262 w_k1_AXP_s106
                                 + 0.999392 w_k1_DIS_s106
                                 + 1.1016 w_k1_FDX_s106
                                 + 0.986198 w_k1_IBM_s106
                                 + 1.00065 w_k1_INTC_s106
                                 + 0.76697 w_k1_KO_s106
                                 + 0.925179 w_k1_MAT_s106
                                 + 1.01297 w_k1_MSFT_s106
                                 + 0.99614 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.04078 w_k1_ADM_s106
                                 + 0.907023 w_k1_AXP_s106
                                 + 1.06169 w_k1_DIS_s106
                                 + 0.921087 w_k1_FDX_s106
                                 + 1.5177 w_k1_IBM_s106
                                 + 0.959109 w_k1_INTC_s106
                                 + 1.06833 w_k1_KO_s106 + 1.37064 w_k1_MAT_s106
                                 + 1.05518 w_k1_MSFT_s106
                                 + 1.01099 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.20688 w_k1_ADM_s106
                                 + 1.21053 w_k1_AXP_s106
                                 + 1.67015 w_k1_DIS_s106
                                 + 1.15075 w_k1_FDX_s106
                                 + 1.30405 w_k1_IBM_s106
                                 + 0.984301 w_k1_INTC_s106
                                 + 1.22795 w_k1_KO_s106 + 1.32178 w_k1_MAT_s106
                                 + 1.12664 w_k1_MSFT_s106
                                 + 1.09384 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.24387 w_k1_ADM_s106
                                 + 1.07764 w_k1_AXP_s106 + 1.0149 w_k1_DIS_s106
                                 + 1.09076 w_k1_FDX_s106
                                 + 1.17264 w_k1_IBM_s106
                                 + 1.01282 w_k1_INTC_s106
                                 + 0.913728 w_k1_KO_s106
                                 + 0.810389 w_k1_MAT_s106
                                 + 0.992709 w_k1_MSFT_s106
                                 + 0.971127 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.919264 w_k1_ADM_s106
                                 + 0.721914 w_k1_AXP_s106
                                 + 0.685251 w_k1_DIS_s106
                                 + 0.733072 w_k1_FDX_s106
                                 + 1.01948 w_k1_IBM_s106
                                 + 0.718117 w_k1_INTC_s106
                                 + 0.708732 w_k1_KO_s106
                                 + 0.805013 w_k1_MAT_s106
                                 + 0.70693 w_k1_MSFT_s106
                                 + 0.854182 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.14674 w_k1_ADM_s106
                                 + 0.960694 w_k1_AXP_s106
                                 + 0.75686 w_k1_DIS_s106
                                 + 0.946478 w_k1_FDX_s106
                                 + 1.00487 w_k1_IBM_s106
                                 + 0.729366 w_k1_INTC_s106
                                 + 0.736476 w_k1_KO_s106
                                 + 0.762102 w_k1_MAT_s106
                                 + 0.744688 w_k1_MSFT_s106
                                 + 0.914928 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.97517 w_k1_ADM_s107
                                 + 0.937439 w_k1_AXP_s107
                                 + 1.11684 w_k1_DIS_s107
                                 + 1.16072 w_k1_FDX_s107
                                 + 0.862202 w_k1_IBM_s107
                                 + 1.29983 w_k1_INTC_s107 + 1.0862 w_k1_KO_s107
                                 + 0.701375 w_k1_MAT_s107
                                 + 1.16119 w_k1_MSFT_s107
                                 + 0.951302 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.903939 w_k1_ADM_s107
                                 + 0.695565 w_k1_AXP_s107
                                 + 0.811375 w_k1_DIS_s107
                                 + 0.812089 w_k1_FDX_s107
                                 + 1.10455 w_k1_IBM_s107
                                 + 1.03822 w_k1_INTC_s107
                                 + 1.05571 w_k1_KO_s107
                                 + 0.759988 w_k1_MAT_s107
                                 + 1.02055 w_k1_MSFT_s107
                                 + 0.927259 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.28456 w_k1_ADM_s107
                                 + 1.4377 w_k1_AXP_s107 + 1.48148 w_k1_DIS_s107
                                 + 1.3916 w_k1_FDX_s107
                                 + 0.991721 w_k1_IBM_s107
                                 + 1.20945 w_k1_INTC_s107
                                 + 1.30264 w_k1_KO_s107 + 1.34485 w_k1_MAT_s107
                                 + 1.40405 w_k1_MSFT_s107
                                 + 1.23025 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.017 w_k1_ADM_s107
                                 + 1.22397 w_k1_AXP_s107
                                 + 0.846056 w_k1_DIS_s107
                                 + 0.924219 w_k1_FDX_s107
                                 + 1.37679 w_k1_IBM_s107
                                 + 0.993762 w_k1_INTC_s107
                                 + 1.17752 w_k1_KO_s107 + 1.34512 w_k1_MAT_s107
                                 + 0.958891 w_k1_MSFT_s107
                                 + 1.12236 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.756851 w_k1_ADM_s107
                                 + 0.761774 w_k1_AXP_s107
                                 + 0.658581 w_k1_DIS_s107
                                 + 0.833446 w_k1_FDX_s107
                                 + 0.720688 w_k1_IBM_s107
                                 + 0.807983 w_k1_INTC_s107
                                 + 0.750282 w_k1_KO_s107
                                 + 0.704433 w_k1_MAT_s107
                                 + 0.797631 w_k1_MSFT_s107
                                 + 0.829206 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.1678 w_k1_ADM_s107
                                 + 0.957515 w_k1_AXP_s107
                                 + 1.07515 w_k1_DIS_s107
                                 + 0.997901 w_k1_FDX_s107
                                 + 1.1872 w_k1_IBM_s107
                                 + 1.08141 w_k1_INTC_s107
                                 + 1.24526 w_k1_KO_s107 + 1.26015 w_k1_MAT_s107
                                 + 1.24683 w_k1_MSFT_s107
                                 + 1.03319 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.08948 w_k1_ADM_s107
                                 + 1.12489 w_k1_AXP_s107
                                 + 1.49302 w_k1_DIS_s107
                                 + 1.14217 w_k1_FDX_s107
                                 + 1.80261 w_k1_IBM_s107
                                 + 1.07863 w_k1_INTC_s107
                                 + 1.66069 w_k1_KO_s107 + 1.61409 w_k1_MAT_s107
                                 + 1.13895 w_k1_MSFT_s107
                                 + 1.26583 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.04456 w_k1_ADM_s107
                                 + 1.16135 w_k1_AXP_s107
                                 + 0.983019 w_k1_DIS_s107
                                 + 1.13312 w_k1_FDX_s107
                                 + 1.09509 w_k1_IBM_s107
                                 + 1.18422 w_k1_INTC_s107
                                 + 1.12183 w_k1_KO_s107 + 1.12898 w_k1_MAT_s107
                                 + 1.03248 w_k1_MSFT_s107
                                 + 1.13864 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.21156 w_k1_ADM_s110
                                 + 1.11534 w_k1_AXP_s110
                                 + 0.999595 w_k1_DIS_s110
                                 + 0.964014 w_k1_FDX_s110
                                 + 1.5152 w_k1_IBM_s110
                                 + 0.919648 w_k1_INTC_s110
                                 + 1.12965 w_k1_KO_s110 + 1.68481 w_k1_MAT_s110
                                 + 1.05038 w_k1_MSFT_s110
                                 + 0.975438 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.27521 w_k1_ADM_s110
                                 + 1.17113 w_k1_AXP_s110 + 1.0358 w_k1_DIS_s110
                                 + 1.01813 w_k1_FDX_s110
                                 + 1.01417 w_k1_IBM_s110
                                 + 1.03421 w_k1_INTC_s110
                                 + 0.98992 w_k1_KO_s110
                                 + 0.919519 w_k1_MAT_s110
                                 + 1.09362 w_k1_MSFT_s110
                                 + 1.09614 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.863691 w_k1_ADM_s110
                                 + 0.717418 w_k1_AXP_s110
                                 + 0.695379 w_k1_DIS_s110
                                 + 0.774547 w_k1_FDX_s110
                                 + 0.706134 w_k1_IBM_s110
                                 + 0.891026 w_k1_INTC_s110
                                 + 0.615857 w_k1_KO_s110
                                 + 0.556494 w_k1_MAT_s110
                                 + 0.85261 w_k1_MSFT_s110
                                 + 0.784283 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.52707 w_k1_ADM_s110
                                 + 1.3602 w_k1_AXP_s110 + 1.45876 w_k1_DIS_s110
                                 + 1.31863 w_k1_FDX_s110
                                 + 1.02416 w_k1_IBM_s110
                                 + 1.12544 w_k1_INTC_s110
                                 + 1.15526 w_k1_KO_s110 + 1.1321 w_k1_MAT_s110
                                 + 1.24443 w_k1_MSFT_s110
                                 + 1.13048 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.49778 w_k1_ADM_s110
                                 + 1.32538 w_k1_AXP_s110
                                 + 1.36264 w_k1_DIS_s110
                                 + 1.13645 w_k1_FDX_s110
                                 + 1.42405 w_k1_IBM_s110
                                 + 1.17881 w_k1_INTC_s110
                                 + 1.29667 w_k1_KO_s110 + 1.55336 w_k1_MAT_s110
                                 + 1.19688 w_k1_MSFT_s110
                                 + 1.21601 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.28664 w_k1_ADM_s110
                                 + 0.889131 w_k1_AXP_s110
                                 + 0.849708 w_k1_DIS_s110
                                 + 1.04278 w_k1_FDX_s110
                                 + 0.681157 w_k1_IBM_s110
                                 + 1.041 w_k1_INTC_s110 + 0.779925 w_k1_KO_s110
                                 + 0.816302 w_k1_MAT_s110
                                 + 1.06735 w_k1_MSFT_s110
                                 + 0.97885 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.893695 w_k1_ADM_s110
                                 + 0.717317 w_k1_AXP_s110
                                 + 0.76058 w_k1_DIS_s110
                                 + 0.793545 w_k1_FDX_s110
                                 + 0.999878 w_k1_IBM_s110
                                 + 0.867836 w_k1_INTC_s110
                                 + 0.870052 w_k1_KO_s110
                                 + 0.790773 w_k1_MAT_s110
                                 + 0.926674 w_k1_MSFT_s110
                                 + 0.923407 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.17429 w_k1_ADM_s110
                                 + 1.06814 w_k1_AXP_s110
                                 + 1.03623 w_k1_DIS_s110
                                 + 1.09206 w_k1_FDX_s110
                                 + 0.896595 w_k1_IBM_s110
                                 + 1.16835 w_k1_INTC_s110
                                 + 0.945428 w_k1_KO_s110
                                 + 0.758497 w_k1_MAT_s110
                                 + 1.17995 w_k1_MSFT_s110
                                 + 1.07172 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.894561 w_k1_ADM_s111
                                 + 0.955608 w_k1_AXP_s111
                                 + 0.790494 w_k1_DIS_s111
                                 + 0.847141 w_k1_FDX_s111
                                 + 1.1584 w_k1_IBM_s111
                                 + 0.884818 w_k1_INTC_s111
                                 + 1.0129 w_k1_KO_s111 + 1.38048 w_k1_MAT_s111
                                 + 0.960816 w_k1_MSFT_s111
                                 + 1.07177 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.36557 w_k1_ADM_s111
                                 + 1.51985 w_k1_AXP_s111
                                 + 1.41043 w_k1_DIS_s111 + 1.1794 w_k1_FDX_s111
                                 + 1.3898 w_k1_IBM_s111
                                 + 1.10641 w_k1_INTC_s111
                                 + 1.20122 w_k1_KO_s111 + 1.35632 w_k1_MAT_s111
                                 + 1.13845 w_k1_MSFT_s111
                                 + 1.28777 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.14584 w_k1_ADM_s111
                                 + 1.51235 w_k1_AXP_s111
                                 + 1.17271 w_k1_DIS_s111
                                 + 1.27027 w_k1_FDX_s111
                                 + 0.952774 w_k1_IBM_s111
                                 + 1.25198 w_k1_INTC_s111
                                 + 1.13002 w_k1_KO_s111 + 1.18128 w_k1_MAT_s111
                                 + 1.28755 w_k1_MSFT_s111
                                 + 1.28606 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.01943 w_k1_ADM_s111
                                 + 1.21339 w_k1_AXP_s111 + 1.216 w_k1_DIS_s111
                                 + 1.02982 w_k1_FDX_s111
                                 + 1.90207 w_k1_IBM_s111
                                 + 1.11323 w_k1_INTC_s111
                                 + 1.26241 w_k1_KO_s111 + 1.80655 w_k1_MAT_s111
                                 + 1.05107 w_k1_MSFT_s111
                                 + 1.20717 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.05478 w_k1_ADM_s111
                                 + 0.946301 w_k1_AXP_s111
                                 + 0.942212 w_k1_DIS_s111
                                 + 0.847341 w_k1_FDX_s111
                                 + 1.4431 w_k1_IBM_s111
                                 + 0.808166 w_k1_INTC_s111
                                 + 1.08538 w_k1_KO_s111 + 1.53697 w_k1_MAT_s111
                                 + 0.862269 w_k1_MSFT_s111
                                 + 1.10516 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.983486 w_k1_ADM_s111
                                 + 0.667713 w_k1_AXP_s111
                                 + 0.629409 w_k1_DIS_s111
                                 + 0.839074 w_k1_FDX_s111
                                 + 0.5589 w_k1_IBM_s111
                                 + 0.97411 w_k1_INTC_s111
                                 + 0.608282 w_k1_KO_s111
                                 + 0.565178 w_k1_MAT_s111
                                 + 0.96275 w_k1_MSFT_s111
                                 + 0.871591 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.794815 w_k1_ADM_s111
                                 + 0.76097 w_k1_AXP_s111
                                 + 0.77456 w_k1_DIS_s111
                                 + 0.869412 w_k1_FDX_s111
                                 + 0.78227 w_k1_IBM_s111
                                 + 0.864425 w_k1_INTC_s111
                                 + 0.739508 w_k1_KO_s111
                                 + 0.697871 w_k1_MAT_s111
                                 + 0.829635 w_k1_MSFT_s111
                                 + 0.908876 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.28589 w_k1_ADM_s111
                                 + 0.945494 w_k1_AXP_s111
                                 + 1.1727 w_k1_DIS_s111 + 1.03827 w_k1_FDX_s111
                                 + 0.993761 w_k1_IBM_s111
                                 + 1.11461 w_k1_INTC_s111
                                 + 1.13849 w_k1_KO_s111
                                 + 0.897704 w_k1_MAT_s111
                                 + 1.26198 w_k1_MSFT_s111
                                 + 1.13209 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.987642 w_k1_ADM_s112
                                 + 1.61093 w_k1_AXP_s112
                                 + 1.57504 w_k1_DIS_s112
                                 + 1.24158 w_k1_FDX_s112
                                 + 1.72924 w_k1_IBM_s112
                                 + 1.34279 w_k1_INTC_s112
                                 + 1.43343 w_k1_KO_s112 + 1.2939 w_k1_MAT_s112
                                 + 1.45693 w_k1_MSFT_s112
                                 + 1.07179 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.26993 w_k1_ADM_s112
                                 + 1.08231 w_k1_AXP_s112
                                 + 0.848136 w_k1_DIS_s112
                                 + 0.95004 w_k1_FDX_s112 + 1.6413 w_k1_IBM_s112
                                 + 0.809516 w_k1_INTC_s112
                                 + 1.11699 w_k1_KO_s112 + 1.28527 w_k1_MAT_s112
                                 + 0.874917 w_k1_MSFT_s112
                                 + 0.998797 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.8379 w_k1_ADM_s112
                                 + 0.753667 w_k1_AXP_s112
                                 + 0.866675 w_k1_DIS_s112
                                 + 0.841267 w_k1_FDX_s112
                                 + 1.00401 w_k1_IBM_s112
                                 + 0.920728 w_k1_INTC_s112
                                 + 0.901129 w_k1_KO_s112
                                 + 0.642449 w_k1_MAT_s112
                                 + 0.835988 w_k1_MSFT_s112
                                 + 0.844353 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.985771 w_k1_ADM_s112
                                 + 0.863841 w_k1_AXP_s112
                                 + 0.751873 w_k1_DIS_s112
                                 + 0.960208 w_k1_FDX_s112
                                 + 0.747778 w_k1_IBM_s112
                                 + 0.973777 w_k1_INTC_s112
                                 + 0.801259 w_k1_KO_s112
                                 + 0.659042 w_k1_MAT_s112
                                 + 1.02138 w_k1_MSFT_s112
                                 + 0.811228 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.44186 w_k1_ADM_s112
                                 + 1.42641 w_k1_AXP_s112
                                 + 1.57578 w_k1_DIS_s112
                                 + 1.48068 w_k1_FDX_s112
                                 + 1.12789 w_k1_IBM_s112
                                 + 1.34385 w_k1_INTC_s112
                                 + 1.45234 w_k1_KO_s112 + 1.27559 w_k1_MAT_s112
                                 + 1.51686 w_k1_MSFT_s112
                                 + 1.13812 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.09439 w_k1_ADM_s112
                                 + 1.15643 w_k1_AXP_s112
                                 + 1.19587 w_k1_DIS_s112
                                 + 1.17688 w_k1_FDX_s112
                                 + 1.00663 w_k1_IBM_s112
                                 + 1.03968 w_k1_INTC_s112
                                 + 1.01036 w_k1_KO_s112 + 1.30555 w_k1_MAT_s112
                                 + 1.18313 w_k1_MSFT_s112
                                 + 1.01646 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.16931 w_k1_ADM_s112
                                 + 1.05032 w_k1_AXP_s112
                                 + 0.99439 w_k1_DIS_s112
                                 + 0.966677 w_k1_FDX_s112
                                 + 1.25094 w_k1_IBM_s112
                                 + 0.831445 w_k1_INTC_s112
                                 + 1.03277 w_k1_KO_s112 + 1.08051 w_k1_MAT_s112
                                 + 0.872511 w_k1_MSFT_s112
                                 + 0.925698 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.01028 w_k1_ADM_s112
                                 + 0.943567 w_k1_AXP_s112
                                 + 1.0486 w_k1_DIS_s112 + 1.00724 w_k1_FDX_s112
                                 + 0.865691 w_k1_IBM_s112
                                 + 0.886835 w_k1_INTC_s112
                                 + 0.963848 w_k1_KO_s112
                                 + 0.646572 w_k1_MAT_s112
                                 + 1.13428 w_k1_MSFT_s112
                                 + 0.994271 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.806443 w_k1_ADM_s113
                                 + 1.43776 w_k1_AXP_s113
                                 + 1.37974 w_k1_DIS_s113
                                 + 1.13975 w_k1_FDX_s113
                                 + 1.59587 w_k1_IBM_s113
                                 + 1.17811 w_k1_INTC_s113
                                 + 1.30682 w_k1_KO_s113 + 1.83541 w_k1_MAT_s113
                                 + 1.04727 w_k1_MSFT_s113 + 1.181 w_k1_PG_s113
                                 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.831657 w_k1_ADM_s113
                                 + 0.881964 w_k1_AXP_s113
                                 + 0.674012 w_k1_DIS_s113
                                 + 0.714134 w_k1_FDX_s113
                                 + 1.19418 w_k1_IBM_s113
                                 + 0.710218 w_k1_INTC_s113
                                 + 0.932835 w_k1_KO_s113 + 1.0603 w_k1_MAT_s113
                                 + 0.626945 w_k1_MSFT_s113
                                 + 1.03304 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.671525 w_k1_ADM_s113
                                 + 0.812187 w_k1_AXP_s113
                                 + 0.616691 w_k1_DIS_s113
                                 + 0.859923 w_k1_FDX_s113
                                 + 0.646183 w_k1_IBM_s113
                                 + 0.948554 w_k1_INTC_s113
                                 + 0.70229 w_k1_KO_s113 + 0.69262 w_k1_MAT_s113
                                 + 0.726401 w_k1_MSFT_s113
                                 + 0.946212 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.1859 w_k1_ADM_s113
                                 + 1.15131 w_k1_AXP_s113
                                 + 1.25771 w_k1_DIS_s113 + 1.1944 w_k1_FDX_s113
                                 + 0.992569 w_k1_IBM_s113
                                 + 1.19089 w_k1_INTC_s113
                                 + 1.11778 w_k1_KO_s113 + 1.00114 w_k1_MAT_s113
                                 + 1.00257 w_k1_MSFT_s113
                                 + 1.27986 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.35561 w_k1_ADM_s113
                                 + 1.07663 w_k1_AXP_s113
                                 + 1.09686 w_k1_DIS_s113
                                 + 1.12666 w_k1_FDX_s113
                                 + 0.73735 w_k1_IBM_s113
                                 + 1.24322 w_k1_INTC_s113
                                 + 0.907309 w_k1_KO_s113
                                 + 0.755414 w_k1_MAT_s113
                                 + 1.01153 w_k1_MSFT_s113
                                 + 1.07845 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.05307 w_k1_ADM_s113
                                 + 0.874543 w_k1_AXP_s113
                                 + 0.942418 w_k1_DIS_s113
                                 + 0.970379 w_k1_FDX_s113
                                 + 1.00703 w_k1_IBM_s113
                                 + 0.899115 w_k1_INTC_s113
                                 + 1.0141 w_k1_KO_s113 + 0.818349 w_k1_MAT_s113
                                 + 0.901579 w_k1_MSFT_s113
                                 + 1.07694 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.03106 w_k1_ADM_s113
                                 + 0.920608 w_k1_AXP_s113
                                 + 0.823095 w_k1_DIS_s113
                                 + 0.823652 w_k1_FDX_s113
                                 + 1.47333 w_k1_IBM_s113
                                 + 0.899624 w_k1_INTC_s113
                                 + 1.02615 w_k1_KO_s113 + 1.74063 w_k1_MAT_s113
                                 + 0.874321 w_k1_MSFT_s113
                                 + 1.04975 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.920359 w_k1_ADM_s113
                                 + 0.686722 w_k1_AXP_s113
                                 + 0.901263 w_k1_DIS_s113
                                 + 0.871484 w_k1_FDX_s113
                                 + 0.699474 w_k1_IBM_s113
                                 + 0.815585 w_k1_INTC_s113
                                 + 0.773993 w_k1_KO_s113
                                 + 0.570496 w_k1_MAT_s113
                                 + 0.82819 w_k1_MSFT_s113
                                 + 0.901747 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.812103 w_k1_ADM_s114
                                 + 0.769958 w_k1_AXP_s114
                                 + 0.592534 w_k1_DIS_s114
                                 + 0.70165 w_k1_FDX_s114
                                 + 0.825548 w_k1_IBM_s114
                                 + 0.902092 w_k1_INTC_s114
                                 + 0.745355 w_k1_KO_s114
                                 + 0.617302 w_k1_MAT_s114
                                 + 0.787281 w_k1_MSFT_s114
                                 + 0.829471 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.993499 w_k1_ADM_s114
                                 + 1.27901 w_k1_AXP_s114
                                 + 1.49017 w_k1_DIS_s114
                                 + 1.10805 w_k1_FDX_s114
                                 + 1.32228 w_k1_IBM_s114
                                 + 1.17967 w_k1_INTC_s114
                                 + 1.25671 w_k1_KO_s114 + 1.38996 w_k1_MAT_s114
                                 + 1.22952 w_k1_MSFT_s114
                                 + 1.15429 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.803204 w_k1_ADM_s114
                                 + 0.893133 w_k1_AXP_s114
                                 + 0.912653 w_k1_DIS_s114
                                 + 0.914671 w_k1_FDX_s114
                                 + 0.882318 w_k1_IBM_s114
                                 + 1.1166 w_k1_INTC_s114
                                 + 0.839553 w_k1_KO_s114
                                 + 0.853501 w_k1_MAT_s114
                                 + 0.895489 w_k1_MSFT_s114
                                 + 1.03811 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.649119 w_k1_ADM_s114
                                 + 0.833403 w_k1_AXP_s114
                                 + 0.939636 w_k1_DIS_s114
                                 + 0.767786 w_k1_FDX_s114
                                 + 1.19923 w_k1_IBM_s114
                                 + 0.863175 w_k1_INTC_s114
                                 + 1.02119 w_k1_KO_s114 + 1.09891 w_k1_MAT_s114
                                 + 0.847741 w_k1_MSFT_s114
                                 + 0.870582 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.851101 w_k1_ADM_s114
                                 + 0.779996 w_k1_AXP_s114
                                 + 0.681802 w_k1_DIS_s114
                                 + 0.756766 w_k1_FDX_s114
                                 + 0.611367 w_k1_IBM_s114
                                 + 0.837529 w_k1_INTC_s114
                                 + 0.600788 w_k1_KO_s114
                                 + 0.490248 w_k1_MAT_s114
                                 + 0.815159 w_k1_MSFT_s114
                                 + 0.882383 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.08594 w_k1_ADM_s114
                                 + 1.10613 w_k1_AXP_s114
                                 + 1.10436 w_k1_DIS_s114
                                 + 1.07335 w_k1_FDX_s114
                                 + 0.908582 w_k1_IBM_s114
                                 + 1.07277 w_k1_INTC_s114
                                 + 0.964263 w_k1_KO_s114
                                 + 0.933999 w_k1_MAT_s114
                                 + 1.06047 w_k1_MSFT_s114
                                 + 1.14276 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.04764 w_k1_ADM_s114
                                 + 1.01419 w_k1_AXP_s114
                                 + 0.952112 w_k1_DIS_s114
                                 + 1.06739 w_k1_FDX_s114
                                 + 0.616721 w_k1_IBM_s114
                                 + 1.06951 w_k1_INTC_s114
                                 + 0.740097 w_k1_KO_s114
                                 + 0.634242 w_k1_MAT_s114
                                 + 1.02779 w_k1_MSFT_s114
                                 + 1.03289 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.876038 w_k1_ADM_s114
                                 + 1.17926 w_k1_AXP_s114
                                 + 1.37738 w_k1_DIS_s114
                                 + 1.26687 w_k1_FDX_s114
                                 + 0.884661 w_k1_IBM_s114
                                 + 1.30659 w_k1_INTC_s114
                                 + 1.01034 w_k1_KO_s114
                                 + 0.991942 w_k1_MAT_s114
                                 + 1.19105 w_k1_MSFT_s114
                                 + 1.12178 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.12322 w_k1_ADM_s115
                                 + 1.0289 w_k1_AXP_s115 + 1.06291 w_k1_DIS_s115
                                 + 0.984116 w_k1_FDX_s115
                                 + 0.97137 w_k1_IBM_s115
                                 + 0.929955 w_k1_INTC_s115
                                 + 1.1122 w_k1_KO_s115 + 0.888246 w_k1_MAT_s115
                                 + 0.911693 w_k1_MSFT_s115
                                 + 0.992412 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.04316 w_k1_ADM_s115
                                 + 0.956455 w_k1_AXP_s115
                                 + 1.07629 w_k1_DIS_s115
                                 + 0.958317 w_k1_FDX_s115
                                 + 1.22885 w_k1_IBM_s115
                                 + 0.856067 w_k1_INTC_s115
                                 + 1.19114 w_k1_KO_s115 + 1.37205 w_k1_MAT_s115
                                 + 0.987453 w_k1_MSFT_s115
                                 + 1.09198 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.775007 w_k1_ADM_s115
                                 + 0.604939 w_k1_AXP_s115
                                 + 0.595873 w_k1_DIS_s115
                                 + 0.70025 w_k1_FDX_s115
                                 + 0.781254 w_k1_IBM_s115
                                 + 0.641191 w_k1_INTC_s115
                                 + 0.738092 w_k1_KO_s115
                                 + 0.567006 w_k1_MAT_s115
                                 + 0.735813 w_k1_MSFT_s115
                                 + 0.780614 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.46917 w_k1_ADM_s115
                                 + 1.59919 w_k1_AXP_s115
                                 + 1.48526 w_k1_DIS_s115
                                 + 1.37199 w_k1_FDX_s115
                                 + 1.32573 w_k1_IBM_s115
                                 + 1.06605 w_k1_INTC_s115
                                 + 1.34856 w_k1_KO_s115 + 1.60124 w_k1_MAT_s115
                                 + 1.13879 w_k1_MSFT_s115
                                 + 1.19876 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.58092 w_k1_ADM_s115
                                 + 1.54981 w_k1_AXP_s115
                                 + 2.15431 w_k1_DIS_s115
                                 + 1.66088 w_k1_FDX_s115
                                 + 1.21722 w_k1_IBM_s115
                                 + 1.32761 w_k1_INTC_s115
                                 + 1.33552 w_k1_KO_s115 + 1.24901 w_k1_MAT_s115
                                 + 1.40919 w_k1_MSFT_s115
                                 + 1.33461 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.36487 w_k1_ADM_s115
                                 + 1.08277 w_k1_AXP_s115
                                 + 1.33133 w_k1_DIS_s115 + 1.2116 w_k1_FDX_s115
                                 + 1.00294 w_k1_IBM_s115
                                 + 0.916817 w_k1_INTC_s115
                                 + 1.15472 w_k1_KO_s115 + 1.21266 w_k1_MAT_s115
                                 + 1.03481 w_k1_MSFT_s115
                                 + 1.12185 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.04998 w_k1_ADM_s115
                                 + 0.787558 w_k1_AXP_s115
                                 + 0.865405 w_k1_DIS_s115
                                 + 0.987989 w_k1_FDX_s115
                                 + 0.732986 w_k1_IBM_s115
                                 + 0.875403 w_k1_INTC_s115
                                 + 0.832255 w_k1_KO_s115
                                 + 0.654548 w_k1_MAT_s115
                                 + 0.916958 w_k1_MSFT_s115
                                 + 0.868127 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.03497 w_k1_ADM_s115
                                 + 1.08813 w_k1_AXP_s115
                                 + 0.750389 w_k1_DIS_s115
                                 + 0.817079 w_k1_FDX_s115
                                 + 1.29616 w_k1_IBM_s115
                                 + 0.868798 w_k1_INTC_s115
                                 + 1.14032 w_k1_KO_s115 + 1.36045 w_k1_MAT_s115
                                 + 0.774477 w_k1_MSFT_s115
                                 + 0.946951 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.66658 w_k1_ADM_s116
                                 + 1.41948 w_k1_AXP_s116
                                 + 1.66641 w_k1_DIS_s116
                                 + 1.50811 w_k1_FDX_s116
                                 + 0.99852 w_k1_IBM_s116
                                 + 1.40721 w_k1_INTC_s116
                                 + 1.20967 w_k1_KO_s116 + 1.03077 w_k1_MAT_s116
                                 + 1.36742 w_k1_MSFT_s116
                                 + 1.20629 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.827357 w_k1_ADM_s116
                                 + 0.768149 w_k1_AXP_s116
                                 + 0.899003 w_k1_DIS_s116
                                 + 0.854194 w_k1_FDX_s116
                                 + 1.39661 w_k1_IBM_s116
                                 + 0.860968 w_k1_INTC_s116
                                 + 0.961347 w_k1_KO_s116
                                 + 1.05014 w_k1_MAT_s116
                                 + 0.846192 w_k1_MSFT_s116
                                 + 0.968776 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.31154 w_k1_ADM_s116
                                 + 1.14445 w_k1_AXP_s116
                                 + 1.22008 w_k1_DIS_s116
                                 + 1.12341 w_k1_FDX_s116
                                 + 1.28573 w_k1_IBM_s116
                                 + 1.24085 w_k1_INTC_s116 + 1.1867 w_k1_KO_s116
                                 + 1.17307 w_k1_MAT_s116
                                 + 1.14419 w_k1_MSFT_s116
                                 + 1.21081 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.992464 w_k1_ADM_s116
                                 + 1.06022 w_k1_AXP_s116
                                 + 1.16252 w_k1_DIS_s116
                                 + 1.30725 w_k1_FDX_s116
                                 + 0.901045 w_k1_IBM_s116
                                 + 1.27036 w_k1_INTC_s116
                                 + 0.951641 w_k1_KO_s116
                                 + 0.838259 w_k1_MAT_s116
                                 + 1.1681 w_k1_MSFT_s116 + 1.02904 w_k1_PG_s116
                                 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.10237 w_k1_ADM_s116
                                 + 1.147 w_k1_AXP_s116 + 1.39897 w_k1_DIS_s116
                                 + 1.15965 w_k1_FDX_s116
                                 + 1.66122 w_k1_IBM_s116
                                 + 1.53851 w_k1_INTC_s116
                                 + 1.29942 w_k1_KO_s116 + 1.01439 w_k1_MAT_s116
                                 + 1.24946 w_k1_MSFT_s116
                                 + 1.07289 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.02093 w_k1_ADM_s116
                                 + 0.88816 w_k1_AXP_s116
                                 + 0.757547 w_k1_DIS_s116
                                 + 0.969685 w_k1_FDX_s116
                                 + 0.776595 w_k1_IBM_s116
                                 + 1.05836 w_k1_INTC_s116
                                 + 0.861234 w_k1_KO_s116
                                 + 0.816861 w_k1_MAT_s116
                                 + 0.871375 w_k1_MSFT_s116
                                 + 1.0243 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.04501 w_k1_ADM_s116
                                 + 0.766644 w_k1_AXP_s116
                                 + 0.806139 w_k1_DIS_s116
                                 + 0.821812 w_k1_FDX_s116
                                 + 1.04273 w_k1_IBM_s116
                                 + 0.94331 w_k1_INTC_s116
                                 + 0.953087 w_k1_KO_s116
                                 + 0.706324 w_k1_MAT_s116
                                 + 0.894421 w_k1_MSFT_s116
                                 + 0.915045 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.972299 w_k1_ADM_s116
                                 + 0.887762 w_k1_AXP_s116
                                 + 1.14779 w_k1_DIS_s116
                                 + 1.10468 w_k1_FDX_s116
                                 + 0.723833 w_k1_IBM_s116
                                 + 1.12642 w_k1_INTC_s116
                                 + 0.906164 w_k1_KO_s116
                                 + 0.545273 w_k1_MAT_s116
                                 + 1.01938 w_k1_MSFT_s116
                                 + 0.959468 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.948356 w_k1_ADM_s117
                                 + 0.802498 w_k1_AXP_s117
                                 + 0.96844 w_k1_DIS_s117
                                 + 0.908264 w_k1_FDX_s117
                                 + 1.12259 w_k1_IBM_s117
                                 + 1.15403 w_k1_INTC_s117
                                 + 1.12798 w_k1_KO_s117
                                 + 0.962571 w_k1_MAT_s117
                                 + 1.0381 w_k1_MSFT_s117 + 1.09799 w_k1_PG_s117
                                 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.926185 w_k1_ADM_s117
                                 + 0.78521 w_k1_AXP_s117
                                 + 0.647025 w_k1_DIS_s117
                                 + 0.755554 w_k1_FDX_s117
                                 + 0.799567 w_k1_IBM_s117
                                 + 0.713548 w_k1_INTC_s117
                                 + 0.775248 w_k1_KO_s117
                                 + 0.694489 w_k1_MAT_s117
                                 + 0.792198 w_k1_MSFT_s117
                                 + 0.855217 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.25224 w_k1_ADM_s117
                                 + 1.18993 w_k1_AXP_s117
                                 + 1.10057 w_k1_DIS_s117
                                 + 1.18559 w_k1_FDX_s117
                                 + 1.26165 w_k1_IBM_s117
                                 + 1.0626 w_k1_INTC_s117 + 1.15733 w_k1_KO_s117
                                 + 1.1557 w_k1_MAT_s117
                                 + 1.22758 w_k1_MSFT_s117
                                 + 1.16599 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.18113 w_k1_ADM_s117
                                 + 1.14789 w_k1_AXP_s117
                                 + 1.22435 w_k1_DIS_s117
                                 + 0.993808 w_k1_FDX_s117
                                 + 1.56006 w_k1_IBM_s117
                                 + 0.851448 w_k1_INTC_s117
                                 + 1.40558 w_k1_KO_s117 + 1.75223 w_k1_MAT_s117
                                 + 0.931561 w_k1_MSFT_s117
                                 + 1.16775 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.12181 w_k1_ADM_s117
                                 + 1.01831 w_k1_AXP_s117
                                 + 0.921966 w_k1_DIS_s117
                                 + 1.04321 w_k1_FDX_s117
                                 + 0.745799 w_k1_IBM_s117
                                 + 0.964327 w_k1_INTC_s117
                                 + 0.983519 w_k1_KO_s117
                                 + 0.970454 w_k1_MAT_s117
                                 + 1.05068 w_k1_MSFT_s117
                                 + 1.02797 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.30202 w_k1_ADM_s117
                                 + 1.35494 w_k1_AXP_s117
                                 + 1.54728 w_k1_DIS_s117
                                 + 1.34059 w_k1_FDX_s117
                                 + 1.14167 w_k1_IBM_s117
                                 + 1.1082 w_k1_INTC_s117 + 1.22495 w_k1_KO_s117
                                 + 1.2108 w_k1_MAT_s117
                                 + 1.19791 w_k1_MSFT_s117
                                 + 1.22474 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.808 w_k1_ADM_s117
                                 + 0.990334 w_k1_AXP_s117
                                 + 0.938769 w_k1_DIS_s117
                                 + 0.868388 w_k1_FDX_s117
                                 + 1.04964 w_k1_IBM_s117
                                 + 0.781385 w_k1_INTC_s117
                                 + 1.06299 w_k1_KO_s117 + 1.21571 w_k1_MAT_s117
                                 + 0.86158 w_k1_MSFT_s117
                                 + 0.972475 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.12417 w_k1_ADM_s117
                                 + 1.15508 w_k1_AXP_s117
                                 + 1.24737 w_k1_DIS_s117
                                 + 1.16986 w_k1_FDX_s117
                                 + 0.889784 w_k1_IBM_s117
                                 + 1.06038 w_k1_INTC_s117
                                 + 1.13767 w_k1_KO_s117 + 1.08001 w_k1_MAT_s117
                                 + 1.07053 w_k1_MSFT_s117
                                 + 1.16676 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0355201937888799 w_k1_ADM_s000
                                 + 0.0206113225621587 w_k1_ADM_s001
                                 + 0.0249646226645089 w_k1_ADM_s002
                                 + 0.0289090214331175 w_k1_ADM_s003
                                 + 0.0754461377027241 w_k1_ADM_s004
                                 + 0.021276108926423 w_k1_ADM_s005
                                 + 0.0296575880971193 w_k1_ADM_s006
                                 + 0.0477904127282072 w_k1_ADM_s007
                                 + 0.0107137412465974 w_k1_ADM_s010
                                 + 0.0242258292478731 w_k1_ADM_s011
                                 + 0.00761865676650665 w_k1_ADM_s012
                                 + 0.0153664324249338 w_k1_ADM_s013
                                 + 0.00940086228089876 w_k1_ADM_s014
                                 + 0.0328535628445588 w_k1_ADM_s015
                                 + 0.0096120085215412 w_k1_ADM_s016
                                 + 0.00903786500542926 w_k1_ADM_s017
                                 + 0.0354914534503614 w_k1_ADM_s100
                                 + 0.0289197737863927 w_k1_ADM_s101
                                 + 0.0194891964934058 w_k1_ADM_s102
                                 + 0.0152968708283399 w_k1_ADM_s103
                                 + 0.0562540728681096 w_k1_ADM_s104
                                 + 0.0299354150193347 w_k1_ADM_s105
                                 + 0.0281614704043471 w_k1_ADM_s106
                                 + 0.0161731112461774 w_k1_ADM_s107
                                 + 0.0749888507554347 w_k1_ADM_s110
                                 + 0.0310819540860889 w_k1_ADM_s111
                                 + 0.0187104317164515 w_k1_ADM_s112
                                 + 0.0318758516495312 w_k1_ADM_s113
                                 + 0.0323830516737352 w_k1_ADM_s114
                                 + 0.0956533429863681 w_k1_ADM_s115
                                 + 0.0310596951169584 w_k1_ADM_s116
                                 + 0.0826023489297864 w_k1_ADM_s117
                                 + 0.0302068987485989 w_k1_AXP_s000
                                 + 0.0203540779466562 w_k1_AXP_s001
                                 + 0.0292236095065154 w_k1_AXP_s002
                                 + 0.0331028703969209 w_k1_AXP_s003
                                 + 0.0763294189299183 w_k1_AXP_s004
                                 + 0.0234386864309847 w_k1_AXP_s005
                                 + 0.0311502402580942 w_k1_AXP_s006
                                 + 0.048868557447357 w_k1_AXP_s007
                                 + 0.00961015753335684 w_k1_AXP_s010
                                 + 0.0256296793545882 w_k1_AXP_s011
                                 + 0.00821493902425341 w_k1_AXP_s012
                                 + 0.0151490356869369 w_k1_AXP_s013
                                 + 0.00811507143392684 w_k1_AXP_s014
                                 + 0.0352195553813835 w_k1_AXP_s015
                                 + 0.0100911076813595 w_k1_AXP_s016
                                 + 0.00981145058434041 w_k1_AXP_s017
                                 + 0.0383258454158775 w_k1_AXP_s100
                                 + 0.0272995103490308 w_k1_AXP_s101
                                 + 0.0211739387031562 w_k1_AXP_s102
                                 + 0.0143152083780476 w_k1_AXP_s103
                                 + 0.0574507067193777 w_k1_AXP_s104
                                 + 0.0354771345025168 w_k1_AXP_s105
                                 + 0.0264334575376171 w_k1_AXP_s106
                                 + 0.0163264774636867 w_k1_AXP_s107
                                 + 0.0643519319199686 w_k1_AXP_s110
                                 + 0.0306057133184888 w_k1_AXP_s111
                                 + 0.0178747979692399 w_k1_AXP_s112
                                 + 0.0323402825659556 w_k1_AXP_s113
                                 + 0.03722926364675 w_k1_AXP_s114
                                 + 0.0852053751529527 w_k1_AXP_s115
                                 + 0.027140068647645 w_k1_AXP_s116
                                 + 0.0799413585981766 w_k1_AXP_s117
                                 + 0.0357007732666248 w_k1_DIS_s000
                                 + 0.0202276940461869 w_k1_DIS_s001
                                 + 0.0272847489666371 w_k1_DIS_s002
                                 + 0.034905801125653 w_k1_DIS_s003
                                 + 0.0721846071428881 w_k1_DIS_s004
                                 + 0.0260004454640971 w_k1_DIS_s005
                                 + 0.0336512967615715 w_k1_DIS_s006
                                 + 0.0458040092009001 w_k1_DIS_s007
                                 + 0.0108544135713501 w_k1_DIS_s010
                                 + 0.0279401000921475 w_k1_DIS_s011
                                 + 0.00890347582376037 w_k1_DIS_s012
                                 + 0.0144616256999462 w_k1_DIS_s013
                                 + 0.00960777955337278 w_k1_DIS_s014
                                 + 0.0335461501723074 w_k1_DIS_s015
                                 + 0.0101546223427626 w_k1_DIS_s016
                                 + 0.00926589628360879 w_k1_DIS_s017
                                 + 0.0404277022402475 w_k1_DIS_s100
                                 + 0.0269654864002298 w_k1_DIS_s101
                                 + 0.0207755219545923 w_k1_DIS_s102
                                 + 0.0154009875648216 w_k1_DIS_s103
                                 + 0.0626151304251617 w_k1_DIS_s104
                                 + 0.0350306541081059 w_k1_DIS_s105
                                 + 0.0282302346527583 w_k1_DIS_s106
                                 + 0.016334024923488 w_k1_DIS_s107
                                 + 0.0627713027726591 w_k1_DIS_s110
                                 + 0.0293499448072464 w_k1_DIS_s111
                                 + 0.0175752416253298 w_k1_DIS_s112
                                 + 0.0319010705190025 w_k1_DIS_s113
                                 + 0.0397258907634744 w_k1_DIS_s114
                                 + 0.0924320227125809 w_k1_DIS_s115
                                 + 0.0289295334936239 w_k1_DIS_s116
                                 + 0.0813957727461497 w_k1_DIS_s117
                                 + 0.0332234533840737 w_k1_FDX_s000
                                 + 0.0205509103531516 w_k1_FDX_s001
                                 + 0.0267656338096259 w_k1_FDX_s002
                                 + 0.0331985120853694 w_k1_FDX_s003
                                 + 0.0714153313134135 w_k1_FDX_s004
                                 + 0.0248813719330439 w_k1_FDX_s005
                                 + 0.0324955726545477 w_k1_FDX_s006
                                 + 0.0452596060001426 w_k1_FDX_s007
                                 + 0.010262318983794 w_k1_FDX_s010
                                 + 0.0260472577200392 w_k1_FDX_s011
                                 + 0.00831726433082356 w_k1_FDX_s012
                                 + 0.0154931844357086 w_k1_FDX_s013
                                 + 0.00903258111963819 w_k1_FDX_s014
                                 + 0.0337828533992332 w_k1_FDX_s015
                                 + 0.0104804481013703 w_k1_FDX_s016
                                 + 0.00895798906084487 w_k1_FDX_s017
                                 + 0.0377297852964341 w_k1_FDX_s100
                                 + 0.0275568753054692 w_k1_FDX_s101
                                 + 0.0208887361958886 w_k1_FDX_s102
                                 + 0.0146948513657722 w_k1_FDX_s103
                                 + 0.0590023674141283 w_k1_FDX_s104
                                 + 0.0318288786439841 w_k1_FDX_s105
                                 + 0.0272336822851404 w_k1_FDX_s106
                                 + 0.0169615277197834 w_k1_FDX_s107
                                 + 0.0618871507716845 w_k1_FDX_s110
                                 + 0.0292530085266627 w_k1_FDX_s111
                                 + 0.0179506289521671 w_k1_FDX_s112
                                 + 0.0317225800594497 w_k1_FDX_s113
                                 + 0.035245576188681 w_k1_FDX_s114
                                 + 0.0877707732512335 w_k1_FDX_s115
                                 + 0.0294884361154477 w_k1_FDX_s116
                                 + 0.0783416609567799 w_k1_FDX_s117
                                 + 0.0329704134438345 w_k1_IBM_s000
                                 + 0.0223707521870364 w_k1_IBM_s001
                                 + 0.0279538281405468 w_k1_IBM_s002
                                 + 0.033836982087704 w_k1_IBM_s003
                                 + 0.0719824858594527 w_k1_IBM_s004
                                 + 0.0244056422610873 w_k1_IBM_s005
                                 + 0.0315825534527177 w_k1_IBM_s006
                                 + 0.0472184570492558 w_k1_IBM_s007
                                 + 0.0100387960971558 w_k1_IBM_s010
                                 + 0.0276383584881872 w_k1_IBM_s011
                                 + 0.00897973176308147 w_k1_IBM_s012
                                 + 0.0144978891672101 w_k1_IBM_s013
                                 + 0.00931553824822008 w_k1_IBM_s014
                                 + 0.0342899293683095 w_k1_IBM_s015
                                 + 0.0100382963384311 w_k1_IBM_s016
                                 + 0.00929957108599528 w_k1_IBM_s017
                                 + 0.0398647423020116 w_k1_IBM_s100
                                 + 0.0271650864095481 w_k1_IBM_s101
                                 + 0.0217178212721637 w_k1_IBM_s102
                                 + 0.0152530002446907 w_k1_IBM_s103
                                 + 0.0646803318727802 w_k1_IBM_s104
                                 + 0.0323636265081554 w_k1_IBM_s105
                                 + 0.028795348194375 w_k1_IBM_s106
                                 + 0.0172981494656635 w_k1_IBM_s107
                                 + 0.0641170385920062 w_k1_IBM_s110
                                 + 0.030793256860751 w_k1_IBM_s111
                                 + 0.0187685073228121 w_k1_IBM_s112
                                 + 0.03297877349888 w_k1_IBM_s113
                                 + 0.035922286790172 w_k1_IBM_s114
                                 + 0.086876239374008 w_k1_IBM_s115
                                 + 0.0286655146621073 w_k1_IBM_s116
                                 + 0.0817785350567803 w_k1_IBM_s117
                                 + 0.0351210827323243 w_k1_INTC_s000
                                 + 0.0232451130641326 w_k1_INTC_s001
                                 + 0.0261543034634715 w_k1_INTC_s002
                                 + 0.0317442954648849 w_k1_INTC_s003
                                 + 0.0652199281327462 w_k1_INTC_s004
                                 + 0.0230374624984961 w_k1_INTC_s005
                                 + 0.0300785698651141 w_k1_INTC_s006
                                 + 0.0444181874781284 w_k1_INTC_s007
                                 + 0.0102845148122184 w_k1_INTC_s010
                                 + 0.0236630492668058 w_k1_INTC_s011
                                 + 0.0068241680750021 w_k1_INTC_s012
                                 + 0.0184854312118423 w_k1_INTC_s013
                                 + 0.0102835806752744 w_k1_INTC_s014
                                 + 0.0415498677165137 w_k1_INTC_s015
                                 + 0.0096328776932705 w_k1_INTC_s016
                                 + 0.01020728610888 w_k1_INTC_s017
                                 + 0.0349404548161436 w_k1_INTC_s100
                                 + 0.0255164904319252 w_k1_INTC_s101
                                 + 0.0205472959773124 w_k1_INTC_s102
                                 + 0.0153012428828884 w_k1_INTC_s103
                                 + 0.0599588621463392 w_k1_INTC_s104
                                 + 0.0326990007301815 w_k1_INTC_s105
                                 + 0.0253429350252041 w_k1_INTC_s106
                                 + 0.0176342492926241 w_k1_INTC_s107
                                 + 0.0628285898347196 w_k1_INTC_s110
                                 + 0.0298705971260983 w_k1_INTC_s111
                                 + 0.0171278437294825 w_k1_INTC_s112
                                 + 0.0323408607082594 w_k1_INTC_s113
                                 + 0.0382619122609835 w_k1_INTC_s114
                                 + 0.0759278621777987 w_k1_INTC_s115
                                 + 0.0315025308182113 w_k1_INTC_s116
                                 + 0.0738786206539555 w_k1_INTC_s117
                                 + 0.0372229524086988 w_k1_KO_s000
                                 + 0.0200724206048226 w_k1_KO_s001
                                 + 0.0264599479599695 w_k1_KO_s002
                                 + 0.0312497610634389 w_k1_KO_s003
                                 + 0.0702641294591633 w_k1_KO_s004
                                 + 0.0268438413064207 w_k1_KO_s005
                                 + 0.0385667060741399 w_k1_KO_s006
                                 + 0.0470122871316974 w_k1_KO_s007
                                 + 0.0102888427320808 w_k1_KO_s010
                                 + 0.0253909031356727 w_k1_KO_s011
                                 + 0.00811989724504575 w_k1_KO_s012
                                 + 0.0137423278222033 w_k1_KO_s013
                                 + 0.0102741991127062 w_k1_KO_s014
                                 + 0.0359263794996187 w_k1_KO_s015
                                 + 0.010195414957372 w_k1_KO_s016
                                 + 0.00924309617020169 w_k1_KO_s017
                                 + 0.0390186325778708 w_k1_KO_s100
                                 + 0.0329235705411847 w_k1_KO_s101
                                 + 0.0216486724793959 w_k1_KO_s102
                                 + 0.0155016427305873 w_k1_KO_s103
                                 + 0.0617984026695243 w_k1_KO_s104
                                 + 0.0352840253142269 w_k1_KO_s105
                                 + 0.0239162615046539 w_k1_KO_s106
                                 + 0.0180573660888951 w_k1_KO_s107
                                 + 0.060208150866416 w_k1_KO_s110
                                 + 0.0289877539868928 w_k1_KO_s111
                                 + 0.0176431484513112 w_k1_KO_s112
                                 + 0.0317179081389438 w_k1_KO_s113
                                 + 0.0350301496468788 w_k1_KO_s114
                                 + 0.0900917115002186 w_k1_KO_s115
                                 + 0.0279519539505677 w_k1_KO_s116
                                 + 0.0846475633319451 w_k1_KO_s117
                                 + 0.0299609681669035 w_k1_MAT_s000
                                 + 0.0195886629879309 w_k1_MAT_s001
                                 + 0.0242404318082557 w_k1_MAT_s002
                                 + 0.033259952815405 w_k1_MAT_s003
                                 + 0.0741576001464969 w_k1_MAT_s004
                                 + 0.0304877875487263 w_k1_MAT_s005
                                 + 0.0334977859379023 w_k1_MAT_s006
                                 + 0.0468959011617343 w_k1_MAT_s007
                                 + 0.0108696200588091 w_k1_MAT_s010
                                 + 0.0315434813827076 w_k1_MAT_s011
                                 + 0.00784638130172962 w_k1_MAT_s012
                                 + 0.0138168593327288 w_k1_MAT_s013
                                 + 0.00995687911565382 w_k1_MAT_s014
                                 + 0.0354856284958516 w_k1_MAT_s015
                                 + 0.0113797033648143 w_k1_MAT_s016
                                 + 0.00831289747003841 w_k1_MAT_s017
                                 + 0.0423511485020396 w_k1_MAT_s100
                                 + 0.0270900949103898 w_k1_MAT_s101
                                 + 0.0257491588399769 w_k1_MAT_s102
                                 + 0.0149822486060142 w_k1_MAT_s103
                                 + 0.061293801438298 w_k1_MAT_s104
                                 + 0.0414202747916247 w_k1_MAT_s105
                                 + 0.0242810272381402 w_k1_MAT_s106
                                 + 0.0165768112401163 w_k1_MAT_s107
                                 + 0.0635076434139188 w_k1_MAT_s110
                                 + 0.0312712388230349 w_k1_MAT_s111
                                 + 0.015960626727671 w_k1_MAT_s112
                                 + 0.0330551282194366 w_k1_MAT_s113
                                 + 0.0357736651504753 w_k1_MAT_s114
                                 + 0.0892463333665144 w_k1_MAT_s115
                                 + 0.0244367109781648 w_k1_MAT_s116
                                 + 0.0853429673282399 w_k1_MAT_s117
                                 + 0.0350158776879427 w_k1_MSFT_s000
                                 + 0.0204446087158116 w_k1_MSFT_s001
                                 + 0.02702545905322 w_k1_MSFT_s002
                                 + 0.031791239824679 w_k1_MSFT_s003
                                 + 0.0676012861188739 w_k1_MSFT_s004
                                 + 0.0252068812175031 w_k1_MSFT_s005
                                 + 0.0333882041510906 w_k1_MSFT_s006
                                 + 0.0439630276422445 w_k1_MSFT_s007
                                 + 0.0110772478193994 w_k1_MSFT_s010
                                 + 0.0249343267989426 w_k1_MSFT_s011
                                 + 0.0078595116713204 w_k1_MSFT_s012
                                 + 0.0146643082399428 w_k1_MSFT_s013
                                 + 0.00946042812311606 w_k1_MSFT_s014
                                 + 0.0365146699932112 w_k1_MSFT_s015
                                 + 0.0100305531503711 w_k1_MSFT_s016
                                 + 0.00970763981632431 w_k1_MSFT_s017
                                 + 0.0370387260414409 w_k1_MSFT_s100
                                 + 0.0266890465613781 w_k1_MSFT_s101
                                 + 0.0209309106895404 w_k1_MSFT_s102
                                 + 0.0142053873999633 w_k1_MSFT_s103
                                 + 0.0569040097477994 w_k1_MSFT_s104
                                 + 0.0315809959965926 w_k1_MSFT_s105
                                 + 0.0260044141004021 w_k1_MSFT_s106
                                 + 0.017263797232412 w_k1_MSFT_s107
                                 + 0.0657722454318573 w_k1_MSFT_s110
                                 + 0.0307545702147844 w_k1_MSFT_s111
                                 + 0.018384961792977 w_k1_MSFT_s112
                                 + 0.028399217676873 w_k1_MSFT_s113
                                 + 0.0367193103755454 w_k1_MSFT_s114
                                 + 0.081360044759417 w_k1_MSFT_s115
                                 + 0.0283990390597276 w_k1_MSFT_s116
                                 + 0.0778695414645926 w_k1_MSFT_s117
                                 + 0.0323830616574118 w_k1_PG_s000
                                 + 0.0198939643094907 w_k1_PG_s001
                                 + 0.0259683459649506 w_k1_PG_s002
                                 + 0.0309792020746598 w_k1_PG_s003
                                 + 0.0698015308966303 w_k1_PG_s004
                                 + 0.0252926402879359 w_k1_PG_s005
                                 + 0.0334131731759833 w_k1_PG_s006
                                 + 0.0467162009753933 w_k1_PG_s007
                                 + 0.0107795114345045 w_k1_PG_s010
                                 + 0.0261291741868323 w_k1_PG_s011
                                 + 0.00827172684477079 w_k1_PG_s012
                                 + 0.0135228599663206 w_k1_PG_s013
                                 + 0.00916333404665539 w_k1_PG_s014
                                 + 0.0316841151044527 w_k1_PG_s015
                                 + 0.0101162248868042 w_k1_PG_s016
                                 + 0.00927014288111334 w_k1_PG_s017
                                 + 0.038400526851275 w_k1_PG_s100
                                 + 0.0285319574077759 w_k1_PG_s101
                                 + 0.021209923606622 w_k1_PG_s102
                                 + 0.0161392419983744 w_k1_PG_s103
                                 + 0.0597917674333806 w_k1_PG_s104
                                 + 0.0340092801754308 w_k1_PG_s105
                                 + 0.026209805823038 w_k1_PG_s106
                                 + 0.0168218284748855 w_k1_PG_s107
                                 + 0.0632681996419529 w_k1_PG_s110
                                 + 0.0320542351236584 w_k1_PG_s111
                                 + 0.0163643752020887 w_k1_PG_s112
                                 + 0.0349479899630213 w_k1_PG_s113
                                 + 0.0373125292601952 w_k1_PG_s114
                                 + 0.0857576536759618 w_k1_PG_s115
                                 + 0.0289766564936053 w_k1_PG_s116
                                 + 0.0831417829764052 w_k1_PG_s117 >= 1.219
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
