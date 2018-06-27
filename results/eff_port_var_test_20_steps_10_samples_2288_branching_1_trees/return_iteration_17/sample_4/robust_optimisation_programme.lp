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
 w_k1_ADM_s0_return_constraint:  1.08839 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.867949 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.10733 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.940779 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.15638 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.956678 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.23412 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.617318 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.14571 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.750858 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.851573 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.36156 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.991455 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.00321 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.999843 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.82485 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.78667 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.786776 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.21019 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.828643 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.06963 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.26927 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.01184 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.895905 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.91917 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.845781 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.953945 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.11309 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.947702 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.38946 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.16781 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.35334 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.11922 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.789272 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.872857 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.854713 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.51785 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.988345 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.2051 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.84061 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.997515 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.994101 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.28036 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.03657 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.22955 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.541407 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.56373 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.719697 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.874003 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.61711 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.04834 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.923746 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.00094 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.779808 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.591393 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.687676 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.55802 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.905847 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.21666 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.36652 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.29357 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.697713 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.96321 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.909602 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.23731 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.53196 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.01785 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.74143 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.13999 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.24129 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.06119 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.797095 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.545827 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.728492 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.78251 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.879883 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.27225 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.857709 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.11419 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.97366 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.26217 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.916867 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.17759 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.452967 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.74821 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.569261 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.61421 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.72541 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.977352 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.825736 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.95868 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.876901 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.524587 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.626335 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.55577 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.789967 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.981756 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.44763 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.09647 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.693123 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.770238 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.847322 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.36108 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.66381 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.08746 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.6762 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.33958 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.28566 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.16986 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.640234 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.59542 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.647041 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 2.21335 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.823809 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.13788 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.898679 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.10189 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.949533 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.1392 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.00685 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.32043 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.595429 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.36016 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.768866 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.750658 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.47201 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.989591 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.91143 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.02257 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.839377 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.751466 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.707919 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.27076 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.826169 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.983147 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.36272 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.10894 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.72028 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.851004 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.841865 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.11315 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.28496 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.02872 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.55849 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.21296 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.0671 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.06099 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.755237 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.718363 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.68717 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.6584 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.860466 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.18565 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.89744 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.795683 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.20643 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.52178 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.892778 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.837504 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.761974 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.65483 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.567849 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.01128 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.19269 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.17556 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.78736 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.839178 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.01152 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.559307 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.80433 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.59966 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.24284 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.31342 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.912593 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.06477 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.903719 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.657597 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.3057 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.54696 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.48058 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.808258 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.913516 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.853631 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.6463 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.14102 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.872069 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.670716 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.33241 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.06967 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.958257 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.21296 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.930521 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.895566 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.856286 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.992308 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.03124 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.4114 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.750682 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.38791 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.891844 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.904282 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.32285 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.22997 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.01483 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.960282 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.994121 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.713086 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.856199 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.3412 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.00044 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.984516 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.15153 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.086 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.896573 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.07152 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.13959 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.2673 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.35642 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.13618 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.54804 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.970896 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.09922 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.986439 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.745921 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.788959 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.637897 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.26509 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.776471 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.19643 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.881869 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.967199 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.949788 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.22499 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.942871 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.905568 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.636597 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.51838 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.605075 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.844868 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.19102 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.12102 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.795048 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.985206 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.08015 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.655863 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.904797 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.62638 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.06174 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.23211 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.15157 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.1414 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.858572 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.789643 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.17373 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.32316 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.62807 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.01889 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.16864 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.926919 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.32136 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.05123 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.788446 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.723704 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.872628 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.35642 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.958769 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.21055 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.802372 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.00602 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.35025 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.74208 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.00594 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.840315 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.406641 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.49976 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.476018 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.822882 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.05228 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.14557 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.613399 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.799248 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.913378 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.45055 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.615312 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.39921 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.965411 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.42175 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.953387 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.25329 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.682279 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.713923 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.09933 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.102 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.72506 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.777458 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.974071 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.875488 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.40935 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.971299 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.871057 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.523464 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.02801 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.33357 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.03108 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.15143 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.915393 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.05212 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.999766 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.08893 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.01214 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.21713 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.697227 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.36888 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.77032 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.7718 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.30274 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.06393 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.905226 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.881935 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.857229 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.652595 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.727664 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.22632 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.756212 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.850895 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.17084 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.00446 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.762726 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.922317 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.904124 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.06281 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.28307 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.06525 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.37519 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.18229 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.10294 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.04645 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.73328 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.744583 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.695248 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.38655 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.843303 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.094 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.921758 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.03038 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.03872 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.09136 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.96232 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.03819 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.728255 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.32493 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.820292 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.816365 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.23243 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.07383 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.912713 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.961044 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.913744 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.748675 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.805661 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.16813 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.910078 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.08109 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.10708 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.16569 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.883203 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.864898 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.08489 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.07066 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.38014 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.09171 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.34196 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.07139 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.18619 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.04749 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.841604 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.800594 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.855528 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.39551 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.949376 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.102999971345036 z_k1_s0000
                                 - 0.139999919628212 z_k1_s0001
                                 - 0.428000616988944 z_k1_s0002
                                 - 0.189999867481149 z_k1_s0003
                                 - 0.218999614059894 z_k1_s0004
                                 - 0.171000373032942 z_k1_s0005
                                 - 0.174999489285342 z_k1_s0006
                                 - 0.215000497807495 z_k1_s0007
                                 - 0.176000346569793 z_k1_s0010
                                 - 0.128999884674764 z_k1_s0011
                                 - 0.0899999397870275 z_k1_s0012
                                 - 0.334999340574305 z_k1_s0013
                                 - 0.220000433212241 z_k1_s0014
                                 - 0.176000346569793 z_k1_s0015
                                 - 0.178999561023359 z_k1_s0016
                                 - 0.145999496041523 z_k1_s0017
                                 - 0.143999858820561 z_k1_s0020
                                 - 0.338998814028526 z_k1_s0021
                                 - 0.184999865586085 z_k1_s0022
                                 - 0.359000175668726 z_k1_s0023
                                 - 0.263999674871103 z_k1_s0024
                                 - 0.139999785392299 z_k1_s0025
                                 - 0.462000425523319 z_k1_s0026
                                 - 0.0959999721803001 z_k1_s0027
                                 - 0.30000032330481 z_k1_s0030
                                 - 0.114999904733572 z_k1_s0031
                                 - 0.230000467066959 z_k1_s0032
                                 - 0.172000163441431 z_k1_s0033
                                 - 0.254999124009413 z_k1_s0034
                                 - 0.22300031704322 z_k1_s0035
                                 - 0.271000171492046 z_k1_s0036
                                 - 0.0780000511792599 z_k1_s0037
                                 - 0.127000246957573 z_k1_s0040
                                 - 0.0609999647811903 z_k1_s0041
                                 - 0.0619999800202268 z_k1_s0042
                                 - 0.14200047554312 z_k1_s0043
                                 - 0.111000003132988 z_k1_s0044
                                 - 0.289999595320404 z_k1_s0045
                                 - 0.225999584822104 z_k1_s0046
                                 - 0.186999955299717 z_k1_s0047
                                 - 0.143999774203353 z_k1_s0050
                                 - 0.463999295832978 z_k1_s0051
                                 - 0.183999582782323 z_k1_s0052
                                 - 0.292998761055627 z_k1_s0053
                                 - 0.339000025648428 z_k1_s0054
                                 - 0.168999628240263 z_k1_s0055
                                 - 0.392999861999845 z_k1_s0056
                                 - 0.119999846936056 z_k1_s0057
                                 - 0.362000025856133 z_k1_s0060
                                 - 0.369000189104502 z_k1_s0061
                                 - 1.77599908131709 z_k1_s0062
                                 - 0.56499999925749 z_k1_s0063
                                 - 1.18599892410126 z_k1_s0064
                                 - 0.518000178375943 z_k1_s0065
                                 - 0.709002337337925 z_k1_s0066
                                 - 0.466000325902728 z_k1_s0067
                                 - 0.566996813124986 z_k1_s0070
                                 - 0.359998912440176 z_k1_s0071
                                 - 0.571997222847942 z_k1_s0072
                                 - 0.695996744807631 z_k1_s0073
                                 - 0.883995305038879 z_k1_s0074
                                 - 0.363998796919807 z_k1_s0075
                                 - 0.393999038763872 z_k1_s0076
                                 - 0.595997416323194 z_k1_s0077
                                 - 0.48400095186998 z_k1_s0100
                                 - 1.23700170831102 z_k1_s0101
                                 - 0.320000484442701 z_k1_s0102
                                 - 0.698002773368734 z_k1_s0103
                                 - 0.922003696879677 z_k1_s0104
                                 - 1.00100303118755 z_k1_s0105
                                 - 0.53300122693813 z_k1_s0106
                                 - 0.649000745097425 z_k1_s0107
                                 - 0.226000551046846 z_k1_s0110
                                 - 0.191000177133264 z_k1_s0111
                                 - 0.179000096408629 z_k1_s0112
                                 - 0.302000093035755 z_k1_s0113
                                 - 0.416000444519596 z_k1_s0114
                                 - 0.204000126118221 z_k1_s0115
                                 - 0.222000247205174 z_k1_s0116
                                 - 0.336999636148955 z_k1_s0117
                                 - 0.266000965215577 z_k1_s0120
                                 - 0.199000041842454 z_k1_s0121
                                 - 0.228000131656182 z_k1_s0122
                                 - 0.530999797392667 z_k1_s0123
                                 - 0.260000981985152 z_k1_s0124
                                 - 0.452000445358754 z_k1_s0125
                                 - 0.219999983148942 z_k1_s0126
                                 - 0.406999289630516 z_k1_s0127
                                 - 0.390000150275023 z_k1_s0130
                                 - 0.507000152357525 z_k1_s0131
                                 - 0.784001880963286 z_k1_s0132
                                 - 0.523998054112886 z_k1_s0133
                                 - 0.314000222426567 z_k1_s0134
                                 - 0.295000240464454 z_k1_s0135
                                 - 0.738998076315777 z_k1_s0136
                                 - 0.747000377142168 z_k1_s0137
                                 - 0.654002269393278 z_k1_s0140
                                 - 0.623000715288911 z_k1_s0141
                                 - 0.43100001655602 z_k1_s0142
                                 - 0.732998559755498 z_k1_s0143
                                 - 0.591999161184545 z_k1_s0144
                                 - 0.889999576412632 z_k1_s0145
                                 - 0.408999941762668 z_k1_s0146
                                 - 0.726998585348224 z_k1_s0147
                                 - 0.319999732159834 z_k1_s0150
                                 - 0.519001162804109 z_k1_s0151
                                 - 0.166000179320639 z_k1_s0152
                                 - 0.176000135588109 z_k1_s0153
                                 - 0.332999939347731 z_k1_s0154
                                 - 0.126000165650626 z_k1_s0155
                                 - 0.126000165650626 z_k1_s0156
                                 - 0.120000041010038 z_k1_s0157
                                 - 0.0949999559744253 z_k1_s0160
                                 - 0.128999922606331 z_k1_s0161
                                 - 0.369999860553014 z_k1_s0162
                                 - 0.343000598342152 z_k1_s0163
                                 - 0.0939998987999764 z_k1_s0164
                                 - 0.128000017531832 z_k1_s0165
                                 - 0.144000019723311 z_k1_s0166
                                 - 0.218000296033815 z_k1_s0167
                                 - 0.908003482397413 z_k1_s0170
                                 - 0.328000917609684 z_k1_s0171
                                 - 0.276000813549714 z_k1_s0172
                                 - 0.352000932983437 z_k1_s0173
                                 - 0.416000832479762 z_k1_s0174
                                 - 0.220000636177278 z_k1_s0175
                                 - 0.862000304548331 z_k1_s0176
                                 - 0.883000371062994 z_k1_s0177
                                 - 0.0610000961196365 z_k1_s1000
                                 - 0.0960000678144443 z_k1_s1001
                                 - 0.0860001089588109 z_k1_s1002
                                 - 0.214000045111282 z_k1_s1003
                                 - 0.219000603039553 z_k1_s1004
                                 - 0.0680000673185799 z_k1_s1005
                                 - 0.191000024043234 z_k1_s1006
                                 - 0.222000706396334 z_k1_s1007
                                 - 0.170999746471993 z_k1_s1010
                                 - 0.327998875614793 z_k1_s1011
                                 - 0.152999803490698 z_k1_s1012
                                 - 0.403999317890645 z_k1_s1013
                                 - 0.154999754466443 z_k1_s1014
                                 - 0.186999738477543 z_k1_s1015
                                 - 0.139999833998664 z_k1_s1016
                                 - 0.382998968196243 z_k1_s1017
                                 - 0.0679997908568736 z_k1_s1020
                                 - 0.0959999223855451 z_k1_s1021
                                 - 0.0950000002771394 z_k1_s1022
                                 - 0.0759998152222465 z_k1_s1023
                                 - 0.06999982007315 z_k1_s1024
                                 - 0.116999211659389 z_k1_s1025
                                 - 0.174999318933544 z_k1_s1026
                                 - 0.227998890668345 z_k1_s1027
                                 - 0.196000457796768 z_k1_s1030
                                 - 0.145001119617194 z_k1_s1031
                                 - 0.212000023656377 z_k1_s1032
                                 - 0.121000326823642 z_k1_s1033
                                 - 0.0770002211059914 z_k1_s1034
                                 - 0.352000556940374 z_k1_s1035
                                 - 0.0760002482397658 z_k1_s1036
                                 - 0.265000751572541 z_k1_s1037
                                 - 0.0580000623864457 z_k1_s1040
                                 - 0.027999997227947 z_k1_s1041
                                 - 0.100999866293695 z_k1_s1042
                                 - 0.0659999325087466 z_k1_s1043
                                 - 0.0790002234573677 z_k1_s1044
                                 - 0.0359999677502243 z_k1_s1045
                                 - 0.0779997376921979 z_k1_s1046
                                 - 0.0560000948558705 z_k1_s1047
                                 - 0.125999541254742 z_k1_s1050
                                 - 0.144999937570597 z_k1_s1051
                                 - 0.057999975028239 z_k1_s1052
                                 - 0.116000328456205 z_k1_s1053
                                 - 0.045999973670677 z_k1_s1054
                                 - 0.197999549400113 z_k1_s1055
                                 - 0.0659999128666591 z_k1_s1056
                                 - 0.191000100441138 z_k1_s1057
                                 - 0.196999148010363 z_k1_s1060
                                 - 0.117999651477745 z_k1_s1061
                                 - 0.138999734909736 z_k1_s1062
                                 - 0.147999770666304 z_k1_s1063
                                 - 0.287998692218405 z_k1_s1064
                                 - 0.0839997207300247 z_k1_s1065
                                 - 0.16999904074066 z_k1_s1066
                                 - 0.0819997536506502 z_k1_s1067
                                 - 0.0840000286167294 z_k1_s1070
                                 - 0.0709999999558105 z_k1_s1071
                                 - 0.184000323969942 z_k1_s1072
                                 - 0.135999700760468 z_k1_s1073
                                 - 0.097999576136249 z_k1_s1074
                                 - 0.041999970058371 z_k1_s1075
                                 - 0.128000186892139 z_k1_s1076
                                 - 0.141999999911621 z_k1_s1077
                                 - 1.3109962892564 z_k1_s1100
                                 - 0.470000270243787 z_k1_s1101
                                 - 1.33599849038649 z_k1_s1102
                                 - 0.690000343187685 z_k1_s1103
                                 - 1.84799826352864 z_k1_s1104
                                 - 0.594000280848528 z_k1_s1105
                                 - 0.897998520589103 z_k1_s1106
                                 - 1.24300372618317 z_k1_s1107
                                 - 0.35600099372927 z_k1_s1110
                                 - 0.806003825321587 z_k1_s1111
                                 - 0.509001809518162 z_k1_s1112
                                 - 1.4710067307548 z_k1_s1113
                                 - 0.482001737590954 z_k1_s1114
                                 - 0.862004972329535 z_k1_s1115
                                 - 1.14900319696944 z_k1_s1116
                                 - 0.488001685541215 z_k1_s1117
                                 - 1.040999424216 z_k1_s1120
                                 - 0.36399978975276 z_k1_s1121
                                 - 1.41299875034797 z_k1_s1122
                                 - 0.553999994508298 z_k1_s1123
                                 - 1.06900012204308 z_k1_s1124
                                 - 1.25799709737647 z_k1_s1125
                                 - 0.630000231110501 z_k1_s1126
                                 - 1.4059985758912 z_k1_s1127
                                 - 0.410000984105391 z_k1_s1130
                                 - 0.39399935648559 z_k1_s1131
                                 - 0.160000144887311 z_k1_s1132
                                 - 0.311000283601587 z_k1_s1133
                                 - 0.437000370024076 z_k1_s1134
                                 - 0.412000001432079 z_k1_s1135
                                 - 0.373000185549068 z_k1_s1136
                                 - 0.666001722005425 z_k1_s1137
                                 - 0.65099879727499 z_k1_s1140
                                 - 0.498001053033605 z_k1_s1141
                                 - 0.350000608976712 z_k1_s1142
                                 - 1.40300458282673 z_k1_s1143
                                 - 1.14900236151472 z_k1_s1144
                                 - 0.444001332170679 z_k1_s1145
                                 - 0.450001231399723 z_k1_s1146
                                 - 1.333004712152 z_k1_s1147
                                 - 0.130000012996978 z_k1_s1150
                                 - 0.139999998127506 z_k1_s1151
                                 - 0.236001134441248 z_k1_s1152
                                 - 0.43200067795952 z_k1_s1153
                                 - 0.16700014365003 z_k1_s1154
                                 - 0.205000004625995 z_k1_s1155
                                 - 0.544001006541697 z_k1_s1156
                                 - 0.209000576318509 z_k1_s1157
                                 - 0.132000155172937 z_k1_s1160
                                 - 0.608999128358331 z_k1_s1161
                                 - 0.411999348317916 z_k1_s1162
                                 - 0.448999512361156 z_k1_s1163
                                 - 0.368000465294268 z_k1_s1164
                                 - 0.172000194022308 z_k1_s1165
                                 - 0.328001235545358 z_k1_s1166
                                 - 0.227000146302634 z_k1_s1167
                                 - 0.535001272121913 z_k1_s1170
                                 - 1.02400381023082 z_k1_s1171
                                 - 0.658001316777424 z_k1_s1172
                                 - 0.852000762034309 z_k1_s1173
                                 - 0.473000759173321 z_k1_s1174
                                 - 0.963003615622412 z_k1_s1175
                                 - 1.27200188461628 z_k1_s1176
                                 - 0.852000762034309 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.803426 w_k1_ADM_s000
                                 + 0.786664 w_k1_AXP_s000
                                 + 0.841149 w_k1_DIS_s000
                                 + 0.775041 w_k1_FDX_s000
                                 + 0.883313 w_k1_IBM_s000
                                 + 0.718375 w_k1_INTC_s000
                                 + 0.989945 w_k1_KO_s000
                                 + 0.678335 w_k1_MAT_s000
                                 + 0.841089 w_k1_MSFT_s000
                                 + 0.912624 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.632677 w_k1_ADM_s000
                                 + 0.701584 w_k1_AXP_s000
                                 + 0.622719 w_k1_DIS_s000
                                 + 0.70964 w_k1_FDX_s000
                                 + 0.729651 w_k1_IBM_s000
                                 + 0.62129 w_k1_INTC_s000
                                 + 0.69608 w_k1_KO_s000
                                 + 0.490925 w_k1_MAT_s000
                                 + 0.79848 w_k1_MSFT_s000
                                 + 0.730347 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.771633 w_k1_ADM_s000
                                 + 1.11505 w_k1_AXP_s000
                                 + 1.38588 w_k1_DIS_s000
                                 + 1.15614 w_k1_FDX_s000
                                 + 0.995736 w_k1_IBM_s000
                                 + 0.865906 w_k1_INTC_s000
                                 + 1.12239 w_k1_KO_s000 + 1.18875 w_k1_MAT_s000
                                 + 1.1146 w_k1_MSFT_s000 + 1.11671 w_k1_PG_s000
                                 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.24515 w_k1_ADM_s000
                                 + 1.26555 w_k1_AXP_s000
                                 + 1.10194 w_k1_DIS_s000
                                 + 1.03666 w_k1_FDX_s000
                                 + 1.36494 w_k1_IBM_s000
                                 + 0.869523 w_k1_INTC_s000
                                 + 1.11374 w_k1_KO_s000 + 1.34937 w_k1_MAT_s000
                                 + 0.963013 w_k1_MSFT_s000
                                 + 1.14397 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.948981 w_k1_ADM_s000
                                 + 0.958844 w_k1_AXP_s000
                                 + 0.78218 w_k1_DIS_s000
                                 + 0.913419 w_k1_FDX_s000
                                 + 0.937997 w_k1_IBM_s000
                                 + 0.95241 w_k1_INTC_s000
                                 + 0.873417 w_k1_KO_s000
                                 + 1.01259 w_k1_MAT_s000
                                 + 0.892307 w_k1_MSFT_s000
                                 + 0.917425 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.30002 w_k1_ADM_s000
                                 + 1.6523 w_k1_AXP_s000 + 1.44403 w_k1_DIS_s000
                                 + 1.28964 w_k1_FDX_s000
                                 + 1.25642 w_k1_IBM_s000
                                 + 1.03138 w_k1_INTC_s000
                                 + 1.21216 w_k1_KO_s000 + 1.07495 w_k1_MAT_s000
                                 + 1.16855 w_k1_MSFT_s000
                                 + 1.23765 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.999439 w_k1_ADM_s000
                                 + 1.01409 w_k1_AXP_s000
                                 + 0.90282 w_k1_DIS_s000
                                 + 0.839283 w_k1_FDX_s000
                                 + 1.22687 w_k1_IBM_s000
                                 + 0.816673 w_k1_INTC_s000
                                 + 1.04151 w_k1_KO_s000 + 1.09683 w_k1_MAT_s000
                                 + 0.855088 w_k1_MSFT_s000
                                 + 1.01145 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.997399 w_k1_ADM_s000
                                 + 0.924814 w_k1_AXP_s000
                                 + 0.891433 w_k1_DIS_s000
                                 + 0.889164 w_k1_FDX_s000
                                 + 0.821496 w_k1_IBM_s000
                                 + 0.812122 w_k1_INTC_s000
                                 + 0.896479 w_k1_KO_s000
                                 + 0.492618 w_k1_MAT_s000
                                 + 0.851067 w_k1_MSFT_s000
                                 + 0.891416 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.16321 w_k1_ADM_s001
                                 + 0.891962 w_k1_AXP_s001
                                 + 1.34525 w_k1_DIS_s001 + 1.0993 w_k1_FDX_s001
                                 + 1.31693 w_k1_IBM_s001
                                 + 1.08194 w_k1_INTC_s001
                                 + 1.31472 w_k1_KO_s001 + 1.53324 w_k1_MAT_s001
                                 + 1.21404 w_k1_MSFT_s001
                                 + 1.15824 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.01217 w_k1_ADM_s001
                                 + 0.95643 w_k1_AXP_s001
                                 + 0.789672 w_k1_DIS_s001
                                 + 0.955002 w_k1_FDX_s001
                                 + 0.712116 w_k1_IBM_s001
                                 + 0.946963 w_k1_INTC_s001
                                 + 0.741845 w_k1_KO_s001
                                 + 0.964042 w_k1_MAT_s001
                                 + 0.876558 w_k1_MSFT_s001
                                 + 0.918766 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.780283 w_k1_ADM_s001
                                 + 0.8578 w_k1_AXP_s001
                                 + 0.881581 w_k1_DIS_s001
                                 + 0.812779 w_k1_FDX_s001
                                 + 1.33225 w_k1_IBM_s001
                                 + 0.940904 w_k1_INTC_s001
                                 + 1.29668 w_k1_KO_s001 + 1.46731 w_k1_MAT_s001
                                 + 0.952464 w_k1_MSFT_s001
                                 + 1.03148 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.27078 w_k1_ADM_s001
                                 + 1.12413 w_k1_AXP_s001
                                 + 1.18789 w_k1_DIS_s001
                                 + 1.24209 w_k1_FDX_s001
                                 + 0.93925 w_k1_IBM_s001
                                 + 1.08915 w_k1_INTC_s001
                                 + 1.09043 w_k1_KO_s001 + 1.3346 w_k1_MAT_s001
                                 + 1.11946 w_k1_MSFT_s001
                                 + 1.03703 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.989735 w_k1_ADM_s001
                                 + 1.04117 w_k1_AXP_s001
                                 + 0.857812 w_k1_DIS_s001
                                 + 0.897424 w_k1_FDX_s001
                                 + 1.12166 w_k1_IBM_s001
                                 + 0.860749 w_k1_INTC_s001
                                 + 1.02787 w_k1_KO_s001 + 1.7492 w_k1_MAT_s001
                                 + 0.804968 w_k1_MSFT_s001
                                 + 1.00045 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.798305 w_k1_ADM_s001
                                 + 0.693003 w_k1_AXP_s001
                                 + 0.605845 w_k1_DIS_s001
                                 + 0.735046 w_k1_FDX_s001
                                 + 0.745194 w_k1_IBM_s001
                                 + 0.773791 w_k1_INTC_s001
                                 + 0.774849 w_k1_KO_s001
                                 + 1.06124 w_k1_MAT_s001
                                 + 0.75596 w_k1_MSFT_s001
                                 + 0.833631 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.37312 w_k1_ADM_s001
                                 + 1.58474 w_k1_AXP_s001
                                 + 1.33833 w_k1_DIS_s001
                                 + 1.34237 w_k1_FDX_s001
                                 + 1.50365 w_k1_IBM_s001
                                 + 1.29957 w_k1_INTC_s001
                                 + 1.31554 w_k1_KO_s001 + 2.97938 w_k1_MAT_s001
                                 + 1.16382 w_k1_MSFT_s001
                                 + 1.29975 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.32036 w_k1_ADM_s001
                                 + 1.09362 w_k1_AXP_s001
                                 + 0.932976 w_k1_DIS_s001
                                 + 1.09971 w_k1_FDX_s001
                                 + 0.751602 w_k1_IBM_s001
                                 + 1.01757 w_k1_INTC_s001
                                 + 0.930789 w_k1_KO_s001
                                 + 1.12179 w_k1_MAT_s001
                                 + 0.950474 w_k1_MSFT_s001
                                 + 0.915024 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.01074 w_k1_ADM_s002
                                 + 1.53999 w_k1_AXP_s002
                                 + 1.04564 w_k1_DIS_s002
                                 + 1.17972 w_k1_FDX_s002
                                 + 1.29425 w_k1_IBM_s002
                                 + 1.15892 w_k1_INTC_s002
                                 + 1.10263 w_k1_KO_s002 + 1.81375 w_k1_MAT_s002
                                 + 1.07558 w_k1_MSFT_s002
                                 + 1.07079 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.949599 w_k1_ADM_s002
                                 + 1.02625 w_k1_AXP_s002
                                 + 0.999645 w_k1_DIS_s002
                                 + 0.813151 w_k1_FDX_s002 + 1.361 w_k1_IBM_s002
                                 + 1.00767 w_k1_INTC_s002
                                 + 1.01216 w_k1_KO_s002 + 1.59505 w_k1_MAT_s002
                                 + 0.900787 w_k1_MSFT_s002
                                 + 0.995437 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.40244 w_k1_ADM_s002
                                 + 1.23759 w_k1_AXP_s002
                                 + 1.05416 w_k1_DIS_s002
                                 + 1.10779 w_k1_FDX_s002
                                 + 1.00019 w_k1_IBM_s002
                                 + 0.937785 w_k1_INTC_s002
                                 + 1.10302 w_k1_KO_s002 + 1.15577 w_k1_MAT_s002
                                 + 0.988269 w_k1_MSFT_s002
                                 + 1.06864 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.03058 w_k1_ADM_s002
                                 + 0.729774 w_k1_AXP_s002
                                 + 0.614176 w_k1_DIS_s002
                                 + 0.700606 w_k1_FDX_s002
                                 + 0.895528 w_k1_IBM_s002
                                 + 0.789658 w_k1_INTC_s002
                                 + 0.823901 w_k1_KO_s002
                                 + 0.812914 w_k1_MAT_s002
                                 + 0.713753 w_k1_MSFT_s002
                                 + 0.820434 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.43079 w_k1_ADM_s002
                                 + 1.56924 w_k1_AXP_s002
                                 + 1.48894 w_k1_DIS_s002
                                 + 1.36212 w_k1_FDX_s002 + 1.1321 w_k1_IBM_s002
                                 + 1.2445 w_k1_INTC_s002 + 1.25883 w_k1_KO_s002
                                 + 1.39018 w_k1_MAT_s002
                                 + 1.23331 w_k1_MSFT_s002 + 1.1413 w_k1_PG_s002
                                 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.23298 w_k1_ADM_s002
                                 + 1.94397 w_k1_AXP_s002
                                 + 1.96577 w_k1_DIS_s002
                                 + 1.44764 w_k1_FDX_s002
                                 + 1.64492 w_k1_IBM_s002 + 1.277 w_k1_INTC_s002
                                 + 1.54542 w_k1_KO_s002 + 1.8589 w_k1_MAT_s002
                                 + 1.32939 w_k1_MSFT_s002
                                 + 1.38032 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.914664 w_k1_ADM_s002
                                 + 0.956945 w_k1_AXP_s002
                                 + 0.899636 w_k1_DIS_s002
                                 + 0.966607 w_k1_FDX_s002
                                 + 0.814208 w_k1_IBM_s002
                                 + 0.9303 w_k1_INTC_s002 + 0.89392 w_k1_KO_s002
                                 + 0.831887 w_k1_MAT_s002
                                 + 0.83385 w_k1_MSFT_s002
                                 + 0.904819 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.05824 w_k1_ADM_s002
                                 + 1.16331 w_k1_AXP_s002
                                 + 1.12373 w_k1_DIS_s002
                                 + 1.18137 w_k1_FDX_s002
                                 + 0.67844 w_k1_IBM_s002
                                 + 1.12851 w_k1_INTC_s002
                                 + 0.842412 w_k1_KO_s002
                                 + 0.934068 w_k1_MAT_s002
                                 + 1.09275 w_k1_MSFT_s002
                                 + 0.96465 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.836338 w_k1_ADM_s003
                                 + 0.686545 w_k1_AXP_s003
                                 + 0.64565 w_k1_DIS_s003
                                 + 0.794565 w_k1_FDX_s003
                                 + 0.749255 w_k1_IBM_s003
                                 + 0.611405 w_k1_INTC_s003
                                 + 0.827421 w_k1_KO_s003
                                 + 0.90465 w_k1_MAT_s003
                                 + 0.709921 w_k1_MSFT_s003
                                 + 0.814806 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.14936 w_k1_ADM_s003
                                 + 0.783819 w_k1_AXP_s003
                                 + 0.798777 w_k1_DIS_s003
                                 + 1.06881 w_k1_FDX_s003
                                 + 0.681095 w_k1_IBM_s003
                                 + 0.869218 w_k1_INTC_s003
                                 + 0.804048 w_k1_KO_s003
                                 + 0.663423 w_k1_MAT_s003
                                 + 0.94453 w_k1_MSFT_s003
                                 + 0.872702 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.26241 w_k1_ADM_s003
                                 + 1.09147 w_k1_AXP_s003
                                 + 0.888323 w_k1_DIS_s003
                                 + 1.07085 w_k1_FDX_s003
                                 + 0.732449 w_k1_IBM_s003
                                 + 0.968444 w_k1_INTC_s003
                                 + 0.934724 w_k1_KO_s003
                                 + 0.862167 w_k1_MAT_s003
                                 + 1.02908 w_k1_MSFT_s003
                                 + 1.05593 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.887315 w_k1_ADM_s003
                                 + 0.700925 w_k1_AXP_s003
                                 + 0.664597 w_k1_DIS_s003
                                 + 0.782587 w_k1_FDX_s003
                                 + 1.13047 w_k1_IBM_s003
                                 + 0.848838 w_k1_INTC_s003
                                 + 0.913802 w_k1_KO_s003
                                 + 0.810153 w_k1_MAT_s003
                                 + 0.766696 w_k1_MSFT_s003
                                 + 0.833115 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.913448 w_k1_ADM_s003
                                 + 1.10808 w_k1_AXP_s003
                                 + 0.85481 w_k1_DIS_s003
                                 + 1.00844 w_k1_FDX_s003
                                 + 1.36527 w_k1_IBM_s003
                                 + 0.946096 w_k1_INTC_s003
                                 + 0.910925 w_k1_KO_s003 + 1.5157 w_k1_MAT_s003
                                 + 0.860727 w_k1_MSFT_s003
                                 + 0.91441 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.31679 w_k1_ADM_s003
                                 + 1.34498 w_k1_AXP_s003
                                 + 1.52685 w_k1_DIS_s003
                                 + 1.29235 w_k1_FDX_s003
                                 + 1.42797 w_k1_IBM_s003
                                 + 1.04537 w_k1_INTC_s003
                                 + 1.30708 w_k1_KO_s003 + 1.72065 w_k1_MAT_s003
                                 + 1.14878 w_k1_MSFT_s003
                                 + 1.20917 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.06559 w_k1_ADM_s003
                                 + 1.02841 w_k1_AXP_s003 + 1.1985 w_k1_DIS_s003
                                 + 1.12334 w_k1_FDX_s003
                                 + 1.04066 w_k1_IBM_s003
                                 + 1.04993 w_k1_INTC_s003
                                 + 1.12674 w_k1_KO_s003 + 1.24561 w_k1_MAT_s003
                                 + 1.26676 w_k1_MSFT_s003
                                 + 1.01522 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.67718 w_k1_ADM_s003
                                 + 1.63091 w_k1_AXP_s003
                                 + 1.65694 w_k1_DIS_s003
                                 + 1.58242 w_k1_FDX_s003 + 1.1243 w_k1_IBM_s003
                                 + 1.11012 w_k1_INTC_s003
                                 + 1.25036 w_k1_KO_s003 + 1.0824 w_k1_MAT_s003
                                 + 1.26173 w_k1_MSFT_s003
                                 + 1.14116 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.910028 w_k1_ADM_s004
                                 + 0.807947 w_k1_AXP_s004
                                 + 0.747805 w_k1_DIS_s004
                                 + 0.883776 w_k1_FDX_s004
                                 + 0.757146 w_k1_IBM_s004
                                 + 0.709737 w_k1_INTC_s004
                                 + 0.625264 w_k1_KO_s004
                                 + 0.717535 w_k1_MAT_s004
                                 + 0.726975 w_k1_MSFT_s004
                                 + 0.885031 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.741202 w_k1_ADM_s004
                                 + 0.574661 w_k1_AXP_s004
                                 + 0.585491 w_k1_DIS_s004
                                 + 0.682391 w_k1_FDX_s004
                                 + 0.748719 w_k1_IBM_s004
                                 + 0.692199 w_k1_INTC_s004
                                 + 0.603054 w_k1_KO_s004
                                 + 0.456155 w_k1_MAT_s004
                                 + 0.62896 w_k1_MSFT_s004
                                 + 0.69957 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.932171 w_k1_ADM_s004
                                 + 0.964464 w_k1_AXP_s004
                                 + 0.861827 w_k1_DIS_s004
                                 + 1.0075 w_k1_FDX_s004
                                 + 0.670777 w_k1_IBM_s004
                                 + 0.980653 w_k1_INTC_s004
                                 + 0.690684 w_k1_KO_s004
                                 + 0.579555 w_k1_MAT_s004
                                 + 0.813084 w_k1_MSFT_s004
                                 + 0.907666 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.904772 w_k1_ADM_s004
                                 + 1.00401 w_k1_AXP_s004
                                 + 1.14206 w_k1_DIS_s004
                                 + 1.14821 w_k1_FDX_s004
                                 + 0.839024 w_k1_IBM_s004
                                 + 1.12004 w_k1_INTC_s004 + 1.1092 w_k1_KO_s004
                                 + 1.01358 w_k1_MAT_s004
                                 + 1.21799 w_k1_MSFT_s004
                                 + 0.980701 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.701377 w_k1_ADM_s004
                                 + 0.757447 w_k1_AXP_s004
                                 + 0.748387 w_k1_DIS_s004
                                 + 0.776657 w_k1_FDX_s004
                                 + 1.27542 w_k1_IBM_s004
                                 + 0.853314 w_k1_INTC_s004
                                 + 0.926026 w_k1_KO_s004
                                 + 1.34077 w_k1_MAT_s004
                                 + 0.91443 w_k1_MSFT_s004
                                 + 1.12896 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.763283 w_k1_ADM_s004
                                 + 0.877705 w_k1_AXP_s004
                                 + 0.915259 w_k1_DIS_s004
                                 + 1.00967 w_k1_FDX_s004
                                 + 1.01522 w_k1_IBM_s004
                                 + 1.04138 w_k1_INTC_s004
                                 + 0.991677 w_k1_KO_s004
                                 + 0.988213 w_k1_MAT_s004
                                 + 0.936904 w_k1_MSFT_s004
                                 + 0.965698 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.17715 w_k1_ADM_s004
                                 + 1.26101 w_k1_AXP_s004
                                 + 1.32799 w_k1_DIS_s004
                                 + 1.16451 w_k1_FDX_s004
                                 + 1.43577 w_k1_IBM_s004
                                 + 1.1475 w_k1_INTC_s004 + 1.18752 w_k1_KO_s004
                                 + 1.69375 w_k1_MAT_s004
                                 + 1.14278 w_k1_MSFT_s004
                                 + 1.44133 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.04065 w_k1_ADM_s004
                                 + 0.993766 w_k1_AXP_s004
                                 + 1.16191 w_k1_DIS_s004
                                 + 1.04666 w_k1_FDX_s004
                                 + 1.06381 w_k1_IBM_s004
                                 + 0.762782 w_k1_INTC_s004
                                 + 1.03275 w_k1_KO_s004 + 1.1605 w_k1_MAT_s004
                                 + 0.997893 w_k1_MSFT_s004
                                 + 1.07556 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.835036 w_k1_ADM_s005
                                 + 1.24095 w_k1_AXP_s005
                                 + 0.85891 w_k1_DIS_s005
                                 + 0.97214 w_k1_FDX_s005 + 0.9914 w_k1_IBM_s005
                                 + 1.11053 w_k1_INTC_s005
                                 + 1.11649 w_k1_KO_s005 + 1.18549 w_k1_MAT_s005
                                 + 0.976035 w_k1_MSFT_s005
                                 + 1.10548 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.860287 w_k1_ADM_s005
                                 + 0.740185 w_k1_AXP_s005
                                 + 0.794758 w_k1_DIS_s005
                                 + 0.991696 w_k1_FDX_s005
                                 + 0.807123 w_k1_IBM_s005
                                 + 1.01938 w_k1_INTC_s005
                                 + 0.971803 w_k1_KO_s005
                                 + 0.933065 w_k1_MAT_s005
                                 + 0.992316 w_k1_MSFT_s005
                                 + 0.903471 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.795039 w_k1_ADM_s005
                                 + 0.718943 w_k1_AXP_s005
                                 + 0.840769 w_k1_DIS_s005
                                 + 0.877344 w_k1_FDX_s005
                                 + 1.10132 w_k1_IBM_s005
                                 + 0.898863 w_k1_INTC_s005
                                 + 1.0966 w_k1_KO_s005 + 0.910787 w_k1_MAT_s005
                                 + 0.837945 w_k1_MSFT_s005
                                 + 0.972936 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.09813 w_k1_ADM_s005
                                 + 1.49788 w_k1_AXP_s005
                                 + 1.31362 w_k1_DIS_s005
                                 + 1.20374 w_k1_FDX_s005
                                 + 1.28851 w_k1_IBM_s005
                                 + 1.43014 w_k1_INTC_s005
                                 + 1.33721 w_k1_KO_s005
                                 + 0.963418 w_k1_MAT_s005
                                 + 1.22662 w_k1_MSFT_s005
                                 + 1.05262 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.750098 w_k1_ADM_s005
                                 + 0.615268 w_k1_AXP_s005
                                 + 0.560747 w_k1_DIS_s005
                                 + 0.704048 w_k1_FDX_s005
                                 + 0.829237 w_k1_IBM_s005
                                 + 0.919845 w_k1_INTC_s005
                                 + 0.892132 w_k1_KO_s005
                                 + 0.824003 w_k1_MAT_s005
                                 + 0.878856 w_k1_MSFT_s005
                                 + 0.809236 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.896663 w_k1_ADM_s005
                                 + 0.939782 w_k1_AXP_s005
                                 + 0.856622 w_k1_DIS_s005
                                 + 0.886949 w_k1_FDX_s005
                                 + 1.51106 w_k1_IBM_s005
                                 + 1.26225 w_k1_INTC_s005
                                 + 1.28269 w_k1_KO_s005 + 1.36053 w_k1_MAT_s005
                                 + 0.929941 w_k1_MSFT_s005
                                 + 1.02623 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.15222 w_k1_ADM_s005
                                 + 0.909265 w_k1_AXP_s005
                                 + 1.20414 w_k1_DIS_s005
                                 + 1.11635 w_k1_FDX_s005
                                 + 0.953618 w_k1_IBM_s005
                                 + 1.17607 w_k1_INTC_s005
                                 + 1.06673 w_k1_KO_s005
                                 + 0.756237 w_k1_MAT_s005
                                 + 1.0898 w_k1_MSFT_s005
                                 + 0.936132 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.26218 w_k1_ADM_s005
                                 + 1.59698 w_k1_AXP_s005
                                 + 1.67885 w_k1_DIS_s005 + 1.6368 w_k1_FDX_s005
                                 + 1.12728 w_k1_IBM_s005
                                 + 1.71625 w_k1_INTC_s005
                                 + 1.35547 w_k1_KO_s005 + 1.47508 w_k1_MAT_s005
                                 + 1.42345 w_k1_MSFT_s005
                                 + 1.20437 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.1728 w_k1_ADM_s006
                                 + 1.2799 w_k1_AXP_s006 + 1.35612 w_k1_DIS_s006
                                 + 1.08624 w_k1_FDX_s006
                                 + 1.39971 w_k1_IBM_s006
                                 + 0.988103 w_k1_INTC_s006
                                 + 1.21007 w_k1_KO_s006 + 1.37322 w_k1_MAT_s006
                                 + 1.02763 w_k1_MSFT_s006
                                 + 1.02559 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.16658 w_k1_ADM_s006
                                 + 1.39764 w_k1_AXP_s006
                                 + 1.84285 w_k1_DIS_s006
                                 + 1.40108 w_k1_FDX_s006
                                 + 1.60302 w_k1_IBM_s006
                                 + 1.20845 w_k1_INTC_s006
                                 + 1.42055 w_k1_KO_s006 + 1.42087 w_k1_MAT_s006
                                 + 1.22283 w_k1_MSFT_s006
                                 + 1.09617 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.0011 w_k1_ADM_s006
                                 + 1.01497 w_k1_AXP_s006
                                 + 0.899648 w_k1_DIS_s006
                                 + 0.954792 w_k1_FDX_s006
                                 + 1.08985 w_k1_IBM_s006
                                 + 0.708033 w_k1_INTC_s006
                                 + 0.8445 w_k1_KO_s006 + 1.08967 w_k1_MAT_s006
                                 + 0.816779 w_k1_MSFT_s006
                                 + 0.924564 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.98985 w_k1_ADM_s006
                                 + 0.95351 w_k1_AXP_s006
                                 + 0.689484 w_k1_DIS_s006
                                 + 0.941856 w_k1_FDX_s006
                                 + 0.680836 w_k1_IBM_s006
                                 + 0.825076 w_k1_INTC_s006
                                 + 0.646664 w_k1_KO_s006
                                 + 0.461513 w_k1_MAT_s006
                                 + 0.853324 w_k1_MSFT_s006
                                 + 0.828352 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.25113 w_k1_ADM_s006
                                 + 1.25912 w_k1_AXP_s006
                                 + 1.26655 w_k1_DIS_s006
                                 + 1.19219 w_k1_FDX_s006
                                 + 1.01302 w_k1_IBM_s006
                                 + 1.07197 w_k1_INTC_s006
                                 + 1.02468 w_k1_KO_s006 + 1.01325 w_k1_MAT_s006
                                 + 1.19516 w_k1_MSFT_s006 + 1.1328 w_k1_PG_s006
                                 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.0354 w_k1_ADM_s006
                                 + 0.933234 w_k1_AXP_s006
                                 + 0.953501 w_k1_DIS_s006
                                 + 0.818443 w_k1_FDX_s006
                                 + 1.57821 w_k1_IBM_s006
                                 + 0.825214 w_k1_INTC_s006
                                 + 0.968682 w_k1_KO_s006
                                 + 1.34084 w_k1_MAT_s006
                                 + 0.862602 w_k1_MSFT_s006
                                 + 0.939297 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.743643 w_k1_ADM_s006
                                 + 0.767264 w_k1_AXP_s006
                                 + 0.564978 w_k1_DIS_s006
                                 + 0.735247 w_k1_FDX_s006
                                 + 0.869268 w_k1_IBM_s006
                                 + 0.654639 w_k1_INTC_s006
                                 + 0.698406 w_k1_KO_s006
                                 + 0.786645 w_k1_MAT_s006
                                 + 0.761379 w_k1_MSFT_s006
                                 + 0.771064 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.996631 w_k1_ADM_s006
                                 + 0.792748 w_k1_AXP_s006
                                 + 1.20775 w_k1_DIS_s006
                                 + 1.04651 w_k1_FDX_s006
                                 + 0.81814 w_k1_IBM_s006
                                 + 1.03937 w_k1_INTC_s006
                                 + 1.03183 w_k1_KO_s006
                                 + 0.653183 w_k1_MAT_s006
                                 + 1.10212 w_k1_MSFT_s006
                                 + 0.989759 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.01817 w_k1_ADM_s007
                                 + 1.32152 w_k1_AXP_s007
                                 + 1.19019 w_k1_DIS_s007
                                 + 1.17134 w_k1_FDX_s007
                                 + 1.31378 w_k1_IBM_s007
                                 + 1.15848 w_k1_INTC_s007
                                 + 1.38984 w_k1_KO_s007 + 1.86398 w_k1_MAT_s007
                                 + 1.12957 w_k1_MSFT_s007
                                 + 1.12262 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.88275 w_k1_ADM_s007
                                 + 0.879651 w_k1_AXP_s007
                                 + 1.09775 w_k1_DIS_s007
                                 + 0.993855 w_k1_FDX_s007
                                 + 1.41204 w_k1_IBM_s007
                                 + 0.918466 w_k1_INTC_s007
                                 + 1.29072 w_k1_KO_s007 + 1.69107 w_k1_MAT_s007
                                 + 1.07398 w_k1_MSFT_s007
                                 + 1.08163 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.27188 w_k1_ADM_s007
                                 + 1.36207 w_k1_AXP_s007 + 1.5365 w_k1_DIS_s007
                                 + 1.33258 w_k1_FDX_s007
                                 + 1.11317 w_k1_IBM_s007
                                 + 1.03512 w_k1_INTC_s007
                                 + 1.42837 w_k1_KO_s007 + 1.0979 w_k1_MAT_s007
                                 + 1.30196 w_k1_MSFT_s007
                                 + 1.26555 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.904428 w_k1_ADM_s007
                                 + 0.80132 w_k1_AXP_s007
                                 + 0.778491 w_k1_DIS_s007
                                 + 0.84362 w_k1_FDX_s007
                                 + 1.19719 w_k1_IBM_s007
                                 + 0.88162 w_k1_INTC_s007
                                 + 0.977281 w_k1_KO_s007
                                 + 1.31306 w_k1_MAT_s007
                                 + 0.840094 w_k1_MSFT_s007
                                 + 0.954438 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.936232 w_k1_ADM_s007
                                 + 0.862776 w_k1_AXP_s007
                                 + 0.802171 w_k1_DIS_s007
                                 + 1.01592 w_k1_FDX_s007
                                 + 0.760674 w_k1_IBM_s007
                                 + 0.967862 w_k1_INTC_s007
                                 + 0.799579 w_k1_KO_s007
                                 + 0.665727 w_k1_MAT_s007
                                 + 0.934833 w_k1_MSFT_s007
                                 + 0.899165 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.06102 w_k1_ADM_s007
                                 + 1.12062 w_k1_AXP_s007
                                 + 1.20027 w_k1_DIS_s007
                                 + 1.28825 w_k1_FDX_s007
                                 + 0.741348 w_k1_IBM_s007
                                 + 1.13792 w_k1_INTC_s007
                                 + 1.11421 w_k1_KO_s007 + 1.11823 w_k1_MAT_s007
                                 + 1.34297 w_k1_MSFT_s007
                                 + 1.12896 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.893675 w_k1_ADM_s007
                                 + 0.923159 w_k1_AXP_s007
                                 + 1.27792 w_k1_DIS_s007
                                 + 0.980943 w_k1_FDX_s007
                                 + 0.960605 w_k1_IBM_s007
                                 + 0.953609 w_k1_INTC_s007
                                 + 1.20782 w_k1_KO_s007 + 1.18972 w_k1_MAT_s007
                                 + 1.18353 w_k1_MSFT_s007
                                 + 1.05902 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.724557 w_k1_ADM_s007
                                 + 0.698309 w_k1_AXP_s007
                                 + 0.68239 w_k1_DIS_s007
                                 + 0.761865 w_k1_FDX_s007
                                 + 0.766511 w_k1_IBM_s007
                                 + 0.699376 w_k1_INTC_s007
                                 + 0.743958 w_k1_KO_s007
                                 + 0.738604 w_k1_MAT_s007
                                 + 0.788465 w_k1_MSFT_s007
                                 + 0.823973 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.24056 w_k1_ADM_s010
                                 + 1.50309 w_k1_AXP_s010
                                 + 1.59225 w_k1_DIS_s010
                                 + 1.27412 w_k1_FDX_s010
                                 + 1.65557 w_k1_IBM_s010
                                 + 1.05815 w_k1_INTC_s010 + 1.3983 w_k1_KO_s010
                                 + 1.91827 w_k1_MAT_s010
                                 + 1.14459 w_k1_MSFT_s010 + 1.2127 w_k1_PG_s010
                                 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.01364 w_k1_ADM_s010
                                 + 0.91901 w_k1_AXP_s010
                                 + 0.954384 w_k1_DIS_s010
                                 + 0.952424 w_k1_FDX_s010
                                 + 0.810409 w_k1_IBM_s010
                                 + 0.97538 w_k1_INTC_s010
                                 + 0.949953 w_k1_KO_s010
                                 + 0.845031 w_k1_MAT_s010
                                 + 1.09597 w_k1_MSFT_s010
                                 + 0.923117 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.866564 w_k1_ADM_s010
                                 + 1.30095 w_k1_AXP_s010
                                 + 0.902075 w_k1_DIS_s010
                                 + 0.930841 w_k1_FDX_s010
                                 + 1.55435 w_k1_IBM_s010
                                 + 1.11827 w_k1_INTC_s010
                                 + 1.18131 w_k1_KO_s010 + 2.033 w_k1_MAT_s010
                                 + 0.982735 w_k1_MSFT_s010
                                 + 1.02387 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.933126 w_k1_ADM_s010
                                 + 0.84993 w_k1_AXP_s010
                                 + 0.628772 w_k1_DIS_s010
                                 + 0.876794 w_k1_FDX_s010
                                 + 0.657612 w_k1_IBM_s010
                                 + 0.864628 w_k1_INTC_s010
                                 + 0.674353 w_k1_KO_s010
                                 + 0.648972 w_k1_MAT_s010
                                 + 0.908209 w_k1_MSFT_s010
                                 + 0.838979 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.15884 w_k1_ADM_s010
                                 + 1.26824 w_k1_AXP_s010
                                 + 1.18139 w_k1_DIS_s010
                                 + 1.09967 w_k1_FDX_s010
                                 + 1.29553 w_k1_IBM_s010
                                 + 1.17799 w_k1_INTC_s010
                                 + 1.23235 w_k1_KO_s010 + 1.3313 w_k1_MAT_s010
                                 + 1.06328 w_k1_MSFT_s010
                                 + 1.12218 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.56311 w_k1_ADM_s010
                                 + 1.36039 w_k1_AXP_s010
                                 + 1.33019 w_k1_DIS_s010
                                 + 1.22951 w_k1_FDX_s010
                                 + 0.931331 w_k1_IBM_s010
                                 + 1.43457 w_k1_INTC_s010
                                 + 1.01011 w_k1_KO_s010
                                 + 0.693175 w_k1_MAT_s010
                                 + 1.19003 w_k1_MSFT_s010
                                 + 1.09709 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.951178 w_k1_ADM_s010
                                 + 0.887269 w_k1_AXP_s010
                                 + 0.785486 w_k1_DIS_s010
                                 + 0.78124 w_k1_FDX_s010
                                 + 0.969362 w_k1_IBM_s010
                                 + 0.872227 w_k1_INTC_s010
                                 + 0.893243 w_k1_KO_s010
                                 + 1.11191 w_k1_MAT_s010
                                 + 0.934685 w_k1_MSFT_s010
                                 + 0.875739 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.875145 w_k1_ADM_s010
                                 + 0.864641 w_k1_AXP_s010
                                 + 0.938363 w_k1_DIS_s010
                                 + 0.905614 w_k1_FDX_s010
                                 + 1.18633 w_k1_IBM_s010
                                 + 0.852345 w_k1_INTC_s010
                                 + 0.944745 w_k1_KO_s010 + 1.1819 w_k1_MAT_s010
                                 + 1.08516 w_k1_MSFT_s010
                                 + 0.998467 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.25571 w_k1_ADM_s011
                                 + 1.07518 w_k1_AXP_s011
                                 + 1.07248 w_k1_DIS_s011
                                 + 0.919332 w_k1_FDX_s011
                                 + 1.32897 w_k1_IBM_s011
                                 + 0.924652 w_k1_INTC_s011
                                 + 1.00525 w_k1_KO_s011 + 1.32273 w_k1_MAT_s011
                                 + 0.912093 w_k1_MSFT_s011
                                 + 0.960311 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.13153 w_k1_ADM_s011
                                 + 0.835644 w_k1_AXP_s011
                                 + 0.981876 w_k1_DIS_s011
                                 + 1.05818 w_k1_FDX_s011
                                 + 0.671154 w_k1_IBM_s011
                                 + 1.00395 w_k1_INTC_s011
                                 + 0.858826 w_k1_KO_s011
                                 + 0.45851 w_k1_MAT_s011
                                 + 1.06886 w_k1_MSFT_s011
                                 + 0.990826 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.41923 w_k1_ADM_s011
                                 + 1.28908 w_k1_AXP_s011
                                 + 1.61579 w_k1_DIS_s011
                                 + 1.34266 w_k1_FDX_s011 + 0.9722 w_k1_IBM_s011
                                 + 1.05715 w_k1_INTC_s011
                                 + 1.36523 w_k1_KO_s011
                                 + 0.782004 w_k1_MAT_s011
                                 + 1.22791 w_k1_MSFT_s011
                                 + 1.28664 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.903095 w_k1_ADM_s011
                                 + 0.731396 w_k1_AXP_s011
                                 + 0.712047 w_k1_DIS_s011
                                 + 0.789937 w_k1_FDX_s011
                                 + 0.79872 w_k1_IBM_s011
                                 + 0.757599 w_k1_INTC_s011
                                 + 0.743518 w_k1_KO_s011
                                 + 0.736614 w_k1_MAT_s011
                                 + 0.834896 w_k1_MSFT_s011
                                 + 0.813485 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.11567 w_k1_ADM_s011
                                 + 1.11342 w_k1_AXP_s011
                                 + 1.09933 w_k1_DIS_s011
                                 + 1.06236 w_k1_FDX_s011
                                 + 1.07437 w_k1_IBM_s011
                                 + 0.76721 w_k1_INTC_s011
                                 + 1.00055 w_k1_KO_s011
                                 + 0.949544 w_k1_MAT_s011
                                 + 1.03082 w_k1_MSFT_s011 + 1.009 w_k1_PG_s011
                                 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.15276 w_k1_ADM_s011
                                 + 1.29493 w_k1_AXP_s011
                                 + 1.12978 w_k1_DIS_s011
                                 + 1.14644 w_k1_FDX_s011
                                 + 0.976087 w_k1_IBM_s011
                                 + 1.2562 w_k1_INTC_s011 + 1.07526 w_k1_KO_s011
                                 + 1.10686 w_k1_MAT_s011
                                 + 1.0449 w_k1_MSFT_s011 + 1.15394 w_k1_PG_s011
                                 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.21166 w_k1_ADM_s011
                                 + 1.37827 w_k1_AXP_s011
                                 + 1.47031 w_k1_DIS_s011 + 1.2537 w_k1_FDX_s011
                                 + 1.39582 w_k1_IBM_s011
                                 + 1.05712 w_k1_INTC_s011
                                 + 1.30299 w_k1_KO_s011 + 1.39296 w_k1_MAT_s011
                                 + 1.19621 w_k1_MSFT_s011
                                 + 1.32324 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.08414 w_k1_ADM_s011
                                 + 0.857825 w_k1_AXP_s011
                                 + 0.900756 w_k1_DIS_s011
                                 + 0.980904 w_k1_FDX_s011
                                 + 0.835499 w_k1_IBM_s011
                                 + 0.864855 w_k1_INTC_s011
                                 + 1.03225 w_k1_KO_s011
                                 + 0.692899 w_k1_MAT_s011
                                 + 0.909132 w_k1_MSFT_s011
                                 + 0.884424 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.22326 w_k1_ADM_s012
                                 + 1.6699 w_k1_AXP_s012 + 1.64376 w_k1_DIS_s012
                                 + 1.15298 w_k1_FDX_s012
                                 + 1.50769 w_k1_IBM_s012
                                 + 1.52792 w_k1_INTC_s012
                                 + 1.49746 w_k1_KO_s012 + 1.88487 w_k1_MAT_s012
                                 + 1.4252 w_k1_MSFT_s012 + 1.2818 w_k1_PG_s012
                                 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.1012 w_k1_ADM_s012
                                 + 1.17224 w_k1_AXP_s012
                                 + 1.47212 w_k1_DIS_s012
                                 + 1.25226 w_k1_FDX_s012
                                 + 0.798428 w_k1_IBM_s012
                                 + 1.27034 w_k1_INTC_s012 + 1.1462 w_k1_KO_s012
                                 + 1.09848 w_k1_MAT_s012
                                 + 1.45511 w_k1_MSFT_s012
                                 + 1.21273 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.14095 w_k1_ADM_s012
                                 + 0.946713 w_k1_AXP_s012
                                 + 1.2566 w_k1_DIS_s012 + 1.0295 w_k1_FDX_s012
                                 + 0.943835 w_k1_IBM_s012
                                 + 1.16962 w_k1_INTC_s012
                                 + 1.04334 w_k1_KO_s012
                                 + 0.706109 w_k1_MAT_s012
                                 + 1.32842 w_k1_MSFT_s012
                                 + 1.02332 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.954584 w_k1_ADM_s012
                                 + 0.970304 w_k1_AXP_s012
                                 + 0.864987 w_k1_DIS_s012
                                 + 0.925677 w_k1_FDX_s012
                                 + 1.21691 w_k1_IBM_s012
                                 + 0.981235 w_k1_INTC_s012
                                 + 1.06779 w_k1_KO_s012 + 1.14258 w_k1_MAT_s012
                                 + 0.928948 w_k1_MSFT_s012
                                 + 0.979138 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.43434 w_k1_ADM_s012
                                 + 1.42424 w_k1_AXP_s012
                                 + 1.62406 w_k1_DIS_s012
                                 + 1.45844 w_k1_FDX_s012
                                 + 1.06963 w_k1_IBM_s012
                                 + 1.4695 w_k1_INTC_s012 + 1.3566 w_k1_KO_s012
                                 + 1.30959 w_k1_MAT_s012
                                 + 1.38133 w_k1_MSFT_s012
                                 + 1.23352 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.931813 w_k1_ADM_s012
                                 + 0.811433 w_k1_AXP_s012
                                 + 0.806684 w_k1_DIS_s012
                                 + 0.829347 w_k1_FDX_s012
                                 + 0.816899 w_k1_IBM_s012
                                 + 1.05217 w_k1_INTC_s012
                                 + 0.852378 w_k1_KO_s012
                                 + 0.73707 w_k1_MAT_s012
                                 + 1.04954 w_k1_MSFT_s012
                                 + 0.879327 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.980908 w_k1_ADM_s012
                                 + 1.20865 w_k1_AXP_s012
                                 + 0.963299 w_k1_DIS_s012
                                 + 1.09282 w_k1_FDX_s012
                                 + 0.830791 w_k1_IBM_s012
                                 + 1.11065 w_k1_INTC_s012
                                 + 0.924768 w_k1_KO_s012
                                 + 0.878594 w_k1_MAT_s012
                                 + 1.04442 w_k1_MSFT_s012
                                 + 0.995806 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.81089 w_k1_ADM_s012
                                 + 0.663916 w_k1_AXP_s012
                                 + 0.61574 w_k1_DIS_s012
                                 + 0.665604 w_k1_FDX_s012
                                 + 0.775646 w_k1_IBM_s012
                                 + 0.852231 w_k1_INTC_s012
                                 + 0.724861 w_k1_KO_s012
                                 + 0.806135 w_k1_MAT_s012
                                 + 0.846161 w_k1_MSFT_s012
                                 + 0.780889 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.07435 w_k1_ADM_s013
                                 + 0.891613 w_k1_AXP_s013
                                 + 0.955039 w_k1_DIS_s013
                                 + 1.05297 w_k1_FDX_s013
                                 + 0.597997 w_k1_IBM_s013
                                 + 0.879422 w_k1_INTC_s013
                                 + 0.806964 w_k1_KO_s013
                                 + 0.637309 w_k1_MAT_s013
                                 + 0.866947 w_k1_MSFT_s013
                                 + 0.919224 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.23027 w_k1_ADM_s013
                                 + 1.24126 w_k1_AXP_s013
                                 + 1.21494 w_k1_DIS_s013
                                 + 1.09084 w_k1_FDX_s013
                                 + 1.33917 w_k1_IBM_s013
                                 + 0.99446 w_k1_INTC_s013
                                 + 1.12085 w_k1_KO_s013 + 1.55425 w_k1_MAT_s013
                                 + 1.10314 w_k1_MSFT_s013
                                 + 1.05099 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.756206 w_k1_ADM_s013
                                 + 0.708114 w_k1_AXP_s013
                                 + 0.688585 w_k1_DIS_s013
                                 + 0.75397 w_k1_FDX_s013
                                 + 0.848045 w_k1_IBM_s013
                                 + 0.832111 w_k1_INTC_s013
                                 + 0.802618 w_k1_KO_s013
                                 + 1.10073 w_k1_MAT_s013
                                 + 0.880947 w_k1_MSFT_s013
                                 + 0.841306 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.02585 w_k1_ADM_s013
                                 + 0.898336 w_k1_AXP_s013
                                 + 0.807773 w_k1_DIS_s013
                                 + 0.836979 w_k1_FDX_s013
                                 + 1.17486 w_k1_IBM_s013
                                 + 0.753621 w_k1_INTC_s013
                                 + 0.818979 w_k1_KO_s013
                                 + 1.26826 w_k1_MAT_s013
                                 + 0.810066 w_k1_MSFT_s013
                                 + 0.927857 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.31039 w_k1_ADM_s013
                                 + 1.54639 w_k1_AXP_s013
                                 + 1.65551 w_k1_DIS_s013
                                 + 1.45493 w_k1_FDX_s013
                                 + 1.14546 w_k1_IBM_s013
                                 + 1.1179 w_k1_INTC_s013 + 1.15732 w_k1_KO_s013
                                 + 1.5073 w_k1_MAT_s013
                                 + 1.29848 w_k1_MSFT_s013
                                 + 1.27091 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.04309 w_k1_ADM_s013
                                 + 1.30365 w_k1_AXP_s013
                                 + 1.02398 w_k1_DIS_s013
                                 + 1.03057 w_k1_FDX_s013
                                 + 1.72805 w_k1_IBM_s013
                                 + 1.03173 w_k1_INTC_s013
                                 + 1.17749 w_k1_KO_s013 + 1.7551 w_k1_MAT_s013
                                 + 0.994037 w_k1_MSFT_s013
                                 + 1.1146 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.22539 w_k1_ADM_s013
                                 + 0.896874 w_k1_AXP_s013
                                 + 0.989719 w_k1_DIS_s013
                                 + 1.07242 w_k1_FDX_s013
                                 + 0.908192 w_k1_IBM_s013
                                 + 0.898245 w_k1_INTC_s013
                                 + 0.888439 w_k1_KO_s013
                                 + 0.967811 w_k1_MAT_s013
                                 + 1.1006 w_k1_MSFT_s013
                                 + 0.977034 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.3344 w_k1_ADM_s013
                                 + 1.24997 w_k1_AXP_s013
                                 + 1.16894 w_k1_DIS_s013
                                 + 1.31217 w_k1_FDX_s013
                                 + 0.942285 w_k1_IBM_s013
                                 + 1.03135 w_k1_INTC_s013
                                 + 0.969402 w_k1_KO_s013
                                 + 1.07587 w_k1_MAT_s013
                                 + 1.27169 w_k1_MSFT_s013
                                 + 1.19469 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.960646 w_k1_ADM_s014
                                 + 1.05179 w_k1_AXP_s014
                                 + 0.930872 w_k1_DIS_s014
                                 + 0.956885 w_k1_FDX_s014
                                 + 1.54002 w_k1_IBM_s014
                                 + 0.942876 w_k1_INTC_s014
                                 + 1.29332 w_k1_KO_s014 + 1.86439 w_k1_MAT_s014
                                 + 0.933892 w_k1_MSFT_s014
                                 + 1.11264 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.920648 w_k1_ADM_s014
                                 + 0.617088 w_k1_AXP_s014
                                 + 0.738208 w_k1_DIS_s014
                                 + 0.74701 w_k1_FDX_s014
                                 + 1.03914 w_k1_IBM_s014
                                 + 0.854126 w_k1_INTC_s014
                                 + 0.840164 w_k1_KO_s014
                                 + 1.05915 w_k1_MAT_s014
                                 + 0.841134 w_k1_MSFT_s014
                                 + 0.852857 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.754517 w_k1_ADM_s014
                                 + 0.603371 w_k1_AXP_s014
                                 + 0.58549 w_k1_DIS_s014
                                 + 0.742699 w_k1_FDX_s014
                                 + 0.725768 w_k1_IBM_s014
                                 + 0.724643 w_k1_INTC_s014
                                 + 0.740278 w_k1_KO_s014
                                 + 0.729747 w_k1_MAT_s014
                                 + 0.73841 w_k1_MSFT_s014 + 0.8819 w_k1_PG_s014
                                 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.36601 w_k1_ADM_s014
                                 + 1.164 w_k1_AXP_s014 + 1.39583 w_k1_DIS_s014
                                 + 1.22777 w_k1_FDX_s014
                                 + 0.889974 w_k1_IBM_s014
                                 + 0.850486 w_k1_INTC_s014
                                 + 0.975175 w_k1_KO_s014
                                 + 1.11285 w_k1_MAT_s014
                                 + 1.1677 w_k1_MSFT_s014 + 1.16283 w_k1_PG_s014
                                 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.38134 w_k1_ADM_s014
                                 + 1.51876 w_k1_AXP_s014
                                 + 1.89138 w_k1_DIS_s014
                                 + 1.47896 w_k1_FDX_s014
                                 + 1.58246 w_k1_IBM_s014
                                 + 1.17781 w_k1_INTC_s014 + 1.3388 w_k1_KO_s014
                                 + 2.096 w_k1_MAT_s014 + 1.26459 w_k1_MSFT_s014
                                 + 1.26501 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.22262 w_k1_ADM_s014
                                 + 1.11147 w_k1_AXP_s014
                                 + 0.92712 w_k1_DIS_s014
                                 + 1.04308 w_k1_FDX_s014
                                 + 1.05595 w_k1_IBM_s014
                                 + 0.817271 w_k1_INTC_s014
                                 + 1.06805 w_k1_KO_s014 + 1.25709 w_k1_MAT_s014
                                 + 0.931007 w_k1_MSFT_s014 + 1.017 w_k1_PG_s014
                                 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.3882 w_k1_ADM_s014
                                 + 0.943897 w_k1_AXP_s014
                                 + 1.0091 w_k1_DIS_s014 + 1.13891 w_k1_FDX_s014
                                 + 0.742937 w_k1_IBM_s014
                                 + 1.00568 w_k1_INTC_s014
                                 + 0.940664 w_k1_KO_s014
                                 + 0.676321 w_k1_MAT_s014
                                 + 1.00145 w_k1_MSFT_s014
                                 + 1.09548 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.09383 w_k1_ADM_s014
                                 + 0.694697 w_k1_AXP_s014
                                 + 0.895722 w_k1_DIS_s014
                                 + 0.884999 w_k1_FDX_s014
                                 + 0.795726 w_k1_IBM_s014
                                 + 0.958121 w_k1_INTC_s014
                                 + 0.935049 w_k1_KO_s014
                                 + 0.658416 w_k1_MAT_s014
                                 + 1.02455 w_k1_MSFT_s014
                                 + 0.993432 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.13961 w_k1_ADM_s015
                                 + 0.615384 w_k1_AXP_s015
                                 + 0.607717 w_k1_DIS_s015
                                 + 0.771904 w_k1_FDX_s015
                                 + 0.757077 w_k1_IBM_s015
                                 + 0.761592 w_k1_INTC_s015
                                 + 0.733604 w_k1_KO_s015
                                 + 0.661789 w_k1_MAT_s015
                                 + 0.856764 w_k1_MSFT_s015
                                 + 0.813326 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.3367 w_k1_ADM_s015
                                 + 1.154 w_k1_AXP_s015 + 1.1575 w_k1_DIS_s015
                                 + 0.972237 w_k1_FDX_s015
                                 + 1.37503 w_k1_IBM_s015
                                 + 0.908616 w_k1_INTC_s015
                                 + 1.16908 w_k1_KO_s015 + 1.65864 w_k1_MAT_s015
                                 + 0.986768 w_k1_MSFT_s015
                                 + 1.03678 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.28111 w_k1_ADM_s015
                                 + 0.747073 w_k1_AXP_s015
                                 + 1.07516 w_k1_DIS_s015
                                 + 1.06142 w_k1_FDX_s015
                                 + 0.836635 w_k1_IBM_s015
                                 + 0.869776 w_k1_INTC_s015
                                 + 0.939986 w_k1_KO_s015
                                 + 0.878799 w_k1_MAT_s015
                                 + 1.22677 w_k1_MSFT_s015
                                 + 1.00481 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.4536 w_k1_ADM_s015
                                 + 1.11631 w_k1_AXP_s015
                                 + 1.47465 w_k1_DIS_s015 + 1.2908 w_k1_FDX_s015
                                 + 1.00016 w_k1_IBM_s015
                                 + 1.1325 w_k1_INTC_s015 + 1.2986 w_k1_KO_s015
                                 + 1.15146 w_k1_MAT_s015
                                 + 1.54909 w_k1_MSFT_s015
                                 + 1.20111 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.874015 w_k1_ADM_s015
                                 + 0.748001 w_k1_AXP_s015
                                 + 0.814324 w_k1_DIS_s015
                                 + 0.860167 w_k1_FDX_s015
                                 + 1.05951 w_k1_IBM_s015
                                 + 0.751851 w_k1_INTC_s015
                                 + 0.915697 w_k1_KO_s015
                                 + 1.16512 w_k1_MAT_s015
                                 + 0.955936 w_k1_MSFT_s015
                                 + 0.899912 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.20171 w_k1_ADM_s015
                                 + 0.953073 w_k1_AXP_s015
                                 + 0.907425 w_k1_DIS_s015
                                 + 1.02199 w_k1_FDX_s015
                                 + 0.976228 w_k1_IBM_s015
                                 + 1.07672 w_k1_INTC_s015
                                 + 1.02106 w_k1_KO_s015
                                 + 0.841579 w_k1_MAT_s015
                                 + 1.13834 w_k1_MSFT_s015
                                 + 1.01609 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.51514 w_k1_ADM_s015
                                 + 0.792756 w_k1_AXP_s015
                                 + 1.03425 w_k1_DIS_s015
                                 + 1.06176 w_k1_FDX_s015
                                 + 0.705001 w_k1_IBM_s015
                                 + 0.915994 w_k1_INTC_s015
                                 + 0.913506 w_k1_KO_s015
                                 + 0.762383 w_k1_MAT_s015
                                 + 1.11402 w_k1_MSFT_s015
                                 + 1.04002 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.22899 w_k1_ADM_s015
                                 + 0.962145 w_k1_AXP_s015
                                 + 1.20727 w_k1_DIS_s015
                                 + 1.12466 w_k1_FDX_s015
                                 + 1.03274 w_k1_IBM_s015
                                 + 1.1052 w_k1_INTC_s015 + 1.05263 w_k1_KO_s015
                                 + 1.03789 w_k1_MAT_s015 + 1.193 w_k1_MSFT_s015
                                 + 1.09914 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.745107 w_k1_ADM_s016
                                 + 0.778524 w_k1_AXP_s016
                                 + 0.715748 w_k1_DIS_s016
                                 + 0.689581 w_k1_FDX_s016
                                 + 0.839228 w_k1_IBM_s016
                                 + 0.668151 w_k1_INTC_s016
                                 + 0.718722 w_k1_KO_s016
                                 + 0.746353 w_k1_MAT_s016
                                 + 0.77018 w_k1_MSFT_s016 + 0.828 w_k1_PG_s016
                                 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.827791 w_k1_ADM_s016
                                 + 0.70663 w_k1_AXP_s016 + 1.0114 w_k1_DIS_s016
                                 + 0.96038 w_k1_FDX_s016
                                 + 0.811203 w_k1_IBM_s016
                                 + 0.926444 w_k1_INTC_s016
                                 + 0.89777 w_k1_KO_s016
                                 + 0.544788 w_k1_MAT_s016
                                 + 0.897347 w_k1_MSFT_s016
                                 + 0.879298 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.0553 w_k1_ADM_s016
                                 + 1.22015 w_k1_AXP_s016
                                 + 1.35766 w_k1_DIS_s016
                                 + 1.03253 w_k1_FDX_s016
                                 + 1.34998 w_k1_IBM_s016
                                 + 1.0252 w_k1_INTC_s016 + 1.19791 w_k1_KO_s016
                                 + 1.11237 w_k1_MAT_s016
                                 + 1.1377 w_k1_MSFT_s016 + 1.07512 w_k1_PG_s016
                                 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.928609 w_k1_ADM_s016
                                 + 0.82262 w_k1_AXP_s016
                                 + 0.995316 w_k1_DIS_s016
                                 + 0.922389 w_k1_FDX_s016
                                 + 1.11458 w_k1_IBM_s016
                                 + 0.869456 w_k1_INTC_s016
                                 + 1.01447 w_k1_KO_s016
                                 + 0.955481 w_k1_MAT_s016
                                 + 0.980285 w_k1_MSFT_s016
                                 + 0.999777 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.695144 w_k1_ADM_s016
                                 + 0.692534 w_k1_AXP_s016
                                 + 0.654233 w_k1_DIS_s016
                                 + 0.831381 w_k1_FDX_s016
                                 + 0.624675 w_k1_IBM_s016
                                 + 0.942061 w_k1_INTC_s016
                                 + 0.648262 w_k1_KO_s016
                                 + 0.48809 w_k1_MAT_s016
                                 + 0.894364 w_k1_MSFT_s016
                                 + 0.713067 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.09995 w_k1_ADM_s016
                                 + 1.10975 w_k1_AXP_s016
                                 + 0.862272 w_k1_DIS_s016
                                 + 1.02757 w_k1_FDX_s016
                                 + 0.90554 w_k1_IBM_s016
                                 + 0.999958 w_k1_INTC_s016
                                 + 0.941054 w_k1_KO_s016
                                 + 0.779041 w_k1_MAT_s016
                                 + 0.954719 w_k1_MSFT_s016
                                 + 0.957989 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.1869 w_k1_ADM_s016
                                 + 1.05555 w_k1_AXP_s016 + 1.0964 w_k1_DIS_s016
                                 + 1.12118 w_k1_FDX_s016
                                 + 0.813498 w_k1_IBM_s016
                                 + 1.03782 w_k1_INTC_s016
                                 + 0.896155 w_k1_KO_s016
                                 + 0.622897 w_k1_MAT_s016
                                 + 1.07042 w_k1_MSFT_s016
                                 + 1.07201 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.24079 w_k1_ADM_s016
                                 + 1.08489 w_k1_AXP_s016
                                 + 1.45495 w_k1_DIS_s016
                                 + 1.22802 w_k1_FDX_s016
                                 + 0.927599 w_k1_IBM_s016
                                 + 1.07829 w_k1_INTC_s016
                                 + 1.05006 w_k1_KO_s016 + 0.90205 w_k1_MAT_s016
                                 + 1.22517 w_k1_MSFT_s016
                                 + 1.01415 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.802176 w_k1_ADM_s017
                                 + 0.736051 w_k1_AXP_s017
                                 + 0.649895 w_k1_DIS_s017
                                 + 0.750344 w_k1_FDX_s017
                                 + 0.878326 w_k1_IBM_s017
                                 + 0.955643 w_k1_INTC_s017
                                 + 0.869974 w_k1_KO_s017
                                 + 0.88363 w_k1_MAT_s017
                                 + 0.730372 w_k1_MSFT_s017
                                 + 0.826792 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.907305 w_k1_ADM_s017
                                 + 0.920624 w_k1_AXP_s017
                                 + 0.836979 w_k1_DIS_s017
                                 + 0.839768 w_k1_FDX_s017
                                 + 1.39981 w_k1_IBM_s017
                                 + 0.997591 w_k1_INTC_s017
                                 + 1.15362 w_k1_KO_s017 + 1.73906 w_k1_MAT_s017
                                 + 0.94204 w_k1_MSFT_s017
                                 + 0.932975 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.66754 w_k1_ADM_s017
                                 + 1.62219 w_k1_AXP_s017
                                 + 1.96031 w_k1_DIS_s017
                                 + 1.45823 w_k1_FDX_s017
                                 + 1.44408 w_k1_IBM_s017
                                 + 1.21246 w_k1_INTC_s017
                                 + 1.59535 w_k1_KO_s017 + 1.58297 w_k1_MAT_s017
                                 + 1.40248 w_k1_MSFT_s017 + 1.4856 w_k1_PG_s017
                                 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.53619 w_k1_ADM_s017
                                 + 1.50424 w_k1_AXP_s017
                                 + 1.38775 w_k1_DIS_s017
                                 + 1.31451 w_k1_FDX_s017
                                 + 1.02973 w_k1_IBM_s017
                                 + 1.38784 w_k1_INTC_s017
                                 + 1.27144 w_k1_KO_s017 + 1.11823 w_k1_MAT_s017
                                 + 1.23909 w_k1_MSFT_s017
                                 + 1.22571 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.11717 w_k1_ADM_s017
                                 + 1.10308 w_k1_AXP_s017
                                 + 1.39951 w_k1_DIS_s017
                                 + 1.08599 w_k1_FDX_s017
                                 + 1.47459 w_k1_IBM_s017
                                 + 1.11222 w_k1_INTC_s017
                                 + 1.28206 w_k1_KO_s017 + 1.27476 w_k1_MAT_s017
                                 + 1.03797 w_k1_MSFT_s017
                                 + 1.05716 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.2019 w_k1_ADM_s017
                                 + 1.72986 w_k1_AXP_s017 + 1.3509 w_k1_DIS_s017
                                 + 1.13862 w_k1_FDX_s017 + 1.6216 w_k1_IBM_s017
                                 + 1.35712 w_k1_INTC_s017
                                 + 1.42515 w_k1_KO_s017 + 2.17091 w_k1_MAT_s017
                                 + 1.13085 w_k1_MSFT_s017 + 1.1455 w_k1_PG_s017
                                 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.27709 w_k1_ADM_s017
                                 + 1.13698 w_k1_AXP_s017
                                 + 0.95755 w_k1_DIS_s017
                                 + 1.11114 w_k1_FDX_s017
                                 + 0.767387 w_k1_IBM_s017
                                 + 1.09881 w_k1_INTC_s017
                                 + 1.02392 w_k1_KO_s017
                                 + 0.938871 w_k1_MAT_s017
                                 + 1.05177 w_k1_MSFT_s017
                                 + 1.06163 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.81917 w_k1_ADM_s017
                                 + 0.826801 w_k1_AXP_s017
                                 + 1.11239 w_k1_DIS_s017 + 0.953 w_k1_FDX_s017
                                 + 0.957715 w_k1_IBM_s017
                                 + 1.13938 w_k1_INTC_s017
                                 + 1.13985 w_k1_KO_s017
                                 + 0.904062 w_k1_MAT_s017
                                 + 1.09091 w_k1_MSFT_s017
                                 + 1.02136 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.31168 w_k1_ADM_s100
                                 + 1.07963 w_k1_AXP_s100
                                 + 1.37637 w_k1_DIS_s100
                                 + 1.33485 w_k1_FDX_s100
                                 + 0.911113 w_k1_IBM_s100
                                 + 1.08808 w_k1_INTC_s100
                                 + 1.18647 w_k1_KO_s100
                                 + 0.882386 w_k1_MAT_s100
                                 + 1.27115 w_k1_MSFT_s100
                                 + 1.06144 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.13114 w_k1_ADM_s100
                                 + 1.10912 w_k1_AXP_s100
                                 + 0.990558 w_k1_DIS_s100
                                 + 1.18073 w_k1_FDX_s100
                                 + 0.825899 w_k1_IBM_s100
                                 + 1.19241 w_k1_INTC_s100
                                 + 0.917668 w_k1_KO_s100
                                 + 1.08406 w_k1_MAT_s100
                                 + 1.19506 w_k1_MSFT_s100
                                 + 1.01199 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.42468 w_k1_ADM_s100
                                 + 1.22629 w_k1_AXP_s100
                                 + 1.24978 w_k1_DIS_s100
                                 + 1.24638 w_k1_FDX_s100
                                 + 1.15375 w_k1_IBM_s100
                                 + 1.40242 w_k1_INTC_s100
                                 + 1.18353 w_k1_KO_s100 + 1.2801 w_k1_MAT_s100
                                 + 1.26989 w_k1_MSFT_s100
                                 + 1.17824 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.10696 w_k1_ADM_s100
                                 + 0.822409 w_k1_AXP_s100
                                 + 0.823906 w_k1_DIS_s100
                                 + 0.887612 w_k1_FDX_s100
                                 + 0.71282 w_k1_IBM_s100
                                 + 0.842065 w_k1_INTC_s100
                                 + 0.782862 w_k1_KO_s100
                                 + 0.78547 w_k1_MAT_s100
                                 + 0.873504 w_k1_MSFT_s100
                                 + 0.864677 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.886046 w_k1_ADM_s100
                                 + 0.780662 w_k1_AXP_s100
                                 + 0.657117 w_k1_DIS_s100
                                 + 0.7334 w_k1_FDX_s100
                                 + 0.952886 w_k1_IBM_s100
                                 + 0.920797 w_k1_INTC_s100
                                 + 0.82128 w_k1_KO_s100
                                 + 0.859412 w_k1_MAT_s100
                                 + 0.853678 w_k1_MSFT_s100
                                 + 0.904461 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.22669 w_k1_ADM_s100
                                 + 1.15378 w_k1_AXP_s100
                                 + 1.87397 w_k1_DIS_s100
                                 + 1.30829 w_k1_FDX_s100
                                 + 1.19311 w_k1_IBM_s100
                                 + 1.41892 w_k1_INTC_s100
                                 + 1.39217 w_k1_KO_s100
                                 + 0.933417 w_k1_MAT_s100
                                 + 1.44045 w_k1_MSFT_s100
                                 + 1.06098 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.2664 w_k1_ADM_s100
                                 + 1.05407 w_k1_AXP_s100
                                 + 1.19583 w_k1_DIS_s100
                                 + 1.05941 w_k1_FDX_s100
                                 + 1.44168 w_k1_IBM_s100
                                 + 1.18859 w_k1_INTC_s100
                                 + 1.18822 w_k1_KO_s100 + 1.10232 w_k1_MAT_s100
                                 + 1.1722 w_k1_MSFT_s100 + 1.12939 w_k1_PG_s100
                                 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.952176 w_k1_ADM_s100
                                 + 1.03653 w_k1_AXP_s100 + 1.0172 w_k1_DIS_s100
                                 + 0.968072 w_k1_FDX_s100
                                 + 1.00136 w_k1_IBM_s100
                                 + 1.04996 w_k1_INTC_s100
                                 + 0.911603 w_k1_KO_s100
                                 + 1.07496 w_k1_MAT_s100
                                 + 0.948123 w_k1_MSFT_s100
                                 + 1.02027 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.910657 w_k1_ADM_s101
                                 + 1.16312 w_k1_AXP_s101
                                 + 0.887393 w_k1_DIS_s101
                                 + 1.03586 w_k1_FDX_s101
                                 + 0.84132 w_k1_IBM_s101
                                 + 0.979728 w_k1_INTC_s101
                                 + 0.840108 w_k1_KO_s101
                                 + 1.30734 w_k1_MAT_s101
                                 + 0.990988 w_k1_MSFT_s101
                                 + 0.947786 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.973859 w_k1_ADM_s101
                                 + 1.47979 w_k1_AXP_s101
                                 + 1.36259 w_k1_DIS_s101
                                 + 1.17521 w_k1_FDX_s101
                                 + 1.52277 w_k1_IBM_s101
                                 + 1.20039 w_k1_INTC_s101
                                 + 1.24132 w_k1_KO_s101 + 1.74508 w_k1_MAT_s101
                                 + 1.26787 w_k1_MSFT_s101
                                 + 1.17171 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.681847 w_k1_ADM_s101
                                 + 0.847537 w_k1_AXP_s101
                                 + 0.580518 w_k1_DIS_s101
                                 + 0.707289 w_k1_FDX_s101
                                 + 1.14973 w_k1_IBM_s101
                                 + 1.03229 w_k1_INTC_s101
                                 + 0.864262 w_k1_KO_s101
                                 + 1.19485 w_k1_MAT_s101
                                 + 0.895601 w_k1_MSFT_s101
                                 + 0.891961 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.947168 w_k1_ADM_s101
                                 + 1.14847 w_k1_AXP_s101 + 1.1189 w_k1_DIS_s101
                                 + 1.03163 w_k1_FDX_s101 + 1.1443 w_k1_IBM_s101
                                 + 0.912098 w_k1_INTC_s101
                                 + 1.01895 w_k1_KO_s101 + 1.31325 w_k1_MAT_s101
                                 + 1.07637 w_k1_MSFT_s101
                                 + 1.12832 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.641815 w_k1_ADM_s101
                                 + 0.564368 w_k1_AXP_s101
                                 + 0.500486 w_k1_DIS_s101
                                 + 0.726624 w_k1_FDX_s101
                                 + 0.632386 w_k1_IBM_s101
                                 + 0.890147 w_k1_INTC_s101
                                 + 0.655598 w_k1_KO_s101
                                 + 0.836172 w_k1_MAT_s101
                                 + 0.831051 w_k1_MSFT_s101
                                 + 0.794385 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.19412 w_k1_ADM_s101
                                 + 1.23258 w_k1_AXP_s101 + 1.1397 w_k1_DIS_s101
                                 + 1.21693 w_k1_FDX_s101
                                 + 0.810827 w_k1_IBM_s101
                                 + 1.08746 w_k1_INTC_s101
                                 + 0.995344 w_k1_KO_s101
                                 + 1.21274 w_k1_MAT_s101
                                 + 1.24599 w_k1_MSFT_s101
                                 + 1.02824 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.26865 w_k1_ADM_s101
                                 + 1.44945 w_k1_AXP_s101
                                 + 1.39513 w_k1_DIS_s101
                                 + 1.42683 w_k1_FDX_s101
                                 + 0.95353 w_k1_IBM_s101
                                 + 1.26398 w_k1_INTC_s101
                                 + 1.29068 w_k1_KO_s101 + 1.46391 w_k1_MAT_s101
                                 + 1.56429 w_k1_MSFT_s101
                                 + 1.18084 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.882671 w_k1_ADM_s101
                                 + 0.835878 w_k1_AXP_s101
                                 + 0.791568 w_k1_DIS_s101
                                 + 0.821822 w_k1_FDX_s101
                                 + 0.788866 w_k1_IBM_s101
                                 + 0.836335 w_k1_INTC_s101
                                 + 0.902038 w_k1_KO_s101
                                 + 1.07802 w_k1_MAT_s101
                                 + 1.01338 w_k1_MSFT_s101
                                 + 0.866755 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.40275 w_k1_ADM_s102
                                 + 1.03908 w_k1_AXP_s102 + 1.4686 w_k1_DIS_s102
                                 + 1.38731 w_k1_FDX_s102
                                 + 1.07394 w_k1_IBM_s102
                                 + 1.31788 w_k1_INTC_s102
                                 + 1.19882 w_k1_KO_s102
                                 + 0.647859 w_k1_MAT_s102
                                 + 1.17384 w_k1_MSFT_s102
                                 + 1.25052 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.967223 w_k1_ADM_s102
                                 + 1.12105 w_k1_AXP_s102
                                 + 0.990749 w_k1_DIS_s102
                                 + 1.26239 w_k1_FDX_s102
                                 + 0.766631 w_k1_IBM_s102
                                 + 1.1264 w_k1_INTC_s102
                                 + 0.840304 w_k1_KO_s102
                                 + 1.02959 w_k1_MAT_s102
                                 + 1.10075 w_k1_MSFT_s102
                                 + 0.976702 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.973258 w_k1_ADM_s102
                                 + 0.842459 w_k1_AXP_s102
                                 + 1.20212 w_k1_DIS_s102
                                 + 1.02771 w_k1_FDX_s102
                                 + 0.917085 w_k1_IBM_s102
                                 + 0.821636 w_k1_INTC_s102
                                 + 0.971672 w_k1_KO_s102
                                 + 0.976888 w_k1_MAT_s102
                                 + 1.04065 w_k1_MSFT_s102
                                 + 1.06695 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.702647 w_k1_ADM_s102
                                 + 1.01061 w_k1_AXP_s102
                                 + 1.40249 w_k1_DIS_s102
                                 + 1.00295 w_k1_FDX_s102
                                 + 1.52843 w_k1_IBM_s102
                                 + 0.969551 w_k1_INTC_s102
                                 + 1.27923 w_k1_KO_s102 + 1.30421 w_k1_MAT_s102
                                 + 1.06494 w_k1_MSFT_s102
                                 + 1.14963 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.32938 w_k1_ADM_s102
                                 + 1.71192 w_k1_AXP_s102
                                 + 1.66204 w_k1_DIS_s102
                                 + 1.38214 w_k1_FDX_s102
                                 + 1.37202 w_k1_IBM_s102
                                 + 1.14116 w_k1_INTC_s102
                                 + 1.59135 w_k1_KO_s102 + 1.77448 w_k1_MAT_s102
                                 + 1.18902 w_k1_MSFT_s102
                                 + 1.31017 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.762141 w_k1_ADM_s102
                                 + 0.607498 w_k1_AXP_s102
                                 + 0.725491 w_k1_DIS_s102
                                 + 0.838003 w_k1_FDX_s102
                                 + 0.663195 w_k1_IBM_s102
                                 + 0.714104 w_k1_INTC_s102
                                 + 0.673628 w_k1_KO_s102
                                 + 0.491671 w_k1_MAT_s102
                                 + 0.68223 w_k1_MSFT_s102
                                 + 0.829872 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.849473 w_k1_ADM_s102
                                 + 0.614382 w_k1_AXP_s102
                                 + 0.726941 w_k1_DIS_s102
                                 + 0.826049 w_k1_FDX_s102
                                 + 0.920991 w_k1_IBM_s102
                                 + 0.82112 w_k1_INTC_s102
                                 + 0.872851 w_k1_KO_s102
                                 + 0.610979 w_k1_MAT_s102
                                 + 0.885122 w_k1_MSFT_s102
                                 + 0.990314 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.897287 w_k1_ADM_s102
                                 + 0.881164 w_k1_AXP_s102
                                 + 0.980824 w_k1_DIS_s102
                                 + 1.00834 w_k1_FDX_s102
                                 + 1.18247 w_k1_IBM_s102
                                 + 1.04299 w_k1_INTC_s102
                                 + 1.18825 w_k1_KO_s102 + 1.17551 w_k1_MAT_s102
                                 + 1.04752 w_k1_MSFT_s102
                                 + 1.11299 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.14265 w_k1_ADM_s103
                                 + 0.850375 w_k1_AXP_s103
                                 + 0.932144 w_k1_DIS_s103
                                 + 0.955753 w_k1_FDX_s103
                                 + 0.617375 w_k1_IBM_s103
                                 + 0.841111 w_k1_INTC_s103
                                 + 0.788891 w_k1_KO_s103
                                 + 0.595535 w_k1_MAT_s103
                                 + 0.906729 w_k1_MSFT_s103
                                 + 0.736001 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.782511 w_k1_ADM_s103
                                 + 0.889975 w_k1_AXP_s103
                                 + 1.13667 w_k1_DIS_s103
                                 + 0.88102 w_k1_FDX_s103
                                 + 0.920475 w_k1_IBM_s103
                                 + 0.944078 w_k1_INTC_s103
                                 + 0.956006 w_k1_KO_s103
                                 + 1.00653 w_k1_MAT_s103
                                 + 1.02101 w_k1_MSFT_s103
                                 + 0.901589 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.00182 w_k1_ADM_s103
                                 + 1.20668 w_k1_AXP_s103
                                 + 1.16411 w_k1_DIS_s103
                                 + 1.06848 w_k1_FDX_s103
                                 + 1.02965 w_k1_IBM_s103
                                 + 0.904147 w_k1_INTC_s103
                                 + 0.914591 w_k1_KO_s103
                                 + 1.34127 w_k1_MAT_s103
                                 + 0.961621 w_k1_MSFT_s103
                                 + 1.00969 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.764007 w_k1_ADM_s103
                                 + 0.690287 w_k1_AXP_s103
                                 + 0.754393 w_k1_DIS_s103
                                 + 0.660141 w_k1_FDX_s103
                                 + 1.08882 w_k1_IBM_s103
                                 + 0.655548 w_k1_INTC_s103
                                 + 0.7816 w_k1_KO_s103 + 0.845249 w_k1_MAT_s103
                                 + 0.730926 w_k1_MSFT_s103
                                 + 0.799531 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.761316 w_k1_ADM_s103
                                 + 0.663494 w_k1_AXP_s103
                                 + 0.640548 w_k1_DIS_s103
                                 + 0.674255 w_k1_FDX_s103
                                 + 0.685934 w_k1_IBM_s103
                                 + 0.646746 w_k1_INTC_s103
                                 + 0.609552 w_k1_KO_s103
                                 + 0.65174 w_k1_MAT_s103
                                 + 0.756373 w_k1_MSFT_s103
                                 + 0.658012 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.20999 w_k1_ADM_s103
                                 + 1.07983 w_k1_AXP_s103
                                 + 1.22965 w_k1_DIS_s103
                                 + 1.04315 w_k1_FDX_s103
                                 + 1.31838 w_k1_IBM_s103
                                 + 0.910271 w_k1_INTC_s103
                                 + 0.955492 w_k1_KO_s103
                                 + 1.32703 w_k1_MAT_s103
                                 + 1.06244 w_k1_MSFT_s103
                                 + 0.940768 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.08189 w_k1_ADM_s103
                                 + 1.60164 w_k1_AXP_s103
                                 + 1.49972 w_k1_DIS_s103
                                 + 1.20627 w_k1_FDX_s103
                                 + 1.70776 w_k1_IBM_s103
                                 + 1.23747 w_k1_INTC_s103
                                 + 1.33391 w_k1_KO_s103 + 1.88191 w_k1_MAT_s103
                                 + 1.58112 w_k1_MSFT_s103
                                 + 1.26714 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.916389 w_k1_ADM_s103
                                 + 0.886381 w_k1_AXP_s103
                                 + 0.897544 w_k1_DIS_s103
                                 + 0.9171 w_k1_FDX_s103
                                 + 0.807134 w_k1_IBM_s103
                                 + 0.834323 w_k1_INTC_s103
                                 + 0.910195 w_k1_KO_s103
                                 + 0.748772 w_k1_MAT_s103
                                 + 0.932053 w_k1_MSFT_s103
                                 + 0.871949 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.24602 w_k1_ADM_s104
                                 + 1.30675 w_k1_AXP_s104
                                 + 1.78122 w_k1_DIS_s104
                                 + 1.38073 w_k1_FDX_s104
                                 + 1.50957 w_k1_IBM_s104 + 1.443 w_k1_INTC_s104
                                 + 1.5372 w_k1_KO_s104 + 1.89203 w_k1_MAT_s104
                                 + 1.57849 w_k1_MSFT_s104
                                 + 1.31904 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.05852 w_k1_ADM_s104
                                 + 0.609718 w_k1_AXP_s104
                                 + 0.748319 w_k1_DIS_s104
                                 + 0.713986 w_k1_FDX_s104
                                 + 1.38263 w_k1_IBM_s104
                                 + 0.952045 w_k1_INTC_s104
                                 + 1.11172 w_k1_KO_s104 + 1.18355 w_k1_MAT_s104
                                 + 0.831343 w_k1_MSFT_s104
                                 + 0.847965 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.20608 w_k1_ADM_s104
                                 + 0.73085 w_k1_AXP_s104
                                 + 0.927888 w_k1_DIS_s104
                                 + 0.963663 w_k1_FDX_s104
                                 + 0.827778 w_k1_IBM_s104
                                 + 1.03508 w_k1_INTC_s104
                                 + 1.07995 w_k1_KO_s104
                                 + 0.941604 w_k1_MAT_s104
                                 + 1.2641 w_k1_MSFT_s104 + 1.12979 w_k1_PG_s104
                                 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.36311 w_k1_ADM_s104
                                 + 1.10338 w_k1_AXP_s104
                                 + 1.21168 w_k1_DIS_s104
                                 + 1.25951 w_k1_FDX_s104
                                 + 1.01292 w_k1_IBM_s104
                                 + 1.31357 w_k1_INTC_s104
                                 + 1.15202 w_k1_KO_s104 + 1.27158 w_k1_MAT_s104
                                 + 1.42126 w_k1_MSFT_s104
                                 + 1.30867 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.31615 w_k1_ADM_s104
                                 + 0.966229 w_k1_AXP_s104
                                 + 1.32796 w_k1_DIS_s104
                                 + 1.06282 w_k1_FDX_s104
                                 + 1.13926 w_k1_IBM_s104
                                 + 1.13573 w_k1_INTC_s104
                                 + 1.60799 w_k1_KO_s104 + 1.62112 w_k1_MAT_s104
                                 + 1.29001 w_k1_MSFT_s104
                                 + 1.34199 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.665013 w_k1_ADM_s104
                                 + 0.689433 w_k1_AXP_s104
                                 + 0.654711 w_k1_DIS_s104
                                 + 0.746374 w_k1_FDX_s104
                                 + 0.824978 w_k1_IBM_s104
                                 + 0.892814 w_k1_INTC_s104
                                 + 0.853734 w_k1_KO_s104
                                 + 1.04915 w_k1_MAT_s104
                                 + 0.92101 w_k1_MSFT_s104
                                 + 0.941874 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.29457 w_k1_ADM_s104
                                 + 0.715468 w_k1_AXP_s104
                                 + 0.791016 w_k1_DIS_s104
                                 + 0.964024 w_k1_FDX_s104
                                 + 0.606714 w_k1_IBM_s104
                                 + 0.993793 w_k1_INTC_s104
                                 + 0.865959 w_k1_KO_s104
                                 + 0.574712 w_k1_MAT_s104
                                 + 1.06791 w_k1_MSFT_s104
                                 + 0.93339 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.12543 w_k1_ADM_s104
                                 + 0.836699 w_k1_AXP_s104
                                 + 0.875495 w_k1_DIS_s104
                                 + 0.857601 w_k1_FDX_s104
                                 + 1.09633 w_k1_IBM_s104
                                 + 0.916135 w_k1_INTC_s104
                                 + 1.10024 w_k1_KO_s104 + 1.03903 w_k1_MAT_s104
                                 + 1.00981 w_k1_MSFT_s104
                                 + 1.09757 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.02916 w_k1_ADM_s105
                                 + 0.95832 w_k1_AXP_s105
                                 + 0.788657 w_k1_DIS_s105
                                 + 0.849893 w_k1_FDX_s105
                                 + 0.646315 w_k1_IBM_s105
                                 + 0.768451 w_k1_INTC_s105
                                 + 0.775433 w_k1_KO_s105
                                 + 0.731565 w_k1_MAT_s105
                                 + 0.831878 w_k1_MSFT_s105
                                 + 0.90128 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.14465 w_k1_ADM_s105
                                 + 0.878241 w_k1_AXP_s105
                                 + 0.716044 w_k1_DIS_s105
                                 + 0.795435 w_k1_FDX_s105
                                 + 0.940094 w_k1_IBM_s105
                                 + 0.79274 w_k1_INTC_s105
                                 + 0.814677 w_k1_KO_s105
                                 + 0.859407 w_k1_MAT_s105
                                 + 0.85608 w_k1_MSFT_s105
                                 + 0.922914 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.762511 w_k1_ADM_s105
                                 + 0.784395 w_k1_AXP_s105
                                 + 0.633331 w_k1_DIS_s105
                                 + 0.674176 w_k1_FDX_s105
                                 + 0.884082 w_k1_IBM_s105
                                 + 0.802585 w_k1_INTC_s105
                                 + 0.727187 w_k1_KO_s105
                                 + 0.97077 w_k1_MAT_s105
                                 + 0.668653 w_k1_MSFT_s105
                                 + 0.814815 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.847131 w_k1_ADM_s105
                                 + 1.36447 w_k1_AXP_s105
                                 + 1.35074 w_k1_DIS_s105
                                 + 1.14778 w_k1_FDX_s105
                                 + 1.41419 w_k1_IBM_s105
                                 + 0.945743 w_k1_INTC_s105
                                 + 1.11608 w_k1_KO_s105 + 1.33237 w_k1_MAT_s105
                                 + 0.950181 w_k1_MSFT_s105
                                 + 0.925796 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.874577 w_k1_ADM_s105
                                 + 0.540384 w_k1_AXP_s105
                                 + 0.49025 w_k1_DIS_s105
                                 + 0.655259 w_k1_FDX_s105
                                 + 0.666075 w_k1_IBM_s105
                                 + 0.648941 w_k1_INTC_s105
                                 + 0.628004 w_k1_KO_s105
                                 + 0.791267 w_k1_MAT_s105
                                 + 0.697028 w_k1_MSFT_s105
                                 + 0.70458 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.33207 w_k1_ADM_s105
                                 + 1.01898 w_k1_AXP_s105
                                 + 0.91257 w_k1_DIS_s105
                                 + 0.929669 w_k1_FDX_s105
                                 + 1.16024 w_k1_IBM_s105
                                 + 0.790869 w_k1_INTC_s105
                                 + 1.00023 w_k1_KO_s105 + 1.26261 w_k1_MAT_s105
                                 + 0.961971 w_k1_MSFT_s105
                                 + 0.896422 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.3532 w_k1_ADM_s105
                                 + 1.39318 w_k1_AXP_s105 + 1.881 w_k1_DIS_s105
                                 + 1.4658 w_k1_FDX_s105 + 1.21368 w_k1_IBM_s105
                                 + 1.53706 w_k1_INTC_s105
                                 + 1.36458 w_k1_KO_s105 + 1.42985 w_k1_MAT_s105
                                 + 1.52036 w_k1_MSFT_s105
                                 + 1.26223 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.21881 w_k1_ADM_s105
                                 + 1.11365 w_k1_AXP_s105
                                 + 1.13868 w_k1_DIS_s105
                                 + 1.16627 w_k1_FDX_s105
                                 + 0.852914 w_k1_IBM_s105
                                 + 1.1141 w_k1_INTC_s105
                                 + 0.848425 w_k1_KO_s105
                                 + 0.680011 w_k1_MAT_s105
                                 + 0.986749 w_k1_MSFT_s105
                                 + 0.889311 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.40865 w_k1_ADM_s106
                                 + 0.938942 w_k1_AXP_s106
                                 + 1.02476 w_k1_DIS_s106 + 1.0768 w_k1_FDX_s106
                                 + 0.756903 w_k1_IBM_s106
                                 + 1.00483 w_k1_INTC_s106
                                 + 0.845147 w_k1_KO_s106
                                 + 0.635178 w_k1_MAT_s106
                                 + 1.09045 w_k1_MSFT_s106
                                 + 0.954107 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.45656 w_k1_ADM_s106
                                 + 1.34635 w_k1_AXP_s106
                                 + 1.65688 w_k1_DIS_s106
                                 + 1.43832 w_k1_FDX_s106
                                 + 1.01223 w_k1_IBM_s106
                                 + 1.45064 w_k1_INTC_s106
                                 + 1.18943 w_k1_KO_s106 + 1.40973 w_k1_MAT_s106
                                 + 1.44446 w_k1_MSFT_s106
                                 + 1.28111 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.67567 w_k1_ADM_s106
                                 + 1.44641 w_k1_AXP_s106
                                 + 1.31876 w_k1_DIS_s106
                                 + 1.21921 w_k1_FDX_s106
                                 + 0.980792 w_k1_IBM_s106
                                 + 0.991924 w_k1_INTC_s106
                                 + 1.22567 w_k1_KO_s106 + 0.89063 w_k1_MAT_s106
                                 + 1.1003 w_k1_MSFT_s106 + 1.04847 w_k1_PG_s106
                                 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.947002 w_k1_ADM_s106
                                 + 0.88616 w_k1_AXP_s106
                                 + 0.997722 w_k1_DIS_s106
                                 + 1.02286 w_k1_FDX_s106
                                 + 1.22013 w_k1_IBM_s106
                                 + 1.25977 w_k1_INTC_s106
                                 + 1.13043 w_k1_KO_s106 + 1.52054 w_k1_MAT_s106
                                 + 1.08134 w_k1_MSFT_s106
                                 + 1.01562 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.0953 w_k1_ADM_s106
                                 + 0.938134 w_k1_AXP_s106
                                 + 0.949543 w_k1_DIS_s106
                                 + 0.916959 w_k1_FDX_s106
                                 + 1.14669 w_k1_IBM_s106
                                 + 1.03066 w_k1_INTC_s106
                                 + 1.06131 w_k1_KO_s106
                                 + 0.809504 w_k1_MAT_s106
                                 + 0.962886 w_k1_MSFT_s106
                                 + 0.964564 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.10241 w_k1_ADM_s106
                                 + 1.35274 w_k1_AXP_s106
                                 + 1.15169 w_k1_DIS_s106
                                 + 1.02995 w_k1_FDX_s106
                                 + 1.58747 w_k1_IBM_s106
                                 + 1.00768 w_k1_INTC_s106
                                 + 1.20761 w_k1_KO_s106 + 1.1738 w_k1_MAT_s106
                                 + 0.985744 w_k1_MSFT_s106
                                 + 0.925275 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.934842 w_k1_ADM_s106
                                 + 0.686841 w_k1_AXP_s106
                                 + 0.695368 w_k1_DIS_s106
                                 + 0.808825 w_k1_FDX_s106
                                 + 0.699664 w_k1_IBM_s106
                                 + 0.829166 w_k1_INTC_s106
                                 + 0.76121 w_k1_KO_s106
                                 + 0.594184 w_k1_MAT_s106
                                 + 0.735582 w_k1_MSFT_s106
                                 + 0.75282 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.27571 w_k1_ADM_s106
                                 + 1.14238 w_k1_AXP_s106
                                 + 1.42127 w_k1_DIS_s106
                                 + 1.15105 w_k1_FDX_s106
                                 + 1.41921 w_k1_IBM_s106
                                 + 1.1089 w_k1_INTC_s106 + 1.27033 w_k1_KO_s106
                                 + 1.25585 w_k1_MAT_s106
                                 + 1.25333 w_k1_MSFT_s106
                                 + 1.09632 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.872605 w_k1_ADM_s107
                                 + 0.779029 w_k1_AXP_s107
                                 + 0.781322 w_k1_DIS_s107
                                 + 0.813293 w_k1_FDX_s107
                                 + 0.991415 w_k1_IBM_s107
                                 + 0.805301 w_k1_INTC_s107
                                 + 0.991833 w_k1_KO_s107
                                 + 0.886169 w_k1_MAT_s107
                                 + 0.814934 w_k1_MSFT_s107
                                 + 0.871995 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.28324 w_k1_ADM_s107
                                 + 1.18742 w_k1_AXP_s107
                                 + 1.06662 w_k1_DIS_s107
                                 + 1.05059 w_k1_FDX_s107
                                 + 1.00873 w_k1_IBM_s107
                                 + 0.78756 w_k1_INTC_s107
                                 + 1.02904 w_k1_KO_s107 + 1.099 w_k1_MAT_s107
                                 + 0.900114 w_k1_MSFT_s107
                                 + 0.97605 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.43696 w_k1_ADM_s107
                                 + 1.23682 w_k1_AXP_s107
                                 + 1.58316 w_k1_DIS_s107
                                 + 1.23208 w_k1_FDX_s107
                                 + 1.38973 w_k1_IBM_s107
                                 + 0.841735 w_k1_INTC_s107
                                 + 1.58602 w_k1_KO_s107 + 1.66617 w_k1_MAT_s107
                                 + 1.27469 w_k1_MSFT_s107
                                 + 1.14853 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.847594 w_k1_ADM_s107
                                 + 0.787166 w_k1_AXP_s107
                                 + 0.8939 w_k1_DIS_s107 + 1.02531 w_k1_FDX_s107
                                 + 0.650706 w_k1_IBM_s107
                                 + 1.08761 w_k1_INTC_s107
                                 + 0.949876 w_k1_KO_s107
                                 + 0.817487 w_k1_MAT_s107
                                 + 1.07637 w_k1_MSFT_s107
                                 + 0.842176 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.08066 w_k1_ADM_s107
                                 + 0.948078 w_k1_AXP_s107
                                 + 1.42253 w_k1_DIS_s107 + 1.2985 w_k1_FDX_s107
                                 + 0.915365 w_k1_IBM_s107
                                 + 0.981934 w_k1_INTC_s107
                                 + 1.21324 w_k1_KO_s107 + 1.31921 w_k1_MAT_s107
                                 + 1.29866 w_k1_MSFT_s107
                                 + 1.06306 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.73104 w_k1_ADM_s107
                                 + 0.686525 w_k1_AXP_s107
                                 + 0.566397 w_k1_DIS_s107
                                 + 0.704206 w_k1_FDX_s107
                                 + 0.696539 w_k1_IBM_s107
                                 + 0.744756 w_k1_INTC_s107
                                 + 0.700099 w_k1_KO_s107
                                 + 0.730307 w_k1_MAT_s107
                                 + 0.769844 w_k1_MSFT_s107
                                 + 0.738841 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.90252 w_k1_ADM_s107
                                 + 0.69219 w_k1_AXP_s107
                                 + 0.86219 w_k1_DIS_s107
                                 + 0.844544 w_k1_FDX_s107
                                 + 0.841032 w_k1_IBM_s107
                                 + 0.722561 w_k1_INTC_s107
                                 + 0.979248 w_k1_KO_s107
                                 + 0.851943 w_k1_MAT_s107
                                 + 0.903 w_k1_MSFT_s107 + 0.839985 w_k1_PG_s107
                                 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.934414 w_k1_ADM_s107
                                 + 0.860546 w_k1_AXP_s107
                                 + 0.843653 w_k1_DIS_s107
                                 + 0.935418 w_k1_FDX_s107
                                 + 1.28551 w_k1_IBM_s107
                                 + 0.872149 w_k1_INTC_s107
                                 + 1.22979 w_k1_KO_s107 + 1.30212 w_k1_MAT_s107
                                 + 0.909641 w_k1_MSFT_s107
                                 + 0.927113 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.06934 w_k1_ADM_s110
                                 + 1.17541 w_k1_AXP_s110
                                 + 0.939553 w_k1_DIS_s110
                                 + 0.962347 w_k1_FDX_s110
                                 + 1.31509 w_k1_IBM_s110
                                 + 1.15862 w_k1_INTC_s110
                                 + 1.27707 w_k1_KO_s110 + 1.44594 w_k1_MAT_s110
                                 + 0.9775 w_k1_MSFT_s110 + 1.06838 w_k1_PG_s110
                                 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.75444 w_k1_ADM_s110
                                 + 1.6161 w_k1_AXP_s110 + 1.84766 w_k1_DIS_s110
                                 + 1.3704 w_k1_FDX_s110 + 1.63786 w_k1_IBM_s110
                                 + 1.62466 w_k1_INTC_s110
                                 + 1.79095 w_k1_KO_s110 + 1.92122 w_k1_MAT_s110
                                 + 1.29084 w_k1_MSFT_s110
                                 + 1.26096 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.2448 w_k1_ADM_s110
                                 + 1.21707 w_k1_AXP_s110
                                 + 1.02831 w_k1_DIS_s110 + 1.2198 w_k1_FDX_s110
                                 + 0.809236 w_k1_IBM_s110
                                 + 1.06723 w_k1_INTC_s110
                                 + 1.03853 w_k1_KO_s110
                                 + 0.922057 w_k1_MAT_s110
                                 + 0.970459 w_k1_MSFT_s110
                                 + 1.07992 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.19725 w_k1_ADM_s110
                                 + 1.14941 w_k1_AXP_s110
                                 + 0.822895 w_k1_DIS_s110
                                 + 0.968562 w_k1_FDX_s110
                                 + 0.898955 w_k1_IBM_s110
                                 + 0.97867 w_k1_INTC_s110
                                 + 0.956569 w_k1_KO_s110
                                 + 1.30355 w_k1_MAT_s110
                                 + 0.916037 w_k1_MSFT_s110
                                 + 1.00282 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.928917 w_k1_ADM_s110
                                 + 0.80655 w_k1_AXP_s110
                                 + 0.695759 w_k1_DIS_s110
                                 + 0.867352 w_k1_FDX_s110
                                 + 0.630454 w_k1_IBM_s110
                                 + 0.98383 w_k1_INTC_s110
                                 + 0.803894 w_k1_KO_s110
                                 + 0.583593 w_k1_MAT_s110
                                 + 0.852992 w_k1_MSFT_s110
                                 + 0.842524 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.00601 w_k1_ADM_s110
                                 + 0.600306 w_k1_AXP_s110
                                 + 0.640525 w_k1_DIS_s110
                                 + 0.676806 w_k1_FDX_s110
                                 + 0.877982 w_k1_IBM_s110
                                 + 0.881514 w_k1_INTC_s110
                                 + 0.890625 w_k1_KO_s110
                                 + 0.802095 w_k1_MAT_s110
                                 + 0.733386 w_k1_MSFT_s110
                                 + 0.824027 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.55289 w_k1_ADM_s110
                                 + 1.61033 w_k1_AXP_s110
                                 + 1.37762 w_k1_DIS_s110
                                 + 1.26998 w_k1_FDX_s110
                                 + 1.31905 w_k1_IBM_s110
                                 + 1.40201 w_k1_INTC_s110
                                 + 1.42044 w_k1_KO_s110 + 1.62409 w_k1_MAT_s110
                                 + 1.14651 w_k1_MSFT_s110
                                 + 1.11712 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.2329 w_k1_ADM_s110
                                 + 0.898145 w_k1_AXP_s110
                                 + 1.14468 w_k1_DIS_s110
                                 + 1.12986 w_k1_FDX_s110
                                 + 0.948492 w_k1_IBM_s110
                                 + 1.32652 w_k1_INTC_s110
                                 + 1.30794 w_k1_KO_s110 + 1.03456 w_k1_MAT_s110
                                 + 1.14025 w_k1_MSFT_s110
                                 + 1.07236 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.709328 w_k1_ADM_s111
                                 + 0.67413 w_k1_AXP_s111
                                 + 0.608946 w_k1_DIS_s111
                                 + 0.776856 w_k1_FDX_s111
                                 + 0.588362 w_k1_IBM_s111
                                 + 0.813562 w_k1_INTC_s111
                                 + 0.721408 w_k1_KO_s111
                                 + 0.695187 w_k1_MAT_s111
                                 + 0.826944 w_k1_MSFT_s111
                                 + 0.77901 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.07444 w_k1_ADM_s111
                                 + 1.21918 w_k1_AXP_s111
                                 + 1.26776 w_k1_DIS_s111
                                 + 1.12321 w_k1_FDX_s111
                                 + 0.927519 w_k1_IBM_s111
                                 + 0.965827 w_k1_INTC_s111
                                 + 1.10885 w_k1_KO_s111 + 1.05061 w_k1_MAT_s111
                                 + 0.99531 w_k1_MSFT_s111
                                 + 1.07555 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.996477 w_k1_ADM_s111
                                 + 1.62137 w_k1_AXP_s111 + 1.4709 w_k1_DIS_s111
                                 + 1.41121 w_k1_FDX_s111
                                 + 1.11602 w_k1_IBM_s111
                                 + 1.29693 w_k1_INTC_s111
                                 + 1.07717 w_k1_KO_s111 + 1.39893 w_k1_MAT_s111
                                 + 1.34531 w_k1_MSFT_s111 + 1.2206 w_k1_PG_s111
                                 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.772366 w_k1_ADM_s111
                                 + 0.962592 w_k1_AXP_s111
                                 + 0.939502 w_k1_DIS_s111
                                 + 0.867096 w_k1_FDX_s111
                                 + 1.37069 w_k1_IBM_s111
                                 + 0.89782 w_k1_INTC_s111
                                 + 1.00694 w_k1_KO_s111 + 1.43142 w_k1_MAT_s111
                                 + 0.826011 w_k1_MSFT_s111
                                 + 0.985225 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.88432 w_k1_ADM_s111
                                 + 0.893283 w_k1_AXP_s111
                                 + 1.10486 w_k1_DIS_s111
                                 + 1.02536 w_k1_FDX_s111
                                 + 0.692313 w_k1_IBM_s111
                                 + 0.813525 w_k1_INTC_s111
                                 + 0.97679 w_k1_KO_s111
                                 + 0.756312 w_k1_MAT_s111
                                 + 0.956981 w_k1_MSFT_s111
                                 + 0.952328 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.942085 w_k1_ADM_s111
                                 + 1.40122 w_k1_AXP_s111
                                 + 1.77594 w_k1_DIS_s111
                                 + 1.27605 w_k1_FDX_s111
                                 + 1.27222 w_k1_IBM_s111
                                 + 1.03081 w_k1_INTC_s111
                                 + 1.49538 w_k1_KO_s111 + 1.97207 w_k1_MAT_s111
                                 + 1.24302 w_k1_MSFT_s111
                                 + 1.30204 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.737268 w_k1_ADM_s111
                                 + 0.708779 w_k1_AXP_s111
                                 + 0.720842 w_k1_DIS_s111
                                 + 0.790001 w_k1_FDX_s111
                                 + 0.869829 w_k1_IBM_s111
                                 + 0.854742 w_k1_INTC_s111
                                 + 0.844625 w_k1_KO_s111
                                 + 0.94417 w_k1_MAT_s111
                                 + 0.863324 w_k1_MSFT_s111
                                 + 0.779828 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.817484 w_k1_ADM_s111
                                 + 1.05677 w_k1_AXP_s111
                                 + 0.845659 w_k1_DIS_s111
                                 + 0.998705 w_k1_FDX_s111
                                 + 0.887319 w_k1_IBM_s111
                                 + 0.915875 w_k1_INTC_s111
                                 + 0.829881 w_k1_KO_s111
                                 + 1.10363 w_k1_MAT_s111
                                 + 0.908835 w_k1_MSFT_s111
                                 + 0.913834 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.928006 w_k1_ADM_s112
                                 + 0.764421 w_k1_AXP_s112
                                 + 1.06914 w_k1_DIS_s112
                                 + 1.01195 w_k1_FDX_s112
                                 + 0.767148 w_k1_IBM_s112
                                 + 1.0563 w_k1_INTC_s112
                                 + 0.968129 w_k1_KO_s112
                                 + 0.660845 w_k1_MAT_s112
                                 + 0.984518 w_k1_MSFT_s112
                                 + 0.887726 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.28246 w_k1_ADM_s112
                                 + 1.48435 w_k1_AXP_s112 + 1.8635 w_k1_DIS_s112
                                 + 1.33939 w_k1_FDX_s112
                                 + 1.44598 w_k1_IBM_s112
                                 + 1.25727 w_k1_INTC_s112
                                 + 1.55028 w_k1_KO_s112 + 1.86767 w_k1_MAT_s112
                                 + 1.43533 w_k1_MSFT_s112
                                 + 1.29526 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.864107 w_k1_ADM_s112
                                 + 0.655266 w_k1_AXP_s112
                                 + 0.600035 w_k1_DIS_s112
                                 + 0.753311 w_k1_FDX_s112
                                 + 0.856614 w_k1_IBM_s112
                                 + 0.743119 w_k1_INTC_s112
                                 + 0.786069 w_k1_KO_s112
                                 + 0.907626 w_k1_MAT_s112
                                 + 0.808892 w_k1_MSFT_s112
                                 + 0.825197 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.930661 w_k1_ADM_s112
                                 + 1.17928 w_k1_AXP_s112
                                 + 0.944633 w_k1_DIS_s112
                                 + 0.903505 w_k1_FDX_s112
                                 + 1.14705 w_k1_IBM_s112
                                 + 0.87817 w_k1_INTC_s112
                                 + 1.10137 w_k1_KO_s112 + 1.5411 w_k1_MAT_s112
                                 + 0.869914 w_k1_MSFT_s112
                                 + 1.00283 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.20843 w_k1_ADM_s112
                                 + 1.04504 w_k1_AXP_s112
                                 + 1.29848 w_k1_DIS_s112
                                 + 1.15586 w_k1_FDX_s112
                                 + 0.887416 w_k1_IBM_s112
                                 + 1.15007 w_k1_INTC_s112
                                 + 1.17213 w_k1_KO_s112 + 1.14024 w_k1_MAT_s112
                                 + 1.1913 w_k1_MSFT_s112 + 1.19224 w_k1_PG_s112
                                 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.08015 w_k1_ADM_s112
                                 + 0.801945 w_k1_AXP_s112
                                 + 0.82663 w_k1_DIS_s112
                                 + 0.957394 w_k1_FDX_s112
                                 + 0.822609 w_k1_IBM_s112
                                 + 0.849279 w_k1_INTC_s112
                                 + 0.889785 w_k1_KO_s112
                                 + 1.01396 w_k1_MAT_s112
                                 + 0.998716 w_k1_MSFT_s112
                                 + 1.0146 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.18844 w_k1_ADM_s112
                                 + 1.29441 w_k1_AXP_s112
                                 + 1.14025 w_k1_DIS_s112
                                 + 1.04961 w_k1_FDX_s112
                                 + 1.43934 w_k1_IBM_s112
                                 + 0.997799 w_k1_INTC_s112
                                 + 1.16938 w_k1_KO_s112 + 1.83727 w_k1_MAT_s112
                                 + 1.06601 w_k1_MSFT_s112
                                 + 1.11369 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.994476 w_k1_ADM_s112
                                 + 0.941852 w_k1_AXP_s112
                                 + 1.15279 w_k1_DIS_s112
                                 + 1.05843 w_k1_FDX_s112 + 1.1968 w_k1_IBM_s112
                                 + 0.978964 w_k1_INTC_s112
                                 + 1.36527 w_k1_KO_s112 + 1.31642 w_k1_MAT_s112
                                 + 1.08599 w_k1_MSFT_s112
                                 + 1.12353 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.17447 w_k1_ADM_s113
                                 + 0.948642 w_k1_AXP_s113
                                 + 1.10258 w_k1_DIS_s113
                                 + 1.22064 w_k1_FDX_s113
                                 + 0.668861 w_k1_IBM_s113
                                 + 1.15216 w_k1_INTC_s113
                                 + 0.841794 w_k1_KO_s113
                                 + 0.617829 w_k1_MAT_s113
                                 + 1.07986 w_k1_MSFT_s113
                                 + 1.06325 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.04336 w_k1_ADM_s113
                                 + 1.0673 w_k1_AXP_s113 + 1.27156 w_k1_DIS_s113
                                 + 0.899244 w_k1_FDX_s113
                                 + 1.59577 w_k1_IBM_s113
                                 + 1.05551 w_k1_INTC_s113
                                 + 1.51509 w_k1_KO_s113 + 1.50254 w_k1_MAT_s113
                                 + 1.04362 w_k1_MSFT_s113
                                 + 1.01227 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.21267 w_k1_ADM_s113
                                 + 1.21877 w_k1_AXP_s113
                                 + 1.37962 w_k1_DIS_s113
                                 + 1.13009 w_k1_FDX_s113
                                 + 1.22034 w_k1_IBM_s113
                                 + 1.19898 w_k1_INTC_s113 + 1.2454 w_k1_KO_s113
                                 + 1.13196 w_k1_MAT_s113
                                 + 1.1052 w_k1_MSFT_s113 + 1.11109 w_k1_PG_s113
                                 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.23742 w_k1_ADM_s113
                                 + 0.896466 w_k1_AXP_s113
                                 + 0.927758 w_k1_DIS_s113
                                 + 0.958332 w_k1_FDX_s113
                                 + 1.02025 w_k1_IBM_s113
                                 + 1.12636 w_k1_INTC_s113
                                 + 1.01216 w_k1_KO_s113
                                 + 0.766514 w_k1_MAT_s113
                                 + 1.02819 w_k1_MSFT_s113
                                 + 1.14961 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.934188 w_k1_ADM_s113
                                 + 0.867854 w_k1_AXP_s113
                                 + 0.732894 w_k1_DIS_s113
                                 + 0.794403 w_k1_FDX_s113
                                 + 1.16482 w_k1_IBM_s113
                                 + 0.836557 w_k1_INTC_s113
                                 + 0.967385 w_k1_KO_s113
                                 + 0.949405 w_k1_MAT_s113
                                 + 0.757467 w_k1_MSFT_s113
                                 + 0.979687 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.47638 w_k1_ADM_s113
                                 + 1.46371 w_k1_AXP_s113
                                 + 1.40775 w_k1_DIS_s113 + 1.4497 w_k1_FDX_s113
                                 + 1.04636 w_k1_IBM_s113
                                 + 1.20101 w_k1_INTC_s113
                                 + 1.11517 w_k1_KO_s113 + 1.09687 w_k1_MAT_s113
                                 + 1.18846 w_k1_MSFT_s113 + 1.3013 w_k1_PG_s113
                                 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.3388 w_k1_ADM_s113
                                 + 1.30405 w_k1_AXP_s113 + 1.0849 w_k1_DIS_s113
                                 + 1.16898 w_k1_FDX_s113
                                 + 1.06059 w_k1_IBM_s113
                                 + 1.10085 w_k1_INTC_s113 + 1.0767 w_k1_KO_s113
                                 + 1.08584 w_k1_MAT_s113
                                 + 1.02117 w_k1_MSFT_s113
                                 + 1.09382 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.877302 w_k1_ADM_s113
                                 + 0.820084 w_k1_AXP_s113
                                 + 0.653484 w_k1_DIS_s113
                                 + 0.833689 w_k1_FDX_s113
                                 + 0.805125 w_k1_IBM_s113
                                 + 0.981337 w_k1_INTC_s113
                                 + 0.745203 w_k1_KO_s113
                                 + 0.737083 w_k1_MAT_s113
                                 + 0.899357 w_k1_MSFT_s113
                                 + 0.893666 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.998976 w_k1_ADM_s114
                                 + 1.07898 w_k1_AXP_s114
                                 + 0.985889 w_k1_DIS_s114
                                 + 1.12012 w_k1_FDX_s114
                                 + 0.587722 w_k1_IBM_s114
                                 + 1.00421 w_k1_INTC_s114
                                 + 0.753899 w_k1_KO_s114
                                 + 0.644513 w_k1_MAT_s114
                                 + 1.0299 w_k1_MSFT_s114
                                 + 0.921149 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.958242 w_k1_ADM_s114
                                 + 1.06378 w_k1_AXP_s114
                                 + 0.944341 w_k1_DIS_s114
                                 + 0.815984 w_k1_FDX_s114
                                 + 1.59188 w_k1_IBM_s114
                                 + 0.894116 w_k1_INTC_s114
                                 + 1.09963 w_k1_KO_s114 + 1.40063 w_k1_MAT_s114
                                 + 0.992884 w_k1_MSFT_s114
                                 + 1.05615 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.572465 w_k1_ADM_s114
                                 + 0.647478 w_k1_AXP_s114
                                 + 0.49913 w_k1_DIS_s114
                                 + 0.553194 w_k1_FDX_s114
                                 + 1.18236 w_k1_IBM_s114
                                 + 0.609828 w_k1_INTC_s114
                                 + 0.782917 w_k1_KO_s114
                                 + 1.18098 w_k1_MAT_s114
                                 + 0.612208 w_k1_MSFT_s114
                                 + 0.738125 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.752988 w_k1_ADM_s114
                                 + 0.871543 w_k1_AXP_s114
                                 + 0.671549 w_k1_DIS_s114
                                 + 0.777838 w_k1_FDX_s114
                                 + 0.760846 w_k1_IBM_s114
                                 + 0.917538 w_k1_INTC_s114
                                 + 0.799973 w_k1_KO_s114
                                 + 0.924366 w_k1_MAT_s114
                                 + 0.815128 w_k1_MSFT_s114
                                 + 0.79775 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.13567 w_k1_ADM_s114
                                 + 1.21513 w_k1_AXP_s114
                                 + 1.36465 w_k1_DIS_s114
                                 + 1.33174 w_k1_FDX_s114
                                 + 0.868375 w_k1_IBM_s114
                                 + 0.986238 w_k1_INTC_s114
                                 + 1.13797 w_k1_KO_s114
                                 + 0.936963 w_k1_MAT_s114
                                 + 1.24743 w_k1_MSFT_s114
                                 + 1.18206 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.33957 w_k1_ADM_s114
                                 + 1.54448 w_k1_AXP_s114
                                 + 1.57099 w_k1_DIS_s114 + 1.2359 w_k1_FDX_s114
                                 + 1.40322 w_k1_IBM_s114
                                 + 1.16765 w_k1_INTC_s114
                                 + 1.21462 w_k1_KO_s114 + 1.43943 w_k1_MAT_s114
                                 + 1.26258 w_k1_MSFT_s114
                                 + 1.21026 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.0214 w_k1_ADM_s114
                                 + 1.25514 w_k1_AXP_s114
                                 + 1.33432 w_k1_DIS_s114
                                 + 0.990625 w_k1_FDX_s114
                                 + 1.41696 w_k1_IBM_s114
                                 + 0.792089 w_k1_INTC_s114
                                 + 1.19223 w_k1_KO_s114 + 1.58018 w_k1_MAT_s114
                                 + 0.983459 w_k1_MSFT_s114
                                 + 1.17489 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.872248 w_k1_ADM_s114
                                 + 0.97982 w_k1_AXP_s114
                                 + 0.880052 w_k1_DIS_s114
                                 + 0.918485 w_k1_FDX_s114
                                 + 0.964024 w_k1_IBM_s114
                                 + 0.882444 w_k1_INTC_s114
                                 + 0.906128 w_k1_KO_s114
                                 + 0.965783 w_k1_MAT_s114
                                 + 0.88216 w_k1_MSFT_s114
                                 + 0.932176 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.34524 w_k1_ADM_s115
                                 + 1.34354 w_k1_AXP_s115
                                 + 1.70703 w_k1_DIS_s115 + 1.4118 w_k1_FDX_s115
                                 + 0.999884 w_k1_IBM_s115
                                 + 1.43438 w_k1_INTC_s115
                                 + 1.33317 w_k1_KO_s115 + 1.297 w_k1_MAT_s115
                                 + 1.48621 w_k1_MSFT_s115
                                 + 1.22965 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.743269 w_k1_ADM_s115
                                 + 1.08406 w_k1_AXP_s115
                                 + 1.25554 w_k1_DIS_s115
                                 + 1.07896 w_k1_FDX_s115
                                 + 1.26875 w_k1_IBM_s115
                                 + 1.33783 w_k1_INTC_s115
                                 + 1.21555 w_k1_KO_s115 + 1.04488 w_k1_MAT_s115
                                 + 1.06123 w_k1_MSFT_s115
                                 + 1.01557 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.872941 w_k1_ADM_s115
                                 + 0.909616 w_k1_AXP_s115
                                 + 0.834301 w_k1_DIS_s115
                                 + 0.841588 w_k1_FDX_s115
                                 + 1.02103 w_k1_IBM_s115
                                 + 1.06429 w_k1_INTC_s115
                                 + 1.11816 w_k1_KO_s115 + 1.10401 w_k1_MAT_s115
                                 + 0.935241 w_k1_MSFT_s115
                                 + 1.02482 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.0422 w_k1_ADM_s115
                                 + 0.829073 w_k1_AXP_s115
                                 + 0.836204 w_k1_DIS_s115
                                 + 0.935267 w_k1_FDX_s115
                                 + 0.718587 w_k1_IBM_s115
                                 + 0.987677 w_k1_INTC_s115
                                 + 0.806523 w_k1_KO_s115
                                 + 0.694717 w_k1_MAT_s115
                                 + 0.878977 w_k1_MSFT_s115
                                 + 0.919602 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.959413 w_k1_ADM_s115
                                 + 1.3058 w_k1_AXP_s115 + 1.80194 w_k1_DIS_s115
                                 + 1.21756 w_k1_FDX_s115 + 1.3389 w_k1_IBM_s115
                                 + 1.59193 w_k1_INTC_s115
                                 + 1.58066 w_k1_KO_s115 + 1.67367 w_k1_MAT_s115
                                 + 1.42193 w_k1_MSFT_s115
                                 + 1.29237 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.985184 w_k1_ADM_s115
                                 + 1.01508 w_k1_AXP_s115
                                 + 1.13145 w_k1_DIS_s115
                                 + 0.880223 w_k1_FDX_s115
                                 + 1.46695 w_k1_IBM_s115
                                 + 1.26473 w_k1_INTC_s115 + 1.325 w_k1_KO_s115
                                 + 1.45455 w_k1_MAT_s115
                                 + 1.03268 w_k1_MSFT_s115
                                 + 1.05013 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.38882 w_k1_ADM_s115
                                 + 1.26005 w_k1_AXP_s115
                                 + 1.23156 w_k1_DIS_s115
                                 + 1.13162 w_k1_FDX_s115
                                 + 1.00822 w_k1_IBM_s115
                                 + 1.12452 w_k1_INTC_s115
                                 + 1.08817 w_k1_KO_s115 + 1.20655 w_k1_MAT_s115
                                 + 1.19332 w_k1_MSFT_s115
                                 + 1.07538 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.923867 w_k1_ADM_s115
                                 + 0.693967 w_k1_AXP_s115
                                 + 0.885548 w_k1_DIS_s115
                                 + 0.823633 w_k1_FDX_s115
                                 + 0.914764 w_k1_IBM_s115
                                 + 1.00558 w_k1_INTC_s115
                                 + 0.911576 w_k1_KO_s115
                                 + 0.764981 w_k1_MAT_s115
                                 + 0.870594 w_k1_MSFT_s115
                                 + 0.826883 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.87024 w_k1_ADM_s116
                                 + 1.55532 w_k1_AXP_s116 + 1.2068 w_k1_DIS_s116
                                 + 1.43582 w_k1_FDX_s116
                                 + 0.982928 w_k1_IBM_s116
                                 + 1.44462 w_k1_INTC_s116
                                 + 1.12066 w_k1_KO_s116 + 1.14063 w_k1_MAT_s116
                                 + 1.20417 w_k1_MSFT_s116
                                 + 1.27439 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.988965 w_k1_ADM_s116
                                 + 1.04821 w_k1_AXP_s116
                                 + 1.05761 w_k1_DIS_s116
                                 + 0.958403 w_k1_FDX_s116
                                 + 1.46379 w_k1_IBM_s116
                                 + 0.984296 w_k1_INTC_s116
                                 + 1.25311 w_k1_KO_s116 + 1.39505 w_k1_MAT_s116
                                 + 1.01298 w_k1_MSFT_s116
                                 + 1.06609 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.771551 w_k1_ADM_s116
                                 + 0.782029 w_k1_AXP_s116
                                 + 0.679977 w_k1_DIS_s116
                                 + 0.813549 w_k1_FDX_s116
                                 + 0.974516 w_k1_IBM_s116
                                 + 0.953284 w_k1_INTC_s116
                                 + 0.852317 w_k1_KO_s116
                                 + 0.968687 w_k1_MAT_s116
                                 + 0.933659 w_k1_MSFT_s116
                                 + 0.882523 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.28021 w_k1_ADM_s116
                                 + 1.2855 w_k1_AXP_s116 + 1.56295 w_k1_DIS_s116
                                 + 1.30659 w_k1_FDX_s116
                                 + 1.42145 w_k1_IBM_s116
                                 + 1.23167 w_k1_INTC_s116
                                 + 1.59478 w_k1_KO_s116 + 1.72777 w_k1_MAT_s116
                                 + 1.45961 w_k1_MSFT_s116
                                 + 1.21369 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.10801 w_k1_ADM_s116
                                 + 0.832162 w_k1_AXP_s116
                                 + 1.17045 w_k1_DIS_s116
                                 + 1.11641 w_k1_FDX_s116
                                 + 0.973975 w_k1_IBM_s116
                                 + 1.14206 w_k1_INTC_s116
                                 + 1.07631 w_k1_KO_s116
                                 + 0.659257 w_k1_MAT_s116
                                 + 1.21837 w_k1_MSFT_s116
                                 + 1.02384 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.725628 w_k1_ADM_s116
                                 + 0.60143 w_k1_AXP_s116
                                 + 0.717446 w_k1_DIS_s116
                                 + 0.715642 w_k1_FDX_s116
                                 + 1.02162 w_k1_IBM_s116
                                 + 1.00194 w_k1_INTC_s116
                                 + 0.754594 w_k1_KO_s116
                                 + 0.634538 w_k1_MAT_s116
                                 + 0.914265 w_k1_MSFT_s116
                                 + 0.780198 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.879738 w_k1_ADM_s116
                                 + 0.767774 w_k1_AXP_s116
                                 + 0.778089 w_k1_DIS_s116
                                 + 0.97653 w_k1_FDX_s116
                                 + 0.62082 w_k1_IBM_s116
                                 + 0.834484 w_k1_INTC_s116
                                 + 0.774463 w_k1_KO_s116
                                 + 0.53767 w_k1_MAT_s116
                                 + 0.906889 w_k1_MSFT_s116
                                 + 0.923948 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.31071 w_k1_ADM_s116
                                 + 1.10873 w_k1_AXP_s116
                                 + 0.760318 w_k1_DIS_s116
                                 + 1.06948 w_k1_FDX_s116
                                 + 0.765486 w_k1_IBM_s116
                                 + 1.08261 w_k1_INTC_s116
                                 + 0.875135 w_k1_KO_s116
                                 + 0.749291 w_k1_MAT_s116
                                 + 1.02458 w_k1_MSFT_s116
                                 + 0.931914 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.829565 w_k1_ADM_s117
                                 + 0.89574 w_k1_AXP_s117
                                 + 1.02251 w_k1_DIS_s117
                                 + 0.881109 w_k1_FDX_s117
                                 + 1.48234 w_k1_IBM_s117
                                 + 0.942024 w_k1_INTC_s117
                                 + 1.17487 w_k1_KO_s117 + 1.19012 w_k1_MAT_s117
                                 + 0.864448 w_k1_MSFT_s117
                                 + 0.984536 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.07896 w_k1_ADM_s117
                                 + 0.984688 w_k1_AXP_s117
                                 + 0.917208 w_k1_DIS_s117
                                 + 1.03392 w_k1_FDX_s117
                                 + 0.774996 w_k1_IBM_s117
                                 + 0.856128 w_k1_INTC_s117
                                 + 0.793552 w_k1_KO_s117
                                 + 0.59216 w_k1_MAT_s117
                                 + 0.864312 w_k1_MSFT_s117
                                 + 0.991874 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.901191 w_k1_ADM_s117
                                 + 0.875043 w_k1_AXP_s117
                                 + 1.33531 w_k1_DIS_s117
                                 + 1.13294 w_k1_FDX_s117
                                 + 1.03159 w_k1_IBM_s117
                                 + 1.35896 w_k1_INTC_s117 + 1.014 w_k1_KO_s117
                                 + 0.693698 w_k1_MAT_s117
                                 + 1.272 w_k1_MSFT_s117 + 1.05326 w_k1_PG_s117
                                 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.04394 w_k1_ADM_s117
                                 + 0.601712 w_k1_AXP_s117
                                 + 0.73335 w_k1_DIS_s117
                                 + 0.766945 w_k1_FDX_s117
                                 + 0.822883 w_k1_IBM_s117
                                 + 0.946791 w_k1_INTC_s117
                                 + 0.780569 w_k1_KO_s117
                                 + 0.706149 w_k1_MAT_s117
                                 + 0.917966 w_k1_MSFT_s117
                                 + 0.867558 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.15906 w_k1_ADM_s117
                                 + 0.893046 w_k1_AXP_s117
                                 + 1.29491 w_k1_DIS_s117
                                 + 1.22442 w_k1_FDX_s117
                                 + 0.659545 w_k1_IBM_s117
                                 + 1.44079 w_k1_INTC_s117
                                 + 0.903703 w_k1_KO_s117
                                 + 0.553323 w_k1_MAT_s117
                                 + 1.31277 w_k1_MSFT_s117
                                 + 1.01303 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.00852 w_k1_ADM_s117
                                 + 1.14643 w_k1_AXP_s117
                                 + 1.37621 w_k1_DIS_s117 + 1.1464 w_k1_FDX_s117
                                 + 1.44583 w_k1_IBM_s117
                                 + 1.26585 w_k1_INTC_s117
                                 + 1.19592 w_k1_KO_s117 + 1.17525 w_k1_MAT_s117
                                 + 1.18926 w_k1_MSFT_s117 + 1.2063 w_k1_PG_s117
                                 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.801859 w_k1_ADM_s117
                                 + 0.880554 w_k1_AXP_s117
                                 + 0.962541 w_k1_DIS_s117
                                 + 0.910077 w_k1_FDX_s117
                                 + 0.971195 w_k1_IBM_s117
                                 + 1.1588 w_k1_INTC_s117
                                 + 0.890507 w_k1_KO_s117
                                 + 0.804203 w_k1_MAT_s117
                                 + 1.06955 w_k1_MSFT_s117
                                 + 0.963867 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.27602 w_k1_ADM_s117
                                 + 1.46141 w_k1_AXP_s117
                                 + 1.44106 w_k1_DIS_s117
                                 + 1.35342 w_k1_FDX_s117
                                 + 1.10326 w_k1_IBM_s117
                                 + 1.26352 w_k1_INTC_s117
                                 + 1.14725 w_k1_KO_s117 + 1.03546 w_k1_MAT_s117
                                 + 1.06869 w_k1_MSFT_s117
                                 + 1.24759 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0155763829299667 w_k1_ADM_s000
                                 + 0.0162803491869606 w_k1_ADM_s001
                                 + 0.0217140016261296 w_k1_ADM_s002
                                 + 0.0177221940143524 w_k1_ADM_s003
                                 + 0.0110690123933593 w_k1_ADM_s004
                                 + 0.0199755779413447 w_k1_ADM_s005
                                 + 0.0620409101505219 w_k1_ADM_s006
                                 + 0.0424985484376595 w_k1_ADM_s007
                                 + 0.0649101347844375 w_k1_ADM_s010
                                 + 0.0236031666961715 w_k1_ADM_s011
                                 + 0.0265148894452473 w_k1_ADM_s012
                                 + 0.0479468677738301 w_k1_ADM_s013
                                 + 0.0579717921390695 w_k1_ADM_s014
                                 + 0.023077713783451 w_k1_ADM_s015
                                 + 0.015340876982589 w_k1_ADM_s016
                                 + 0.0458029998145619 w_k1_ADM_s017
                                 + 0.0127874003391546 w_k1_ADM_s100
                                 + 0.018005786228517 w_k1_ADM_s101
                                 + 0.00869564486035892 w_k1_ADM_s102
                                 + 0.0145186219113397 w_k1_ADM_s103
                                 + 0.00605603779376709 w_k1_ADM_s104
                                 + 0.0106422465287432 w_k1_ADM_s105
                                 + 0.0149402743550851 w_k1_ADM_s106
                                 + 0.00928899917462793 w_k1_ADM_s107
                                 + 0.0995684131994203 w_k1_ADM_s110
                                 + 0.0524627099357766 w_k1_ADM_s111
                                 + 0.07967023878722 w_k1_ADM_s112
                                 + 0.0357164872593808 w_k1_ADM_s113
                                 + 0.0590634674135313 w_k1_ADM_s114
                                 + 0.0224597809568603 w_k1_ADM_s115
                                 + 0.028604858331567 w_k1_ADM_s116
                                 + 0.0665768085950301 w_k1_ADM_s117
                                 + 0.0176577557810959 w_k1_AXP_s000
                                 + 0.0152851209031426 w_k1_AXP_s001
                                 + 0.0230081981077428 w_k1_AXP_s002
                                 + 0.0165610140222347 w_k1_AXP_s003
                                 + 0.011494630636887 w_k1_AXP_s004
                                 + 0.0200968108632684 w_k1_AXP_s005
                                 + 0.0621051533374456 w_k1_AXP_s006
                                 + 0.0435329578420754 w_k1_AXP_s007
                                 + 0.0643901665910005 w_k1_AXP_s010
                                 + 0.0217663862066554 w_k1_AXP_s011
                                 + 0.0268174093288143 w_k1_AXP_s012
                                 + 0.044695973265322 w_k1_AXP_s013
                                 + 0.0496499257637779 w_k1_AXP_s014
                                 + 0.0170085149066267 w_k1_AXP_s015
                                 + 0.0149438115045289 w_k1_AXP_s016
                                 + 0.0449711722600883 w_k1_AXP_s017
                                 + 0.0113465011968816 w_k1_AXP_s100
                                 + 0.0211894898234016 w_k1_AXP_s101
                                 + 0.00834449439295302 w_k1_AXP_s102
                                 + 0.0142286901675181 w_k1_AXP_s103
                                 + 0.00443315854015468 w_k1_AXP_s104
                                 + 0.00983138880211726 w_k1_AXP_s105
                                 + 0.0127029002520539 w_k1_AXP_s106
                                 + 0.0081691815225935 w_k1_AXP_s107
                                 + 0.0912917903361102 w_k1_AXP_s110
                                 + 0.0643243171218972 w_k1_AXP_s111
                                 + 0.0718098960109489 w_k1_AXP_s112
                                 + 0.0329815351294165 w_k1_AXP_s113
                                 + 0.0663443239960761 w_k1_AXP_s114
                                 + 0.0215592728492876 w_k1_AXP_s115
                                 + 0.0265624054190681 w_k1_AXP_s116
                                 + 0.0646760128432677 w_k1_AXP_s117
                                 + 0.0174422324636743 w_k1_DIS_s000
                                 + 0.0148703911665157 w_k1_DIS_s001
                                 + 0.0209675654961189 w_k1_DIS_s002
                                 + 0.0161667893078023 w_k1_DIS_s003
                                 + 0.0121219067560373 w_k1_DIS_s004
                                 + 0.0204159081991463 w_k1_DIS_s005
                                 + 0.0611768319610769 w_k1_DIS_s006
                                 + 0.0454693784027152 w_k1_DIS_s007
                                 + 0.0612720764824657 w_k1_DIS_s010
                                 + 0.0225194504581835 w_k1_DIS_s011
                                 + 0.0272541680379739 w_k1_DIS_s012
                                 + 0.0437806744640173 w_k1_DIS_s013
                                 + 0.0535292634076646 w_k1_DIS_s014
                                 + 0.0189392213967739 w_k1_DIS_s015
                                 + 0.016891243636786 w_k1_DIS_s016
                                 + 0.0458121861163277 w_k1_DIS_s017
                                 + 0.0118841125753084 w_k1_DIS_s100
                                 + 0.0192871244737611 w_k1_DIS_s101
                                 + 0.00967832041006283 w_k1_DIS_s102
                                 + 0.0151958096979018 w_k1_DIS_s103
                                 + 0.00537156718391663 w_k1_DIS_s104
                                 + 0.00941489987709299 w_k1_DIS_s105
                                 + 0.0133332249045935 w_k1_DIS_s106
                                 + 0.00947588376862004 w_k1_DIS_s107
                                 + 0.0836794531688136 w_k1_DIS_s110
                                 + 0.0667365564815163 w_k1_DIS_s111
                                 + 0.0792961539548814 w_k1_DIS_s112
                                 + 0.0320248242960043 w_k1_DIS_s113
                                 + 0.0626804432570865 w_k1_DIS_s114
                                 + 0.0234374761327018 w_k1_DIS_s115
                                 + 0.0276722867052678 w_k1_DIS_s116
                                 + 0.0737964593774993 w_k1_DIS_s117
                                 + 0.0162958476825635 w_k1_FDX_s000
                                 + 0.01533564486885 w_k1_FDX_s001
                                 + 0.0202424827515527 w_k1_FDX_s002
                                 + 0.0171538384710859 w_k1_FDX_s003
                                 + 0.0121729391369152 w_k1_FDX_s004
                                 + 0.0213796693149836 w_k1_FDX_s005
                                 + 0.0598493156708207 w_k1_FDX_s006
                                 + 0.0457887173038264 w_k1_FDX_s007
                                 + 0.0595348870397208 w_k1_FDX_s010
                                 + 0.0217348156081527 w_k1_FDX_s011
                                 + 0.0254753580254641 w_k1_FDX_s012
                                 + 0.0452697831326376 w_k1_FDX_s013
                                 + 0.05224340631181 w_k1_FDX_s014
                                 + 0.0183892574225169 w_k1_FDX_s015
                                 + 0.0152665241876273 w_k1_FDX_s016
                                 + 0.0432351329278165 w_k1_FDX_s017
                                 + 0.0115875281587683 w_k1_FDX_s100
                                 + 0.0194229732483859 w_k1_FDX_s101
                                 + 0.0095863625234227 w_k1_FDX_s102
                                 + 0.0137528814773736 w_k1_FDX_s103
                                 + 0.00514583255207941 w_k1_FDX_s104
                                 + 0.0092838550087735 w_k1_FDX_s105
                                 + 0.0128518664926449 w_k1_FDX_s106
                                 + 0.0090681372385477 w_k1_FDX_s107
                                 + 0.0875342908248225 w_k1_FDX_s110
                                 + 0.0616495570500956 w_k1_FDX_s111
                                 + 0.0769536528176071 w_k1_FDX_s112
                                 + 0.0326932011962656 w_k1_FDX_s113
                                 + 0.0616952713389091 w_k1_FDX_s114
                                 + 0.0210876072180619 w_k1_FDX_s115
                                 + 0.0279204011042161 w_k1_FDX_s116
                                 + 0.0692291965661507 w_k1_FDX_s117
                                 + 0.0169023927856493 w_k1_IBM_s000
                                 + 0.0151499986003404 w_k1_IBM_s001
                                 + 0.0212473607054772 w_k1_IBM_s002
                                 + 0.0170230215199091 w_k1_IBM_s003
                                 + 0.0126196038870855 w_k1_IBM_s004
                                 + 0.0214396524628542 w_k1_IBM_s005
                                 + 0.0643497491961218 w_k1_IBM_s006
                                 + 0.0450081047592239 w_k1_IBM_s007
                                 + 0.0617353131884818 w_k1_IBM_s010
                                 + 0.0208127140598403 w_k1_IBM_s011
                                 + 0.0256711260753983 w_k1_IBM_s012
                                 + 0.0443716079450934 w_k1_IBM_s013
                                 + 0.0537868012981117 w_k1_IBM_s014
                                 + 0.0195939720233829 w_k1_IBM_s015
                                 + 0.0156015487322428 w_k1_IBM_s016
                                 + 0.0449502943398174 w_k1_IBM_s017
                                 + 0.0117410818941729 w_k1_IBM_s100
                                 + 0.0196681025548777 w_k1_IBM_s101
                                 + 0.00954316776031216 w_k1_IBM_s102
                                 + 0.0146507795289869 w_k1_IBM_s103
                                 + 0.00505146002119812 w_k1_IBM_s104
                                 + 0.00936448066813293 w_k1_IBM_s105
                                 + 0.01284370157793 w_k1_IBM_s106
                                 + 0.00908259982063371 w_k1_IBM_s107
                                 + 0.0824537357618901 w_k1_IBM_s110
                                 + 0.0640419723825112 w_k1_IBM_s111
                                 + 0.077437684223584 w_k1_IBM_s112
                                 + 0.0328746063321605 w_k1_IBM_s113
                                 + 0.0620014013077093 w_k1_IBM_s114
                                 + 0.0212298438722121 w_k1_IBM_s115
                                 + 0.0297246104884056 w_k1_IBM_s116
                                 + 0.0684618219840654 w_k1_IBM_s117
                                 + 0.0139925849086609 w_k1_INTC_s000
                                 + 0.0146886557165656 w_k1_INTC_s001
                                 + 0.020109239567759 w_k1_INTC_s002
                                 + 0.0149761665145242 w_k1_INTC_s003
                                 + 0.0115089992666557 w_k1_INTC_s004
                                 + 0.0241061968089185 w_k1_INTC_s005
                                 + 0.0517454909271466 w_k1_INTC_s006
                                 + 0.0425552593796553 w_k1_INTC_s007
                                 + 0.0622024267173028 w_k1_INTC_s010
                                 + 0.0192031279087841 w_k1_INTC_s011
                                 + 0.028957881526144 w_k1_INTC_s012
                                 + 0.0398404022314837 w_k1_INTC_s013
                                 + 0.0461699935966614 w_k1_INTC_s014
                                 + 0.0169305779889696 w_k1_INTC_s015
                                 + 0.0146159522502854 w_k1_INTC_s016
                                 + 0.0473259808324815 w_k1_INTC_s017
                                 + 0.0123990898030326 w_k1_INTC_s100
                                 + 0.0192628668119346 w_k1_INTC_s101
                                 + 0.00894417597484168 w_k1_INTC_s102
                                 + 0.0125811094891089 w_k1_INTC_s103
                                 + 0.00552273259548625 w_k1_INTC_s104
                                 + 0.008687101451003 w_k1_INTC_s105
                                 + 0.0130680903460285 w_k1_INTC_s106
                                 + 0.00770198160709666 w_k1_INTC_s107
                                 + 0.0963324536836212 w_k1_INTC_s110
                                 + 0.0575866176031372 w_k1_INTC_s111
                                 + 0.0739663717867046 w_k1_INTC_s112
                                 + 0.0335497347416095 w_k1_INTC_s113
                                 + 0.0578413138414752 w_k1_INTC_s114
                                 + 0.0239864540305909 w_k1_INTC_s115
                                 + 0.028479739835885 w_k1_INTC_s116
                                 + 0.0753255517273363 w_k1_INTC_s117
                                 + 0.0166497446304715 w_k1_KO_s000
                                 + 0.0154296489320906 w_k1_KO_s001
                                 + 0.0204429088741795 w_k1_KO_s002
                                 + 0.0163949226950079 w_k1_KO_s003
                                 + 0.0116840232524106 w_k1_KO_s004
                                 + 0.0230635287068576 w_k1_KO_s005
                                 + 0.0552047640263696 w_k1_KO_s006
                                 + 0.0478157286135653 w_k1_KO_s007
                                 + 0.05937186302885 w_k1_KO_s010
                                 + 0.021328561757548 w_k1_KO_s011
                                 + 0.0266775700862422 w_k1_KO_s012
                                 + 0.0403284156818027 w_k1_KO_s013
                                 + 0.0521076203480823 w_k1_KO_s014
                                 + 0.0190109690864738 w_k1_KO_s015
                                 + 0.0151463238825405 w_k1_KO_s016
                                 + 0.0479217306302342 w_k1_KO_s017
                                 + 0.0113364221405104 w_k1_KO_s100
                                 + 0.0190861804615948 w_k1_KO_s101
                                 + 0.00965594882943565 w_k1_KO_s102
                                 + 0.0130756131172852 w_k1_KO_s103
                                 + 0.00592317985672204 w_k1_KO_s104
                                 + 0.00866519468916166 w_k1_KO_s105
                                 + 0.0128518326188814 w_k1_KO_s106
                                 + 0.0102566213561441 w_k1_KO_s107
                                 + 0.0947944199401899 w_k1_KO_s110
                                 + 0.0631535592803956 w_k1_KO_s111
                                 + 0.0832162822056606 w_k1_KO_s112
                                 + 0.0323624050650165 w_k1_KO_s113
                                 + 0.0602599242946864 w_k1_KO_s114
                                 + 0.0227387556416574 w_k1_KO_s115
                                 + 0.0295683317164449 w_k1_KO_s116
                                 + 0.0646272787979704 w_k1_KO_s117
                                 + 0.016071952759889 w_k1_MAT_s000
                                 + 0.0224205186213045 w_k1_MAT_s001
                                 + 0.0240880868478387 w_k1_MAT_s002
                                 + 0.0187752958216817 w_k1_MAT_s003
                                 + 0.0133402089076582 w_k1_MAT_s004
                                 + 0.0203699301687071 w_k1_MAT_s005
                                 + 0.0597580014134371 w_k1_MAT_s006
                                 + 0.0511202384649217 w_k1_MAT_s007
                                 + 0.0635833170990483 w_k1_MAT_s010
                                 + 0.0191250396894797 w_k1_MAT_s011
                                 + 0.0268271519053708 w_k1_MAT_s012
                                 + 0.0507402897541116 w_k1_MAT_s013
                                 + 0.0612432747447687 w_k1_MAT_s014
                                 + 0.0213577750872486 w_k1_MAT_s015
                                 + 0.0131243043115234 w_k1_MAT_s016
                                 + 0.0481876837573505 w_k1_MAT_s017
                                 + 0.0113694401018816 w_k1_MAT_s100
                                 + 0.0248351922804818 w_k1_MAT_s101
                                 + 0.00891493200846848 w_k1_MAT_s102
                                 + 0.0150804797964081 w_k1_MAT_s103
                                 + 0.00590754699260254 w_k1_MAT_s104
                                 + 0.00938297426077142 w_k1_MAT_s105
                                 + 0.0117603914488834 w_k1_MAT_s106
                                 + 0.0102412582454975 w_k1_MAT_s107
                                 + 0.0922922845923607 w_k1_MAT_s110
                                 + 0.0759987479031435 w_k1_MAT_s111
                                 + 0.0900685515304496 w_k1_MAT_s112
                                 + 0.0302752797310498 w_k1_MAT_s113
                                 + 0.0654148362910495 w_k1_MAT_s114
                                 + 0.0226949081379212 w_k1_MAT_s115
                                 + 0.0287320700400942 w_k1_MAT_s116
                                 + 0.0559983613279806 w_k1_MAT_s117
                                 + 0.01586298275294 w_k1_MSFT_s000
                                 + 0.0144472193093463 w_k1_MSFT_s001
                                 + 0.019011669555301 w_k1_MSFT_s002
                                 + 0.0160752822453587 w_k1_MSFT_s003
                                 + 0.0117213615899816 w_k1_MSFT_s004
                                 + 0.0216875933709417 w_k1_MSFT_s005
                                 + 0.0567364763459893 w_k1_MSFT_s006
                                 + 0.046079746482799 w_k1_MSFT_s007
                                 + 0.0623213711700867 w_k1_MSFT_s010
                                 + 0.0209613802691423 w_k1_MSFT_s011
                                 + 0.0287252167337449 w_k1_MSFT_s012
                                 + 0.0447679707633222 w_k1_MSFT_s013
                                 + 0.0504064315864999 w_k1_MSFT_s014
                                 + 0.0200786649201557 w_k1_MSFT_s015
                                 + 0.015736140405727 w_k1_MSFT_s016
                                 + 0.0434590085952947 w_k1_MSFT_s017
                                 + 0.0120768758470014 w_k1_MSFT_s100
                                 + 0.0212613449379775 w_k1_MSFT_s101
                                 + 0.00922070485936268 w_k1_MSFT_s102
                                 + 0.0141745346982696 w_k1_MSFT_s103
                                 + 0.0061122067945625 w_k1_MSFT_s104
                                 + 0.008892968381506 w_k1_MSFT_s105
                                 + 0.012861760929507 w_k1_MSFT_s106
                                 + 0.00917646689327687 w_k1_MSFT_s107
                                 + 0.0827565013233941 w_k1_MSFT_s110
                                 + 0.0596468049162556 w_k1_MSFT_s111
                                 + 0.0790060684003683 w_k1_MSFT_s112
                                 + 0.0315105760399508 w_k1_MSFT_s113
                                 + 0.0613519084696404 w_k1_MSFT_s114
                                 + 0.0222249996211912 w_k1_MSFT_s115
                                 + 0.0295154064837728 w_k1_MSFT_s116
                                 + 0.0700382216397879 w_k1_MSFT_s117
                                 + 0.0167276793435398 w_k1_PG_s000
                                 + 0.0149567535412591 w_k1_PG_s001
                                 + 0.0198906024663949 w_k1_PG_s002
                                 + 0.015979173944975 w_k1_PG_s003
                                 + 0.0128284442941029 w_k1_PG_s004
                                 + 0.0202602242784077 w_k1_PG_s005
                                 + 0.0572376402738446 w_k1_PG_s006
                                 + 0.0452822077388662 w_k1_PG_s007
                                 + 0.058897141375909 w_k1_PG_s010
                                 + 0.0212921851377729 w_k1_PG_s011
                                 + 0.0259060371071699 w_k1_PG_s012
                                 + 0.0437946375499228 w_k1_PG_s013
                                 + 0.0531574057884167 w_k1_PG_s014
                                 + 0.0186718597786006 w_k1_PG_s015
                                 + 0.0149791299278039 w_k1_PG_s016
                                 + 0.0440700135461373 w_k1_PG_s017
                                 + 0.0116070712975769 w_k1_PG_s100
                                 + 0.0195139486204195 w_k1_PG_s101
                                 + 0.00983401127424544 w_k1_PG_s102
                                 + 0.0129497378789421 w_k1_PG_s103
                                 + 0.00574921333333595 w_k1_PG_s104
                                 + 0.00865102602523506 w_k1_PG_s105
                                 + 0.0120857003869239 w_k1_PG_s106
                                 + 0.00842791743177313 w_k1_PG_s107
                                 + 0.0851058613171428 w_k1_PG_s110
                                 + 0.0613815206577506 w_k1_PG_s111
                                 + 0.0794934186467969 w_k1_PG_s112
                                 + 0.0333750821318011 w_k1_PG_s113
                                 + 0.0617006553976266 w_k1_PG_s114
                                 + 0.0213009114060077 w_k1_PG_s115
                                 + 0.027515811812194 w_k1_PG_s116
                                 + 0.0690444080355314 w_k1_PG_s117 >= 1.2
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
