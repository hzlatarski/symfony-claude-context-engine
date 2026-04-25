<?php
namespace App\Controller;

use App\Service\SessionService;

class PromotedController
{
    public function __construct(
        private readonly SessionService $session,
    ) {}

    public function run(): void
    {
        $this->session->start();

        $unknown = $this->session;
        $unknown->dynamic();
    }
}
