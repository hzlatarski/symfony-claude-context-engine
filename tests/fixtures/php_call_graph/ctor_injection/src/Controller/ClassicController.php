<?php
namespace App\Controller;

use App\Service\SessionService;

class ClassicController
{
    private SessionService $session;

    public function __construct(SessionService $session)
    {
        $this->session = $session;
    }

    public function run(): void
    {
        $this->session->start();
    }
}
